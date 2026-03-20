from __future__ import annotations

import json
from typing import Any, AsyncIterator, Literal, TypedDict

from langgraph.graph import END, StateGraph
from ollama import AsyncClient

from app.retrieval import LightRAGStore


class WorkflowState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    intent: str
    contexts: list[dict[str, Any]]
    answer: str
    node: str
    path: list[str]


class QAWorkflow:
    def __init__(self, store: LightRAGStore, ollama_base_url: str, ollama_model: str) -> None:
        self.store = store
        self.client = AsyncClient(host=ollama_base_url)
        self.model = ollama_model
        self.graph = self._build_graph()
        self.greeting_graph = self._build_greeting_graph()

    async def greetings(self) -> dict[str, Any]:
        return await self.greeting_graph.ainvoke({})

    async def invoke(self, question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        return await self.graph.ainvoke(
            {
                "question": question,
                "history": self._normalize_history(history),
            }
        )

    async def stream(self, question: str, history: list[dict[str, str]] | None = None) -> AsyncIterator[dict[str, Any]]:
        state: WorkflowState = {
            "question": question,
            "history": self._normalize_history(history),
        }
        intent_state = await self.intent_classifier(state)
        state.update(intent_state)
        route = self.route_intent(state)
        if route == "general_question":
            async for event in self.stream_general_question(state):
                yield event
            return
        async for event in self.stream_lightrag_qa(state):
            yield event

    def _build_graph(self):
        workflow = StateGraph(WorkflowState)
        workflow.add_node("intent_classifier", self.intent_classifier)
        workflow.add_node("general_question", self.general_question)
        workflow.add_node("lightrag_qa", self.lightrag_qa)
        workflow.set_entry_point("intent_classifier")
        workflow.add_conditional_edges(
            "intent_classifier",
            self.route_intent,
            {
                "general_question": "general_question",
                "lightrag_qa": "lightrag_qa",
            },
        )
        workflow.add_edge("general_question", END)
        workflow.add_edge("lightrag_qa", END)
        return workflow.compile()

    def _build_greeting_graph(self):
        workflow = StateGraph(WorkflowState)
        workflow.add_node("greetings", self.greetings_node)
        workflow.set_entry_point("greetings")
        workflow.add_edge("greetings", END)
        return workflow.compile()

    async def greetings_node(self, state: WorkflowState) -> WorkflowState:
        del state
        response = await self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a concise welcome message for a document Q&A application. "
                        "Invite the user to ask about the loaded documents and mention that follow-up questions are supported. "
                        "Answer in concise English."
                    ),
                }
            ],
            options={"temperature": 0.4},
        )
        return {
            "answer": response["message"]["content"],
            "intent": "greetings",
            "node": "greetings",
            "path": ["greetings"],
            "contexts": [],
        }

    async def intent_classifier(self, state: WorkflowState) -> WorkflowState:
        question = state["question"]
        history = state.get("history", [])
        lowered = question.strip().lower()
        doc_cues = (
            "document",
            "docs",
            "pdf",
            "file",
            "files",
            "paper",
            "papers",
            "dataset",
            "datasets",
            "method",
            "methods",
            "metric",
            "metrics",
            "problem",
            "problems",
            "solution",
            "solutions",
            "mentioned",
            "according to",
            "in the documents",
            "in these documents",
            "from the documents",
            "based on the documents",
        )
        if any(cue in lowered for cue in doc_cues):
            return {"intent": "lightrag_qa"}

        response = await self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user request into exactly one label: lightrag_qa or general_question. "
                        "Use lightrag_qa when the user is asking about the ingested documents, their topics, "
                        "or asking a follow-up to a previous document-grounded answer. "
                        "Use general_question for casual conversation or general world knowledge unrelated to the documents. "
                        "Return JSON only in the form {\"intent\":\"lightrag_qa\"} or {\"intent\":\"general_question\"}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "history": history[-8:],
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
            options={"temperature": 0},
            format="json",
        )
        content = response["message"]["content"]
        try:
            payload = json.loads(content)
            intent = payload.get("intent", "lightrag_qa")
        except json.JSONDecodeError:
            intent = "lightrag_qa"
        if intent not in {"lightrag_qa", "general_question"}:
            intent = "lightrag_qa"
        return {"intent": intent}

    def route_intent(self, state: WorkflowState) -> Literal["general_question", "lightrag_qa"]:
        return "general_question" if state.get("intent") == "general_question" else "lightrag_qa"

    async def lightrag_qa(self, state: WorkflowState) -> WorkflowState:
        result = await self.store.query(
            state["question"],
            history=state.get("history", []),
        )
        return {
            "answer": result["answer"],
            "intent": "lightrag_qa",
            "node": "lightrag_qa",
            "path": ["intent_classifier", "lightrag_qa"],
            "contexts": result.get("contexts", []),
        }

    async def general_question(self, state: WorkflowState) -> WorkflowState:
        response = await self._general_question_chat(state)
        return {
            "answer": response["message"]["content"],
            "intent": "general_question",
            "node": "general_question",
            "path": ["intent_classifier", "general_question"],
            "contexts": [],
        }

    async def stream_general_question(self, state: WorkflowState) -> AsyncIterator[dict[str, Any]]:
        answer_parts: list[str] = []
        response_stream = await self._general_question_chat(state, stream=True)
        async for chunk in response_stream:
            token = chunk["message"]["content"]
            if not token:
                continue
            answer_parts.append(token)
            yield {
                "type": "token",
                "token": token,
                "intent": "general_question",
                "node": "general_question",
                "path": ["intent_classifier", "general_question"],
            }
        yield {
            "type": "answer",
            "answer": "".join(answer_parts),
            "intent": "general_question",
            "node": "general_question",
            "path": ["intent_classifier", "general_question"],
            "contexts": [],
        }

    async def stream_lightrag_qa(self, state: WorkflowState) -> AsyncIterator[dict[str, Any]]:
        final_payload: dict[str, Any] | None = None
        async for event in self.store.stream_query(
            state["question"],
            history=state.get("history", []),
        ):
            if "token" in event:
                yield {
                    "type": "token",
                    "token": event["token"],
                    "intent": "lightrag_qa",
                    "node": "lightrag_qa",
                    "path": ["intent_classifier", "lightrag_qa"],
                }
                continue
            final_payload = event
        if final_payload is None:
            final_payload = {"answer": "", "contexts": []}
        yield {
            "type": "answer",
            "answer": final_payload.get("answer", ""),
            "intent": "lightrag_qa",
            "node": "lightrag_qa",
            "path": ["intent_classifier", "lightrag_qa"],
            "contexts": final_payload.get("contexts", []),
        }

    async def _general_question_chat(self, state: WorkflowState, stream: bool = False):
        response = await self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user directly in concise English. "
                        "You can use the recent conversation history for follow-up questions. "
                        "Do not mention document retrieval or the knowledge base unless the user asks."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": state["question"],
                            "history": state.get("history", [])[-8:],
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
            options={"temperature": 0.2},
            stream=stream,
        )
        return response

    @staticmethod
    def _normalize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in history or []:
            role = item.get("role", "").strip()
            content = item.get("content", "").strip()
            if role and content:
                normalized.append({"role": role, "content": content})
        return normalized
