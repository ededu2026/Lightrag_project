from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:3b", alias="OLLAMA_MODEL")
    lightrag_working_dir: Path = Field(default=Path("rag_storage"), alias="LIGHTRAG_WORKING_DIR")
    lightrag_query_mode: str = Field(default="mix", alias="LIGHTRAG_QUERY_MODE")
    stella_model_name: str = Field(
        default="dunzhang/stella_en_400M_v5",
        alias="STELLA_MODEL_NAME",
    )
    stella_revision: str | None = Field(default=None, alias="STELLA_REVISION")
    stella_query_prompt_name: str = Field(
        default="s2p_query",
        alias="STELLA_QUERY_PROMPT_NAME",
    )
    stella_device: str = Field(default="auto", alias="STELLA_DEVICE")
    stella_batch_size: int = Field(default=4, alias="STELLA_BATCH_SIZE")
    lightrag_llm_timeout: int = Field(default=1800, alias="LLM_TIMEOUT")
    lightrag_llm_max_async: int = Field(default=1, alias="MAX_ASYNC")
    lightrag_embedding_timeout: int = Field(default=600, alias="EMBEDDING_TIMEOUT")
    lightrag_embedding_max_async: int = Field(default=1, alias="EMBEDDING_FUNC_MAX_ASYNC")
    lightrag_embedding_batch_num: int = Field(default=2, alias="EMBEDDING_BATCH_NUM")
    huggingface_token: str | None = Field(default=None, alias="HF_TOKEN")
    raw_data_dir: Path = Field(default=Path("data"), alias="RAW_DATA_DIR")
    data_dir: Path = Field(default=Path("parsed_data"), alias="DATA_DIR")
    parsed_assets_dir: Path = Field(default=Path("parsed_assets"), alias="PARSED_ASSETS_DIR")
    enable_image_ocr: bool = Field(default=True, alias="ENABLE_IMAGE_OCR")
    enable_page_ocr_fallback: bool = Field(default=False, alias="ENABLE_PAGE_OCR_FALLBACK")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
