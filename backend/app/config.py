from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always resolve .env relative to this file (backend/app/config.py → backend/.env)
_ENV_PATH = str(Path(__file__).resolve().parent.parent / ".env")


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    extraction_engine: str = "docling"
    upload_dir: str = "./uploads"
    max_upload_mb: int = 50
    agent_model: str = "claude-opus-4-5"
    agent_max_turns: int = 40
    hybrid_url: str = "http://localhost:5002"
    # If true, the opendataloader_hybrid extractor will spawn the
    # `opendataloader-pdf-hybrid` server on demand if nothing is listening at
    # `hybrid_url`. Set to false in environments where the backend runs as a
    # managed service.
    hybrid_autostart: bool = True
    # How long to wait for the spawned backend to become healthy. Loading the
    # Docling-Fast layout model on first run can take ~30–60 s.
    hybrid_startup_timeout: int = 120

    # Comma-separated list; "*" keeps the open default but is unsafe in prod.
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Render-endpoint clamp. High DPI on a multi-page PDF can exhaust memory.
    render_min_dpi: int = 36
    render_max_dpi: int = 300

    # Whether /extract returns raw exception details in the response body.
    # Keep False in any environment reachable from untrusted clients.
    expose_error_details: bool = False

    # VLM settings. vlm_model is a LiteLLM model string ("<provider>/<model>").
    # Any vision-capable model supported by LiteLLM works.
    vlm_model: str = "gemini/gemini-2.5-flash"
    vlm_dpi: int = 200  # higher DPI = better OCR, more tokens
    vlm_max_concurrent: int = 4  # parallel page calls

    # API keys for VLM / agent providers (set in .env)
    gemini_api_key: str = ""

    # OpenDataLoader extraction options (apply to both `opendataloader` and
    # `opendataloader_hybrid` engines). Values mirror the upstream CLI flags.
    odl_table_method: str = "default"      # default | cluster
    odl_reading_order: str = "xycut"       # xycut | off
    odl_use_struct_tree: bool = False      # use tagged-PDF structure tree
    odl_keep_line_breaks: bool = False     # preserve original line breaks

    # LiteParse (LlamaIndex) — local PDFium text + bundled Tesseract OCR.
    # OCR only kicks in on pages with no extractable text layer; leaving it on
    # lets the engine handle scanned PDFs at the cost of some latency.
    liteparse_ocr_enabled: bool = True
    liteparse_ocr_language: str = "eng"    # Tesseract language code(s), e.g. "eng+fra"
    liteparse_dpi: int = 150               # render DPI used when a page falls back to OCR

    model_config = SettingsConfigDict(env_file=_ENV_PATH, env_file_encoding="utf-8", extra="ignore")

    @field_validator("max_upload_mb")
    @classmethod
    def _positive_upload(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_upload_mb must be positive")
        return v

    @field_validator("render_max_dpi")
    @classmethod
    def _dpi_bounds(cls, v: int) -> int:
        if v < 36 or v > 600:
            raise ValueError("render_max_dpi must be between 36 and 600")
        return v

    @field_validator("odl_table_method")
    @classmethod
    def _odl_table_method_choices(cls, v: str) -> str:
        if v not in ("default", "cluster"):
            raise ValueError("odl_table_method must be 'default' or 'cluster'")
        return v

    @field_validator("odl_reading_order")
    @classmethod
    def _odl_reading_order_choices(cls, v: str) -> str:
        if v not in ("xycut", "off"):
            raise ValueError("odl_reading_order must be 'xycut' or 'off'")
        return v

    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


settings = Settings()
