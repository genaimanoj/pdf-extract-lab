from typing import Protocol, runtime_checkable

from ..schema import ExtractionResult


@runtime_checkable
class Extractor(Protocol):
    name: str

    async def extract(self, pdf_path: str) -> ExtractionResult: ...
