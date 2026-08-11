from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

from app.domain.models import (
    ImageKind,
    ParsedBlock,
    ParsedImage,
    ParserUnavailableError,
    UnifiedDocument,
)


class PlainTextParser:
    """Small parser used by tests and local contract checks."""

    name = "plain_text"
    version = "1"

    def supports(self, media_type: str, filename: str) -> bool:
        return filename.lower().endswith((".txt", ".md"))

    def parse(
        self,
        file_obj: BinaryIO,
        *,
        filename: str,
        document_id: str,
        version_id: str,
    ) -> UnifiedDocument:
        text = file_obj.read().decode("utf-8")
        blocks = [
            ParsedBlock(
                text=paragraph,
                page_no=1,
                section_path=[],
                ordinal=index,
            )
            for index, paragraph in enumerate(text.split("\n\n"))
            if paragraph.strip()
        ]
        return UnifiedDocument(blocks=blocks, images=[], raw={"text": text})


class DoclingAdapter:
    name = "docling"

    def __init__(
        self,
        artifacts_path: Path | None = None,
        *,
        ocr_engine: str = "rapidocr",
        rapidocr_backend: str = "torch",
        ocr_languages: list[str] | None = None,
    ) -> None:
        self.artifacts_path = artifacts_path
        self.ocr_engine = ocr_engine
        self.rapidocr_backend = rapidocr_backend
        self.ocr_languages = ocr_languages or ["chinese"]

    @property
    def version(self) -> str:
        try:
            docling_version = version("docling")
        except PackageNotFoundError:
            return "not-installed"
        ocr_version = self.ocr_engine
        if self.ocr_engine == "rapidocr":
            ocr_version = f"{ocr_version}-{self.rapidocr_backend}"
        languages = "-".join(self.ocr_languages)
        return f"{docling_version}+ocr-{ocr_version}-{languages}"

    def supports(self, media_type: str, filename: str) -> bool:
        return filename.lower().endswith((".pdf", ".docx", ".pptx"))

    def parse(
        self,
        file_obj: BinaryIO,
        *,
        filename: str,
        document_id: str,
        version_id: str,
    ) -> UnifiedDocument:
        try:
            from docling.datamodel.base_models import DocumentStream, InputFormat
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling_core.types.doc import PictureItem, TableItem, TextItem
        except ImportError as exc:
            raise ParserUnavailableError(
                "Docling is not installed; run `pip install -e .[docling]`"
            ) from exc

        options = self._pdf_pipeline_options()
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )
        stream = DocumentStream(name=filename, stream=BytesIO(file_obj.read()))
        result = converter.convert(stream)
        document = result.document

        blocks: list[ParsedBlock] = []
        images: list[ParsedImage] = []
        section_path: list[str] = []
        picture_pages: set[int] = set()
        for item, level in document.iterate_items():
            page_no, bbox = _provenance(item)
            label = str(getattr(item, "label", "")).lower()
            if isinstance(item, TextItem):
                text = str(getattr(item, "text", "")).strip()
                if not text:
                    continue
                if "section_header" in label or "title" in label:
                    level_index = max(0, min(int(level or 1) - 1, 9))
                    section_path = section_path[:level_index] + [text]
                    continue
                blocks.append(
                    ParsedBlock(
                        text=text,
                        page_no=page_no,
                        section_path=list(section_path),
                        ordinal=len(blocks),
                        bbox=bbox,
                        kind=label or "paragraph",
                    )
                )
            elif isinstance(item, TableItem):
                try:
                    text = item.export_to_dataframe(doc=document).to_markdown(index=False)
                except Exception:
                    text = str(getattr(item, "text", ""))
                if text.strip():
                    blocks.append(
                        ParsedBlock(
                            text=text,
                            page_no=page_no,
                            section_path=list(section_path),
                            ordinal=len(blocks),
                            bbox=bbox,
                            kind="table",
                        )
                    )
            elif isinstance(item, PictureItem):
                image = item.get_image(document)
                if image is None:
                    continue
                content = _pil_to_png(image)
                caption = _picture_caption(item, document)
                images.append(
                    ParsedImage(
                        content=content,
                        media_type="image/png",
                        page_no=page_no,
                        ordinal=len(images),
                        kind=ImageKind.FIGURE,
                        caption=caption,
                        bbox=bbox,
                        section_path=list(section_path),
                    )
                )
                if page_no:
                    picture_pages.add(page_no)

        for page_key, page in document.pages.items():
            page_no = int(getattr(page, "page_no", page_key))
            page_image = getattr(getattr(page, "image", None), "pil_image", None)
            if page_image is None or page_no in picture_pages:
                continue
            images.append(
                ParsedImage(
                    content=_pil_to_png(page_image),
                    media_type="image/png",
                    page_no=page_no,
                    ordinal=len(images),
                    kind=ImageKind.PAGE_RENDER,
                    section_path=[],
                )
            )
        try:
            raw = document.export_to_dict()
        except AttributeError:
            raw = {"markdown": document.export_to_markdown()}
        return UnifiedDocument(blocks=blocks, images=images, raw=raw)

    def _pdf_pipeline_options(self) -> Any:
        from docling.datamodel.pipeline_options import (
            OcrAutoOptions,
            PdfPipelineOptions,
            RapidOcrOptions,
            TableStructureV2Options,
        )

        options = PdfPipelineOptions()
        options.artifacts_path = self.artifacts_path
        options.do_ocr = True
        if self.ocr_engine == "rapidocr":
            options.ocr_options = RapidOcrOptions(
                lang=self.ocr_languages,
                backend=self.rapidocr_backend,
            )
        else:
            options.ocr_options = OcrAutoOptions(lang=self.ocr_languages)
        options.do_table_structure = True
        options.table_structure_options = TableStructureV2Options()
        options.generate_picture_images = True
        options.generate_page_images = True
        return options


class MinerUAdapter:
    name = "mineru_http"
    version = "skeleton-v1"

    def __init__(self, base_url: str, enabled: bool = False) -> None:
        self.base_url = base_url
        self.enabled = enabled

    def supports(self, media_type: str, filename: str) -> bool:
        return filename.lower().endswith(".pdf")

    def parse(
        self,
        file_obj: BinaryIO,
        *,
        filename: str,
        document_id: str,
        version_id: str,
    ) -> UnifiedDocument:
        if not self.enabled:
            raise ParserUnavailableError(
                "MinerU adapter is disabled pending license approval"
            )
        raise ParserUnavailableError(
            "MinerU HTTP contract must be configured after license approval"
        )


def _provenance(item: Any) -> tuple[int | None, list[float] | None]:
    provenance = getattr(item, "prov", None) or []
    if not provenance:
        return None, None
    first = provenance[0]
    page_no = getattr(first, "page_no", None)
    bbox = getattr(first, "bbox", None)
    if bbox is None:
        return page_no, None
    values = [
        getattr(bbox, key, None)
        for key in ("l", "t", "r", "b")
    ]
    return page_no, values if all(value is not None for value in values) else None


def _picture_caption(item: Any, document: Any) -> str | None:
    try:
        value = item.caption_text(document)
        return str(value)[:300] if value else None
    except (AttributeError, TypeError):
        return None


def _pil_to_png(image: Any) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
