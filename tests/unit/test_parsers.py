from pathlib import Path

import pytest

from app.infrastructure.parsers import DoclingAdapter


def test_docling_adapter_uses_explicit_rapidocr_options(tmp_path: Path) -> None:
    pipeline_options = pytest.importorskip("docling.datamodel.pipeline_options")
    adapter = DoclingAdapter(
        tmp_path,
        ocr_engine="rapidocr",
        rapidocr_backend="torch",
        ocr_languages=["chinese"],
    )

    options = adapter._pdf_pipeline_options()

    assert isinstance(options.ocr_options, pipeline_options.RapidOcrOptions)
    assert options.ocr_options.backend == "torch"
    assert options.ocr_options.lang == ["chinese"]
    assert options.artifacts_path == tmp_path
    assert "+ocr-rapidocr-torch-chinese" in adapter.version


def test_docling_adapter_can_explicitly_fall_back_to_auto_ocr() -> None:
    pipeline_options = pytest.importorskip("docling.datamodel.pipeline_options")
    adapter = DoclingAdapter(
        ocr_engine="auto",
        ocr_languages=["chinese"],
    )

    options = adapter._pdf_pipeline_options()

    assert isinstance(options.ocr_options, pipeline_options.OcrAutoOptions)
    assert options.ocr_options.lang == ["chinese"]
    assert "+ocr-auto-chinese" in adapter.version
