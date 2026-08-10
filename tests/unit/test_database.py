from app.infrastructure.database import DocumentORM


def test_document_source_type_accepts_standard_office_mime_types() -> None:
    column = DocumentORM.__table__.c.source_type

    assert column.type.length >= len(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
