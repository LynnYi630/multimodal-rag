from app.domain.models import ParsedBlock
from app.domain.services import (
    StructureAwareChunker,
    deterministic_node_id,
    reciprocal_rank_fusion,
)


def test_deterministic_node_id_is_stable_and_sensitive() -> None:
    first = deterministic_node_id("version", "text", 1, 2, "hash")
    assert first == deterministic_node_id("version", "text", 1, 2, "hash")
    assert first != deterministic_node_id("version", "text", 1, 3, "hash")


def test_rrf_merges_rankings_deterministically() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert fused[0][0] == "b"
    assert {item_id for item_id, _ in fused} == {"a", "b", "c"}


def test_chunker_respects_structural_boundary() -> None:
    blocks = [
        ParsedBlock("第一段内容。" * 20, 1, ["第一章"], 0),
        ParsedBlock("第二段内容。" * 20, 1, ["第一章"], 1),
        ParsedBlock("第三段内容。" * 20, 2, ["第二章"], 2),
    ]
    chunks = StructureAwareChunker(target_tokens=500).chunk(blocks)
    assert len(chunks) == 2
    assert chunks[0].section_path == ["第一章"]
    assert chunks[1].section_path == ["第二章"]

