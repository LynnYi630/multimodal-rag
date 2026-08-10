from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence

from app.domain.models import ParsedBlock

ID_NAMESPACE = uuid.UUID("6f709ee7-1cc9-45df-8e72-59280632039e")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def deterministic_version_id(
    document_id: str,
    file_hash: str,
    parser_version: str,
    embedding_model: str,
) -> str:
    value = f"{document_id}:{file_hash}:{parser_version}:{embedding_model}"
    return str(uuid.uuid5(ID_NAMESPACE, value))


def deterministic_node_id(
    version_id: str,
    node_type: str,
    page_no: int | None,
    ordinal: int,
    content_hash: str,
) -> str:
    return str(
        uuid.uuid5(
            ID_NAMESPACE,
            f"{version_id}:{node_type}:{page_no}:{ordinal}:{content_hash}",
        )
    )


def deterministic_asset_id(version_id: str, object_key: str, content_hash: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"{version_id}:{object_key}:{content_hash}"))


def safe_filename(filename: str) -> str:
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", name).strip("._")
    return cleaned[:180] or "document"


def estimate_tokens(text: str) -> int:
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    non_chinese = len(re.findall(r"\w+", text))
    return chinese + math.ceil(non_chinese * 1.25)


class StructureAwareChunker:
    def __init__(
        self,
        target_tokens: int = 600,
        hard_limit: int = 900,
        min_tokens: int = 80,
    ) -> None:
        self.target_tokens = target_tokens
        self.hard_limit = hard_limit
        self.min_tokens = min_tokens

    def chunk(self, blocks: Sequence[ParsedBlock]) -> list[ParsedBlock]:
        chunks: list[ParsedBlock] = []
        pending: list[ParsedBlock] = []
        pending_tokens = 0

        def flush() -> None:
            nonlocal pending_tokens
            if not pending:
                return
            first = pending[0]
            text = "\n\n".join(block.text.strip() for block in pending if block.text.strip())
            if text:
                chunks.append(
                    ParsedBlock(
                        text=text,
                        page_no=first.page_no,
                        section_path=list(first.section_path),
                        ordinal=len(chunks),
                        bbox=first.bbox if len(pending) == 1 else None,
                        kind="chunk",
                    )
                )
            pending.clear()
            pending_tokens = 0

        for block in blocks:
            if not block.text.strip():
                continue
            units = self._split_oversized(block)
            for unit in units:
                unit_tokens = estimate_tokens(unit.text)
                boundary_changed = bool(
                    pending
                    and (
                        pending[-1].section_path != unit.section_path
                        or pending[-1].page_no != unit.page_no
                    )
                )
                if pending and (
                    boundary_changed
                    or pending_tokens + unit_tokens > self.hard_limit
                    or pending_tokens >= self.target_tokens
                ):
                    flush()
                pending.append(unit)
                pending_tokens += unit_tokens
        flush()

        if len(chunks) > 1 and estimate_tokens(chunks[-1].text) < self.min_tokens:
            previous = chunks[-2]
            last = chunks[-1]
            if (
                previous.section_path == last.section_path
                and estimate_tokens(previous.text + last.text) <= self.hard_limit
            ):
                previous.text = f"{previous.text}\n\n{last.text}"
                chunks.pop()
        for index, chunk in enumerate(chunks):
            chunk.ordinal = index
        return chunks

    def _split_oversized(self, block: ParsedBlock) -> list[ParsedBlock]:
        if estimate_tokens(block.text) <= self.hard_limit:
            return [block]
        paragraphs = re.split(r"(?<=[。！？.!?])\s+|\n+", block.text)
        pieces: list[ParsedBlock] = []
        current: list[str] = []
        current_tokens = 0
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            paragraph_tokens = estimate_tokens(paragraph)
            if current and current_tokens + paragraph_tokens > self.hard_limit:
                pieces.append(self._copy_block(block, " ".join(current), len(pieces)))
                current, current_tokens = [], 0
            if paragraph_tokens > self.hard_limit:
                char_limit = max(200, self.hard_limit * 2)
                for start in range(0, len(paragraph), char_limit):
                    pieces.append(
                        self._copy_block(block, paragraph[start : start + char_limit], len(pieces))
                    )
            else:
                current.append(paragraph)
                current_tokens += paragraph_tokens
        if current:
            pieces.append(self._copy_block(block, " ".join(current), len(pieces)))
        return pieces

    @staticmethod
    def _copy_block(block: ParsedBlock, text: str, ordinal: int) -> ParsedBlock:
        return ParsedBlock(
            text=text,
            page_no=block.page_no,
            section_path=list(block.section_path),
            ordinal=ordinal,
            bbox=block.bbox,
            kind=block.kind,
        )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
            best_rank[item_id] = min(best_rank.get(item_id, rank), rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))
    return ordered[:limit] if limit is not None else ordered


def normalize(values: Iterable[float]) -> list[float]:
    result = list(values)
    norm = math.sqrt(sum(value * value for value in result))
    return [value / norm for value in result] if norm else result

