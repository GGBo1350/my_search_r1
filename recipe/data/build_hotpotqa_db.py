#!/usr/bin/env python3
"""构建 V3 全局检索使用的 SQLite FTS5 数据库。

支持两种语料来源：

1. Hugging Face HotpotQA distractor 的 ``DatasetDict.save_to_disk`` 目录；
2. HotpotQA 官方预处理 Wikipedia 目录中的 ``.bz2``/JSONL 段落文件。

第一种模式只合并数据集实际出现过的候选段落，不等同于官方 fullwiki；第二种模式
才用于导入官方发布的完整 Wikipedia 段落语料。
"""

from __future__ import annotations

import argparse
import bz2
import json
import logging
import os
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from datasets import DatasetDict, load_from_disk

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class CorpusDocument:
    title: str
    sentences: tuple[str, ...]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return [value]


def _normalize_document(title: Any, text: Any) -> CorpusDocument | None:
    normalized_title = str(title or "").strip()
    if not normalized_title:
        return None

    if isinstance(text, str):
        raw_sentences = [text]
    else:
        raw_sentences = _as_list(text)
    sentences = tuple(str(sentence).strip() for sentence in raw_sentences if str(sentence).strip())
    if not sentences:
        return None
    return CorpusDocument(title=normalized_title, sentences=sentences)


def iter_dataset_documents(dataset_dir: Path, splits: list[str]) -> Iterator[CorpusDocument]:
    """遍历 HotpotQA DatasetDict 中每条样本附带的候选段落。"""
    loaded = load_from_disk(str(dataset_dir))
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"需要 DatasetDict，实际得到 {type(loaded).__name__}: {dataset_dir}")

    for split in splits:
        if split not in loaded:
            raise KeyError(f"数据集中不存在 split={split!r}，可用值：{list(loaded.keys())}")
        logger.info("读取数据集 split=%s，共 %d 条样本", split, len(loaded[split]))
        for row in loaded[split]:
            context = row.get("context") or {}
            titles = _as_list(context.get("title"))
            sentence_groups = _as_list(context.get("sentences"))
            if len(titles) != len(sentence_groups):
                raise ValueError("context.title 与 context.sentences 长度不一致")
            for title, sentences in zip(titles, sentence_groups, strict=True):
                document = _normalize_document(title, sentences)
                if document is not None:
                    yield document


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="r", encoding="utf-8")


def _wiki_files(wiki_path: Path) -> list[Path]:
    if wiki_path.is_file():
        return [wiki_path]
    if not wiki_path.is_dir():
        raise FileNotFoundError(f"Wikipedia 语料路径不存在：{wiki_path}")
    files = sorted(
        path
        for path in wiki_path.rglob("*")
        if path.is_file() and (path.suffix.lower() in {".bz2", ".jsonl", ".json"})
    )
    if not files:
        raise FileNotFoundError(f"没有找到 .bz2/.jsonl/.json 语料文件：{wiki_path}")
    return files


def iter_wikipedia_documents(wiki_path: Path) -> Iterator[CorpusDocument]:
    """遍历 HotpotQA 官方预处理 Wikipedia 的逐行 JSON 段落文件。"""
    files = _wiki_files(wiki_path)
    logger.info("找到 %d 个 Wikipedia 语料文件", len(files))
    for file_index, path in enumerate(files, start=1):
        logger.info("读取 Wikipedia 文件 %d/%d：%s", file_index, len(files), path)
        with _open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"无效 JSON：{path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    continue
                document = _normalize_document(record.get("title"), record.get("text"))
                if document is not None:
                    yield document


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-262144;

        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL COLLATE NOCASE UNIQUE,
            text TEXT NOT NULL,
            sentences_json TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE documents_fts USING fts5(
            title,
            text,
            content='documents',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def build_database(
    documents: Iterable[CorpusDocument],
    output_path: Path,
    source: str,
    overwrite: bool = False,
    commit_every: int = 10_000,
    max_documents: int | None = None,
) -> tuple[int, int]:
    """写入并索引语料，返回 ``(读取段落数, 去重后段落数)``。"""
    if commit_every < 1:
        raise ValueError("commit_every 必须为正整数")
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents 必须为正整数")

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"数据库已经存在：{output_path}；如需覆盖请传入 --overwrite")

    temporary_path = output_path.with_name(f".{output_path.name}.building-{os.getpid()}")
    if temporary_path.exists():
        temporary_path.unlink()

    connection = sqlite3.connect(temporary_path)
    raw_count = 0
    try:
        _initialize_database(connection)
        statement = """
            INSERT INTO documents(title, text, sentences_json)
            VALUES (?, ?, ?)
            ON CONFLICT(title) DO UPDATE SET
                text=excluded.text,
                sentences_json=excluded.sentences_json
            WHERE length(excluded.text) > length(documents.text)
        """
        for document in documents:
            if max_documents is not None and raw_count >= max_documents:
                break
            body = " ".join(document.sentences)
            connection.execute(
                statement,
                (document.title, body, json.dumps(document.sentences, ensure_ascii=False)),
            )
            raw_count += 1
            if raw_count % commit_every == 0:
                connection.commit()
                logger.info("已读取 %s 个段落", f"{raw_count:,}")

        connection.commit()
        unique_count = int(connection.execute("SELECT count(*) FROM documents").fetchone()[0])
        if unique_count == 0:
            raise ValueError("没有可写入数据库的有效段落")

        logger.info("开始构建 SQLite FTS5 索引：%s 个去重段落", f"{unique_count:,}")
        connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')")
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", SCHEMA_VERSION),
                ("source", source),
                ("raw_document_count", str(raw_count)),
                ("document_count", str(unique_count)),
                ("created_at", datetime.now(timezone.utc).isoformat()),
            ],
        )
        connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('optimize')")
        connection.commit()
    except BaseException:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    os.replace(temporary_path, output_path)
    return raw_count, unique_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset-dir",
        type=Path,
        help="HotpotQA DatasetDict.save_to_disk 目录；构建 distractor 候选段落并集",
    )
    source_group.add_argument(
        "--wiki-path",
        type=Path,
        help="HotpotQA 官方预处理 Wikipedia 的目录或单个 .bz2/JSONL 文件",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hotpotqa_global/hotpotqa.sqlite"),
        help="输出 SQLite 数据库",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="dataset 模式需要合并的 split",
    )
    parser.add_argument("--commit-every", type=int, default=10_000)
    parser.add_argument("--max-documents", type=int, help="只构建前 N 个段落，用于冒烟检查")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.dataset_dir is not None:
        source = f"hotpotqa_dataset:{args.dataset_dir}"
        documents = iter_dataset_documents(args.dataset_dir, args.splits)
    else:
        source = f"hotpotqa_official_wikipedia:{args.wiki_path}"
        documents = iter_wikipedia_documents(args.wiki_path)

    raw_count, unique_count = build_database(
        documents=documents,
        output_path=args.output,
        source=source,
        overwrite=args.overwrite,
        commit_every=args.commit_every,
        max_documents=args.max_documents,
    )
    output_path = args.output.expanduser().resolve()
    print(f"数据库构建完成：{output_path}")
    print(f"读取段落：{raw_count:,}")
    print(f"标题去重后：{unique_count:,}")
    print(f"文件大小：{output_path.stat().st_size / (1024**3):.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
