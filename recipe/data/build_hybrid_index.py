#!/usr/bin/env python3
"""从清洗后的训练集与验证集构建 SQLite + FAISS 混合索引。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def _context_from_extra_info(extra_info: dict[str, Any]) -> dict[str, Any]:
    return extra_info["tools_kwargs"]["search"]["create_kwargs"]["context"]


def collect_documents(parquet_paths: list[Path]) -> dict[str, tuple[str, tuple[str, ...]]]:
    """按大小写无关标题去重；重复标题保留正文更完整的一份。"""
    documents: dict[str, tuple[str, tuple[str, ...]]] = {}
    for path in parquet_paths:
        table = pq.read_table(path, columns=["extra_info"])
        logger.info("读取 %s：%d 条样本", path, table.num_rows)
        for row in table.to_pylist():
            context = _context_from_extra_info(row["extra_info"])
            titles = list(context["title"])
            sentence_groups = list(context["sentences"])
            if len(titles) != len(sentence_groups):
                raise ValueError(f"title/sentences 长度不一致：{path}")
            for title, sentences in zip(titles, sentence_groups, strict=True):
                normalized_title = str(title).strip()
                normalized_sentences = tuple(str(value).strip() for value in sentences if str(value).strip())
                if not normalized_title or not normalized_sentences:
                    continue
                key = normalized_title.casefold()
                previous = documents.get(key)
                if previous is None or len(" ".join(normalized_sentences)) > len(" ".join(previous[1])):
                    documents[key] = (normalized_title, normalized_sentences)
    if not documents:
        raise ValueError("没有从 Parquet 中提取到有效段落")
    return documents


def build_sqlite(documents: dict[str, tuple[str, tuple[str, ...]]], path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL COLLATE NOCASE UNIQUE,
                text TEXT NOT NULL,
                sentences_json TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                title, text, content='documents', content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        rows = [
            (title, " ".join(sentences), json.dumps(sentences, ensure_ascii=False))
            for title, sentences in sorted(documents.values(), key=lambda item: item[0].casefold())
        ]
        connection.executemany(
            "INSERT INTO documents(title, text, sentences_json) VALUES (?, ?, ?)", rows
        )
        connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')")
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", "hybrid-1"),
                ("document_count", str(len(rows))),
                ("created_at", datetime.now(timezone.utc).isoformat()),
            ],
        )
        connection.execute("INSERT INTO documents_fts(documents_fts) VALUES ('optimize')")
        connection.commit()
    finally:
        connection.close()


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


def encode_corpus(
    database_path: Path,
    model_path: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=dtype).to(device)
    model.eval()
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute("SELECT id, title, text FROM documents ORDER BY id").fetchall()
    finally:
        connection.close()

    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        passages = [f"{title}\n{text}" for _, title, text in batch]
        encoded = tokenizer(
            passages,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            pooled = last_token_pool(hidden, encoded["attention_mask"])
            embeddings = functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(embeddings.float().cpu().numpy())
        logger.info("向量化进度：%d/%d", min(start + batch_size, len(rows)), len(rows))
    ids = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    return np.ascontiguousarray(np.concatenate(all_embeddings).astype(np.float32)), ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/hotpotqa_v3_2k/train.parquet"))
    parser.add_argument("--validation", type=Path, default=Path("data/hotpotqa_v3_2k/validation.parquet"))
    parser.add_argument("--model-path", help="Qwen3-Embedding-0.6B 本地目录；省略时从 ModelScope 下载")
    parser.add_argument("--model-id", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--model-cache-dir", type=Path, default=Path("data/modelscope"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/hotpotqa_v3_2k/hybrid_index"))
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if args.batch_size < 1 or args.max_length < 32:
        raise ValueError("batch-size 必须大于 0，max-length 必须至少为 32")
    for path in (args.train, args.validation):
        if not path.is_file():
            raise FileNotFoundError(path)
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("请先安装 faiss-cpu：pip install faiss-cpu") from exc

    model_path = args.model_path
    if not model_path:
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise ImportError("自动下载模型需要 modelscope") from exc
        args.model_cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = snapshot_download(args.model_id, cache_dir=str(args.model_cache_dir.resolve()))
        logger.info("Embedding 模型目录：%s", model_path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "documents.sqlite"
    index_path = output_dir / "documents.faiss"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (database_path, index_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"索引文件已经存在；如需重建请添加 --overwrite：{existing}")
    for path in existing:
        path.unlink()

    documents = collect_documents([args.train, args.validation])
    logger.info("共提取 %d 个标题去重后的段落", len(documents))
    build_sqlite(documents, database_path)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings, ids = encode_corpus(
        database_path, model_path, device, args.batch_size, args.max_length
    )
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(embeddings.shape[1]))
    index.add_with_ids(embeddings, ids)
    temporary_index = index_path.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(temporary_index))
    os.replace(temporary_index, index_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "hybrid-1",
                "train_file": str(args.train.resolve()),
                "validation_file": str(args.validation.resolve()),
                "model_path": str(Path(model_path).expanduser().resolve()),
                "document_count": len(documents),
                "embedding_dimension": int(embeddings.shape[1]),
                "distance": "inner_product_on_l2_normalized_vectors",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"混合索引构建完成：{output_dir}")
    print(f"去重段落数：{len(documents):,}；向量维度：{embeddings.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
