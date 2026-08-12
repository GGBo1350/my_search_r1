"""Local HotpotQA paragraph search for Search-R1 V3.

The default ``sample`` backend searches the distractor paragraphs attached to
the current example. The optional ``sqlite`` backend searches a shared global
HotpotQA/Wikipedia FTS5 database. The ``hybrid`` backend combines SQLite FTS5
and a FAISS dense index through reciprocal-rank fusion (RRF). No backend
receives supporting-fact labels or accesses the network during training.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

# Importing the parser registers ``search_r1_v3`` with verl.
import recipe.core.my_tool_parser  # noqa: F401
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]", re.IGNORECASE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(str(text).lower()) if token not in _STOP_WORDS]


@dataclass(frozen=True)
class _Document:
    title: str
    sentences: tuple[str, ...]
    term_counts: Counter[str]
    length: int
    doc_id: int = 0


@dataclass(frozen=True)
class _SearchIndex:
    documents: tuple[_Document, ...]
    document_frequencies: Counter[str]
    average_length: float


class HotpotSearchTool(BaseTool):
    """Search sample context, SQLite FTS5, or a SQLite + FAISS hybrid corpus."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None):
        super().__init__(config, tool_schema)
        self.backend = str(config.get("backend", "sample")).strip().lower()
        if self.backend not in {"sample", "sqlite", "hybrid"}:
            raise ValueError(f"Unsupported HotpotSearchTool backend: {self.backend!r}")
        self._indices: dict[str, _SearchIndex] = {}
        self._instances: set[str] = set()
        self._database: sqlite3.Connection | None = None
        self._database_lock = threading.RLock()
        self._embedding_lock = threading.RLock()
        self._reranker_lock = threading.RLock()
        self._faiss_index: Any = None
        self._embedding_model: Any = None
        self._embedding_tokenizer: Any = None
        self._reranker_model: Any = None
        self._reranker_tokenizer: Any = None
        self.topk = max(1, int(config.get("topk", 1)))
        self.title_weight = max(1, int(config.get("title_weight", 3)))
        self.k1 = float(config.get("bm25_k1", 1.5))
        self.b = float(config.get("bm25_b", 0.75))
        self.max_document_chars = max(200, int(config.get("max_document_chars", 2400)))
        # Keep the tool-side limit in sync with the reward-side query score
        # (my_reward.py basic_valid: 1 < len(query) <= MAX_QUERY_CHARS) so an
        # over-long query is rejected here instead of silently degrading
        # retrieval. 70 characters covers a concise entity-focused query with
        # a relation and a qualifier, while still rejecting verbose repetition.
        self.max_query_chars = max(1, int(config.get("max_query_chars", 70)))
        if self.backend in {"sqlite", "hybrid"}:
            configured_path = os.environ.get("HOTPOTQA_DB_PATH") or config.get("database_path")
            if not configured_path:
                raise ValueError(f"{self.backend} backend requires database_path or HOTPOTQA_DB_PATH")
            self.database_path = Path(os.path.expandvars(str(configured_path))).expanduser().resolve()
            self._database = self._open_database(self.database_path)
            if self.backend == "hybrid":
                configured_index = os.environ.get("HOTPOTQA_FAISS_INDEX_PATH") or config.get("faiss_index_path")
                configured_model = os.environ.get("HOTPOTQA_EMBEDDING_MODEL_PATH") or config.get(
                    "embedding_model_path"
                )
                if not configured_index or not configured_model:
                    raise ValueError(
                        "hybrid backend requires faiss_index_path and embedding_model_path "
                        "(or their HOTPOTQA_* environment variables)"
                    )
                self.faiss_index_path = Path(os.path.expandvars(str(configured_index))).expanduser().resolve()
                self.embedding_model_path = os.path.expandvars(str(configured_model))
                self.embedding_device = str(
                    os.environ.get("HOTPOTQA_EMBEDDING_DEVICE") or config.get("embedding_device", "cpu")
                )
                self.embedding_max_length = max(32, int(config.get("embedding_max_length", 512)))
                self.hybrid_candidate_topk = max(self.topk, int(config.get("hybrid_candidate_topk", 20)))
                self.rrf_k = max(1, int(config.get("rrf_k", 60)))
                self.bm25_weight = max(0.0, float(config.get("bm25_weight", 1.0)))
                self.dense_weight = max(0.0, float(config.get("dense_weight", 1.0)))
                if self.bm25_weight == 0.0 and self.dense_weight == 0.0:
                    raise ValueError("bm25_weight and dense_weight cannot both be zero")
                self.query_instruction = str(
                    config.get(
                        "embedding_query_instruction",
                        "Given a web search query, retrieve relevant passages that answer the query",
                    )
                ).strip()
                self.reranker_enabled = bool(config.get("reranker_enabled", False))
                if self.reranker_enabled:
                    configured_reranker = os.environ.get("HOTPOTQA_RERANKER_MODEL_PATH") or config.get(
                        "reranker_model_path"
                    )
                    if not configured_reranker:
                        raise ValueError(
                            "reranker_enabled requires reranker_model_path or HOTPOTQA_RERANKER_MODEL_PATH"
                        )
                    self.reranker_model_path = os.path.expandvars(str(configured_reranker))
                    self.reranker_device = str(
                        os.environ.get("HOTPOTQA_RERANKER_DEVICE")
                        or config.get("reranker_device", self.embedding_device)
                    )
                    self.reranker_topk = max(self.topk, int(config.get("reranker_topk", 5)))
                    self.reranker_max_length = max(128, int(config.get("reranker_max_length", 1024)))
                    self.reranker_batch_size = max(1, int(config.get("reranker_batch_size", 5)))
                    self.reranker_instruction = str(
                        config.get("reranker_instruction", self.query_instruction)
                    ).strip()
                self._faiss_index = self._open_faiss_index(self.faiss_index_path)
        else:
            self.database_path = None

    @staticmethod
    def _open_database(database_path: Path) -> sqlite3.Connection:
        if not database_path.is_file():
            raise FileNotFoundError(f"HotpotQA search database does not exist: {database_path}")
        uri = f"{database_path.as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents_fts'"
        ).fetchone()
        if table is None:
            connection.close()
            raise ValueError(f"Not a Search-R1 HotpotQA FTS5 database: {database_path}")
        return connection

    @staticmethod
    def _open_faiss_index(index_path: Path) -> Any:
        if not index_path.is_file():
            raise FileNotFoundError(f"HotpotQA FAISS index does not exist: {index_path}")
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("hybrid retrieval requires faiss-cpu (or faiss-gpu)") from exc
        return faiss.read_index(str(index_path))

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        configured_schema = getattr(self, "tool_schema", None)
        if configured_schema is not None:
            return configured_schema
        return OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the local document collection for passages relevant to a factual query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A concise, entity-focused search query.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        )

    async def create(
        self,
        instance_id: str | None = None,
        create_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[str, ToolResponse]:
        del kwargs
        create_kwargs = create_kwargs or {}
        resolved_id = instance_id or str(uuid4())
        if self.backend in {"sqlite", "hybrid"}:
            self._instances.add(resolved_id)
            return resolved_id, ToolResponse()

        context = create_kwargs.get("context")
        if not isinstance(context, dict):
            raise ValueError("HotpotSearchTool requires create_kwargs.context")

        titles = list(context.get("title") or [])
        sentence_groups = list(context.get("sentences") or [])
        if not titles or len(titles) != len(sentence_groups):
            raise ValueError("context.title and context.sentences must be non-empty lists of equal length")

        documents: list[_Document] = []
        document_frequencies: Counter[str] = Counter()
        for title, sentences in zip(titles, sentence_groups, strict=True):
            normalized_sentences = tuple(str(sentence) for sentence in (sentences or []))
            title_tokens = _tokenize(str(title)) * self.title_weight
            body_tokens = _tokenize(" ".join(normalized_sentences))
            term_counts = Counter(title_tokens + body_tokens)
            document_frequencies.update(term_counts.keys())
            documents.append(
                _Document(
                    title=str(title),
                    sentences=normalized_sentences,
                    term_counts=term_counts,
                    length=max(1, sum(term_counts.values())),
                )
            )

        average_length = sum(document.length for document in documents) / len(documents)
        self._indices[resolved_id] = _SearchIndex(
            documents=tuple(documents),
            document_frequencies=document_frequencies,
            average_length=average_length,
        )
        return resolved_id, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        del kwargs
        if parameters.get("_parse_error"):
            return (
                ToolResponse(
                    text=(
                        "<information><format_error>Could not parse your "
                        "tool call. Please use the correct format: "
                        "<tool_call>{\"name\": \"search\", \"arguments\": "
                        "{\"query\": \"your search query\"}}</tool_call>"
                        "</format_error></information>"
                    )
                ),
                0.0,
                {"format_error": True},
            )
        query = str(parameters.get("query") or "").strip()
        if not query:
            return (
                ToolResponse(
                    text=(
                        "<information><format_error>Invalid or empty query. "
                        "Please retry using the correct format: "
                        "<tool_call>{\"name\": \"search\", \"arguments\": "
                        "{\"query\": \"your search query\"}}</tool_call>"
                        "</format_error></information>"
                    )
                ),
                0.0,
                {"format_error": True},
            )
        if len(query) > self.max_query_chars:
            return (
                ToolResponse(
                    text=(
                        "<information><format_error>Query is too long "
                        f"({len(query)} characters, maximum {self.max_query_chars}). "
                        "Please provide a concise, entity-focused search query."
                        "</format_error></information>"
                    )
                ),
                0.0,
                {"format_error": True},
            )

        query_terms = list(dict.fromkeys(_tokenize(query)))
        if self.backend == "hybrid":
            if instance_id not in self._instances:
                raise KeyError(f"Unknown search instance: {instance_id}")
            selected = self._search_hybrid(query, query_terms)
        elif self.backend == "sqlite":
            if instance_id not in self._instances:
                raise KeyError(f"Unknown search instance: {instance_id}")
            selected = self._search_database(query_terms)
        else:
            index = self._indices.get(instance_id)
            if index is None:
                raise KeyError(f"Unknown search instance: {instance_id}")
            scored = [(self._bm25_score(document, query_terms, index), document) for document in index.documents]
            scored.sort(key=lambda item: (-item[0], item[1].title.casefold()))
            selected = [(score, document) for score, document in scored[: self.topk] if score > 0]

        if not selected:
            response = f"<information><query>{html.escape(query)}</query><no_results /></information>"
            return ToolResponse(text=response), 0.0, {"results_count": 0}

        rendered = ["<information>", f"<query>{html.escape(query)}</query>"]
        # BM25 rank and score are retrieval-internal signals, not evidence.
        # Exposing them can make the policy treat a high-ranked distractor as
        # more trustworthy than a lower-ranked supporting document.
        for _, document in selected:
            rendered.append(f'<document title="{html.escape(document.title, quote=True)}">')
            remaining = self.max_document_chars
            for sentence in document.sentences:
                if remaining <= 0:
                    break
                clipped = sentence[:remaining]
                rendered.append(html.escape(clipped))
                remaining -= len(clipped)
            rendered.append("</document>")
        rendered.append("</information>")
        response = "\n".join(rendered)
        return (
            ToolResponse(text=response),
            0.0,
            {
                "results_count": len(selected),
                "retrieved_titles": [document.title for _, document in selected],
            },
        )

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        del kwargs
        self._indices.pop(instance_id, None)
        self._instances.discard(instance_id)

    def _search_database(
        self, query_terms: list[str], limit: int | None = None
    ) -> list[tuple[float, _Document]]:
        if not query_terms or self._database is None:
            return []
        escaped_terms = [term.replace('"', '""') for term in query_terms]
        fts_query = " OR ".join(f'"{term}"' for term in escaped_terms)
        title_weight = float(self.title_weight)
        sql = f"""
            SELECT
                documents.id AS doc_id,
                documents.title AS title,
                documents.sentences_json AS sentences_json,
                bm25(documents_fts, {title_weight:.6f}, 1.0) AS rank_score
            FROM documents_fts
            JOIN documents ON documents.id = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY rank_score, documents.title COLLATE NOCASE
            LIMIT ?
        """
        with self._database_lock:
            rows = self._database.execute(sql, (fts_query, limit or self.topk)).fetchall()

        selected: list[tuple[float, _Document]] = []
        for row in rows:
            try:
                decoded_sentences = json.loads(row["sentences_json"])
            except (TypeError, json.JSONDecodeError):
                decoded_sentences = []
            sentences = tuple(str(sentence) for sentence in decoded_sentences)
            rank_score = float(row["rank_score"])
            selected.append(
                (
                    -rank_score,
                    _Document(
                        title=str(row["title"]),
                        sentences=sentences,
                        term_counts=Counter(),
                        length=1,
                        doc_id=int(row["doc_id"]),
                    ),
                )
            )
        return selected

    def _search_hybrid(self, query: str, query_terms: list[str]) -> list[tuple[float, _Document]]:
        lexical = self._search_database(query_terms, self.hybrid_candidate_topk)
        dense = self._search_dense(query, self.hybrid_candidate_topk)
        fused_scores: dict[int, float] = {}
        documents: dict[int, _Document] = {}
        for ranking, weight in ((lexical, self.bm25_weight), (dense, self.dense_weight)):
            for rank, (_, document) in enumerate(ranking, start=1):
                documents[document.doc_id] = document
                fused_scores[document.doc_id] = fused_scores.get(document.doc_id, 0.0) + weight / (
                    self.rrf_k + rank
                )
        ranked = sorted(
            fused_scores.items(),
            key=lambda item: (-item[1], documents[item[0]].title.casefold()),
        )
        candidates = [(score, documents[doc_id]) for doc_id, score in ranked]
        if self.reranker_enabled:
            return self._rerank(query, candidates[: self.reranker_topk])[: self.topk]
        return candidates[: self.topk]

    def _rerank(
        self, query: str, candidates: list[tuple[float, _Document]]
    ) -> list[tuple[float, _Document]]:
        if not candidates:
            return []
        with self._reranker_lock:
            self._load_reranker()
            import torch
            import torch.nn.functional as functional

            prefix = (
                '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
                'and the Instruct provided. Note that the answer can only be "yes" or "no".'
                '<|im_end|>\n<|im_start|>user\n'
            )
            suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
            prefix_tokens = self._reranker_tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = self._reranker_tokenizer.encode(suffix, add_special_tokens=False)
            token_false_id = self._reranker_tokenizer.convert_tokens_to_ids("no")
            token_true_id = self._reranker_tokenizer.convert_tokens_to_ids("yes")
            scored: list[tuple[float, int, _Document]] = []
            for start in range(0, len(candidates), self.reranker_batch_size):
                batch = candidates[start : start + self.reranker_batch_size]
                pairs = [
                    (
                        f"<Instruct>: {self.reranker_instruction}\n<Query>: {query}\n<Document>: "
                        f"{document.title}\n{' '.join(document.sentences)}"
                    )
                    for _, document in batch
                ]
                inputs = self._reranker_tokenizer(
                    pairs,
                    padding=False,
                    truncation="longest_first",
                    return_attention_mask=False,
                    max_length=self.reranker_max_length - len(prefix_tokens) - len(suffix_tokens),
                )
                for index, token_ids in enumerate(inputs["input_ids"]):
                    inputs["input_ids"][index] = prefix_tokens + token_ids + suffix_tokens
                inputs = self._reranker_tokenizer.pad(
                    inputs, padding=True, return_tensors="pt"
                )
                inputs = {name: tensor.to(self.reranker_device) for name, tensor in inputs.items()}
                with torch.inference_mode():
                    logits = self._reranker_model(**inputs).logits[:, -1, :]
                    binary_logits = torch.stack(
                        [logits[:, token_false_id], logits[:, token_true_id]], dim=1
                    )
                    scores = functional.softmax(binary_logits, dim=1)[:, 1].float().cpu().tolist()
                for offset, (score, (_, document)) in enumerate(zip(scores, batch, strict=True)):
                    scored.append((float(score), start + offset, document))
            scored.sort(key=lambda item: (-item[0], item[1], item[2].title.casefold()))
            return [(score, document) for score, _, document in scored]

    def _load_reranker(self) -> None:
        if self._reranker_model is not None and self._reranker_tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("reranking requires torch and transformers") from exc
        self._reranker_tokenizer = AutoTokenizer.from_pretrained(
            self.reranker_model_path, padding_side="left", trust_remote_code=True
        )
        self._reranker_model = AutoModelForCausalLM.from_pretrained(
            self.reranker_model_path,
            trust_remote_code=True,
            dtype=torch.float32 if self.reranker_device.startswith("cpu") else torch.bfloat16,
        ).to(self.reranker_device)
        self._reranker_model.eval()

    def _search_dense(self, query: str, limit: int) -> list[tuple[float, _Document]]:
        if self._faiss_index is None or self._database is None:
            return []
        vector = self._encode_query(query)
        scores, ids = self._faiss_index.search(vector, limit)
        ranked_ids = [int(doc_id) for doc_id in ids[0] if int(doc_id) >= 0]
        if not ranked_ids:
            return []

        placeholders = ",".join("?" for _ in ranked_ids)
        with self._database_lock:
            rows = self._database.execute(
                f"SELECT id, title, sentences_json FROM documents WHERE id IN ({placeholders})",
                ranked_ids,
            ).fetchall()
        by_id: dict[int, _Document] = {}
        for row in rows:
            try:
                sentences = tuple(str(value) for value in json.loads(row["sentences_json"]))
            except (TypeError, json.JSONDecodeError):
                sentences = ()
            doc_id = int(row["id"])
            by_id[doc_id] = _Document(
                title=str(row["title"]),
                sentences=sentences,
                term_counts=Counter(),
                length=1,
                doc_id=doc_id,
            )
        return [
            (float(score), by_id[int(doc_id)])
            for score, doc_id in zip(scores[0], ids[0], strict=True)
            if int(doc_id) in by_id
        ]

    def _encode_query(self, query: str) -> Any:
        with self._embedding_lock:
            if self._embedding_model is None or self._embedding_tokenizer is None:
                try:
                    import torch
                    from transformers import AutoModel, AutoTokenizer
                except ImportError as exc:
                    raise ImportError("hybrid retrieval requires torch and transformers") from exc
                self._embedding_tokenizer = AutoTokenizer.from_pretrained(
                    self.embedding_model_path, trust_remote_code=True
                )
                self._embedding_model = AutoModel.from_pretrained(
                    self.embedding_model_path,
                    trust_remote_code=True,
                    dtype=torch.float32 if self.embedding_device.startswith("cpu") else torch.bfloat16,
                ).to(self.embedding_device)
                self._embedding_model.eval()

            import torch
            import torch.nn.functional as functional

            prompt = f"Instruct: {self.query_instruction}\nQuery: {query}" if self.query_instruction else query
            encoded = self._embedding_tokenizer(
                [prompt],
                padding=True,
                truncation=True,
                max_length=self.embedding_max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(self.embedding_device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                hidden = self._embedding_model(**encoded).last_hidden_state
                pooled = self._last_token_pool(hidden, encoded["attention_mask"])
                normalized = functional.normalize(pooled, p=2, dim=1)
            return normalized.float().cpu().numpy()

    @staticmethod
    def _last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
        if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
            return last_hidden_states[:, -1]
        import torch

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
        return last_hidden_states[batch_indices, sequence_lengths]

    def __del__(self) -> None:
        database = getattr(self, "_database", None)
        if database is not None:
            try:
                database.close()
            except sqlite3.Error:
                pass

    def _bm25_score(self, document: _Document, query_terms: list[str], index: _SearchIndex) -> float:
        score = 0.0
        document_count = len(index.documents)
        for term in query_terms:
            frequency = document.term_counts.get(term, 0)
            if not frequency:
                continue
            document_frequency = index.document_frequencies.get(term, 0)
            inverse_document_frequency = math.log(
                1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            length_normalizer = self.k1 * (1.0 - self.b + self.b * document.length / index.average_length)
            score += inverse_document_frequency * frequency * (self.k1 + 1.0) / (frequency + length_normalizer)
        return score
