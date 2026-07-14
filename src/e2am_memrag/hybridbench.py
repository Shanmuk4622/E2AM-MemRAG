from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .utils import atomic_write_json, canonical_json, sha256_file


BENCHMARK_SCHEMA_VERSION = 1
TASK_TYPES = (
    "no_retrieval",
    "knowledge_only",
    "memory_only",
    "knowledge_memory",
    "temporal_update",
    "authority_conflict",
    "multi_hop",
    "deleted_or_missing",
)
SPLITS = ("pilot", "train", "calibration", "validation", "test")


def stable_id(namespace: str, *parts: object, length: int = 20) -> str:
    raw = canonical_json({"namespace": namespace, "parts": list(parts)}).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def tokenize(value: str) -> list[str]:
    # Keep punctuation inside identifiers (PRJ-ABC, 2e-5) but never attach
    # sentence-final periods, which would break exact lexical retrieval.
    return re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", value.lower())


def _opaque(rng: random.Random, prefix: str, width: int = 6) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return prefix + "-" + "".join(rng.choice(alphabet) for _ in range(width))


def _split_for_index(index: int) -> str:
    # Per task, every twenty scenario groups contain 10% pilot, 50% train,
    # 15% calibration, 10% validation, and 15% sealed test. The first five
    # occurrences cover every split, so the 40-query smoke profile is complete.
    slot = (index // len(TASK_TYPES)) % 20
    pattern = (
        "pilot",
        "train",
        "calibration",
        "validation",
        "test",
        "pilot",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "calibration",
        "calibration",
        "validation",
        "test",
        "test",
    )
    return pattern[slot]


_QUESTION_TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "no_retrieval": {
        "pilot": ("Copy only this token: {direct_token}.",),
        "train": ("Return exactly the token {direct_token}.", "Echo only {direct_token}."),
        "calibration": ("Give precisely this supplied token: {direct_token}.",),
        "validation": ("Reply with this literal identifier and nothing else: {direct_token}." ,),
        "test": ("Produce the exact code shown here: {direct_token}.",),
    },
    "knowledge_only": {
        "pilot": ("What optimizer is specified for {project}?",),
        "train": ("Which optimizer does the current specification require for {project}?",),
        "calibration": ("Consult project evidence and return {project}'s required optimizer.",),
        "validation": ("Read the approved specification for {project}; what optimizer is mandated?",),
        "test": ("According to authoritative project documentation, name {project}'s optimizer.",),
    },
    "memory_only": {
        "pilot": ("Recall my chosen seed for {project}.",),
        "train": ("Which seed did I choose for the last run of {project}?",),
        "calibration": ("Use active memory to state the seed I selected for {project}.",),
        "validation": ("From my experiment history, recover the selected seed for {project}.",),
        "test": ("What seed is recorded in my prior decision for {project}?",),
    },
    "knowledge_memory": {
        "pilot": ("Give {project}'s optimizer and my seed.",),
        "train": ("For {project}, combine the documented optimizer with my chosen seed.",),
        "calibration": ("Join specification and memory into an optimizer-seed pair for {project}.",),
        "validation": ("Using current docs and my run history, give optimizer and seed for {project}.",),
        "test": ("Reconstruct {project}'s optimizer-seed pair from documentation plus memory.",),
    },
    "temporal_update": {
        "pilot": ("Give the current batch size for {project}.",),
        "train": ("What is the latest approved batch size for {project}?",),
        "calibration": ("After temporal resolution, what batch size applies to {project}?",),
        "validation": ("Resolve the revision history and report {project}'s current batch size.",),
        "test": ("Use the newest authoritative specification: which batch size now applies to {project}?",),
    },
    "authority_conflict": {
        "pilot": ("Which learning rate wins the evidence conflict for {project}?",),
        "train": ("Conflicting notes exist for {project}; which learning rate is authoritative?",),
        "calibration": ("Use source authority to choose {project}'s learning rate.",),
        "validation": ("Resolve the source conflict and state the approved learning rate for {project}.",),
        "test": ("Prefer authoritative evidence over an unofficial note: give {project}'s learning rate.",),
    },
    "multi_hop": {
        "pilot": ("Find {project}'s scheduler through its dataset alias.",),
        "train": ("Follow {project}'s dataset alias to find the required scheduler.",),
        "calibration": ("Traverse project-to-dataset-to-scheduler for {project}.",),
        "validation": ("Resolve the dataset code used by {project}, then report its scheduler.",),
        "test": ("Two documents are needed: which scheduler corresponds to {project}'s dataset?",),
    },
    "deleted_or_missing": {
        "pilot": ("Is an active private label available for {project}?",),
        "train": ("What private label remains stored for {project}?",),
        "calibration": ("Answer only if {project}'s private label was not deleted.",),
        "validation": ("Recover the private label for {project}, if valid evidence still exists.",),
        "test": ("State {project}'s private label only when it remains supported by active evidence.",),
    },
}


def generate_hybridbench(
    scenario_count: int = 192,
    *,
    seed: int = 4622,
) -> dict[str, list[dict[str, Any]]]:
    """Create controlled RAG/memory tasks whose answers use opaque held-out facts.

    The generator is deterministic and does not call an LLM. Opaque facts prevent
    parametric memorization, while explicit timestamps, authority and tombstones
    make temporal/conflict/deletion failures measurable.
    """
    if scenario_count < len(TASK_TYPES) * len(SPLITS):
        raise ValueError("scenario_count must cover every task and split")
    rng = random.Random(seed)
    documents: list[dict[str, Any]] = []
    memory_events: list[dict[str, Any]] = []
    public_queries: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    optimizers = ("AdamW", "Lion", "Adafactor", "RMSprop")
    schedulers = ("cosine", "linear-warmup", "one-cycle", "constant")
    seeds = (4622, 1701, 31415, 27182, 8080)
    batch_sizes = (4, 8, 12, 16)
    learning_rates = ("1e-5", "2e-5", "3e-5", "5e-5")

    for index in range(scenario_count):
        split = _split_for_index(index)
        task_type = TASK_TYPES[index % len(TASK_TYPES)]
        scenario_id = f"scenario-{index:04d}-{stable_id('scenario', seed, index, length=8)}"
        project = _opaque(rng, "PRJ")
        direct_token = _opaque(rng, "ANS")
        dataset_code = _opaque(rng, "DATA")
        private_label = _opaque(rng, "PRIVATE")
        # Facts are drawn independently of task type and split.  Index-based
        # cycling would make some task answers constant and let a query-only
        # classifier appear to solve retrieval-required examples.
        optimizer = rng.choice(optimizers)
        scheduler = rng.choice(schedulers)
        chosen_seed = rng.choice(seeds)
        old_batch = rng.choice(batch_sizes)
        new_batch = rng.choice([value for value in batch_sizes if value != old_batch])
        approved_lr = rng.choice(learning_rates)
        unofficial_lr = rng.choice(
            [value for value in learning_rates if value != approved_lr]
        )

        def add_doc(
            suffix: str,
            text: str,
            *,
            timestamp: str,
            authority: int,
            kind: str,
        ) -> str:
            doc_id = f"doc-{stable_id('doc', scenario_id, suffix)}"
            documents.append(
                {
                    "doc_id": doc_id,
                    "scenario_id": scenario_id,
                    "text": text,
                    "timestamp": timestamp,
                    "authority": authority,
                    "kind": kind,
                    "active": True,
                }
            )
            return doc_id

        def add_memory(
            suffix: str,
            text: str,
            *,
            timestamp: str,
            event_type: str = "fact",
            tombstone_target: str | None = None,
        ) -> str:
            event_id = f"mem-{stable_id('memory', scenario_id, suffix)}"
            memory_events.append(
                {
                    "event_id": event_id,
                    "scenario_id": scenario_id,
                    "text": text,
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "tombstone_target": tombstone_target,
                    "entity": project,
                }
            )
            return event_id

        spec_doc = add_doc(
            "current-spec",
            f"Official specification for {project}. Required optimizer: {optimizer}.",
            timestamp="2026-05-01T00:00:00Z",
            authority=100,
            kind="official_spec",
        )
        add_doc(
            "old-revision",
            f"Archived specification for {project}. Batch size: {old_batch}.",
            timestamp="2025-01-01T00:00:00Z",
            authority=100,
            kind="archived_spec",
        )
        current_doc = add_doc(
            "new-revision",
            f"Current approved specification for {project}. Batch size: {new_batch}.",
            timestamp="2026-06-01T00:00:00Z",
            authority=100,
            kind="official_spec",
        )
        approved_doc = add_doc(
            "approved-lr",
            f"Signed research record for {project}. Approved learning rate: {approved_lr}.",
            timestamp="2026-04-01T00:00:00Z",
            authority=100,
            kind="signed_record",
        )
        add_doc(
            "unofficial-lr",
            f"Unofficial scratch note for {project}. Suggested learning rate: {unofficial_lr}.",
            timestamp="2026-06-15T00:00:00Z",
            authority=10,
            kind="unofficial_note",
        )
        alias_doc = add_doc(
            "dataset-alias",
            f"Project {project} uses dataset alias {dataset_code}.",
            timestamp="2026-03-01T00:00:00Z",
            authority=90,
            kind="registry",
        )
        schedule_doc = add_doc(
            "dataset-scheduler",
            f"Dataset alias {dataset_code} requires the {scheduler} scheduler.",
            timestamp="2026-03-02T00:00:00Z",
            authority=90,
            kind="registry",
        )
        seed_event = add_memory(
            "chosen-seed",
            f"I selected seed {chosen_seed} for the last run of {project}.",
            timestamp="2026-05-20T10:00:00Z",
        )
        private_event = add_memory(
            "private-label",
            f"The private label for {project} is {private_label}.",
            timestamp="2026-04-10T10:00:00Z",
        )
        tombstone_event = add_memory(
            "delete-private-label",
            f"Delete the stored private label for {project}.",
            timestamp="2026-05-10T10:00:00Z",
            event_type="tombstone",
            tombstone_target=private_event,
        )

        required_docs: list[str] = []
        required_memory: list[str] = []
        if task_type == "no_retrieval":
            answer = direct_token
        elif task_type == "knowledge_only":
            answer = optimizer
            required_docs = [spec_doc]
        elif task_type == "memory_only":
            answer = str(chosen_seed)
            required_memory = [seed_event]
        elif task_type == "knowledge_memory":
            answer = f"{optimizer}; seed {chosen_seed}"
            required_docs = [spec_doc]
            required_memory = [seed_event]
        elif task_type == "temporal_update":
            answer = str(new_batch)
            required_docs = [current_doc]
        elif task_type == "authority_conflict":
            answer = approved_lr
            required_docs = [approved_doc]
        elif task_type == "multi_hop":
            answer = scheduler
            required_docs = [alias_doc, schedule_doc]
        else:
            answer = "INSUFFICIENT_EVIDENCE"
            # The deleted fact and its tombstone are audit metadata, not evidence the
            # answerer is allowed to retrieve.  A correct abstention must therefore
            # not be penalized for omitting an intentionally unavailable citation.
            required_memory = []

        template_options = _QUESTION_TEMPLATES[task_type][split]
        template = template_options[index % len(template_options)]
        question = template.format(
            project=project,
            direct_token=direct_token,
        )
        query_id = f"query-{stable_id('query', scenario_id, task_type)}"
        public_queries.append(
            {
                "query_id": query_id,
                "scenario_id": scenario_id,
                "split": split,
                "task_type": task_type,
                "question": question,
                "query_time": "2026-07-01T00:00:00Z",
                "template_family": f"{task_type}-{split}",
            }
        )
        labels.append(
            {
                "query_id": query_id,
                "answer": answer,
                "required_doc_ids": required_docs,
                "required_memory_ids": required_memory,
                "forbidden_memory_ids": (
                    [private_event, tombstone_event]
                    if task_type == "deleted_or_missing"
                    else []
                ),
                "should_abstain": answer == "INSUFFICIENT_EVIDENCE",
            }
        )

    return {
        "documents": documents,
        "memory_events": memory_events,
        "queries": public_queries,
        "labels": labels,
    }


def _shingles(text: str, width: int = 3) -> set[tuple[str, ...]]:
    tokens = tokenize(text)
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def leakage_audit(
    queries: Sequence[Mapping[str, Any]],
    *,
    near_duplicate_threshold: float = 0.95,
) -> dict[str, Any]:
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_templates: dict[str, set[str]] = defaultdict(set)
    normalized: dict[str, str] = {}
    for query in queries:
        split = str(query["split"])
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        query_id = str(query["query_id"])
        normalized[query_id] = normalize_text(str(query["question"]))
        split_groups[split].add(str(query["scenario_id"]))
        split_templates[split].add(str(query["template_family"]))

    group_overlap = {
        f"{left}:{right}": sorted(split_groups[left] & split_groups[right])
        for i, left in enumerate(SPLITS)
        for right in SPLITS[i + 1 :]
    }
    template_overlap = {
        f"{left}:{right}": sorted(split_templates[left] & split_templates[right])
        for i, left in enumerate(SPLITS)
        for right in SPLITS[i + 1 :]
    }
    exact_duplicates: list[tuple[str, str]] = []
    near_duplicates: list[dict[str, Any]] = []
    by_split = {split: [q for q in queries if q["split"] == split] for split in SPLITS}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            for query_a in by_split[left]:
                id_a = str(query_a["query_id"])
                shingles_a = _shingles(str(query_a["question"]))
                for query_b in by_split[right]:
                    id_b = str(query_b["query_id"])
                    if normalized[id_a] == normalized[id_b]:
                        exact_duplicates.append((id_a, id_b))
                    shingles_b = _shingles(str(query_b["question"]))
                    union = shingles_a | shingles_b
                    score = len(shingles_a & shingles_b) / len(union) if union else 1.0
                    if score >= near_duplicate_threshold:
                        near_duplicates.append(
                            {"left": id_a, "right": id_b, "jaccard": round(score, 6)}
                        )
    hard_pass = not any(group_overlap.values()) and not any(template_overlap.values())
    hard_pass = hard_pass and not exact_duplicates and not near_duplicates
    return {
        "schema_version": 1,
        "hard_pass": hard_pass,
        "group_overlap": group_overlap,
        "template_overlap": template_overlap,
        "exact_cross_split_duplicates": exact_duplicates,
        "near_cross_split_duplicates": near_duplicates,
        "counts": {split: len(by_split[split]) for split in SPLITS},
        "near_duplicate_threshold": near_duplicate_threshold,
    }


@dataclass(frozen=True)
class BM25Index:
    ids: tuple[str, ...]
    tokenized: tuple[tuple[str, ...], ...]
    document_frequency: Mapping[str, int]
    average_length: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        id_field: str,
        text_field: str = "text",
    ) -> "BM25Index":
        ids = tuple(str(record[id_field]) for record in records)
        if len(ids) != len(set(ids)):
            raise ValueError("BM25 record IDs must be unique")
        tokenized = tuple(tuple(tokenize(str(record[text_field]))) for record in records)
        df: Counter[str] = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        average = sum(len(tokens) for tokens in tokenized) / max(1, len(tokenized))
        return cls(ids=ids, tokenized=tokenized, document_frequency=dict(df), average_length=average)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = tokenize(query)
        count = len(self.ids)
        scores: list[tuple[str, float]] = []
        for record_id, tokens in zip(self.ids, self.tokenized):
            frequencies = Counter(tokens)
            length_norm = 1 - self.b + self.b * len(tokens) / max(self.average_length, 1e-9)
            score = 0.0
            for term in query_terms:
                df = int(self.document_frequency.get(term, 0))
                if not df:
                    continue
                idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
                tf = frequencies[term]
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
            scores.append((record_id, score))
        return sorted(scores, key=lambda item: (-item[1], item[0]))[:top_k]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "ids": list(self.ids),
            "tokenized": [list(tokens) for tokens in self.tokenized],
            "document_frequency": dict(sorted(self.document_frequency.items())),
            "average_length": self.average_length,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BM25Index":
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported BM25 index schema")
        return cls(
            ids=tuple(str(item) for item in value["ids"]),
            tokenized=tuple(tuple(str(token) for token in row) for row in value["tokenized"]),
            document_frequency={str(k): int(v) for k, v in value["document_frequency"].items()},
            average_length=float(value["average_length"]),
            k1=float(value["k1"]),
            b=float(value["b"]),
        )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    from .utils import atomic_write_text

    atomic_write_text(destination, raw)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {number} is not an object")
                rows.append(value)
    return rows


def freeze_dataset(root: str | Path, data: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    for name in ("documents", "memory_events", "queries", "labels"):
        path = destination / f"{name}.jsonl"
        write_jsonl(path, data[name])
        inventory.append(
            {
                "logical_name": name,
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": len(data[name]),
            }
        )
    audit = leakage_audit(data["queries"])
    atomic_write_json(destination / "leakage_audit.json", audit)
    if not audit["hard_pass"]:
        raise RuntimeError(f"HybridBench leakage gate failed: {audit}")
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": "E2AM-HybridBench",
        "inventory": inventory,
        "leakage_audit_sha256": sha256_file(destination / "leakage_audit.json"),
        "task_types": list(TASK_TYPES),
        "split_counts": audit["counts"],
    }
    payload["freeze_sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    atomic_write_json(destination / "BENCHMARK_FREEZE.json", payload)
    return payload
