from __future__ import annotations

import unittest

from e2am_memrag.hybridbench import generate_hybridbench
from e2am_memrag.rag_engine import (
    EvidenceRetriever,
    ExtractiveMockGenerator,
    HashingEncoder,
    active_memory_events,
    execute_route,
    parse_generation,
    routes_for_lane,
    score_answer,
)


class RagEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = generate_hybridbench(40)
        cls.labels = {row["query_id"]: row for row in cls.data["labels"]}
        cls.retriever = EvidenceRetriever(
            cls.data["documents"],
            cls.data["memory_events"],
            encoder=HashingEncoder(128),
        )

    def test_tombstoned_memory_is_not_retrievable(self) -> None:
        active_ids = {row["event_id"] for row in active_memory_events(self.data["memory_events"])}
        tombstoned = {
            row["tombstone_target"]
            for row in self.data["memory_events"]
            if row["event_type"] == "tombstone"
        }
        self.assertFalse(active_ids & tombstoned)

    def test_graph_memory_prefers_newest_linked_event_in_utc(self) -> None:
        events = [
            {
                "event_id": "mem-older",
                "text": "PRJ-TEMP latest status latest status latest status.",
                "timestamp": "2026-05-01T12:30:00Z",
                "event_type": "fact",
                "tombstone_target": None,
                "entity": "PRJ-TEMP",
            },
            {
                "event_id": "mem-newer",
                "text": "A short current update.",
                "timestamp": "2026-05-01T09:00:00-04:00",
                "event_type": "fact",
                "tombstone_target": None,
                "entity": "PRJ-TEMP",
            },
            {
                "event_id": "mem-unlinked-future",
                "text": "PRJ-TEMP latest status.",
                "timestamp": "2030-01-01T00:00:00Z",
                "event_type": "fact",
                "tombstone_target": None,
                "entity": "ORG-OTHER",
            },
        ]
        retriever = EvidenceRetriever(
            [
                {
                    "doc_id": "doc-control",
                    "text": "Control document.",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "authority": 1,
                }
            ],
            events,
            encoder=HashingEncoder(64),
        )
        route = next(route for route in routes_for_lane("lane-c") if route.memory == "graph")

        result = retriever.retrieve("What is the latest status for PRJ-TEMP?", route)

        self.assertEqual(result["memory_ids"][0], "mem-newer")
        self.assertLess(
            result["memory_ids"].index("mem-older"),
            result["memory_ids"].index("mem-unlinked-future"),
        )

    def test_graph_memory_recency_excludes_tombstones_and_is_order_stable(self) -> None:
        events = [
            {
                "event_id": "mem-old",
                "text": "Earlier PRJ-STABLE decision.",
                "timestamp": "2025-01-01T00:00:00Z",
                "event_type": "fact",
                "tombstone_target": None,
                "entity": "PRJ-STABLE",
            },
            {
                "event_id": "mem-current",
                "text": "Current PRJ-STABLE decision.",
                "timestamp": "2026-01-01T00:00:00Z",
                "event_type": "fact",
                "tombstone_target": None,
                "entity": "PRJ-STABLE",
            },
            {
                "event_id": "mem-deleted-newest",
                "text": "Deleted PRJ-STABLE decision.",
                "timestamp": "2027-01-01T00:00:00Z",
                "event_type": "fact",
                "tombstone_target": None,
                "entity": "PRJ-STABLE",
            },
            {
                "event_id": "mem-delete-marker",
                "text": "Delete the newest PRJ-STABLE decision.",
                "timestamp": "2027-02-01T00:00:00Z",
                "event_type": "tombstone",
                "tombstone_target": "mem-deleted-newest",
                "entity": "PRJ-STABLE",
            },
        ]
        documents = [
            {
                "doc_id": "doc-control",
                "text": "Control document.",
                "timestamp": "2026-01-01T00:00:00Z",
                "authority": 1,
            }
        ]
        route = next(route for route in routes_for_lane("lane-c") if route.memory == "graph")
        forward = EvidenceRetriever(
            documents,
            events,
            encoder=HashingEncoder(64),
        ).retrieve("Recall the PRJ-STABLE decision.", route)["memory_ids"]
        reversed_order = EvidenceRetriever(
            documents,
            list(reversed(events)),
            encoder=HashingEncoder(64),
        ).retrieve("Recall the PRJ-STABLE decision.", route)["memory_ids"]

        self.assertEqual(forward, reversed_order)
        self.assertEqual(forward[0], "mem-current")
        self.assertNotIn("mem-deleted-newest", forward)
        self.assertNotIn("mem-delete-marker", forward)

    def test_route_lanes_are_fixed_and_non_overlapping(self) -> None:
        lanes = [routes_for_lane(f"lane-{letter}") for letter in "abcd"]
        route_ids = [route.route_id for lane in lanes for route in lane]
        self.assertEqual(len(route_ids), 22)
        self.assertEqual(len(route_ids), len(set(route_ids)))

    def test_controlled_knowledge_route_executes_and_scores(self) -> None:
        query = next(row for row in self.data["queries"] if row["task_type"] == "knowledge_only")
        route = routes_for_lane("lane-a")[1]
        result = execute_route(
            query=query,
            label=self.labels[query["query_id"]],
            route=route,
            retriever=self.retriever,
            generator=ExtractiveMockGenerator(),
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertTrue(result["metrics"]["success"])
        self.assertTrue(result["retrieval"]["knowledge_ids"])

    def test_generation_parser_never_treats_malformed_text_as_json(self) -> None:
        result = parse_generation("not json; insufficient evidence")
        self.assertFalse(result["parse_ok"])
        self.assertTrue(result["abstain"])

    def test_scoring_rejects_unknown_and_forbidden_citations(self) -> None:
        label = {
            "answer": "Lion",
            "required_doc_ids": ["required"],
            "required_memory_ids": [],
            "forbidden_memory_ids": ["deleted"],
            "should_abstain": False,
        }
        unknown = score_answer(
            {
                "answer": "Lion",
                "citations": ["required", "unknown"],
                "abstain": False,
                "parse_ok": True,
            },
            label,
            retrieved_ids={"required"},
        )
        self.assertFalse(unknown["success"])
        forbidden = score_answer(
            {
                "answer": "Lion",
                "citations": ["required"],
                "abstain": False,
                "parse_ok": True,
            },
            label,
            retrieved_ids={"required", "deleted"},
        )
        self.assertFalse(forbidden["success"])


if __name__ == "__main__":
    unittest.main()
