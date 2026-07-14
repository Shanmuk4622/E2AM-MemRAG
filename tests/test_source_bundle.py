from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from e2am_memrag.source_bundle import _hub_call, hub_source_paths


class _HttpError(RuntimeError):
    def __init__(self, status: int, retry_after: str | None = None) -> None:
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        self.response = SimpleNamespace(status_code=status, headers=headers)
        super().__init__(f"HTTP {status}")


class SourceBundleTests(unittest.TestCase):
    def test_paths_are_content_addressed(self) -> None:
        archive = "a" * 64
        source = "b" * 64
        environment = "c" * 64
        paths = hub_source_paths(
            archive_sha256=archive,
            experiment_id="bootstrap-v1",
            source_tree_sha256=source,
            environment_sha256=environment,
        )
        self.assertEqual(
            paths.archive,
            f"source-bundles/{archive}/e2am-memrag-runtime.zip",
        )
        self.assertEqual(
            paths.environment_pin,
            f"preflight-pins/bootstrap-v1/{source}/{environment}.json",
        )
        self.assertEqual(paths.pointer, "source-bundles/LATEST.json")

    def test_unsafe_paths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            hub_source_paths(
                archive_sha256="not-a-sha",
                experiment_id="bootstrap-v1",
                source_tree_sha256="b" * 64,
                environment_sha256="c" * 64,
            )
        with self.assertRaises(ValueError):
            hub_source_paths(
                archive_sha256="a" * 64,
                experiment_id="../escape",
                source_tree_sha256="b" * 64,
                environment_sha256="c" * 64,
            )

    def test_rate_limit_retries_with_retry_after(self) -> None:
        calls = 0

        def operation() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise _HttpError(429, "0")
            return "ok"

        with patch("e2am_memrag.source_bundle.time.sleep") as sleep:
            self.assertEqual(_hub_call(operation), "ok")
        self.assertEqual(calls, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_authentication_failure_is_not_retried(self) -> None:
        calls = 0

        def operation() -> None:
            nonlocal calls
            calls += 1
            raise _HttpError(401)

        with patch("e2am_memrag.source_bundle.time.sleep") as sleep:
            with self.assertRaises(_HttpError):
                _hub_call(operation)
        self.assertEqual(calls, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
