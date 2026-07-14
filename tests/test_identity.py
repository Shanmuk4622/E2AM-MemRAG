from __future__ import annotations

import unittest

from e2am_memrag.identity import RunIdentity, assigned_shard, make_config_hash, make_unit_id


class IdentityTests(unittest.TestCase):
    def test_hashes_are_stable_across_key_order(self) -> None:
        self.assertEqual(make_config_hash({"b": 2, "a": 1}), make_config_hash({"a": 1, "b": 2}))
        self.assertEqual(make_unit_id({"b": 2, "a": 1}), make_unit_id({"a": 1, "b": 2}))

    def test_every_unit_has_exactly_one_owner(self) -> None:
        identities = [RunIdentity("exp", "abc123def456", f"w-{i}", i, 4) for i in range(4)]
        for value in range(100):
            unit_id = make_unit_id({"value": value})
            owners = [identity for identity in identities if identity.owns(unit_id)]
            self.assertEqual(len(owners), 1)
            self.assertEqual(owners[0].shard_index, assigned_shard(unit_id, 4))


if __name__ == "__main__":
    unittest.main()

