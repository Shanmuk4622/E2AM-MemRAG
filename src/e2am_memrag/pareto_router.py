from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .rag_engine import ROUTES
from .utils import canonical_json


QUERY_FEATURES = (
    "query_tokens",
    "query_characters",
    "digit_count",
    "entity_code_count",
    "temporal_terms",
    "conflict_terms",
    "memory_terms",
    "multi_hop_terms",
)
PROBE_FEATURES = (
    "doc_top",
    "doc_gap",
    "doc_count",
    "memory_top",
    "memory_gap",
    "memory_count",
    "authority_max",
    "authority_range",
)


def _energy_joules(trace: Mapping[str, Any]) -> float | None:
    energy = trace.get("generation", {}).get("energy")
    if not isinstance(energy, Mapping) or not energy.get("available"):
        return None
    value = energy.get("energy_joules")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _vector(
    features: Mapping[str, Any],
    route_id: str,
    route_ids: Sequence[str],
    *,
    include_probe: bool,
) -> list[float]:
    names = QUERY_FEATURES + (PROBE_FEATURES if include_probe else ())
    values = [float(features.get(name, 0.0) or 0.0) for name in names]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Router features must all be finite")
    values.extend(1.0 if candidate == route_id else 0.0 for candidate in route_ids)
    return values


@dataclass
class SeedModels:
    success_model: Any
    success_calibrator: Any
    energy_model: Any
    latency_model: Any


class ConstantSuccessModel:
    """Pickle-safe fallback for a route trace set with one observed outcome class."""

    def __init__(self, probability: float) -> None:
        self.probability = min(1.0, max(0.0, float(probability)))

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[1.0 - self.probability, self.probability] for _ in rows]


class ConstantCalibrator:
    """Calibration fallback when calibration labels or raw scores are constant."""

    def __init__(self, probability: float) -> None:
        self.probability = min(1.0, max(0.0, float(probability)))

    def predict(self, values: Sequence[float]) -> list[float]:
        return [self.probability for _ in values]


@dataclass
class RouterBundle:
    schema_version: int
    route_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    models: list[SeedModels]
    tau: float
    latency_limit_seconds: float
    safe_route_id: str
    include_probe: bool
    training_hash: str

    def __post_init__(self) -> None:
        known = {route.route_id: route for route in ROUTES}
        if not self.route_ids or len(self.route_ids) != len(set(self.route_ids)):
            raise ValueError("Router route_ids must be non-empty and unique")
        if set(self.route_ids) - set(known):
            raise ValueError("Router contains unknown route IDs")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Router seeds must be non-empty and unique")
        if len(self.models) != len(self.seeds):
            raise ValueError("Router model/seed counts differ")
        if self.safe_route_id not in self.route_ids:
            raise ValueError("Router safe route is absent from route_ids")
        if known[self.safe_route_id].offline_only:
            raise ValueError("Router safe route cannot be offline-only")
        if not math.isfinite(self.tau) or not 0.0 <= self.tau <= 1.0:
            raise ValueError("Router tau must be finite and within [0, 1]")
        if (
            not math.isfinite(self.latency_limit_seconds)
            or self.latency_limit_seconds <= 0
        ):
            raise ValueError("Router latency limit must be finite and positive")

    def predict_actions(
        self,
        features: Mapping[str, Any],
        *,
        candidate_route_ids: Sequence[str] | None = None,
    ) -> list[dict[str, float | str]]:
        candidates = (
            self.route_ids
            if candidate_route_ids is None
            else tuple(candidate_route_ids)
        )
        unknown = set(candidates) - set(self.route_ids)
        if unknown:
            raise ValueError(f"Unknown route IDs: {sorted(unknown)}")
        actions = []
        for route_id in candidates:
            vector = [_vector(features, route_id, self.route_ids, include_probe=self.include_probe)]
            probabilities = []
            energies = []
            latencies = []
            for seed_model in self.models:
                raw = float(seed_model.success_model.predict_proba(vector)[0][1])
                calibrated = float(seed_model.success_calibrator.predict([raw])[0])
                if not math.isfinite(calibrated):
                    calibrated = 0.0
                probabilities.append(min(1.0, max(0.0, calibrated)))
                try:
                    predicted_energy = math.exp(
                        float(seed_model.energy_model.predict(vector)[0])
                    )
                    predicted_latency = math.exp(
                        float(seed_model.latency_model.predict(vector)[0])
                    )
                except (OverflowError, TypeError, ValueError):
                    predicted_energy = 1.0e30
                    predicted_latency = 1.0e30
                energies.append(
                    predicted_energy if math.isfinite(predicted_energy) else 1.0e30
                )
                latencies.append(
                    predicted_latency if math.isfinite(predicted_latency) else 1.0e30
                )
            probability = statistics.fmean(probabilities)
            deviation = statistics.pstdev(probabilities) if len(probabilities) > 1 else 0.0
            actions.append(
                {
                    "route_id": route_id,
                    "success_probability": probability,
                    "success_lower_bound": min(probabilities),
                    "success_seed_standard_deviation": deviation,
                    "success_lower_bound_method": "minimum calibrated grouped-bootstrap seed",
                    "predicted_energy_joules": statistics.fmean(energies),
                    "predicted_latency_seconds": statistics.fmean(latencies),
                    "cost_prediction_quantile": 0.90,
                }
            )
        return actions

    def choose(
        self,
        query_features: Mapping[str, Any],
        *,
        probe_features: Mapping[str, Any] | None = None,
        allow_offline: bool = False,
    ) -> dict[str, Any]:
        route_by_id = {route.route_id: route for route in ROUTES}
        query_only = {
            name: float(query_features.get(name, 0.0) or 0.0)
            for name in QUERY_FEATURES
        }
        allowed = [
            route_id
            for route_id in self.route_ids
            if allow_offline or not route_by_id[route_id].offline_only
        ]
        direct_ids = [
            route_id
            for route_id in allowed
            if route_by_id[route_id].knowledge == "none"
            and route_by_id[route_id].memory == "none"
        ]
        stage0 = self.predict_actions(query_only, candidate_route_ids=direct_ids)
        feasible_direct = [
            action
            for action in stage0
            if action["success_lower_bound"] >= self.tau
            and action["predicted_latency_seconds"] <= self.latency_limit_seconds
        ]
        if feasible_direct:
            chosen = min(feasible_direct, key=lambda item: item["predicted_energy_joules"])
            return {"stage": 0, "probe_required": False, "chosen": chosen, "actions": stage0}

        if probe_features is None:
            return {
                "stage": 0,
                "probe_required": True,
                "chosen": None,
                "actions": stage0,
            }

        combined = {**query_only, **probe_features}
        # Once the cheap stage-0 direct decision fails, stage 1 considers only
        # actions that actually use the charged probe.  This keeps training and
        # deployment feature availability identical.
        stage1_ids = [route_id for route_id in allowed if route_id not in direct_ids]
        stage1 = self.predict_actions(
            combined, candidate_route_ids=stage1_ids or allowed
        )
        feasible = [
            action
            for action in stage1
            if action["success_lower_bound"] >= self.tau
            and action["predicted_latency_seconds"] <= self.latency_limit_seconds
        ]
        if feasible:
            chosen = min(feasible, key=lambda item: item["predicted_energy_joules"])
            fallback = False
        else:
            chosen = next(
                action for action in stage1 if action["route_id"] == self.safe_route_id
            )
            fallback = True
        return {
            "stage": 1,
            "probe_required": True,
            "chosen": chosen,
            "actions": stage1,
            "safe_fallback": fallback,
        }


def _rows_for_split(
    traces: Sequence[Mapping[str, Any]],
    query_splits: Mapping[str, str],
    split: str,
) -> list[Mapping[str, Any]]:
    return [trace for trace in traces if query_splits.get(str(trace["query_id"])) == split]


def _success_arrays(
    traces: Sequence[Mapping[str, Any]],
    route_ids: Sequence[str],
    *,
    include_probe: bool,
) -> tuple[list[list[float]], list[int]]:
    vectors: list[list[float]] = []
    successes: list[int] = []
    for trace in traces:
        route_id = str(trace["route_id"])
        features = trace.get("features")
        if route_id not in route_ids or not isinstance(features, Mapping):
            continue
        vectors.append(
            _vector(features, route_id, route_ids, include_probe=include_probe)
        )
        successes.append(1 if trace.get("success") else 0)
    if len(vectors) < max(20, len(route_ids) * 2):
        raise RuntimeError("Too few trace outcomes to train the router success model")
    return vectors, successes


def _cost_arrays(
    traces: Sequence[Mapping[str, Any]],
    route_ids: Sequence[str],
    *,
    include_probe: bool,
) -> tuple[list[list[float]], list[float], list[float]]:
    vectors: list[list[float]] = []
    energies: list[float] = []
    latencies: list[float] = []
    for trace in traces:
        energy = _energy_joules(trace)
        try:
            latency_value = float(trace.get("total_seconds")) + float(
                trace.get("probe_seconds", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            continue
        route_id = str(trace.get("route_id"))
        features = trace.get("features")
        if (
            route_id not in route_ids
            or not isinstance(features, Mapping)
            or energy is None
            or not math.isfinite(latency_value)
            or latency_value <= 0
        ):
            continue
        vectors.append(_vector(features, route_id, route_ids, include_probe=include_probe))
        energies.append(math.log(max(energy, 1e-6)))
        latencies.append(math.log(max(latency_value, 1e-6)))
    if len(vectors) < max(20, len(route_ids) * 2):
        raise RuntimeError("Too few finite telemetry rows to train router cost models")
    return vectors, energies, latencies


def _select_safe_route(validation: Sequence[Mapping[str, Any]], route_ids: Sequence[str]) -> str:
    by_route: dict[str, list[Mapping[str, Any]]] = {route_id: [] for route_id in route_ids}
    for trace in validation:
        route_id = str(trace.get("route_id"))
        if route_id in by_route:
            by_route[route_id].append(trace)
    ranked = []
    for route_id, rows in by_route.items():
        if not rows:
            continue
        success_rate = sum(bool(row.get("success")) for row in rows) / len(rows)
        finite_energy = [value for row in rows if (value := _energy_joules(row)) is not None]
        median_energy = statistics.median(finite_energy) if finite_energy else float("inf")
        ranked.append((route_id, success_rate, median_energy))
    if not ranked:
        raise RuntimeError("No validation route has usable outcomes")
    return min(ranked, key=lambda item: (-item[1], item[2], item[0]))[0]


def _group_bootstrap(
    traces: Sequence[Mapping[str, Any]], seed: int
) -> list[Mapping[str, Any]]:
    """Resample whole queries so route outcomes remain paired within a bootstrap."""

    by_query: dict[str, list[Mapping[str, Any]]] = {}
    for trace in traces:
        by_query.setdefault(str(trace["query_id"]), []).append(trace)
    query_ids = sorted(by_query)
    if not query_ids:
        return []
    rng = random.Random(int(seed))
    sampled: list[Mapping[str, Any]] = []
    for _ in query_ids:
        sampled.extend(by_query[rng.choice(query_ids)])
    return sampled


def fit_router(
    traces: Sequence[Mapping[str, Any]],
    query_splits: Mapping[str, str],
    *,
    route_ids: Sequence[str] | None = None,
    seeds: Sequence[int] = (4622, 1701, 31415, 27182, 65537),
    latency_limit_seconds: float = 12.0,
    include_probe: bool = True,
    pretrained_seed_models: Mapping[int, SeedModels] | None = None,
    checkpoint_callback: Callable[[int, SeedModels], None] | None = None,
) -> RouterBundle:
    """Fit action-conditional success/energy/latency models without test access."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.isotonic import IsotonicRegression

    if not seeds or len(set(map(int, seeds))) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    pretrained_seed_models = dict(pretrained_seed_models or {})
    if set(pretrained_seed_models) - set(map(int, seeds)):
        raise ValueError("A pretrained router seed is outside the requested seed set")
    if any(not isinstance(model, SeedModels) for model in pretrained_seed_models.values()):
        raise TypeError("pretrained_seed_models must contain SeedModels values")
    if not math.isfinite(latency_limit_seconds) or latency_limit_seconds <= 0:
        raise ValueError("latency_limit_seconds must be finite and positive")
    if any(query_splits.get(str(trace["query_id"])) == "test" for trace in traces):
        raise RuntimeError("Router fitting input contains sealed test traces")
    route_ids = tuple(route_ids or (route.route_id for route in ROUTES))
    known_routes = {route.route_id for route in ROUTES}
    if (
        not route_ids
        or len(route_ids) != len(set(route_ids))
        or set(route_ids) - known_routes
    ):
        raise ValueError("route_ids must be a non-empty subset of the frozen catalog")
    training = _rows_for_split(traces, query_splits, "train")
    calibration = _rows_for_split(traces, query_splits, "calibration")
    validation = _rows_for_split(traces, query_splits, "validation")
    # Calibration and validation have separate roles; test remains sealed. Pilot
    # pruning occurs in an earlier frozen stage.
    x_cal, y_cal = _success_arrays(
        calibration, route_ids, include_probe=include_probe
    )
    models: list[SeedModels] = []
    for seed in seeds:
        if int(seed) in pretrained_seed_models:
            models.append(pretrained_seed_models[int(seed)])
            continue
        bootstrap = _group_bootstrap(training, int(seed))
        x_train, y_train = _success_arrays(
            bootstrap, route_ids, include_probe=include_probe
        )
        x_cost, e_train, l_train = _cost_arrays(
            bootstrap, route_ids, include_probe=include_probe
        )
        if len(set(y_train)) == 1:
            success_model = ConstantSuccessModel(float(y_train[0]))
        else:
            success_model = HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=180,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=int(seed),
            ).fit(x_train, y_train)
        raw_calibration = [float(row[1]) for row in success_model.predict_proba(x_cal)]
        if len(set(y_cal)) == 1 or len(set(raw_calibration)) < 2:
            calibrator = ConstantCalibrator(sum(y_cal) / len(y_cal))
        else:
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                raw_calibration, y_cal
            )
        energy_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.90,
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=int(seed),
        ).fit(x_cost, e_train)
        latency_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.90,
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=int(seed),
        ).fit(x_cost, l_train)
        seed_models = SeedModels(
            success_model, calibrator, energy_model, latency_model
        )
        models.append(seed_models)
        if checkpoint_callback is not None:
            checkpoint_callback(int(seed), seed_models)

    route_by_id = {route.route_id: route for route in ROUTES}
    deployable_escalations = [
        route_id
        for route_id in route_ids
        if not route_by_id[route_id].offline_only
        and not (
            route_by_id[route_id].knowledge == "none"
            and route_by_id[route_id].memory == "none"
        )
    ]
    safe_route = _select_safe_route(
        validation, deployable_escalations or route_ids
    )
    training_descriptor = [
        {
            "unit_id": trace.get("unit_id"),
            "query_id": trace.get("query_id"),
            "route_id": trace.get("route_id"),
            "status": trace.get("status"),
            "success": trace.get("success"),
            "split": query_splits.get(str(trace.get("query_id"))),
            "features": trace.get("features"),
            "energy_joules": _energy_joules(trace),
            "total_seconds": trace.get("total_seconds"),
            "probe_seconds": trace.get("probe_seconds", 0.0),
            "spec_hash": trace.get("spec_hash"),
        }
        for trace in sorted(traces, key=lambda row: str(row.get("unit_id")))
    ]
    training_hash = hashlib.sha256(
        canonical_json(
            {
                "rows": training_descriptor,
                "route_ids": list(route_ids),
                "seeds": [int(seed) for seed in seeds],
                "latency_limit_seconds": latency_limit_seconds,
                "include_probe": include_probe,
                "model_family": "hist-gradient-boosting-calibrated-q90-v2",
            }
        ).encode()
    ).hexdigest()
    # Conservative fixed candidates; choose the lowest threshold whose validation
    # success is at least 0.80. This never observes test labels.
    candidate_taus = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
    tau = 0.75
    provisional = RouterBundle(
        schema_version=2,
        route_ids=route_ids,
        seeds=tuple(int(seed) for seed in seeds),
        models=models,
        tau=tau,
        latency_limit_seconds=latency_limit_seconds,
        safe_route_id=safe_route,
        include_probe=include_probe,
        training_hash=training_hash,
    )
    lookup = {(str(row["query_id"]), str(row["route_id"])): row for row in validation}
    query_feature_rows: dict[str, dict[str, float]] = {}
    probe_feature_rows: dict[str, dict[str, float]] = {}
    for row in validation:
        query_id = str(row["query_id"])
        features = row.get("features", {})
        query_feature_rows.setdefault(
            query_id,
            {name: float(features.get(name, 0.0) or 0.0) for name in QUERY_FEATURES},
        )
        candidate_probe = {
            name: float(features.get(name, 0.0) or 0.0) for name in PROBE_FEATURES
        }
        if any(candidate_probe.values()):
            probe_feature_rows[query_id] = candidate_probe
    selected_tau: float | None = None
    for candidate in candidate_taus:
        provisional.tau = candidate
        outcomes = []
        execution = []
        abstentions = []
        for query_id, features in query_feature_rows.items():
            decision = provisional.choose(
                features,
                probe_features=probe_feature_rows.get(
                    query_id, {name: 0.0 for name in PROBE_FEATURES}
                ),
            )
            actual = lookup.get((query_id, str(decision["chosen"]["route_id"])))
            if actual is None:
                raise RuntimeError(
                    "Validation trace matrix is missing a router-selected action"
                )
            outcomes.append(bool(actual.get("success")))
            execution.append(actual.get("status") == "SUCCESS")
            abstentions.append(
                bool(actual.get("answer", {}).get("abstain", False))
            )
        if (
            outcomes
            and sum(outcomes) / len(outcomes) >= 0.80
            and sum(execution) / len(execution) >= 0.90
            and sum(abstentions) / len(abstentions) <= 0.20
        ):
            selected_tau = candidate
            break
    if selected_tau is None:
        raise RuntimeError(
            "No validation threshold satisfies the frozen success, execution-coverage, "
            "and abstention constraints"
        )
    provisional.tau = selected_tau
    return provisional


def save_router(bundle: RouterBundle, path: str | Path) -> dict[str, Any]:
    import joblib

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    manifest = {
        "schema_version": bundle.schema_version,
        "route_ids": list(bundle.route_ids),
        "seeds": list(bundle.seeds),
        "tau": bundle.tau,
        "latency_limit_seconds": bundle.latency_limit_seconds,
        "safe_route_id": bundle.safe_route_id,
        "include_probe": bundle.include_probe,
        "training_hash": bundle.training_hash,
        "model_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    manifest["policy_sha256"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    return manifest


def load_router(path: str | Path, manifest: Mapping[str, Any]) -> RouterBundle:
    import joblib

    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != manifest.get("model_sha256"):
        raise RuntimeError("Router artifact checksum mismatch")
    unsigned = dict(manifest)
    supplied_policy_hash = unsigned.pop("policy_sha256", None)
    expected_policy_hash = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    if supplied_policy_hash != expected_policy_hash:
        raise RuntimeError("Router policy manifest checksum mismatch")
    bundle = joblib.load(source)
    if not isinstance(bundle, RouterBundle):
        raise RuntimeError("Router artifact has an unexpected type")
    expected_fields = {
        "schema_version": bundle.schema_version,
        "route_ids": list(bundle.route_ids),
        "seeds": list(bundle.seeds),
        "tau": bundle.tau,
        "latency_limit_seconds": bundle.latency_limit_seconds,
        "safe_route_id": bundle.safe_route_id,
        "include_probe": bundle.include_probe,
        "training_hash": bundle.training_hash,
        "model_sha256": digest,
    }
    if unsigned != expected_fields:
        raise RuntimeError("Router policy manifest does not match the serialized bundle")
    return bundle
