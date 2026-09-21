"""Run the preregistered cheap hypothesis tournament on cached EURUSD payoffs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import t
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from scripts.research.fx_fast4_diagnostic import _fx_day, model_feature_names, model_matrix
from scripts.research.fx_fast4_events import DIRECTIONAL_FEATURES
from scripts.research.fx_fast4_materialize import _sha256
from scripts.research.fx_fast5_materialize import DEFAULT_FEATURE_PANEL
from scripts.research.fx_fast5_model import _read_features
from scripts.research.fx_fast9_materialize import DEFAULT_OUTPUT as PAYOFF_ROOT

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast10" / "tournament"
PAIR = "EURUSD"
PURGE = pd.Timedelta(hours=6)
TRAIN_END = pd.Timestamp("2021-01-01", tz="UTC") - PURGE
SELECTION_START = pd.Timestamp("2021-01-01", tz="UTC") + PURGE
SELECTION_END = pd.Timestamp("2021-03-15", tz="UTC") - PURGE
AUDIT_START = pd.Timestamp("2021-03-15", tz="UTC") + PURGE
AUDIT_END = pd.Timestamp("2021-04-01", tz="UTC") - PURGE
TAILS = (("low10", 0.10, -1), ("low25", 0.25, -1), ("high75", 0.75, 1), ("high90", 0.90, 1))
TOP_ATOMIC = 30
MAX_AUDIT = 25
MODEL_TOP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)
RANDOM_SEED = 20260921


@dataclass
class Candidate:
    name: str
    family: str
    specification: dict[str, object]
    mask: np.ndarray


@dataclass(frozen=True)
class TournamentData:
    matrix: np.ndarray
    names: tuple[str, ...]
    index: pd.DatetimeIndex
    base_r: np.ndarray
    stress_r: np.ndarray


def load_data(
    feature_panel: Path = DEFAULT_FEATURE_PANEL,
    payoff_root: Path = PAYOFF_ROOT,
) -> TournamentData:
    features = _read_features(feature_panel, PAIR)
    paths = [payoff_root / f"{PAIR}-{year}-managed-payoffs.parquet" for year in (2019, 2020, 2021)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"tournament payoff partitions missing: {missing}")
    payoffs = pd.concat(pd.read_parquet(path) for path in paths).sort_index()
    names = model_feature_names(features)
    matrices: list[np.ndarray] = []
    indexes: list[pd.DatetimeIndex] = []
    base_parts: list[np.ndarray] = []
    stress_parts: list[np.ndarray] = []
    for side, side_name in ((1, "long"), (-1, "short")):
        required = [f"{side_name}_result_r_base", f"{side_name}_result_r_stress"]
        valid = payoffs[required].notna().all(axis=1)
        selected = payoffs.loc[valid]
        selected_features = features.loc[selected.index]
        finite = np.isfinite(selected_features[names].to_numpy(dtype=np.float32)).all(axis=1)
        selected = selected.loc[finite]
        selected_features = selected_features.loc[finite]
        directional = model_matrix(selected_features, names, side)
        matrices.append(
            np.concatenate(
                (directional, np.full((len(directional), 1), side, dtype=np.float32)), axis=1
            )
        )
        indexes.append(pd.DatetimeIndex(selected.index))
        base_parts.append(selected[required[0]].to_numpy(dtype=np.float32))
        stress_parts.append(selected[required[1]].to_numpy(dtype=np.float32))
    matrix = np.concatenate(matrices)
    index = indexes[0].append(indexes[1:])
    base = np.concatenate(base_parts)
    stress = np.concatenate(stress_parts)
    order = np.argsort(index.asi8, kind="stable")
    return TournamentData(
        matrix=matrix[order],
        names=tuple([*names, "side"]),
        index=index[order],
        base_r=base[order],
        stress_r=stress[order],
    )


def time_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "train": np.asarray(index < TRAIN_END),
        "selection": np.asarray((index >= SELECTION_START) & (index < SELECTION_END)),
        "audit": np.asarray((index >= AUDIT_START) & (index < AUDIT_END)),
    }


def mean_metrics(mask: np.ndarray, base_r: np.ndarray, stress_r: np.ndarray) -> dict[str, float]:
    rows = int(mask.sum())
    if rows == 0:
        return {
            "rows": 0,
            "mean_base_r": float("nan"),
            "mean_stress_r": float("nan"),
            "conservative_mean_r": float("nan"),
        }
    mean_base = float(base_r[mask].mean())
    mean_stress = float(stress_r[mask].mean())
    return {
        "rows": rows,
        "mean_base_r": mean_base,
        "mean_stress_r": mean_stress,
        "conservative_mean_r": min(mean_base, mean_stress),
    }


def enumerate_tail_candidates(data: TournamentData, train: np.ndarray) -> list[Candidate]:
    candidates: list[Candidate] = []
    for column, feature in enumerate(data.names):
        values = data.matrix[:, column]
        training = values[train]
        for label, quantile, direction in TAILS:
            threshold = float(np.quantile(training, quantile))
            mask = values <= threshold if direction < 0 else values >= threshold
            candidates.append(
                Candidate(
                    name=f"tail:{feature}:{label}",
                    family="tail",
                    specification={
                        "feature": feature,
                        "column": column,
                        "tail": label,
                        "threshold": threshold,
                    },
                    mask=mask,
                )
            )
    return candidates


def _top_training_atoms(
    candidates: list[Candidate], train: np.ndarray, data: TournamentData
) -> list[Candidate]:
    scored = []
    for candidate in candidates:
        metrics = mean_metrics(candidate.mask & train, data.base_r, data.stress_r)
        if metrics["rows"] >= 500:
            scored.append((metrics["conservative_mean_r"], candidate))
    return [candidate for _, candidate in sorted(scored, key=lambda item: item[0], reverse=True)[:TOP_ATOMIC]]


def enumerate_intersections(
    atoms: list[Candidate], train: np.ndarray, data: TournamentData
) -> list[Candidate]:
    intersections: list[Candidate] = []
    for left, right in itertools.combinations(atoms, 2):
        if left.specification["feature"] == right.specification["feature"]:
            continue
        mask = left.mask & right.mask
        if int((mask & train).sum()) < 500:
            continue
        intersections.append(
            Candidate(
                name=f"and:{left.name}+{right.name}",
                family="intersection",
                specification={"left": left.specification, "right": right.specification},
                mask=mask,
            )
        )
    return intersections


def enumerate_tree_regimes(
    data: TournamentData, train: np.ndarray
) -> list[Candidate]:
    target = np.minimum(data.base_r, data.stress_r)
    candidates: list[Candidate] = []
    for depth in (2, 3, 4):
        model = DecisionTreeRegressor(
            max_depth=depth, min_samples_leaf=500, random_state=RANDOM_SEED
        )
        model.fit(data.matrix[train], target[train])
        leaves = model.apply(data.matrix)
        for leaf in np.unique(leaves[train]):
            mask = leaves == leaf
            candidates.append(
                Candidate(
                    name=f"tree:d{depth}:leaf{int(leaf)}",
                    family="tree",
                    specification={"depth": depth, "leaf": int(leaf)},
                    mask=mask,
                )
            )
    return candidates


def enumerate_cluster_regimes(
    data: TournamentData, train: np.ndarray
) -> list[Candidate]:
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(data.matrix[train])
    scaled_training = scaler.transform(data.matrix[train]).astype(np.float32)
    scaled_all = scaler.transform(data.matrix).astype(np.float32)
    candidates: list[Candidate] = []
    for clusters in (4, 8, 16):
        model = MiniBatchKMeans(
            n_clusters=clusters,
            batch_size=8_192,
            n_init=5,
            random_state=RANDOM_SEED,
        )
        model.fit(scaled_training)
        labels = model.predict(scaled_all)
        for label in range(clusters):
            candidates.append(
                Candidate(
                    name=f"cluster:k{clusters}:c{label}",
                    family="cluster",
                    specification={"clusters": clusters, "label": label},
                    mask=labels == label,
                )
            )
    del scaled_training, scaled_all
    gc.collect()
    return candidates


def _xgb_regressor() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=100,
        reg_lambda=10.0,
        tree_method="hist",
        n_jobs=10,
        random_state=RANDOM_SEED,
    )


def _xgb_classifier() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=100,
        reg_lambda=10.0,
        tree_method="hist",
        n_jobs=10,
        random_state=RANDOM_SEED,
    )


def _rank_candidates(name: str, score: np.ndarray, train: np.ndarray) -> list[Candidate]:
    candidates = []
    for fraction in MODEL_TOP_FRACTIONS:
        threshold = float(np.quantile(score[train], 1 - fraction))
        candidates.append(
            Candidate(
                name=f"model:{name}:top{fraction:g}",
                family="model",
                specification={"model": name, "top_fraction": fraction, "threshold": threshold},
                mask=score >= threshold,
            )
        )
    return candidates


def enumerate_model_rankings(
    data: TournamentData, train: np.ndarray
) -> list[Candidate]:
    x_train = data.matrix[train]
    base = data.base_r[train]
    stress = data.stress_r[train]
    candidates: list[Candidate] = []

    standard = StandardScaler().fit(x_train)
    standardized_train = standard.transform(x_train)
    standardized_all = standard.transform(data.matrix)
    ridge_base = Ridge(alpha=100.0).fit(standardized_train, base)
    ridge_stress = Ridge(alpha=100.0).fit(standardized_train, stress)
    ridge_score = np.minimum(
        ridge_base.predict(standardized_all), ridge_stress.predict(standardized_all)
    )
    candidates.extend(_rank_candidates("ridge", ridge_score, train))
    del standardized_train, standardized_all, ridge_base, ridge_stress

    regressors = {
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=200,
            max_features=0.8,
            n_jobs=10,
            random_state=RANDOM_SEED,
        ),
        "hist_gradient": lambda: HistGradientBoostingRegressor(
            max_iter=200,
            max_leaf_nodes=15,
            learning_rate=0.05,
            min_samples_leaf=200,
            l2_regularization=10.0,
            random_state=RANDOM_SEED,
        ),
        "xgboost_regression": _xgb_regressor,
    }
    for name, factory in regressors.items():
        base_model = factory()
        stress_model = factory()
        base_model.fit(x_train, base)
        stress_model.fit(x_train, stress)
        score = np.minimum(base_model.predict(data.matrix), stress_model.predict(data.matrix))
        candidates.extend(_rank_candidates(name, score, train))
        del base_model, stress_model, score
        gc.collect()

    for name, target in (
        ("xgboost_positive", (base > 0) & (stress > 0)),
        ("xgboost_runner", (base > 0.1) & (stress > 0.1)),
    ):
        model = _xgb_classifier()
        model.fit(x_train, target.astype(np.int8))
        score = model.predict_proba(data.matrix)[:, 1]
        candidates.extend(_rank_candidates(name, score, train))
        del model, score
        gc.collect()
    return candidates


def _mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


def select_for_audit(
    candidates: list[Candidate], masks: dict[str, np.ndarray], data: TournamentData
) -> tuple[list[Candidate], list[dict[str, object]]]:
    selected_rows: list[tuple[float, Candidate, dict[str, float], dict[str, float]]] = []
    seen: set[str] = set()
    for candidate in candidates:
        train_mask = candidate.mask & masks["train"]
        selection_mask = candidate.mask & masks["selection"]
        train_metrics = mean_metrics(train_mask, data.base_r, data.stress_r)
        selection_metrics = mean_metrics(selection_mask, data.base_r, data.stress_r)
        if train_metrics["rows"] < 500 or selection_metrics["rows"] < 100:
            continue
        digest = _mask_digest(selection_mask)
        if digest in seen:
            continue
        seen.add(digest)
        selected_rows.append(
            (
                selection_metrics["conservative_mean_r"],
                candidate,
                train_metrics,
                selection_metrics,
            )
        )
    ranked = sorted(selected_rows, key=lambda item: item[0], reverse=True)
    finalists = [item[1] for item in ranked[:MAX_AUDIT]]
    selection_report = [
        {
            "rank": rank,
            "name": item[1].name,
            "family": item[1].family,
            "specification": item[1].specification,
            "train": item[2],
            "selection": item[3],
        }
        for rank, item in enumerate(ranked[:MAX_AUDIT], start=1)
    ]
    return finalists, selection_report


def clustered_lower_bound(
    values: np.ndarray, index: pd.DatetimeIndex, alpha: float
) -> tuple[float, int]:
    realized = np.asarray(values, dtype=np.float64)
    days = _fx_day(index)
    unique_days, inverse = np.unique(days, return_inverse=True)
    clusters = len(unique_days)
    mean = float(realized.mean())
    if clusters < 2:
        return float("nan"), clusters
    scores = np.bincount(inverse, weights=realized - mean)
    variance = clusters / (clusters - 1) * float(np.square(scores).sum()) / len(realized) ** 2
    standard_error = np.sqrt(max(variance, 0.0))
    critical = float(t.ppf(1 - alpha, df=clusters - 1))
    return mean - critical * standard_error, clusters


def audit_finalists(
    finalists: list[Candidate], masks: dict[str, np.ndarray], data: TournamentData
) -> list[dict[str, object]]:
    if not finalists:
        return []
    alpha = 0.05 / len(finalists)
    unconditional = mean_metrics(masks["audit"], data.base_r, data.stress_r)
    results = []
    for candidate in finalists:
        mask = candidate.mask & masks["audit"]
        metrics = mean_metrics(mask, data.base_r, data.stress_r)
        selected_index = data.index[mask]
        base_lcb, days = clustered_lower_bound(data.base_r[mask], selected_index, alpha)
        stress_lcb, stress_days = clustered_lower_bound(data.stress_r[mask], selected_index, alpha)
        base_lift = metrics["mean_base_r"] - unconditional["mean_base_r"]
        stress_lift = metrics["mean_stress_r"] - unconditional["mean_stress_r"]
        gate = (
            metrics["rows"] >= 50
            and days >= 8
            and stress_days >= 8
            and metrics["mean_base_r"] > 0
            and metrics["mean_stress_r"] > 0
            and base_lcb > 0
            and stress_lcb > 0
            and base_lift >= 0.05
            and stress_lift >= 0.05
        )
        results.append(
            {
                "name": candidate.name,
                "family": candidate.family,
                "specification": candidate.specification,
                **metrics,
                "fx_days": days,
                "bonferroni_alpha": alpha,
                "lower_95_base_r": base_lcb,
                "lower_95_stress_r": stress_lcb,
                "base_lift": base_lift,
                "stress_lift": stress_lift,
                "pilot_gate": gate,
            }
        )
    return results


def run_tournament(
    feature_panel: Path = DEFAULT_FEATURE_PANEL,
    payoff_root: Path = PAYOFF_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    data = load_data(feature_panel, payoff_root)
    masks = time_masks(data.index)
    tails = enumerate_tail_candidates(data, masks["train"])
    atoms = _top_training_atoms(tails, masks["train"], data)
    families = {
        "tail": tails,
        "intersection": enumerate_intersections(atoms, masks["train"], data),
        "tree": enumerate_tree_regimes(data, masks["train"]),
        "cluster": enumerate_cluster_regimes(data, masks["train"]),
        "model": enumerate_model_rankings(data, masks["train"]),
    }
    all_candidates = [candidate for candidates in families.values() for candidate in candidates]
    finalists, selection = select_for_audit(all_candidates, masks, data)
    audit = audit_finalists(finalists, masks, data)
    passed = [item for item in audit if item["pilot_gate"]]
    winner = max(
        passed,
        key=lambda item: min(item["mean_base_r"], item["mean_stress_r"]),
        default=None,
    )
    report = {
        "kind": "round10_cheap_exhaustive_hypothesis_tournament",
        "protected_samples_opened": False,
        "q2_opened": False,
        "rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "features": list(data.names),
        "generated_hypotheses": len(all_candidates),
        "generated_by_family": {name: len(values) for name, values in families.items()},
        "audited_hypotheses": len(finalists),
        "audited_by_family": dict(Counter(candidate.family for candidate in finalists)),
        "selection": selection,
        "audit": audit,
        "winner": winner,
        "status": "p0_pass" if winner else "information_set_exhausted",
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "tournament-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(
        f"round10: {len(all_candidates)} generated, {len(finalists)} audited, "
        f"{len(passed)} passed, status={report['status']}, sha256={_sha256(report_path)}\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-panel", type=Path, default=DEFAULT_FEATURE_PANEL)
    parser.add_argument("--payoff-root", type=Path, default=PAYOFF_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    run_tournament(arguments.feature_panel, arguments.payoff_root, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
