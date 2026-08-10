"""Simulator-trained Distilled-WolfGuard baseline."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from wolfbench.agents.wolfguard import WolfGuardConfig
from wolfbench.defense.policy import make_intervention


ACTION_LABELS = ["none", "warning", "cooldown", "block"]
FEATURE_NAMES = [
    "day_norm",
    "price_gap",
    "abs_price_gap",
    "spread_bps_scaled",
    "log_volume",
    "log_real_volume",
    "volume_share",
    "depth_imbalance",
    "cancel_rate",
    "wash_share",
    "log_msg_volume",
    "msg_volume_share",
    "sentiment",
    "harmful_msg_share",
    "log_cascade_size",
    "recent_return",
]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _log1p_nonnegative(value: Any) -> float:
    return float(np.log1p(max(_float(value), 0.0)))


def extract_public_features(summary: dict[str, Any], asset: str) -> np.ndarray:
    """Extract the public per-asset feature vector used by Distilled-WolfGuard."""
    market_all = summary.get("market", {}) or {}
    social_all = summary.get("social", {}) or {}
    market = market_all.get(asset, {}) or {}
    social = social_all.get(asset, {}) or {}
    recent_return = summary.get("recent_return", {}) or {}

    volumes = [_float(m.get("volume", 0.0)) for m in market_all.values()]
    msg_volumes = [_float(s.get("msg_volume", 0.0)) for s in social_all.values()]
    max_volume = max(volumes) if volumes else 0.0
    max_msg_volume = max(msg_volumes) if msg_volumes else 0.0

    price = _float(market.get("price", 0.0))
    fundamental = max(_float(market.get("fundamental", 1.0), 1.0), 1e-9)
    price_gap = (price - fundamental) / fundamental
    volume = _float(market.get("volume", 0.0))
    msg_volume = _float(social.get("msg_volume", 0.0))

    return np.array([
        _float(summary.get("day", 0.0)) / 30.0,
        price_gap,
        abs(price_gap),
        _float(market.get("spread_bps", 0.0)) / 100.0,
        _log1p_nonnegative(volume),
        _log1p_nonnegative(market.get("real_volume", 0.0)),
        volume / max(max_volume, 1e-9),
        _float(market.get("depth_imbalance", 0.0)),
        _float(market.get("cancel_rate", 0.0)),
        _float(market.get("wash_share", 0.0)),
        _log1p_nonnegative(msg_volume),
        msg_volume / max(max_msg_volume, 1e-9),
        _float(social.get("sentiment", 0.0)),
        _float(social.get("harmful_msg_share", 0.0)),
        _log1p_nonnegative(social.get("cascade_size", 0.0)),
        _float(recent_return.get(asset, 0.0)),
    ], dtype=float)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


@dataclass
class DistilledWolfGuardModel:
    weights: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    action_labels: list[str] = field(default_factory=lambda: list(ACTION_LABELS))
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, summary: dict[str, Any], asset: str) -> dict[str, float]:
        x = extract_public_features(summary, asset)
        xs = (x - self.feature_mean) / self.feature_scale
        probs = _softmax((xs @ self.weights + self.bias)[None, :])[0]
        return {label: float(prob) for label, prob in zip(self.action_labels, probs)}

    def predict_action(self, summary: dict[str, Any], asset: str) -> tuple[str, float, dict[str, float]]:
        probs = self.predict_proba(summary, asset)
        action = max(probs, key=probs.get)
        risk = 1.0 - probs.get("none", 0.0)
        if action != "none":
            risk = max(risk, probs[action])
        return action, float(np.clip(risk, 0.0, 1.0)), probs

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "distilled_wolfguard_softmax",
            "action_labels": self.action_labels,
            "feature_names": self.feature_names,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "DistilledWolfGuardModel":
        payload = json.loads(Path(path).read_text())
        return cls(
            weights=np.array(payload["weights"], dtype=float),
            bias=np.array(payload["bias"], dtype=float),
            feature_mean=np.array(payload["feature_mean"], dtype=float),
            feature_scale=np.array(payload["feature_scale"], dtype=float),
            action_labels=list(payload.get("action_labels", ACTION_LABELS)),
            feature_names=list(payload.get("feature_names", FEATURE_NAMES)),
            metadata=dict(payload.get("metadata", {})),
        )


def _records_to_xy(records: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    label_to_id = {label: idx for idx, label in enumerate(ACTION_LABELS)}
    for record in records:
        label = record.get("oracle_label")
        if not record.get("label_available", label is not None) or label not in label_to_id:
            continue
        observation = record.get("observation", {}) or {}
        asset = str(record.get("asset", ""))
        if not asset or asset not in observation.get("market", {}):
            continue
        xs.append(extract_public_features(observation, asset))
        ys.append(label_to_id[label])
    if not xs:
        raise ValueError("No labeled trajectory records found for Distilled-WolfGuard training.")
    return np.vstack(xs), np.array(ys, dtype=int)


def train_distilled_model(
    records: Iterable[dict[str, Any]],
    epochs: int = 300,
    lr: float = 0.1,
    l2: float = 1e-4,
    seed: int = 0,
    class_balance: bool = True,
) -> DistilledWolfGuardModel:
    """Train a deterministic multinomial logistic Distilled-WolfGuard."""
    x, y = _records_to_xy(records)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale = np.where(feature_scale <= 1e-9, 1.0, feature_scale)
    xs = (x - feature_mean) / feature_scale

    rng = np.random.default_rng(seed)
    n_features = xs.shape[1]
    n_classes = len(ACTION_LABELS)
    weights = rng.normal(0.0, 0.01, size=(n_features, n_classes))
    bias = np.zeros(n_classes, dtype=float)

    counts = np.bincount(y, minlength=n_classes).astype(float)
    if class_balance:
        sample_weights = len(y) / np.maximum(counts[y] * n_classes, 1.0)
    else:
        sample_weights = np.ones_like(y, dtype=float)
    sample_weights = sample_weights / np.mean(sample_weights)

    final_loss = 0.0
    for _ in range(int(epochs)):
        logits = xs @ weights + bias
        probs = _softmax(logits)
        final_loss = float(
            -np.mean(sample_weights * np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12)))
            + l2 * float(np.sum(weights * weights))
        )
        delta = probs
        delta[np.arange(len(y)), y] -= 1.0
        delta *= sample_weights[:, None] / len(y)
        grad_w = xs.T @ delta + 2.0 * l2 * weights
        grad_b = delta.sum(axis=0)
        weights -= lr * grad_w
        bias -= lr * grad_b

    metadata = {
        "epochs": int(epochs),
        "lr": float(lr),
        "l2": float(l2),
        "seed": int(seed),
        "class_balance": bool(class_balance),
        "n_train_records": int(len(y)),
        "class_counts": {label: int(counts[idx]) for idx, label in enumerate(ACTION_LABELS)},
        "training_loss": final_loss,
    }
    return DistilledWolfGuardModel(
        weights=weights,
        bias=bias,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        metadata=metadata,
    )


def evaluate_distilled_model(
    model: DistilledWolfGuardModel,
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    label_to_id = {label: idx for idx, label in enumerate(ACTION_LABELS)}
    confusion = np.zeros((len(ACTION_LABELS), len(ACTION_LABELS)), dtype=int)
    total = 0
    correct = 0
    for record in records:
        label = record.get("oracle_label")
        if not record.get("label_available", label is not None) or label not in label_to_id:
            continue
        observation = record.get("observation", {}) or {}
        asset = str(record.get("asset", ""))
        if not asset or asset not in observation.get("market", {}):
            continue
        pred, _, _ = model.predict_action(observation, asset)
        yi = label_to_id[label]
        pi = label_to_id[pred]
        confusion[yi, pi] += 1
        total += 1
        correct += int(yi == pi)

    per_class = {}
    f1_values = []
    for idx, label in enumerate(ACTION_LABELS):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)

    return {
        "n_eval_records": total,
        "accuracy": float(correct / max(total, 1)),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


@dataclass
class DistilledWolfGuardPolicy:
    """Open, reproducible WolfGuard baseline trained from trajectory JSONL."""
    model_path: str | None = None
    name: str = "Distilled-WolfGuard"
    config: WolfGuardConfig = field(default_factory=WolfGuardConfig)
    clean_baseline: dict[str, dict[str, float]] = field(default_factory=dict)
    _model: DistilledWolfGuardModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        path = self.model_path or os.getenv(
            "WOLFBENCH_DISTILLED_MODEL",
            "paperoutputs/benchmark/distilled_wolfguard/model.json",
        )
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Distilled-WolfGuard model not found at {path}. "
                "Run `wolfbench train-distilled --dataset ... --out ...` first "
                "or set WOLFBENCH_DISTILLED_MODEL."
            )
        self.model_path = str(path)
        self._model = DistilledWolfGuardModel.load(path)

    def fit_baseline(self, baseline: dict[str, dict[str, float]]) -> None:
        self.clean_baseline = baseline

    def decide(self, day: int, summary: dict[str, Any]) -> dict[str, dict]:
        out = {}
        public_summary = dict(summary)
        public_summary.pop("oracle_view", None)
        for asset in public_summary.get("market", {}):
            action, risk, probs = self._model.predict_action(public_summary, asset)
            out[asset] = make_intervention(
                asset,
                action,
                risk=risk,
                reason="distilled",
                components={f"p_{label}": prob for label, prob in probs.items()},
            )
        return out