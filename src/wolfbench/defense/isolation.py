"""Process-isolated defense policy adapter for official evaluation."""
from __future__ import annotations

import importlib
import json
import math
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

from wolfbench.defense.baselines import get_policy


def _json_clean(obj: Any) -> Any:
    return json.loads(json.dumps(_json_safe(obj), allow_nan=False))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    try:
        val = float(obj)
    except (TypeError, ValueError):
        return str(obj)
    return val if math.isfinite(val) else 0.0


def _import_policy(spec: str, kwargs: dict[str, Any]) -> Any:
    if ":" in spec:
        mod_name, cls_name = spec.split(":", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)(**kwargs)
    return get_policy(spec, **kwargs)


def _policy_worker(conn, spec: str, kwargs: dict[str, Any]) -> None:
    policy = _import_policy(spec, kwargs)
    conn.send({"ok": True, "name": getattr(policy, "name", spec)})
    try:
        while True:
            msg = conn.recv()
            cmd = msg.get("cmd")
            if cmd == "close":
                conn.send({"ok": True})
                break
            if cmd == "fit_baseline":
                if hasattr(policy, "fit_baseline"):
                    policy.fit_baseline(_json_clean(msg.get("baseline", {})))
                conn.send({"ok": True})
            elif cmd == "decide":
                actions = policy.decide(int(msg["day"]), _json_clean(msg["summary"]))
                conn.send({"ok": True, "actions": _json_clean(actions)})
            else:
                conn.send({"ok": False, "error": f"unknown command: {cmd}"})
    except BaseException as exc:
        conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        conn.close()


@dataclass
class SubprocessWolfGuardPolicy:
    """Defense adapter that keeps submitted code out of the env process."""
    spec: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(
            target=_policy_worker,
            args=(child_conn, self.spec, self.kwargs),
            daemon=True,
        )
        self._proc.start()
        ready = self._recv("startup")
        self.name = ready.get("name", self.spec)

    def _recv(self, label: str) -> dict[str, Any]:
        if not self._parent_conn.poll(self.timeout_s):
            self.close()
            raise TimeoutError(f"policy subprocess timed out during {label}")
        msg = self._parent_conn.recv()
        if not msg.get("ok", False):
            self.close()
            raise RuntimeError(f"policy subprocess failed during {label}: {msg.get('error')}")
        return msg

    def fit_baseline(self, baseline: dict[str, dict[str, float]]) -> None:
        self._parent_conn.send({"cmd": "fit_baseline", "baseline": _json_clean(baseline)})
        self._recv("fit_baseline")

    def decide(self, day: int, summary: dict[str, Any]) -> dict[str, dict]:
        self._parent_conn.send({"cmd": "decide", "day": int(day), "summary": _json_clean(summary)})
        return self._recv("decide").get("actions", {})

    def close(self) -> None:
        if getattr(self, "_proc", None) is None:
            return
        if self._proc.is_alive():
            try:
                self._parent_conn.send({"cmd": "close"})
                if self._parent_conn.poll(1.0):
                    self._parent_conn.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()