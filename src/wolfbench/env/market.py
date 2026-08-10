"""Order types and stylized market with depth, spread, and short-horizon dynamics.

This is intentionally not a full ITCH/OUCH simulator. Each "day" runs a small
number of intra-day clearing rounds. Spoof orders can be observed in the depth
imbalance and cancelled within ``cancel_latency_steps``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Side = Literal["buy", "sell"]


@dataclass
class Order:
    agent_id: str
    asset: str
    side: Side
    quantity: float
    is_spoof: bool = False
    cancel_after_steps: int = 0     # 0 = real order
    is_wash: bool = False
    counterparty_id: str | None = None  # for wash trades, the colluder
    is_harmful: bool = False        # comes from a harmful agent


@dataclass
class TradeRecord:
    day: int
    step: int
    asset: str
    buyer_id: str
    seller_id: str
    quantity: float
    price: float
    is_wash: bool = False
    buyer_is_harmful: bool = False
    seller_is_harmful: bool = False


@dataclass
class AssetState:
    asset_id: str
    fundamental: float
    fundamental_vol: float
    base_liquidity: float
    price: float
    last_price: float = 0.0
    volume_today: float = 0.0
    real_volume_today: float = 0.0   # excludes wash
    spoof_buy_size: float = 0.0
    spoof_sell_size: float = 0.0
    cancel_count: int = 0
    order_count: int = 0
    spread_bps: float = 20.0

    history: dict[str, list[float]] = field(default_factory=lambda: {
        "price": [],
        "fundamental": [],
        "volume": [],
        "real_volume": [],
        "spread_bps": [],
        "depth_imbalance": [],
        "cancel_rate": [],
        "wash_share": [],
    })


class MarketEnv:
    """Multi-asset market with daily clearing.

    Pricing model per day:
        net_pressure = (real_buy - real_sell) + imbalance_signal_perceived * 0
        impact = net_pressure / liquidity
        price_{t+1} = price_t * (1 + impact + fundamental_drift + noise)
    Spoof and wash do NOT enter ``net_pressure`` directly (they were not real
    intent), but they leave observable footprints (depth imbalance, cancel
    rate, wash share) that other agents can read.
    """

    def __init__(self, scenario, rng: np.random.Generator, liquidity_scale: float = 1.0):
        self.rng = rng
        self.assets: dict[str, AssetState] = {}
        for a in scenario.assets:
            self.assets[a.id] = AssetState(
                asset_id=a.id,
                fundamental=a.fundamental,
                fundamental_vol=a.fundamental_vol,
                base_liquidity=a.initial_liquidity * liquidity_scale,
                price=a.fundamental,
                last_price=a.fundamental,
                spread_bps=scenario.market_makers["base_spread_bps"],
            )
        self.base_spread_bps = scenario.market_makers["base_spread_bps"]
        self.inventory_aversion = scenario.market_makers["inventory_aversion"]
        self.trades: list[TradeRecord] = []

    # ------------------------------------------------------------------ daily

    def begin_day(self) -> None:
        for s in self.assets.values():
            s.last_price = s.price
            s.volume_today = 0.0
            s.real_volume_today = 0.0
            s.spoof_buy_size = 0.0
            s.spoof_sell_size = 0.0
            s.cancel_count = 0
            s.order_count = 0

    def end_day(self, day: int) -> None:
        for s in self.assets.values():
            # fundamental random walk
            drift = self.rng.normal(0.0, s.fundamental_vol)
            s.fundamental *= float(np.exp(drift))

            # spread reacts to inventory aversion and noise
            stress = abs(s.price - s.fundamental) / max(s.fundamental, 1e-6)
            s.spread_bps = self.base_spread_bps * (1.0 + 4.0 * stress)

            depth_imb = self._depth_imbalance(s)
            cancel_rate = (s.cancel_count / max(s.order_count, 1))
            wash_share = 0.0
            if s.volume_today > 0:
                wash_share = 1.0 - (s.real_volume_today / s.volume_today)

            s.history["price"].append(s.price)
            s.history["fundamental"].append(s.fundamental)
            s.history["volume"].append(s.volume_today)
            s.history["real_volume"].append(s.real_volume_today)
            s.history["spread_bps"].append(s.spread_bps)
            s.history["depth_imbalance"].append(depth_imb)
            s.history["cancel_rate"].append(cancel_rate)
            s.history["wash_share"].append(wash_share)

    # ------------------------------------------------------------------ orders

    def submit_orders(self, day: int, step: int, orders: list[Order]) -> list[TradeRecord]:
        """Net real orders into a daily impact; record spoof/wash footprints."""
        new_trades: list[TradeRecord] = []
        # group by asset
        by_asset: dict[str, list[Order]] = {a: [] for a in self.assets}
        for o in orders:
            if o.asset in by_asset:
                by_asset[o.asset].append(o)

        for asset_id, asset_orders in by_asset.items():
            s = self.assets[asset_id]
            real_buy = 0.0
            real_sell = 0.0

            # wash trades cross internally and add only to volume
            washes: list[Order] = [o for o in asset_orders if o.is_wash]
            spoofs = [o for o in asset_orders if o.is_spoof]
            reals = [o for o in asset_orders if not o.is_spoof and not o.is_wash]

            for o in reals:
                s.order_count += 1
                if o.side == "buy":
                    real_buy += o.quantity
                else:
                    real_sell += o.quantity

            # spoofs: book pressure visible, then cancelled
            for o in spoofs:
                s.order_count += 1
                s.cancel_count += 1
                if o.side == "buy":
                    s.spoof_buy_size += o.quantity
                else:
                    s.spoof_sell_size += o.quantity

            # match real orders against pooled liquidity at mid; price impact applied
            net = real_buy - real_sell
            matched = min(real_buy, real_sell)
            # pseudo-counterparty matching -- record for wealth bookkeeping
            if matched > 0 and reals:
                buyers = [o for o in reals if o.side == "buy"]
                sellers = [o for o in reals if o.side == "sell"]
                self._match_pool(day, step, asset_id, buyers, sellers, matched, s.price, new_trades)
            # remaining net order trades against market makers at mid +/- spread
            spread = s.price * (s.spread_bps / 10000.0)
            if net > 0:
                # buyers lift offer
                self._match_with_mm(day, step, asset_id, [o for o in reals if o.side == "buy"],
                                    abs(net), s.price + spread / 2, "buy", new_trades)
            elif net < 0:
                self._match_with_mm(day, step, asset_id, [o for o in reals if o.side == "sell"],
                                    abs(net), s.price - spread / 2, "sell", new_trades)

            # wash volume
            for o in washes:
                # wash matched at current mid, both sides held by colluders
                qty = o.quantity
                price = s.price
                s.volume_today += qty
                # do not affect real_volume_today
                tr = TradeRecord(
                    day=day, step=step, asset=asset_id,
                    buyer_id=o.agent_id, seller_id=o.counterparty_id or o.agent_id,
                    quantity=qty, price=price, is_wash=True,
                    buyer_is_harmful=True, seller_is_harmful=True,
                )
                new_trades.append(tr)

            # apply price impact from net real flow (saturating, bounded per day)
            liquidity = s.base_liquidity
            gap = (s.price - s.fundamental) / max(s.fundamental, 1e-6)
            # MMs widen effective depth as price diverges from fundamental
            eff_liquidity = liquidity * (1.0 + 5.0 * abs(gap))
            raw = (net / max(eff_liquidity, 1e-6))
            impact = 0.06 * float(np.tanh(raw))   # cap ±6% per day from flow
            noise = self.rng.normal(0.0, s.fundamental_vol * 0.5)
            # mean-reversion toward fundamental, log-shaped
            mean_rev = 0.05 * float(np.tanh(2.0 * (s.fundamental - s.price) / max(s.price, 1e-6)))
            s.price = max(0.01, s.price * (1.0 + impact + noise + mean_rev))

        self.trades.extend(new_trades)
        return new_trades

    # -------------------------------------------------------------- internals

    def _match_pool(self, day, step, asset_id, buyers, sellers, total_qty, price, out):
        if total_qty <= 0 or not buyers or not sellers:
            return
        s = self.assets[asset_id]
        # proportional fill
        b_total = sum(o.quantity for o in buyers)
        s_total = sum(o.quantity for o in sellers)
        scale_b = total_qty / b_total if b_total > 0 else 0.0
        scale_s = total_qty / s_total if s_total > 0 else 0.0
        bi, si = 0, 0
        b_left = buyers[0].quantity * scale_b if buyers else 0.0
        s_left = sellers[0].quantity * scale_s if sellers else 0.0
        while bi < len(buyers) and si < len(sellers):
            q = min(b_left, s_left)
            if q > 1e-9:
                out.append(TradeRecord(
                    day=day, step=step, asset=asset_id,
                    buyer_id=buyers[bi].agent_id, seller_id=sellers[si].agent_id,
                    quantity=q, price=price,
                    buyer_is_harmful=buyers[bi].is_harmful,
                    seller_is_harmful=sellers[si].is_harmful,
                ))
                s.volume_today += q
                s.real_volume_today += q
            b_left -= q
            s_left -= q
            if b_left <= 1e-9:
                bi += 1
                if bi < len(buyers):
                    b_left = buyers[bi].quantity * scale_b
            if s_left <= 1e-9:
                si += 1
                if si < len(sellers):
                    s_left = sellers[si].quantity * scale_s

    def _match_with_mm(self, day, step, asset_id, takers, total_qty, price, side, out):
        if total_qty <= 0 or not takers:
            return
        s = self.assets[asset_id]
        t_total = sum(o.quantity for o in takers)
        if t_total <= 0:
            return
        for o in takers:
            q = total_qty * (o.quantity / t_total)
            if side == "buy":
                tr = TradeRecord(day=day, step=step, asset=asset_id,
                                 buyer_id=o.agent_id, seller_id="market_maker",
                                 quantity=q, price=price,
                                 buyer_is_harmful=o.is_harmful)
            else:
                tr = TradeRecord(day=day, step=step, asset=asset_id,
                                 buyer_id="market_maker", seller_id=o.agent_id,
                                 quantity=q, price=price,
                                 seller_is_harmful=o.is_harmful)
            out.append(tr)
            s.volume_today += q
            s.real_volume_today += q

    def _depth_imbalance(self, s: AssetState) -> float:
        total = s.spoof_buy_size + s.spoof_sell_size
        if total <= 1e-9:
            return 0.0
        return (s.spoof_buy_size - s.spoof_sell_size) / total

    # ------------------------------------------------------------------ views

    def snapshot(self) -> dict[str, dict[str, float]]:
        out = {}
        for aid, s in self.assets.items():
            out[aid] = {
                "price": s.price,
                "fundamental": s.fundamental,
                "spread_bps": s.spread_bps,
                "volume": s.volume_today,
                "real_volume": s.real_volume_today,
                "depth_imbalance": self._depth_imbalance(s),
                "cancel_rate": s.cancel_count / max(s.order_count, 1),
                "wash_share": (1.0 - s.real_volume_today / s.volume_today) if s.volume_today > 0 else 0.0,
            }
        return out
