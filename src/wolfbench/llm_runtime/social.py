"""One-hop language delivery; subsequent forwarding requires an agent action."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from wolfbench.agents.social_game import BoundedSocialEnv


class LanguageSocialEnv(BoundedSocialEnv):
    game_version = "language_society_v1"

    def __init__(self, graph, scenario, rng, benign_ids):
        super().__init__(graph, scenario, rng)
        self.benign_ids = set(benign_ids)
        self.messages_by_id: dict[str, Any] = {}

    def step(self, day, messages, market_returns):
        # Transport is not a decision-making bot. Disable the old automatic
        # second hop for EVERY source, including harmful amplifier accounts.
        routed = []
        for index, message in enumerate(messages):
            mid = message.message_id or f"d{day}:{message.sender_id}:{index}"
            message = replace(message, message_id=mid, is_bot=False)
            self.messages_by_id[mid] = message
            routed.append(message)
        start = len(self.message_events)
        super().step(day, routed, market_returns)
        for event in self.message_events[start:]:
            source = self.messages_by_id[event["message_id"]]
            event["text"] = source.text
            event["parent_message_id"] = source.parent_message_id

    def _exposure_event(self, *args, **kwargs):
        event = super()._exposure_event(*args, **kwargs)
        message = self.messages_by_id[event["message_id"]]
        event["text"] = message.text if self.content_visible else ""
        event["parent_message_id"] = message.parent_message_id
        return event

    def _deliver(self, event, harmful):
        super()._deliver(event, harmful)
        # The numerator and denominator both refer to benign population agents.
        if event["recipient_id"] not in self.benign_ids:
            self.state.cascade_size[event["asset"]].discard(event["recipient_id"])
