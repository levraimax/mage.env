"""Custom MDP-style XMage player interface."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession

from magebench.common.log import get_logger, setup_logging
from magebench.pilot.bridge_transport import build_bridge_launch_args, spawn_bridge_http
from magebench.pilot.pilot_state import BoardCursorTracker
from magebench.pilot.tool_error import extract_text_content

logger = get_logger(__name__)


@dataclass(frozen=True)
class MDPChoice:
    """One option in the pending XMage decision."""

    index: int
    name: str
    action: str | None = None
    mana_cost: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class MDPObservation:
    """State and legal choices presented to an MDP policy."""

    raw: dict[str, Any]

    @property
    def action_pending(self) -> bool:
        return bool(self.raw.get("action_pending"))

    @property
    def game_over(self) -> bool:
        return bool(self.raw.get("game_over"))

    @property
    def action_type(self) -> str:
        return str(self.raw.get("action_type", ""))

    @property
    def response_type(self) -> str:
        return str(self.raw.get("response_type", ""))

    @property
    def message(self) -> str:
        return str(self.raw.get("message", ""))

    @property
    def context(self) -> str:
        return str(self.raw.get("context", ""))

    @property
    def board(self) -> list[dict[str, Any]]:
        board = self.raw.get("board")
        return board if isinstance(board, list) else []

    @property
    def choices(self) -> list[MDPChoice]:
        raw_choices = self.raw.get("choices")
        if not isinstance(raw_choices, list):
            return []
        choices: list[MDPChoice] = []
        for index, choice in enumerate(raw_choices):
            if not isinstance(choice, dict):
                continue
            choices.append(
                MDPChoice(
                    index=index,
                    name=str(choice.get("name", choice.get("text", ""))),
                    action=choice.get("action") if isinstance(choice.get("action"), str) else None,
                    mana_cost=choice.get("mana_cost") if isinstance(choice.get("mana_cost"), str) else None,
                    raw=choice,
                )
            )
        return choices

    @property
    def hand(self) -> list[dict[str, Any]]:
        for player in self.board:
            if player.get("is_you"):
                hand = player.get("hand")
                return hand if isinstance(hand, list) else []
        return []


@dataclass(frozen=True)
class MDPAction:
    """An action returned by an MDP policy."""

    choice: int | str | bool | None = None
    amount: int | None = None
    amounts: list[int] | None = None
    pile: int | None = None
    text: str | None = None
    mana_plan: str | None = None
    auto_tap: bool | None = None
    attackers: str | None = None
    blockers: str | None = None

    def as_mcp_arguments(self) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        if self.choice is not None:
            arguments["choice"] = "yes" if self.choice is True else "no" if self.choice is False else str(self.choice)
        if self.amount is not None:
            arguments["amount"] = self.amount
        if self.amounts is not None:
            arguments["amounts"] = self.amounts
        if self.pile is not None:
            arguments["pile"] = self.pile
        if self.text is not None:
            arguments["text"] = self.text
        if self.mana_plan is not None:
            arguments["mana_plan"] = self.mana_plan
        if self.auto_tap is not None:
            arguments["auto_tap"] = self.auto_tap
        if self.attackers is not None:
            arguments["attackers"] = self.attackers
        if self.blockers is not None:
            arguments["blockers"] = self.blockers
        return arguments


class BaseMDP:
    """Base class for a custom MDP policy.

    Subclasses implement ``act``. Override ``on_start`` and ``on_game_over`` for
    lifecycle hooks. The class receives structured observations and returns
    structured actions; MCP details stay outside the policy implementation.
    """

    def on_start(self, username: str) -> None:
        del username

    def act(self, observation: MDPObservation) -> MDPAction:
        raise NotImplementedError("MDP subclasses must implement act(observation)")

    def on_game_over(self, observation: MDPObservation) -> None:
        del observation


class FirstChoiceMDP(BaseMDP):
    """Minimal deterministic policy for testing and as a starting example."""

    def act(self, observation: MDPObservation) -> MDPAction:
        if observation.response_type == "boolean":
            return MDPAction(choice=True)
        if observation.choices:
            return MDPAction(choice=observation.choices[0].index)
        return MDPAction(choice="no")


def load_mdp(policy_spec: str) -> BaseMDP:
    """Load an MDP policy from a Python file or dotted module."""
    policy_path = Path(policy_spec)
    module_name: str

    if policy_path.exists():
        module_name = f"_magebench_mdp_{policy_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, policy_path)
        assert spec is not None and spec.loader is not None, f"Cannot load MDP module: {policy_path}"
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(policy_spec)

    factory = getattr(module, "create_mdp", None)
    if factory is not None:
        assert callable(factory), f"{policy_spec}: create_mdp must be callable"
        policy = factory()
    else:
        classes = [
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if cls is not BaseMDP and issubclass(cls, BaseMDP)
        ]
        assert classes, f"{policy_spec}: define create_mdp() or a BaseMDP subclass"
        assert len(classes) == 1, f"{policy_spec}: define exactly one BaseMDP subclass or create_mdp()"
        policy = classes[0]()

    assert isinstance(policy, BaseMDP), f"{policy_spec}: policy must implement BaseMDP"
    return policy


async def call_tool(session: ClientSession, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await session.call_tool(name, arguments)
    text = extract_text_content(name, result)
    data = json.loads(text)
    assert isinstance(data, dict), f"{name} returned non-object JSON: {text}"
    return data


async def run_mdp(
    server: str,
    port: int,
    username: str,
    project_root: Path,
    deck_path: Path | None,
    policy_spec: str,
) -> None:
    policy = load_mdp(policy_spec)
    policy.on_start(username)

    launch_args = build_bridge_launch_args(
        server=server,
        port=port,
        username=username,
        deck_path=deck_path,
        heap_size_mb=512,
    )

    tracker = BoardCursorTracker()

    async with spawn_bridge_http(
        mvn_args=launch_args.mvn_args,
        project_root=project_root,
        jvm_args=launch_args.jvm_args,
    ) as session:
        await session.initialize()

        while True:
            pass_args: dict[str, Any] = {}
            tracker.inject("pass_priority", pass_args)
            data = await call_tool(session, "pass_priority", pass_args)
            tracker.extract(json.dumps(data))

            observation = MDPObservation(raw=data)
            if observation.game_over:
                policy.on_game_over(observation)
                return

            if not observation.action_pending:
                await asyncio.sleep(0.1)
                continue

            if not data.get("choices"):
                choice_args: dict[str, Any] = {}
                tracker.inject("get_action_choices", choice_args)
                data = await call_tool(session, "get_action_choices", choice_args)
                tracker.extract(json.dumps(data))
                observation = MDPObservation(raw=data)

            action = policy.act(observation)
            assert isinstance(action, MDPAction), "MDP act() must return MDPAction"
            await call_tool(session, "choose_action", action.as_mcp_arguments())


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Custom MDP player for XMage")
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--port", type=int, default=17171)
    parser.add_argument("--username", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--policy", required=True, help="Python file or dotted module containing create_mdp()")
    args = parser.parse_args()

    asyncio.run(
        run_mdp(
            server=args.server,
            port=args.port,
            username=args.username,
            project_root=args.project_root.resolve(),
            deck_path=args.deck,
            policy_spec=args.policy,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
