"""Tests for the custom MDP player interface."""

import json
from pathlib import Path

from magebench.orchestration.config import Config, MDPPlayer
from magebench.pilot.mdp import FirstChoiceMDP, MDPAction, MDPObservation, load_mdp


def test_mdp_observation_exposes_choices_and_hand():
    observation = MDPObservation(
        {
            "action_pending": True,
            "action_type": "GAME_SELECT",
            "response_type": "select",
            "context": "T3 PRECOMBAT_MAIN",
            "board": [{"is_you": True, "hand": [{"name": "Lightning Bolt"}]}],
            "choices": [{"name": "Mountain", "action": "land"}, {"name": "Bolt", "action": "cast"}],
        }
    )

    assert observation.action_pending
    assert observation.context == "T3 PRECOMBAT_MAIN"
    assert observation.hand == [{"name": "Lightning Bolt"}]
    assert [choice.name for choice in observation.choices] == ["Mountain", "Bolt"]


def test_mdp_action_serializes_to_mcp_arguments():
    action = MDPAction(choice=2, mana_plan="p1,p2", auto_tap=True)

    assert action.as_mcp_arguments() == {
        "choice": "2",
        "mana_plan": "p1,p2",
        "auto_tap": True,
    }
    assert MDPAction(choice=True).as_mcp_arguments() == {"choice": "yes"}
    assert MDPAction(choice=False).as_mcp_arguments() == {"choice": "no"}


def test_first_choice_policy():
    policy = FirstChoiceMDP()
    observation = MDPObservation({"action_pending": True, "response_type": "boolean"})
    assert policy.act(observation) == MDPAction(choice=True)


def test_mdp_player_loads_from_config(tmp_path: Path):
    policy_file = tmp_path / "policy.py"
    policy_file.write_text("class Policy:\n    pass\n")
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "gameType": "Two Player Duel",
        "players": [{"type": "mdp", "name": "Agent", "deck": "random", "policy": "policy.py"}],
    }))

    config = Config(config_file=config_file)
    config.load_config()

    assert len(config.mdp_players) == 1
    player = config.mdp_players[0]
    assert isinstance(player, MDPPlayer)
    assert player.name == "Agent"
    assert player.policy == str(policy_file)


def test_load_mdp_from_python_file(tmp_path: Path):
    policy_file = tmp_path / "policy.py"
    policy_file.write_text(
        "from magebench.pilot.mdp import BaseMDP, MDPAction\n"
        "class MyPolicy(BaseMDP):\n"
        "    def act(self, observation):\n"
        "        return MDPAction(choice=1)\n"
    )

    policy = load_mdp(str(policy_file))
    assert policy.__class__.__name__ == "MyPolicy"
