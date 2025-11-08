from typing import Any, Dict, Optional

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.environment.env import ActionType, ObsType, PokeEnv
from poke_env.battle import AbstractBattle, Battle

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from gymnasium.spaces import Box, Discrete, Space
import numpy as np
import datetime


class ExampleEnv(SinglesEnv):
    LOW = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
    HIGH = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(
                np.array(self.LOW, dtype=np.float32),
                np.array(self.HIGH, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, Battle)
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=battle.opponent_active_pokemon._data.type_chart,
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )


def make_env():
    battle_format = "gen5randombattle"
    opponent = RandomPlayer(battle_format=battle_format)
    poke_env = ExampleEnv(
        battle_format=battle_format,
        # account_configuration1=AccountConfiguration("autoeevee", "password"),
        strict=False,
    )
    return SingleAgentWrapper(env=poke_env, opponent=opponent)


def main():
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
    )
    model.learn(total_timesteps=100000)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model.save(f"saves/autoeeveeV0_{timestamp}")


if __name__ == "__main__":
    main()
