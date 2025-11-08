#!/usr/bin/env python3
"""Simple online reinforcement learning script for local Pokémon Showdown via poke-env."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - better error for missing torch
    raise SystemExit(
        "This script requires PyTorch. Install it first, e.g. `pip install torch`."
    ) from exc

from poke_env.player.baselines import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration


class HashingTokenizer:
    """Hash tokens to a fixed vocabulary so we can embed them later."""

    def __init__(
        self,
        vocab_size: int = 8192,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
    ) -> None:
        if vocab_size < 4:
            raise ValueError("vocab_size must be >= 4")
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self._fixed = {pad_token: 0, unk_token: 1}
        self._mod = vocab_size - len(self._fixed)

    def encode(self, token: str) -> int:
        if token in self._fixed:
            return self._fixed[token]
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        return len(self._fixed) + (value % self._mod)

    def encode_many(self, tokens: Sequence[str]) -> np.ndarray:
        encoded = np.zeros(len(tokens), dtype=np.int64)
        for idx, token in enumerate(tokens):
            encoded[idx] = self.encode(token)
        return encoded

    @property
    def pad_index(self) -> int:
        return self._fixed[self.pad_token]


class ObservationBuilder:
    """Turn battle state into a fixed-length list of discrete tokens."""

    def __init__(self, tokenizer: HashingTokenizer, max_moves: int = 4, max_switches: int = 5) -> None:
        self.tokenizer = tokenizer
        self.max_moves = max_moves
        self.max_switches = max_switches
        self.observation_length = self._template_length()

    def _template_length(self) -> int:
        # Mirror the logic of build() but fill with pad tokens.
        tokens = []
        tokens.extend(["format:<pad>", "turn:0"])
        tokens.extend(["self_species:<pad>", "self_status:<pad>", "self_hp:0"])
        tokens.extend(["opp_species:<pad>", "opp_status:<pad>", "opp_hp:0"])
        for idx in range(self.max_moves):
            tokens.extend(
                [
                    f"move{idx}_id:<pad>",
                    f"move{idx}_type:<pad>",
                    f"move{idx}_cat:<pad>",
                    f"move{idx}_pp:0",
                ]
            )
        for idx in range(self.max_switches):
            tokens.append(f"switch{idx}_species:<pad>")
        tokens.extend(
            [
                "flag_force_switch:0",
                "flag_can_mega:0",
                "flag_can_dmax:0",
                "flag_can_z:0",
                "flag_can_tera:0",
                "flag_trapped:0",
            ]
        )
        return len(tokens)

    def build(self, battle) -> np.ndarray:
        tokens: List[str] = []
        tokens.append(f"format:{battle.format}")
        tokens.append(f"turn:{min(battle.turn, 255)}")

        active = battle.active_pokemon
        if active is None:
            tokens.extend(["self_species:<unk>", "self_status:<unk>", "self_hp:0"])
        else:
            status = getattr(active.status, "name", "none") or "none"
            hp_bucket = int(round((active.current_hp_fraction or 0.0) * 10))
            tokens.extend(
                [
                    f"self_species:{active.species}",
                    f"self_status:{status}",
                    f"self_hp:{hp_bucket}",
                ]
            )

        opponent = battle.opponent_active_pokemon
        if opponent is None:
            tokens.extend(["opp_species:<unk>", "opp_status:<unk>", "opp_hp:0"])
        else:
            opp_status = getattr(opponent.status, "name", "none") or "none"
            opp_hp_bucket = int(round((opponent.current_hp_fraction or 0.0) * 10))
            tokens.extend(
                [
                    f"opp_species:{opponent.species}",
                    f"opp_status:{opp_status}",
                    f"opp_hp:{opp_hp_bucket}",
                ]
            )

        moves = battle.available_moves or []
        for idx in range(self.max_moves):
            if idx < len(moves):
                move = moves[idx]
                tokens.extend(
                    [
                        f"move{idx}_id:{move.id}",
                        f"move{idx}_type:{getattr(move.type, 'name', 'none')}",
                        f"move{idx}_cat:{getattr(move.category, 'name', 'none')}",
                        f"move{idx}_pp:{move.current_pp if move.current_pp is not None else 0}",
                    ]
                )
            else:
                tokens.extend(
                    [
                        f"move{idx}_id:<pad>",
                        f"move{idx}_type:<pad>",
                        f"move{idx}_cat:<pad>",
                        f"move{idx}_pp:0",
                    ]
                )

        switches = [s for s in (battle.available_switches or []) if s is not None]
        for idx in range(self.max_switches):
            if idx < len(switches):
                tokens.append(f"switch{idx}_species:{switches[idx].species}")
            else:
                tokens.append(f"switch{idx}_species:<pad>")

        tokens.extend(
            [
                f"flag_force_switch:{int(bool(getattr(battle, 'force_switch', False)))}",
                f"flag_can_mega:{int(bool(getattr(battle, 'can_mega_evolve', False)))}",
                f"flag_can_dmax:{int(bool(getattr(battle, 'can_dynamax', False)))}",
                f"flag_can_z:{int(bool(getattr(battle, 'can_z_move', False)))}",
                f"flag_can_tera:{int(bool(getattr(battle, 'can_tera', False)))}",
                f"flag_trapped:{int(bool(getattr(battle, 'trapped', False)))}",
            ]
        )

        assert len(tokens) == self.observation_length
        return self.tokenizer.encode_many(tokens)

    def empty(self) -> np.ndarray:
        pad_tokens = [self.tokenizer.pad_token] * self.observation_length
        return self.tokenizer.encode_many(pad_tokens)


class ActionMapper:
    """Map action indices to actual battle orders."""

    NUM_ACTIONS = 13

    def __init__(self, max_moves: int = 4, max_switches: int = 5) -> None:
        self.max_moves = max_moves
        self.max_switches = max_switches

    def valid_actions(self, battle) -> List[int]:
        valid: List[int] = []
        switches = [s for s in (battle.available_switches or []) if s is not None]
        moves = battle.available_moves or []

        if getattr(battle, "force_switch", False):
            for idx in range(min(len(switches), self.max_switches)):
                valid.append(4 + idx)
            return valid or [4]

        for idx in range(min(len(moves), self.max_moves)):
            valid.append(idx)

        for idx in range(min(len(switches), self.max_switches)):
            valid.append(4 + idx)

        if battle.can_mega_evolve and moves:
            valid.append(9)
        if battle.can_dynamax and moves:
            valid.append(10)
        if battle.can_z_move and battle.active_pokemon and battle.active_pokemon.available_z_moves:
            valid.append(11)
        if battle.can_tera and moves:
            valid.append(12)

        return valid or [0]

    def to_order(self, player: Player, battle, action_idx: int):
        switches = [s for s in (battle.available_switches or []) if s is not None]
        moves = battle.available_moves or []

        if getattr(battle, "force_switch", False):
            if switches:
                idx = min(max(action_idx - 4, 0), len(switches) - 1)
                return player.create_order(switches[idx])
            return player.choose_default_move()

        if 0 <= action_idx <= 3 and action_idx < len(moves):
            return player.create_order(moves[action_idx])

        if 4 <= action_idx <= 8:
            switch_idx = action_idx - 4
            if switch_idx < len(switches):
                return player.create_order(switches[switch_idx])

        if action_idx == 9 and battle.can_mega_evolve and moves:
            return player.create_order(moves[0], mega=True)

        if action_idx == 10 and battle.can_dynamax and moves:
            return player.create_order(moves[0], dynamax=True)

        if action_idx == 11 and battle.can_z_move and battle.active_pokemon:
            z_moves = battle.active_pokemon.available_z_moves
            if z_moves:
                return player.create_order(z_moves[0], z_move=True)

        if action_idx == 12 and battle.can_tera and moves:
            return player.create_order(moves[0], terastallize=True)

        return player.choose_random_move(battle)


@dataclass
class BattleState:
    last_obs: np.ndarray | None
    last_action: int | None
    last_team_hp: float
    last_opp_hp: float


class ReplayBuffer:
    """Cyclic buffer for experience replay."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Embedding-based Q-network for discrete action values."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        hidden_dim: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(seq_len * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        emb = self.embedding(tokens)
        flat = emb.view(emb.size(0), self.seq_len * emb.size(-1))
        return self.net(flat)


class OnlineRLAgent(Player):
    """Minimal DQN-style agent operating directly on poke-env battles."""

    def __init__(
        self,
        *,
        account: AccountConfiguration,
        server: ServerConfiguration,
        battle_format: str,
        observation_builder: ObservationBuilder,
        action_mapper: ActionMapper,
        q_network: QNetwork,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.1,
        epsilon_decay_steps: int = 50_000,
        batch_size: int = 32,
        replay_capacity: int = 5_000,
        target_update_steps: int = 1_000,
        win_reward: float = 1.0,
        loss_penalty: float = 1.0,
        max_concurrent_battles: int = 1,
    ) -> None:
        super().__init__(
            account_configuration=account,
            battle_format=battle_format,
            server_configuration=server,
            max_concurrent_battles=max_concurrent_battles,
            start_timer_on_battle_start=True,
        )
        self.observation_builder = observation_builder
        self.action_mapper = action_mapper
        self.q_network = q_network
        hidden_dim = q_network.net[2].in_features
        self.target_network = QNetwork(
            vocab_size=q_network.embedding.num_embeddings,
            seq_len=q_network.seq_len,
            embed_dim=q_network.embedding.embedding_dim,
            hidden_dim=hidden_dim,
            num_actions=ActionMapper.NUM_ACTIONS,
        )
        self.target_network.load_state_dict(q_network.state_dict())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.eval()
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(replay_capacity)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=3e-4)
        self.loss_fn = F.smooth_l1_loss
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self.target_update_steps = target_update_steps
        self._global_step = 0
        self._epsilon_start = epsilon_start
        self._epsilon_final = epsilon_final
        self._epsilon_decay_steps = epsilon_decay_steps
        self._battle_states: dict[str, BattleState] = {}
        self._blank_state_tokens = observation_builder.empty()

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self._global_step / float(max(1, self._epsilon_decay_steps)))
        return self._epsilon_start + frac * (self._epsilon_final - self._epsilon_start)

    def _team_hp(self, battle) -> float:
        total = 0.0
        for mon in battle.team.values():
            if mon.current_hp_fraction is not None:
                total += mon.current_hp_fraction
            elif mon.fainted:
                total += 0.0
            else:
                total += 1.0
        return total

    def _opponent_hp(self, battle) -> float:
        total = 0.0
        for mon in battle.opponent_team.values():
            if mon.current_hp_fraction is not None:
                total += mon.current_hp_fraction
            elif mon.fainted:
                total += 0.0
            else:
                total += 1.0
        return total

    def _init_battle_state(self, battle) -> BattleState:
        return BattleState(
            last_obs=None,
            last_action=None,
            last_team_hp=self._team_hp(battle),
            last_opp_hp=self._opponent_hp(battle),
        )

    def _store_transition(
        self,
        state_tokens: np.ndarray,
        action: int,
        reward: float,
        next_tokens: np.ndarray,
        done: bool,
    ) -> None:
        state_tensor = torch.tensor(state_tokens, dtype=torch.long)
        next_tensor = torch.tensor(next_tokens, dtype=torch.long)
        self.buffer.add(state_tensor, action, reward, next_tensor, done)

    def _optimize(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q
        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._global_step += 1
        if self._global_step % self.target_update_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _select_action(self, observation: np.ndarray, battle) -> int:
        valid_actions = self.action_mapper.valid_actions(battle)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        obs_tensor = torch.tensor(observation, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)[0]
        valid_tensor = torch.tensor(valid_actions, dtype=torch.long, device=self.device)
        best_idx = valid_tensor[torch.argmax(q_values[valid_tensor])].item()
        return best_idx

    def choose_move(self, battle):  # type: ignore[override]
        state = self._battle_states.setdefault(battle.battle_tag, self._init_battle_state(battle))
        current_obs = self.observation_builder.build(battle)
        current_team_hp = self._team_hp(battle)
        current_opp_hp = self._opponent_hp(battle)
        reward = (current_team_hp - state.last_team_hp) - (current_opp_hp - state.last_opp_hp)

        if state.last_obs is not None and state.last_action is not None:
            self._store_transition(state.last_obs, state.last_action, reward, current_obs, False)
            self._optimize()

        action = self._select_action(current_obs, battle)
        state.last_obs = current_obs
        state.last_action = action
        state.last_team_hp = current_team_hp
        state.last_opp_hp = current_opp_hp
        return self.action_mapper.to_order(self, battle, action)

    def _battle_finished_callback(self, battle):  # type: ignore[override]
        state = self._battle_states.pop(battle.battle_tag, None)
        if state is None or state.last_obs is None or state.last_action is None:
            return
        final_team_hp = self._team_hp(battle)
        final_opp_hp = self._opponent_hp(battle)
        reward = (final_team_hp - state.last_team_hp) - (final_opp_hp - state.last_opp_hp)
        if battle.won:
            reward += self.win_reward
        elif battle.lost:
            reward -= self.loss_penalty
        self._store_transition(
            state.last_obs,
            state.last_action,
            reward,
            self._blank_state_tokens.copy(),
            True,
        )
        # Push a few extra updates at the end of a battle to consolidate learning.
        for _ in range(4):
            self._optimize()


async def run_training(args: argparse.Namespace) -> None:
    websocket_url = f"ws://{args.host}:{args.port}/showdown/websocket"
    auth_url = f"http://{args.host}:{args.port}/action.php?"
    server = ServerConfiguration(websocket_url, auth_url)

    tokenizer = HashingTokenizer(vocab_size=args.vocab_size)
    obs_builder = ObservationBuilder(tokenizer)
    action_mapper = ActionMapper()

    q_network = QNetwork(
        vocab_size=tokenizer.vocab_size,
        seq_len=obs_builder.observation_length,
        embed_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_actions=ActionMapper.NUM_ACTIONS,
    )

    agent_account = AccountConfiguration(args.username, None)
    opponent_account = AccountConfiguration(args.opponent_username, None)

    agent = OnlineRLAgent(
        account=agent_account,
        server=server,
        battle_format=args.battle_format,
        observation_builder=obs_builder,
        action_mapper=action_mapper,
        q_network=q_network,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay_steps=args.epsilon_decay_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_update_steps=args.target_update_steps,
        win_reward=args.win_reward,
        loss_penalty=args.loss_penalty,
    )

    opponent = RandomPlayer(
        account_configuration=opponent_account,
        battle_format=args.battle_format,
        server_configuration=server,
    )

    await agent.battle_against(opponent, n_battles=args.battles)

    if args.save_path:
        torch.save(agent.q_network.state_dict(), args.save_path)
        print(f"Saved trained weights to {args.save_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an online RL agent on a local Pokémon Showdown server.")
    parser.add_argument("--host", default="127.0.0.1", help="Pokemon Showdown host.")
    parser.add_argument("--port", type=int, default=8000, help="Pokemon Showdown port.")
    parser.add_argument("--battle-format", default="gen9randombattle", help="Showdown battle format to use.")
    parser.add_argument("--battles", type=int, default=50, help="Number of battles to play during training.")
    parser.add_argument("--username", default="RLAgent", help="Agent username on the server.")
    parser.add_argument(
        "--opponent-username",
        default="RLRandomOpponent",
        help="Opponent username on the server.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial ε for exploration.")
    parser.add_argument("--epsilon-final", type=float, default=0.05, help="Final ε for exploration.")
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=50_000,
        help="Number of agent steps used for ε decay.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=5_000)
    parser.add_argument("--target-update-steps", type=int, default=1_000)
    parser.add_argument("--win-reward", type=float, default=1.0)
    parser.add_argument("--loss-penalty", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--save-path", default="", help="Optional path to store the trained model weights.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args(None if argv is None else list(argv))


async def _async_main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    await run_training(args)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main(sys.argv[1:])
