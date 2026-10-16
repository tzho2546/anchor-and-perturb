"""
qmix.py - Hardened QMIX architecture & open environment interface.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMultiAgentEnv:
    """Standard open interface for multi-agent reinforcement learning environments."""
    def __init__(self) -> None:
        self.n_agents: int = 0
        self.obs_dim: int = 0
        self.state_dim: int = 0
        self.n_actions: int = 0
        self.episode_limit: int = 0

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        raise NotImplementedError

    def step(
        self, actions: List[int]
    ) -> Tuple[float, bool, Dict[str, Any], List[np.ndarray], np.ndarray]:
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        return [1] * self.n_actions

    def get_env_info(self) -> Dict[str, Any]:
        return {
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "episode_limit": self.episode_limit,
        }


class RNNAgent(nn.Module):
    """GRU-based individual agent network for decentralized execution."""
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def forward(
        self, inputs: torch.Tensor, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class QMixer(nn.Module):
    """Mixing network with non-negative hypernetwork weights."""
    def __init__(
        self, n_agents: int, state_dim: int, embed_dim: int = 32, hypernet_dim: int = 64
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_dim),
            nn.ReLU(),
            nn.Linear(hypernet_dim, n_agents * embed_dim),
        )
        self.hyper_b_1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w_final = nn.Sequential(
            nn.Linear(state_dim, hypernet_dim),
            nn.ReLU(),
            nn.Linear(hypernet_dim, embed_dim),
        )
        self.hyper_b_final = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = agent_qs.shape
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        w1 = torch.abs(self.hyper_w_1(states)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w_final = torch.abs(self.hyper_w_final(states)).view(-1, self.embed_dim, 1)
        b_final = self.hyper_b_final(states).view(-1, 1, 1)

        v = torch.bmm(hidden, w_final) + b_final
        return v.view(bs, seq_len, 1)


class EpisodeReplayBuffer:
    """Stores full episodes for recurrent training."""
    def __init__(
        self, capacity: int, episode_limit: int, n_agents: int, obs_dim: int, state_dim: int, n_actions: int
    ):
        self.capacity = capacity
        self.episode_limit = episode_limit
        self.size = 0
        self.idx = 0

        self.buf_obs = np.zeros((capacity, episode_limit + 1, n_agents, obs_dim), dtype=np.float32)
        self.buf_actions = np.zeros((capacity, episode_limit, n_agents), dtype=np.int64)
        self.buf_rewards = np.zeros((capacity, episode_limit, 1), dtype=np.float32)
        self.buf_states = np.zeros((capacity, episode_limit + 1, state_dim), dtype=np.float32)
        self.buf_avail_actions = np.zeros((capacity, episode_limit + 1, n_agents, n_actions), dtype=np.float32)
        self.buf_terminated = np.zeros((capacity, episode_limit, 1), dtype=np.float32)
        self.buf_filled = np.zeros((capacity, episode_limit, 1), dtype=np.float32)

    def insert(self, ep_data: Dict[str, np.ndarray]) -> None:
        self.buf_obs[self.idx] = ep_data["obs"]
        self.buf_actions[self.idx] = ep_data["actions"]
        self.buf_rewards[self.idx] = ep_data["rewards"]
        self.buf_states[self.idx] = ep_data["states"]
        self.buf_avail_actions[self.idx] = ep_data["avail_actions"]
        self.buf_terminated[self.idx] = ep_data["terminated"]
        self.buf_filled[self.idx] = ep_data["filled"]

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.choice(self.size, batch_size, replace=False)
        return {
            "obs": torch.FloatTensor(self.buf_obs[indices]),
            "actions": torch.LongTensor(self.buf_actions[indices]),
            "rewards": torch.FloatTensor(self.buf_rewards[indices]),
            "states": torch.FloatTensor(self.buf_states[indices]),
            "avail_actions": torch.FloatTensor(self.buf_avail_actions[indices]),
            "terminated": torch.FloatTensor(self.buf_terminated[indices]),
            "filled": torch.FloatTensor(self.buf_filled[indices]),
        }


class QMIXLearner:
    """Manages networks, parameter updates, and target tracking."""
    def __init__(
        self, env_info: Dict[str, Any], lr: float = 5e-4, gamma: float = 0.99, target_update_interval: int = 200
    ):
        self.n_agents = env_info["n_agents"]
        self.obs_dim = env_info["obs_dim"]
        self.state_dim = env_info["state_dim"]
        self.n_actions = env_info["n_actions"]
        self.gamma = gamma
        self.target_update_interval = target_update_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_step_count = 0

        input_shape = self.obs_dim + self.n_agents

        self.eval_rnn = RNNAgent(input_shape, 64, self.n_actions).to(self.device)
        self.target_rnn = RNNAgent(input_shape, 64, self.n_actions).to(self.device)
        self.eval_mixer = QMixer(self.n_agents, self.state_dim).to(self.device)
        self.target_mixer = QMixer(self.n_agents, self.state_dim).to(self.device)

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_mixer.load_state_dict(self.eval_mixer.state_dict())

        self.params = list(self.eval_rnn.parameters()) + list(self.eval_mixer.parameters())
        self.optimizer = torch.optim.RMSprop(self.params, lr=lr, alpha=0.99, eps=1e-5)

    def _build_inputs(self, batch_obs: torch.Tensor, bs: int) -> torch.Tensor:
        """Appends one-hot agent IDs to observations. Handles [bs, n_agents, obs_dim]."""
        eye = torch.eye(self.n_agents, device=self.device).unsqueeze(0).expand(bs, -1, -1)
        return torch.cat([batch_obs, eye], dim=-1)

    def select_action(
        self, obs: List[np.ndarray], hidden_states: torch.Tensor, avail_actions: List[List[int]], epsilon: float = 0.0
    ) -> Tuple[List[int], torch.Tensor]:
        obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(self.device)
        inputs = self._build_inputs(obs_tensor, 1).view(self.n_agents, -1)
        hidden_states = hidden_states.to(self.device)

        with torch.no_grad():
            q_vals, next_hidden = self.eval_rnn(inputs, hidden_states)
            q_vals = q_vals.cpu().numpy()

        actions = []
        for i in range(self.n_agents):
            avail = np.array(avail_actions[i])
            if np.random.rand() < epsilon:
                valid = np.where(avail == 1)[0]
                actions.append(np.random.choice(valid))
            else:
                q_vals[i][avail == 0] = -1e9
                actions.append(int(np.argmax(q_vals[i])))

        return actions, next_hidden

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        bs = batch["obs"].shape[0]
        seq_len = batch["actions"].shape[1]

        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        states = batch["states"].to(self.device)
        avail_actions = batch["avail_actions"].to(self.device)
        terminated = batch["terminated"].to(self.device)
        mask = batch["filled"].to(self.device)

        mac_out = []
        hidden = self.eval_rnn.init_hidden(bs * self.n_agents).to(self.device)

        for t in range(seq_len + 1):
            inputs = self._build_inputs(obs[:, t], bs).view(bs * self.n_agents, -1)
            q, hidden = self.eval_rnn(inputs, hidden)
            mac_out.append(q.view(bs, self.n_agents, self.n_actions))

        mac_out = torch.stack(mac_out, dim=1)
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions.unsqueeze(3)).squeeze(3)

        target_mac_out = []
        target_hidden = self.target_rnn.init_hidden(bs * self.n_agents).to(self.device)

        with torch.no_grad():
            for t in range(seq_len + 1):
                inputs = self._build_inputs(obs[:, t], bs).view(bs * self.n_agents, -1)
                t_q, target_hidden = self.target_rnn(inputs, target_hidden)
                target_mac_out.append(t_q.view(bs, self.n_agents, self.n_actions))
            target_mac_out = torch.stack(target_mac_out, dim=1)

            # Double Q-Learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_mac_out[:, 1:], dim=3, index=cur_max_actions).squeeze(3)

        q_tot = self.eval_mixer(chosen_action_qvals, states[:, :-1])

        with torch.no_grad():
            target_q_tot = self.target_mixer(target_max_qvals, states[:, 1:])
            y = rewards + self.gamma * (1.0 - terminated) * target_q_tot

        td_error = (q_tot - y) * mask
        loss = (td_error ** 2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_interval == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_mixer.load_state_dict(self.eval_mixer.state_dict())

        return float(loss.item())