import json
import random
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from smac.env import StarCraft2Env
from qmix import BaseMultiAgentEnv, EpisodeReplayBuffer, QMIXLearner
from json_collector import JSONCollector

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class SMACWrapper(BaseMultiAgentEnv):
    def __init__(self, map_name: str = "2s3z"):
        super().__init__()
        self.env = StarCraft2Env(map_name=map_name)
        env_info = self.env.get_env_info()

        self.n_agents = env_info["n_agents"]
        self.obs_dim = env_info["obs_shape"]
        self.n_actions = env_info["n_actions"]
        self.episode_limit = env_info["episode_limit"]
        self.state_dim = env_info["state_shape"]

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        self.env.reset()
        obs_list = [np.array(o, dtype=np.float32) for o in self.env.get_obs()]
        state = np.array(self.env.get_state(), dtype=np.float32)
        return obs_list, state

    def step(
            self, actions: List[int]
    ) -> Tuple[float, bool, Dict[str, Any], List[np.ndarray], np.ndarray]:
        reward, terminated, env_info = self.env.step(actions)
        obs_list = [np.array(o, dtype=np.float32) for o in self.env.get_obs()]
        state = np.array(self.env.get_state(), dtype=np.float32)
        return float(reward), bool(terminated), env_info, obs_list, state

    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        try:
            return self.env.get_avail_agent_actions(agent_id)
        except Exception:
            avail = [0] * self.n_actions
            avail[0] = 1
            return avail

    def get_env_info(self) -> Dict[str, Any]:
        return {
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "episode_limit": self.episode_limit,
        }

    def close(self):
        self.env.close()


class VectorEpsilonQMIXLearner(QMIXLearner):
    def select_action(
            self,
            obs: List[np.ndarray],
            hidden_states: torch.Tensor,
            avail_actions: List[List[int]],
            epsilon: Union[float, np.ndarray, List[float]] = 0.0,
    ) -> Tuple[List[int], torch.Tensor, np.ndarray]:
        if np.isscalar(epsilon):
            eps_vec = np.full(self.n_agents, float(epsilon), dtype=np.float32)
        else:
            eps_vec = np.asarray(epsilon, dtype=np.float32)

        obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(self.device)
        inputs = self._build_inputs(obs_tensor, 1).view(self.n_agents, -1)
        hidden_states = hidden_states.to(self.device)

        with torch.no_grad():
            q_vals, next_hidden = self.eval_rnn(inputs, hidden_states)
            q_vals = q_vals.cpu().numpy()

        actions = []
        for i in range(self.n_agents):
            avail = np.array(avail_actions[i])
            if np.random.rand() < eps_vec[i]:
                valid = np.where(avail == 1)[0]
                actions.append(int(np.random.choice(valid)))
            else:
                q_vals[i][avail == 0] = -1e9
                actions.append(int(np.argmax(q_vals[i])))

        return actions, next_hidden, q_vals


class LazinessHeuristicProbe:
    """
    Environment-agnostic AnP trigger.
    Tracks action entropy (H_i) and Q-value spread (c_i) over a rolling window
    to calculate the Laziness Disparity Index (L_i).
    """

    def __init__(
            self,
            n_agents: int,
            n_actions: int,
            window_size: int = 100,
            beta: float = 1.0,
            threshold: float = 0.7  # Trigger threshold for max L_i
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.window_size = window_size
        self.beta = beta
        self.threshold = threshold

        # Rolling history arrays
        self.action_history = np.zeros((n_agents, window_size), dtype=np.int64)
        self.spread_history = np.zeros((n_agents, window_size), dtype=np.float32)
        self.ptr = 0
        self.filled_size = 0

        self.latest_L = np.zeros(n_agents, dtype=np.float32)
        self.flag_counts = np.zeros(n_agents, dtype=np.int64)
        self.detection_log = []

    def update_step(self, actions: List[int], q_vals: np.ndarray, avail_actions: List[List[int]]):
        idx = self.ptr % self.window_size

        for i in range(self.n_agents):
            self.action_history[i, idx] = actions[i]

            # Q-value spread proxy for marginal contribution (c_i)
            avail = np.array(avail_actions[i])
            valid_qs = q_vals[i][avail == 1]
            if len(valid_qs) > 0:
                spread = np.max(valid_qs) - np.mean(valid_qs)
            else:
                spread = 0.0
            self.spread_history[i, idx] = spread

        self.ptr += 1
        self.filled_size = min(self.filled_size + 1, self.window_size)

    def check(self, episode: int, borderline: int = 1600) -> Tuple[bool, int, np.ndarray]:
        if self.filled_size == 0:
            return False, 0, self.latest_L

        # 1. Marginal Contribution Proxy (c_i)
        c_i = np.mean(self.spread_history[:, :self.filled_size], axis=1)
        max_c = np.max(c_i)

        # 2. Exploratory Atrophy (H_i) via empirical action entropy
        h_i = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            counts = np.bincount(self.action_history[i, :self.filled_size], minlength=self.n_actions)
            probs = counts / self.filled_size
            probs = probs[probs > 0]
            h_i[i] = -np.sum(probs * np.log(probs))

        # 3. Laziness Disparity Index (L_i)
        L_i = (1.0 - (c_i / (max_c + 1e-5))) * np.exp(-self.beta * h_i)
        self.latest_L = L_i

        max_agent = int(np.argmax(L_i))
        max_L = float(L_i[max_agent])

        # Trigger condition limits intervention to later training stages
        is_flagged = bool(max_L > self.threshold) and episode > borderline

        if is_flagged:
            self.flag_counts[max_agent] += 1
            self.detection_log.append({
                "episode": episode,
                "flagged_agent": max_agent,
                "max_L": max_L,
                "L_i": [float(round(x, 4)) for x in L_i],
                "c_i": [float(round(x, 4)) for x in c_i],
                "h_i": [float(round(x, 4)) for x in h_i],
            })

        return is_flagged, max_agent, L_i


def evaluate_greedy(env, learner, env_info, num_episodes=20) -> Tuple[float, float]:
    eval_rewards = []
    eval_wins = []

    for _ in range(num_episodes):
        obs, state = env.reset()
        hidden = learner.eval_rnn.init_hidden(env_info["n_agents"]).to(learner.device)
        ep_reward = 0.0
        won = False

        for _ in range(env_info["episode_limit"]):
            avail = [env.get_avail_agent_actions(i) for i in range(env_info["n_agents"])]
            actions, hidden, _ = learner.select_action(obs, hidden, avail, epsilon=0.0)
            reward, terminated, info, next_obs, next_state = env.step(actions)
            ep_reward += reward
            obs, state = next_obs, next_state
            if terminated:
                won = bool(info.get("battle_won", False))
                break

        eval_rewards.append(ep_reward)
        eval_wins.append(float(won))

    return float(np.mean(eval_rewards)), float(np.mean(eval_wins))


def run_probe_detection(
        map_name: str = "2s3z",
        seed: int = 42,
        total_episodes: int = 3000,
        threshold: float = 0.75,
        beta: float = 1.0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    env = SMACWrapper(map_name)
    env_info = env.get_env_info()

    tag = f"probe_detect_seed{seed}_{map_name}"
    collector = JSONCollector(tag)

    base_lr = 5e-4
    batch_size = 32
    explore_stop = 800

    learner = VectorEpsilonQMIXLearner(env_info=env_info, lr=base_lr, target_update_interval=200)

    # Initialize the new heuristic probe
    probe = LazinessHeuristicProbe(
        n_agents=env_info["n_agents"],
        n_actions=env_info["n_actions"],
        window_size=100,
        beta=beta,
        threshold=threshold
    )

    buffer = EpisodeReplayBuffer(
        capacity=2000,
        episode_limit=env_info["episode_limit"],
        n_agents=env_info["n_agents"],
        obs_dim=env_info["obs_dim"],
        state_dim=env_info["state_dim"],
        n_actions=env_info["n_actions"],
    )

    print("\n" + "#" * 70)
    print(f"STARTING PROBE DETECTION: Map={map_name} | Seed={seed} | Agents={env_info['n_agents']}")
    print(f"Trigger Threshold: {probe.threshold:.3f} | Beta: {probe.beta:.3f}")
    print("Mode: Diagnostic Only (No Interventions Applied)")
    print("#" * 70 + "\n")

    episode_rewards = []
    eval_score = 0.0
    win_rate = 0.0

    for episode in range(total_episodes):
        if episode < explore_stop:
            current_eps = max(0.05, 1.0 - (0.95 * (episode / explore_stop)))
        else:
            current_eps = 0.05

        eps_vec = np.full(env_info["n_agents"], current_eps, dtype=np.float32)

        ep_data = {
            "obs": np.zeros((env_info["episode_limit"] + 1, env_info["n_agents"], env_info["obs_dim"]),
                            dtype=np.float32),
            "actions": np.zeros((env_info["episode_limit"], env_info["n_agents"]), dtype=np.int64),
            "rewards": np.zeros((env_info["episode_limit"], 1), dtype=np.float32),
            "states": np.zeros((env_info["episode_limit"] + 1, env_info["state_dim"]), dtype=np.float32),
            "avail_actions": np.zeros((env_info["episode_limit"] + 1, env_info["n_agents"], env_info["n_actions"]),
                                      dtype=np.float32),
            "terminated": np.zeros((env_info["episode_limit"], 1), dtype=np.float32),
            "filled": np.zeros((env_info["episode_limit"], 1), dtype=np.float32),
        }

        obs, state = env.reset()
        hidden = learner.eval_rnn.init_hidden(env_info["n_agents"]).to(learner.device)
        total_reward = 0.0
        step = 0

        while step < env_info["episode_limit"]:
            avail = [env.get_avail_agent_actions(i) for i in range(env_info["n_agents"])]
            ep_data["obs"][step] = np.array(obs)
            ep_data["states"][step] = np.array(state)
            ep_data["avail_actions"][step] = np.array(avail)

            actions, hidden, q_vals = learner.select_action(obs, hidden, avail, epsilon=eps_vec)

            # Feed step data into the heuristic tracker
            probe.update_step(actions, q_vals, avail)

            reward, terminated, info, next_obs, next_state = env.step(actions)

            truncated = (step + 1 >= env_info["episode_limit"]) or info.get("episode_limit", False)
            true_terminated = terminated and not truncated

            ep_data["actions"][step] = actions
            ep_data["rewards"][step] = reward
            ep_data["terminated"][step] = float(true_terminated)
            ep_data["filled"][step] = 1.0

            total_reward += reward
            obs, state = next_obs, next_state
            step += 1
            if terminated:
                break

        ep_data["obs"][step] = np.array(obs)
        ep_data["states"][step] = np.array(state)
        ep_data["avail_actions"][step] = np.array(
            [env.get_avail_agent_actions(i) for i in range(env_info["n_agents"])]
        )

        buffer.insert(ep_data)
        episode_rewards.append(total_reward)

        if buffer.size >= batch_size:
            learner.train_step(buffer.sample(batch_size))

        # Check for anomalies using the new L_i heuristic
        is_flagged, max_agent, latest_L = probe.check(episode)

        if (episode + 1) % 25 == 0:
            eval_score, win_rate = evaluate_greedy(env, learner, env_info, num_episodes=20)
            avg_train = float(np.mean(episode_rewards[-25:]))

            collector.collect(
                episode=episode,
                total_episodes=total_episodes,
                status="FLAGGED" if is_flagged else "NORMAL",
                avg_train=avg_train,
                eval_score=eval_score,
                win_rate=win_rate,
                epsilon=float(np.max(eps_vec)),
                is_pulse=False,
            )

        avg_train = float(np.mean(episode_rewards[-25:])) if len(episode_rewards) > 0 else 0.0
        status_tag = f"ALERT(A{max_agent})" if is_flagged else "NORMAL   "
        L_str = " ".join([f"{x:.4f}" for x in latest_L])

        print(
            f"\rEp {episode + 1:04d}/{total_episodes} [{status_tag}] | "
            f"Train: {avg_train:5.2f} | Win: {win_rate * 100:4.1f}% | "
            f"L_i: [{L_str}]",
            end="",
            flush=True,
        )

        if (episode + 1) % 25 == 0:
            print()

    # Save isolated detection dump using the new heuristic keys
    with open("probe_detections.json", "w") as f:
        json.dump(
            {
                "map": map_name,
                "seed": seed,
                "threshold": float(probe.threshold),
                "beta": float(probe.beta),
                "flag_counts": probe.flag_counts.tolist(),
                "detections": probe.detection_log,
            },
            f,
            indent=2,
        )

    env.close()
    collector.close()

    print("\n\n" + "=" * 70)
    print("PROBE DETECTION AUDIT COMPLETE")
    print(f"Total Detections Logged: {len(probe.detection_log)}")
    print(f"Flag Counts Per Agent: {probe.flag_counts.tolist()}")
    print("Full telemetry saved to probe_detections.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_probe_detection(
        map_name="2s3z",
        seed=11,
        total_episodes=3000,
        threshold=0.75,
        beta=1.0,
    )