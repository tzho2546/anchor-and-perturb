import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from smac.env import StarCraft2Env
from qmix import BaseMultiAgentEnv, EpisodeReplayBuffer, QMIXLearner
from json_collector import JSONCollector

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class SMACWrapper(BaseMultiAgentEnv):
    def __init__(self, map_name: str = "3s5z", seed: Optional[int] = None):
        super().__init__()
        self.env = StarCraft2Env(map_name=map_name, seed=seed)
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
            avail = np.asarray(avail_actions[i])
            if np.random.rand() < eps_vec[i]:
                valid = np.where(avail == 1)[0]
                actions.append(int(np.random.choice(valid)))
            else:
                masked_q = np.where(avail == 1, q_vals[i], -1e9)
                actions.append(int(np.argmax(masked_q)))

        return actions, next_hidden, q_vals


class LazinessHeuristicProbe:
    def __init__(
            self,
            n_agents: int,
            n_actions: int,
            window_size: int = 1000,
            beta: float = 1.0,
            threshold: float = 0.75
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.window_size = window_size
        self.beta = beta
        self.threshold = threshold

        self.action_history = np.zeros((n_agents, window_size), dtype=np.int64)
        self.spread_history = np.zeros((n_agents, window_size), dtype=np.float32)
        self.alive_mask = np.zeros((n_agents, window_size), dtype=bool)

        self.global_ptr = 0
        self.filled_size = 0

        self.latest_L = np.zeros(n_agents, dtype=np.float32)
        self.flag_counts = np.zeros(n_agents, dtype=np.int64)
        self.detection_log = []

    def reset_history(self):
        self.global_ptr = 0
        self.filled_size = 0
        self.alive_mask.fill(False)
        self.latest_L.fill(0.0)

    def update_step(self, actions: List[int], q_vals: np.ndarray, avail_actions: List[List[int]]):
        idx = self.global_ptr % self.window_size

        for i in range(self.n_agents):
            avail = np.asarray(avail_actions[i])
            # In SMAC, action 0 (no-op) is only available when the unit is dead.
            # Mark dead steps in alive_mask while keeping all agents on the same global clock.
            if avail[0] == 1:
                self.alive_mask[i, idx] = False
                self.action_history[i, idx] = 0
                self.spread_history[i, idx] = 0.0
                continue

            self.alive_mask[i, idx] = True
            self.action_history[i, idx] = actions[i]

            valid_qs = q_vals[i][avail == 1]
            if len(valid_qs) > 1:
                spread = float(np.max(valid_qs) - np.mean(valid_qs))
            else:
                spread = 0.0
            self.spread_history[i, idx] = spread

        self.global_ptr += 1
        if self.filled_size < self.window_size:
            self.filled_size += 1

    def check(
            self,
            episode: int,
            borderline: int = 1600,
            recent_win_rate: float = 0.0,
            win_rate_ceiling: float = 0.80,
    ) -> Tuple[bool, int, np.ndarray]:
        # Require a full global window of environment steps before diagnosing
        if self.filled_size < self.window_size:
            return False, 0, self.latest_L

        c_i = np.zeros(self.n_agents, dtype=np.float32)
        h_i = np.zeros(self.n_agents, dtype=np.float32)

        for i in range(self.n_agents):
            mask = self.alive_mask[i, :self.filled_size]
            n_alive = int(np.sum(mask))

            if n_alive == 0:
                c_i[i] = 0.0
                h_i[i] = 0.0
                continue

            # Weight Q-spread by survival ratio over the shared window so early deaths
            # reflect lower cumulative episode contribution without warping the timeline
            alive_spreads = self.spread_history[i, :self.filled_size][mask]
            c_i[i] = float(np.sum(alive_spreads) / float(self.filled_size))

            # Compute action entropy strictly over alive decisions within the shared window
            alive_actions = self.action_history[i, :self.filled_size][mask]
            counts = np.bincount(alive_actions, minlength=self.n_actions)
            probs = counts / float(n_alive)
            probs = probs[probs > 0]
            h_i[i] = float(-np.sum(probs * np.log(probs)))

        max_c = float(np.max(c_i))
        L_i = (1.0 - (c_i / (max_c + 1e-5))) * np.exp(-self.beta * h_i)
        self.latest_L = L_i

        max_agent = int(np.argmax(L_i))
        max_L = float(L_i[max_agent])

        # Trigger only past borderline, above threshold, and when policy isn't already near-optimal
        is_flagged = (
            bool(max_L > self.threshold)
            and (episode >= borderline)
            and (recent_win_rate < win_rate_ceiling)
        )

        if is_flagged:
            self.flag_counts[max_agent] += 1
            self.detection_log.append({
                "episode": episode,
                "flagged_agent": max_agent,
                "max_L": max_L,
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


def run_experiment(
        mode: str,
        map_name: str = "3s5z",
        seed: int = 42,
        total_episodes: int = 4000,
        explore_stop: int = 800,
        pulse_start_pct: float = 0.505,
        pulse_duration_pct: float = 0.083,
        auto_threshold: float = 0.75,
        auto_borderline: int = 1600,
        auto_pulse_duration: int = 350,
        pulse_max_count: int = 2,
        phi: float = 1.5,  # Annealing aggressiveness parameter
):
    use_anp = mode in ["anp", "anp_r", "anp_auto", "anp_auto_r"]
    use_elite = mode in ["vanilla_r", "anp_r", "anp_auto_r"]
    is_auto = mode in ["anp_auto", "anp_auto_r"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    env = SMACWrapper(map_name=map_name, seed=seed)
    env_info = env.get_env_info()

    tag = f"qmix_{mode}_seed{seed}_{map_name}"
    collector = JSONCollector(tag)

    probe = None
    if use_anp:
        probe = LazinessHeuristicProbe(
            n_agents=env_info["n_agents"],
            n_actions=env_info["n_actions"],
            window_size=1000,
            beta=1.0,
            threshold=auto_threshold,
        )

    if is_auto:
        injection_start = total_episodes + 1
        injection_end = total_episodes + 1
    elif use_anp:
        injection_start = min(int(total_episodes * pulse_start_pct), total_episodes)
        injection_end = min(injection_start + int(total_episodes * pulse_duration_pct), total_episodes)
    else:
        injection_start = auto_borderline
        injection_end = total_episodes + 1

    target_agent = None
    last_target_agent = None

    print("\n" + "#" * 70)
    print(f"STARTING RUN: Map={map_name} | Mode={mode.upper()} | Auto={is_auto} | Reservoir={use_elite} | Seed={seed}")
    if is_auto:
        print(f"Auto Triggers -> L_i Threshold: {auto_threshold} | Borderline: Ep {auto_borderline}")
        print(f"Pulse Settings -> Duration: {auto_pulse_duration} Eps | Max Count (K): {pulse_max_count}")
    elif use_anp:
        print(f"Horizon: {total_episodes} | Static Pulse Window: [{injection_start}, {injection_end}]")
    else:
        print(f"Horizon: {total_episodes} | Nominal Explore Stop: Ep {explore_stop}")
    print("#" * 70 + "\n")

    base_lr = 5e-4
    train_steps_per_episode = 1
    batch_size = 32

    learner = VectorEpsilonQMIXLearner(env_info=env_info, lr=base_lr, target_update_interval=200)

    buffer = EpisodeReplayBuffer(
        capacity=2000,
        episode_limit=env_info["episode_limit"],
        n_agents=env_info["n_agents"],
        obs_dim=env_info["obs_dim"],
        state_dim=env_info["state_dim"],
        n_actions=env_info["n_actions"],
    )

    elite_buffer = None
    if use_elite:
        elite_buffer = EpisodeReplayBuffer(
            capacity=500,
            episode_limit=env_info["episode_limit"],
            n_agents=env_info["n_agents"],
            obs_dim=env_info["obs_dim"],
            state_dim=env_info["state_dim"],
            n_actions=env_info["n_actions"],
        )

    episode_rewards = []
    intervention_active = False
    eval_score = 0.0
    win_rate = 0.0
    pulse_count = 0
    cooldown_episodes = 200
    reward_threshold = 9.3

    for episode in range(total_episodes):
        if episode < explore_stop:
            current_nominal_eps = max(0.05, 1.0 - (0.95 * (episode / explore_stop)))
        else:
            current_nominal_eps = 0.05

        is_post_pulse_cooldown = (
                use_anp and (pulse_count > 0) and (injection_end <= episode < injection_end + cooldown_episodes)
        )

        # 1. Action exploration scheduling (AnP Rollout Interface)
        if use_anp and (injection_start <= episode < injection_end):
            if not intervention_active:
                if not is_auto and target_agent is None:
                    # In static AnP mode, use probe L_i if ready, else fallback to random
                    if probe is not None and np.max(probe.latest_L) > 0:
                        target_agent = int(np.argmax(probe.latest_L))
                    else:
                        target_agent = np.random.randint(0, env_info["n_agents"])
                    pulse_count += 1  # Ensure static AnP also triggers post-pulse cooldown

                print("\n" + "=" * 60)
                if is_auto:
                    print(f">>> [AUTO TRIGGER]: L_i Threshold Exceeded. Targeting Agent {target_agent} at Ep {episode}")
                else:
                    print(
                        f">>> [SUPERVISOR TRIGGER]: Injecting Rank-1 Yank Vector (Agent {target_agent}) at Ep {episode}")
                print(f">>> M=1 Exploration Active (phi={phi}) | N-1 Strictly Anchored (Greedy)")
                print("=" * 60 + "\n")
                intervention_active = True

            progress = (episode - injection_start) / max(1, (injection_end - injection_start))
            # Non-linear annealing governed by phi
            pulse_epsilon = 0.05 + 0.25 * ((1.0 - progress) ** phi)

            # Strictly anchor non-targeted agents to 0.0 (Pure Greedy Stationary Background)
            eps_vec = np.zeros(env_info["n_agents"], dtype=np.float32)
            eps_vec[target_agent] = pulse_epsilon
            status = "PULSE"

        elif is_post_pulse_cooldown:
            if intervention_active:
                print("\n" + "=" * 60)
                print(
                    f">>> [SUPERVISOR RELEASE]: Pulse Completed at Ep {episode}. Starting Smooth Un-Anchor Ramp (phi={phi})")
                print("=" * 60 + "\n")
                intervention_active = False
                last_target_agent = target_agent
                target_agent = None

                if probe is not None:
                    probe.reset_history()

            # Smoothly un-anchor N-1 teammates from 0.0 back to current_nominal_eps using phi
            cooldown_progress = (episode - injection_end) / max(1, cooldown_episodes)
            anchor_release_eps = current_nominal_eps * (cooldown_progress ** phi)

            eps_vec = np.full(env_info["n_agents"], anchor_release_eps, dtype=np.float32)
            if last_target_agent is not None:
                eps_vec[last_target_agent] = current_nominal_eps
            status = "COOLDOWN"

        else:
            if intervention_active:
                intervention_active = False
                last_target_agent = target_agent
                target_agent = None
                if probe is not None:
                    probe.reset_history()

            eps_vec = np.full(env_info["n_agents"], current_nominal_eps, dtype=np.float32)
            status = "NOMINAL"

        # 2. Environment rollout
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
        won = False

        while step < env_info["episode_limit"]:
            avail = [env.get_avail_agent_actions(i) for i in range(env_info["n_agents"])]
            ep_data["obs"][step] = np.array(obs)
            ep_data["states"][step] = np.array(state)
            ep_data["avail_actions"][step] = np.array(avail)

            actions, hidden, q_vals = learner.select_action(obs, hidden, avail, epsilon=eps_vec)

            if probe is not None and not intervention_active:
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
                won = bool(info.get("battle_won", False))
                break

        ep_data["obs"][step] = np.array(obs)
        ep_data["states"][step] = np.array(state)
        ep_data["avail_actions"][step] = np.array([env.get_avail_agent_actions(i) for i in range(env_info["n_agents"])])

        buffer.insert(ep_data)
        episode_rewards.append(total_reward)

        is_post_pulse_cooldown = (
            use_anp and (pulse_count > 0) and (injection_end <= episode < injection_end + cooldown_episodes)
        )

        # 3. Check for automated AnP trigger (blocked during active pulse and post-pulse cooldown)
        if (
            is_auto
            and episode >= auto_borderline
            and not intervention_active
            and not is_post_pulse_cooldown
            and pulse_count < pulse_max_count
        ):
            is_flagged, max_agent, _ = probe.check(
                episode=episode,
                borderline=auto_borderline,
                recent_win_rate=win_rate,
            )
            if is_flagged:
                injection_start = episode + 1
                injection_end = min(injection_start + auto_pulse_duration, total_episodes)
                target_agent = max_agent
                pulse_count += 1

        # 4. Experience replay sampling
        can_collect_elite = (
            (episode >= injection_start)
            if use_anp
            else (episode >= auto_borderline)
        )
        if use_elite and can_collect_elite and won and (total_reward >= reward_threshold):
            elite_buffer.insert(ep_data)

        if buffer.size >= batch_size:
            for _ in range(train_steps_per_episode):
                if use_elite and (episode >= injection_start) and (elite_buffer.size >= 40):
                    n_elite = min(4, elite_buffer.size)
                    n_nominal = batch_size - n_elite

                    nom_batch = buffer.sample(n_nominal)
                    elite_batch = elite_buffer.sample(n_elite)

                    batch = {}
                    for k in nom_batch.keys():
                        batch[k] = torch.cat([nom_batch[k], elite_batch[k]], dim=0)
                    learner.train_step(batch)
                else:
                    learner.train_step(buffer.sample(batch_size))

        # 5. Checkpoint evaluation
        if (episode + 1) % 25 == 0:
            eval_score, win_rate = evaluate_greedy(env, learner, env_info, num_episodes=20)
            avg_train_25 = float(np.mean(episode_rewards[-25:]))

            is_pulse_active = bool(status == "PULSE")

            collector.collect(
                episode=episode,
                total_episodes=total_episodes,
                status=status,
                avg_train=avg_train_25,
                eval_score=eval_score,
                win_rate=win_rate,
                epsilon=float(np.max(eps_vec)),
                is_pulse=is_pulse_active,
            )

        # 6. Console progress stream
        avg_train = float(np.mean(episode_rewards[-25:])) if len(episode_rewards) > 0 else 0.0
        disp_eps = float(np.max(eps_vec))
        elite_disp = f"{elite_buffer.size:03d}/500" if use_elite else "OFF"

        if intervention_active:
            status_tag = f"PULSE(A{target_agent})"
        elif is_auto and probe is not None and np.max(probe.latest_L) > auto_threshold:
            status_tag = "L_i ALERT"
        else:
            status_tag = status

        print(
            f"\rEp {episode + 1:04d}/{total_episodes} [{status_tag:9s}] | "
            f"Train: {avg_train:5.2f} | Eval Reward: {eval_score:5.2f} | "
            f"Win Rate: {win_rate * 100:5.1f}% | Elite: {elite_disp} | Epsilon: {disp_eps:.3f}",
            end="",
            flush=True,
        )

        if (episode + 1) % 25 == 0:
            print()
            if probe:
                L_i_current = np.max(probe.latest_L)
                print(f"Current L_I = {L_i_current}")

    env.close()
    print(f"\nFinished run for mode: {mode.upper()}\n")
    collector.close()


if __name__ == "__main__":
    mode = ("anp_auto")
    map_name = "MMM"
    seed = 15
    total_episodes = 4000
    explore_stop = 1000

    pulse_start_pct = 0.500
    pulse_duration_pct = 0.083

    auto_threshold = 0.053
    auto_borderline = 2000
    auto_pulse_duration = 200
    pulse_max_count = 1
    phi = 1.5  # >1.0 front-loads perturbation decay & delays anchor release shock

    run_experiment(
        mode=mode,
        map_name=map_name,
        seed=seed,
        total_episodes=total_episodes,
        explore_stop=explore_stop,
        pulse_start_pct=pulse_start_pct,
        pulse_duration_pct=pulse_duration_pct,
        auto_threshold=auto_threshold,
        auto_borderline=auto_borderline,
        auto_pulse_duration=auto_pulse_duration,
        pulse_max_count=pulse_max_count,
        phi=phi,
    )