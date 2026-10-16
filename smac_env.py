from typing import Any, Dict, List, Tuple
import numpy as np
from smac.env import StarCraft2Env
from qmix import BaseMultiAgentEnv


class SMACWrapper(BaseMultiAgentEnv):
    """
    SMAC wrapper implementing the exact interface expected by the training loop
    demonstrated in simple_spread_v3.py.
    """

    def __init__(self, map_name: str = "3m"):
        super().__init__()
        self.env = StarCraft2Env(map_name=map_name)
        env_info = self.env.get_env_info()

        # Map SMAC environment properties to the expected interface attributes
        self.n_agents = env_info["n_agents"]
        self.obs_dim = env_info["obs_shape"]
        self.n_actions = env_info["n_actions"]
        self.episode_limit = env_info["episode_limit"]
        self.state_dim = env_info["state_shape"]

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Resets the environment and returns the initial observations and global state.
        """
        self.env.reset()
        obs_list = self.env.get_obs()
        state = self.env.get_state()
        return obs_list, state

    def step(self, actions: List[int]) -> Tuple[float, bool, Dict[str, Any], List[np.ndarray], np.ndarray]:
        """
        Executes a step using a list of discrete actions.
        """
        reward, terminated, env_info = self.env.step(actions)

        obs_list = self.env.get_obs()
        state = self.env.get_state()

        return float(reward), terminated, env_info, obs_list, state

    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        """
        Returns a mask of available actions for a specific agent.
        """
        return self.env.get_avail_agent_actions(agent_id)

    def get_env_info(self) -> Dict[str, Any]:
        """
        Returns environment metadata required for initializing replay buffers and networks.
        """
        return {
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "episode_limit": self.episode_limit,
        }

    def close(self):
        """
        Closes the environment resources.
        """
        self.env.close()