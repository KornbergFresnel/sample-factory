import pettingzoo
import importlib

from typing import Sequence, Dict, Any, List
from collections import defaultdict

import supersuit
import gym

from sample_factory.envs.env_registry import global_env_registry


ATARI_W = ATARI_H = 84


class MAAtariSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout
        self.has_timer = False


ATARI_ENVS = [
    # MAAtariSpec('atari_montezuma', 'MontezumaRevengeNoFrameskip-v4', default_timeout=18000),

    MAAtariSpec('maatari_pong', 'basketball_pong_v1'),
    # AtariSpec('atari_qbert', 'QbertNoFrameskip-v4'),
    # AtariSpec('atari_breakout', 'BreakoutNoFrameskip-v4'),
    # AtariSpec('atari_spaceinvaders', 'SpaceInvadersNoFrameskip-v4'),

    # AtariSpec('atari_asteroids', 'AsteroidsNoFrameskip-v4'),
    # AtariSpec('atari_gravitar', 'GravitarNoFrameskip-v4'),
    # AtariSpec('atari_mspacman', 'MsPacmanNoFrameskip-v4'),
    # AtariSpec('atari_seaquest', 'SeaQuestNoFrameskip-v4'),
]


def atari_env_by_name(name):
    for cfg in ATARI_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Atari env')



def nested_env_creator(ori_creator: type, wrappers: Sequence[Dict]) -> type:
    """Wrap original atari environment creator with multiple wrappers"""

    def creator(**env_config):
        env = ori_creator(**env_config)
        # parse wrappers
        for wconfig in wrappers:
            name = wconfig["name"]
            params = wconfig["params"]

            wrapper = getattr(
                supersuit, name
            )  # importlib.import_module(f"supersuit.{env_desc['wrapper']['name']}")

            if isinstance(params, Sequence):
                env = wrapper(env, *params)
            elif isinstance(params, Dict):
                env = wrapper(env, **params)
            else:
                raise TypeError(f"Unexpected type: {type(params)}")
        return env

    return creator


def load_env_config(env_id):
    return dict(
        basketball_pong_v1={
            "num_players": 2,
            "obs_type": "rgb_image",
            "wrappers": [
                {"name": "resize_v0", "params": [12, 12]},
                {"name": "dtype_v0", "params": ["float32"]},
                {
                    "name": "normalize_obs_v0",
                    "params": {"env_min": 0.0, "env_max": 1.0},
                },
            ],
        }
    )[env_id]


def make_parallel_env(env_id, parallel=True, **env_configs) -> Any:
    # filter maatari
    env_id = env_id[8:]
    env_configs["env_config"] = load_env_config(env_id)
    # print("------ configs:", env_configs)
    env_configs = env_configs["env_config"]
    env_module = env_module = importlib.import_module(f"pettingzoo.atari.{env_id}")
    ori_caller = env_module.env if not parallel else env_module.parallel_env
    wrappers = (
        env_configs.pop("wrappers") if env_configs.get("wrappers") is not None else []
    )
    wrapped_caller = nested_env_creator(ori_caller, wrappers)

    return wrapped_caller(**env_configs)


class MAAatri(gym.Env):
    """Since sample-factory support only homogeneous agent settings"""

    def __init__(self, full_env_name, cfg):
        self.name = full_env_name
        self.cfg = cfg
        self.curr_episode_steps = 0
        self.res = 10  # 10 * 10 images
        self.channels = 3

        self.env = make_parallel_env(full_env_name, **cfg)

        self.agents = self.env.possible_agents
        self.observation_space = self.env.observation_spaces[self.agents[0]]
        self.action_space = self.env.action_spaces[self.agents[0]]
        self.num_agents = len(self.agents)

    def step(self, action: List):
        actions = dict(zip(self.agents, action))

        obs, reward, done, info = self.env.step(actions)
        obs_seq = [obs[a] for a in self.agents]
        reward_seq = [reward[a] for a in self.agents]
        done_seq = [done[a] for a in self.agents]
        info_seq = [info[a] for a in self.agents]

        return obs_seq, reward_seq, done_seq, info_seq

    def reset(self, **kwargs):
        obs = self.env.reset()
        return [obs[a] for a in self.agents]


def make_env(env_name, cfg, **kwargs):
    return MAAatri(env_name, cfg)