from typing import Union, Optional

import gymnasium
import numpy as np
from itertools import product
import logging
import imageio

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs import ObsPlusPrevActRewWrapper
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY

import activesearch


@ENV_REGISTRY.register('activesearch_lightzero')
class ActiveSearchEnv(BaseEnv):

    def __init__(self, cfg: dict = {}) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = self._cfg.get('replay_path', None)
        self._num_uavs = self._cfg.num_uavs
        self._num_people = self._cfg.num_people
        # action_space
        self._single_action_shape = self._cfg.single_action_shape   # e.g. 7
        self._real_single_action_space = list(range(self._single_action_shape))
        self._real_action_space = np.array(list(product(self._real_single_action_space, repeat=self._num_uavs)))
        one_uav_action_n = len(self._real_action_space)
        self._action_space = gymnasium.spaces.Discrete(one_uav_action_n)
        # obs_space
        self._observation_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(self._cfg.observation_space,))
        self._action_space.seed(0)  # default seed
        self._reward_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32)
        self._continuous = False

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gymnasium.make('active-search-v0')
            if self._replay_path is not None:
                self.enable_save_replay(self._replay_path)
            # if hasattr(self._cfg, 'obs_plus_prev_action_reward') and self._cfg.obs_plus_prev_action_reward:
            #     self._env = ObsPlusPrevActRewWrapper(self._env)
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
            self._action_space.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
            self._action_space.seed(self._seed)
        self._eval_episode_return = 0
        # process obs
        raw_obs, info = self._env.reset()
        action_mask = np.ones(self._action_space.n, 'int8')
        obs = {'observation': raw_obs, 'action_mask': action_mask, 'to_play': -1}

        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        if isinstance(action, np.ndarray) and action.shape == (1, ):
            action = action.squeeze()  # 0-dim array
        real_action = self._real_action_space[action]
        assert isinstance(real_action, np.ndarray) and len(real_action) == self._num_uavs, "illegal action!"
        raw_obs, rew, done, _, info = self._env.step(real_action)

        self._eval_episode_return += rew
        if self._replay_path is not None:
            # please only turn on when eval
            self.render()
        if done:
            info['eval_episode_return'] = self._eval_episode_return
            self.display_frames_as_gif(self._video_frames, self._replay_path)
            self._video_frames = []
            # logging.INFO('one game finish!')

        action_mask = np.ones(self._action_space.n, 'int8')
        obs = {'observation': raw_obs, 'action_mask': action_mask, 'to_play': -1}
        rew = to_ndarray([rew]).astype(np.float32)
        return BaseEnvTimestep(obs, rew, done, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self._env.config['save_replay'] = True
        self._video_frames = []

    def render(self):
        # TODO(rjy): should save gif
        self._video_frames.append(self._env.render())

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gymnasium.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "LightZero Active Search Env"
    
    def display_frames_as_gif(self, frames, path):
        imageio.mimsave(path+'/video.gif', frames, duration=20)