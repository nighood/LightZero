import numpy as np
from easydict import EasyDict
from zoo.active_search.envs.active_search_env import ActiveSearchEnv

mcfg=EasyDict(
        env_name='active-search-v0',
        num_uavs = 2,
        num_people = 15,
        single_action_shape = 7,
        observation_space = 90006
        )

def test_naive(cfg):
    env = ActiveSearchEnv(cfg)
    env.seed(314)
    assert env._seed == 314
    obs = env.reset()
    assert obs['observation'].shape == (90006,)
    for i in range(10):
        random_action = env.random_action()
        timestep = env.step(random_action)
        print(timestep)
        assert isinstance(timestep.obs['observation'], np.ndarray)
        assert isinstance(timestep.done, bool)
        assert timestep.obs['observation'].shape == (90006,)
        assert timestep.reward.shape == (1, )
    print(env.observation_space, env.action_space, env.reward_space)
    env.close()

test_naive(mcfg)