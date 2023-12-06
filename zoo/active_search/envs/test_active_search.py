import numpy as np
from easydict import EasyDict
from zoo.active_search.envs.active_search_env import ActiveSearchEnv

width, height, h_3d = 80, 80, 100
uav_action_lst = [(0, 0, 0), (5, 0, 0), (0, 5, 0), (0, 0, 2), (-5, 0, 0), (0, -5, 0), (0, 0, -2)]
num_uavs = 2
num_people = 15
search_banjing_max = 10
seed = 0

mcfg=EasyDict(
        env_name='active-search-v0',
        num_uavs = num_uavs,
        num_people = num_people,
        single_action_shape = len(uav_action_lst),
        observation_space = width*height+3*num_uavs,
        replay_path = 'Xia_result/video',
        env_config= EasyDict({
            # "width": width,
            # "height": height,
            # "h_3d": h_3d,
            # "seed": seed,
            # "num_uavs": num_uavs,
            # "num_people": num_people,
            # "EP_MAX_TIME": 200,
            # "uav_action_lst": uav_action_lst,
            # "observation_size": width*height+3*num_uavs,
            # "search_banjing_max": search_banjing_max,
            "save_replay": True,
        })
        # uav_config = EasyDict({
        #     "width": width,
        #     "height": height,
        #     "h_3d": h_3d,
        #     "seed": seed,
        #     "h_min": uav_h_min,
        #     "uav_action_lst": uav_action_lst,
        #     "search_banjing_max": search_banjing_max,
        # })
        # people_config = EasyDict({
        #     "width": width,
        #     "height": height,
        #     "h_3d": h_3d,
        #     "seed": seed,
        #     "search_banjing_max": search_banjing_max,
        # })
        )

def test_naive(cfg):
    env = ActiveSearchEnv(cfg)
    env.seed(314)
    assert env._seed == 314
    obs = env.reset()
    assert obs['observation'].shape == (width*height+3*num_uavs,)
    for i in range(100000):
        random_action = env.random_action()
        timestep = env.step(random_action)
        print(timestep)
        assert isinstance(timestep.obs['observation'], np.ndarray)
        assert isinstance(timestep.done, bool)
        assert timestep.obs['observation'].shape == (width*height+3*num_uavs,)
        assert timestep.reward.shape == (1, )
        if timestep.done:
            break
    print(env.observation_space, env.action_space, env.reward_space)
    env.close()

test_naive(mcfg)