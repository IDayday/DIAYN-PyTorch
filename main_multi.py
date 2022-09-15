import gym
from agent import SACAgent, get_model_params
from config import get_config
import torch
import random
import numpy as np
import mujoco_py
import os
import torch.multiprocessing as mp
from children_process import learn, simulation, evaluate


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

backup_file = ['./main_multi.py','./agent.py','./model.py','./children_process.py','./config.py']


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return

def main():
    # setting
    configs = get_config()
    # get state or action shape from the env
    test_env = gym.make(configs["env_name"])
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    # if clip is needed
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    configs.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds,
                   "backup_file": backup_file})
    print("configs:", configs)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    # seed in major process
    seed_everything(configs["seed"])

    # initialize agent(including model parameters)
    # p_z is uniform
    p_z = np.full(configs["n_skills"], 1 / configs["n_skills"])
    ag = SACAgent(p_z=p_z, **configs)
    params_list = get_model_params(ag)
    current_params_list = []
    for pa in params_list:
        pa.share_memory()
        current_params_list.append(pa)

    data_queue = mp.Queue(maxsize=100)   # FIFO 
    signal_queue = mp.Queue()
    simlog_queue = mp.Queue()
    evalsignal_queue = mp.Queue()

    # multiprocessing
    processes = [] 
    p = mp.Process(target=learn, args=(current_params_list, configs, data_queue, signal_queue, simlog_queue, evalsignal_queue))
    p.start()
    processes.append(p)
    for rank in range(1, configs["num_processes"] + 1):
        p = mp.Process(target=simulation, args=(rank, current_params_list, configs, data_queue, signal_queue, simlog_queue))
        p.start()
        processes.append(p)

    # p = mp.Process(target=evaluate, args=(model_params, args, signal_queue, evalsignal_queue))
    # p.start()
    # processes.append(p)

    for p in processes:
        p.join()






    # env = gym.make(params["env_name"])


    # agent = SACAgent(p_z=p_z, **params)
    # logger = Logger(agent, **params)

    # if params["do_train"]:

    #     if not params["train_from_scratch"]:
    #         episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
    #         agent.hard_update_target_network()
    #         min_episode = episode
    #         np.random.set_state(np_rng_state)
    #         env.np_random.set_state(env_rng_states[0])
    #         env.observation_space.np_random.set_state(env_rng_states[1])
    #         env.action_space.np_random.set_state(env_rng_states[2])
    #         agent.set_rng_states(torch_rng_state, random_rng_state)
    #         print("Keep training from previous run.")

    #     else:
    #         min_episode = 0
    #         last_logq_zs = 0
    #         np.random.seed(params["seed"])
    #         env.seed(params["seed"])
    #         env.observation_space.seed(params["seed"])
    #         env.action_space.seed(params["seed"])
    #         print("Training from scratch.")

    #     logger.on()
    #     for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
    #         z = np.random.choice(params["n_skills"], p=p_z)
    #         state = env.reset()
    #         state = concat_state_latent(state, z, params["n_skills"])
    #         episode_reward = 0
    #         logq_zses = []

    #         max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
    #         for step in range(1, 1 + max_n_steps):

    #             action = agent.choose_action(state)
    #             next_state, reward, done, _ = env.step(action)
    #             next_state = concat_state_latent(next_state, z, params["n_skills"])
    #             agent.store(state, z, done, action, next_state)
    #             logq_zs = agent.train()
    #             if logq_zs is None:
    #                 logq_zses.append(last_logq_zs)
    #             else:
    #                 logq_zses.append(logq_zs)
    #             episode_reward += reward
    #             state = next_state
    #             if done:
    #                 break

    #         logger.log(episode,
    #                    episode_reward,
    #                    z,
    #                    sum(logq_zses) / len(logq_zses),
    #                    step,
    #                    np.random.get_state(),
    #                    env.np_random.get_state(),
    #                    env.observation_space.np_random.get_state(),
    #                    env.action_space.np_random.get_state(),
    #                    *agent.get_rng_states(),
    #                    )

    # else:
    #     logger.load_weights()
    #     player = Play(env, agent, n_skills=params["n_skills"])
    #     player.evaluate()


if __name__ == "__main__":
    # if you want to put the model on GPU and share the params, you should use "spawn"
    # 4 processes will used 9GB memory on GPU
    # 10 processes will used 18.5GB memory on GPU
    mp.set_start_method('spawn')
    # also you can put the model on CPU, and only use GPU when training
    main()
