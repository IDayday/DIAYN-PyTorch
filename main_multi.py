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


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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

def main(configs):
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


if __name__ == "__main__":
    # setting
    configs = get_config()
    # if you want to put the model on GPU and share the params, you should use "spawn"
    # 4 processes will used 9GB memory on GPU
    # 10 processes will used 18.5GB memory on GPU
    # if configs["forward"] == 'GPU':
    #     mp.set_start_method('spawn')
    mp.set_start_method('spawn')
    # also you can put the model on CPU, and only use GPU when training
    main(configs)
