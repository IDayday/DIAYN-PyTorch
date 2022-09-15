import os
import numpy as np
import random
import gym
import time
from tqdm import tqdm
import torch
from copy import deepcopy
import logging
import shutil
from collections import namedtuple
from model import save_weights
from agent import SACAgent, set_model_params, concat_obs_latent, update_model_params, get_model_params
from torch.utils.tensorboard import SummaryWriter

Transition = namedtuple('Transition', ('obs', 'z', 'done', 'action', 'next_obs'))

def learn(model_params, args, data_queue, signal_queue, simlog_queue, evalsignal_queue):
    # save code files and logs
    save_dir = './checkpoints/' + args["env_name"] + "/" + time.strftime("%m-%d %H-%M-%S", time.localtime())
    os.makedirs(save_dir)
    save_dir += '/'
    for file in args["backup_file"]:
        shutil.copy(file, save_dir + file.split('/')[-1])
    exp_dir = save_dir + 'exp'
    model_dir = save_dir + 'model/'
    log_dir = save_dir + 'train.log'
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    # init agent
    p_z = np.full(args["n_skills"], 1 / args["n_skills"])
    ag = SACAgent(p_z=p_z, **args)
    ag = set_model_params(ag, model_params)

    # logging setting
    writer = SummaryWriter(exp_dir)
    logging.basicConfig(filename=log_dir, filemode='a', level=logging.INFO)
    episode_reward_max = -np.inf
    running_logq_z = 0


    start = time.time()
    for i_episode in tqdm(range(1, args["max_n_episodes"] + 1)):
        logq_zses = []
        print("simulation start : {}/{}".format(i_episode, args["max_n_episodes"]))

        # get data
        for r in range(args["num_processes"]):
            ret, simlog = data_queue.get()
            rank, temp, episode_reward, step, episode_time, z = simlog
            info = 'Rank {} Finished simulation: {} | Reward: {:.2f} | Step: {:d} | Costtime: {:.2f}'.format(
                        rank, temp, episode_reward, step, episode_time)
            print(info)
            ret_clone = deepcopy(ret)
            ag.memory.extend(ret)
            logging.info(info)
            print("memory length: ",len(ag.memory))
            del ret
            del ret_clone

            episode_reward_max = max(episode_reward_max, episode_reward)
            writer.add_histogram(str(z)+ "episode reward", episode_reward)
            writer.add_histogram(str(z)+ "episode step", step)
            writer.add_histogram("Total Rewards", episode_reward)

            if len(ag.memory) >= args["warmup_size"]:
                signal_queue.put(1)
                for epoch in range(1, step//args["num_processes"] + 1):
                    logq_zs = ag.train()
                    logq_zses.append(logq_zs)
                logq_z = sum(logq_zses) / len(logq_zses)
                if running_logq_z == 0:
                    running_logq_z = logq_z
                else:
                    running_logq_z = 0.99*running_logq_z + 0.01*logq_z

                # update global model parameter
                updated_model_params = get_model_params(ag)
                update_model_params(model_params, updated_model_params)
                # save weights
                save_weights(model_dir, i_episode, ag, episode_reward_max, running_logq_z)
                # expect for samples from new model params
                while not data_queue.empty():
                    data_queue.get()
                _ = signal_queue.get()
        
        # save log
        writer.add_scalar("Max episode reward", episode_reward_max, i_episode)
        writer.add_scalar("Running logq(z|s)", running_logq_z, i_episode)


    end = round((time.time() - start)/60,2)
    print(f"learning completed! cost time: {end} min")
    signal_queue.put("end")
    signal_queue.put("end")


def simulation(rank, model_params, args, data_queue, signal_queue, simlog_queue):
    print('process id:', os.getpid())
    print(f'rank {rank} simulation start')
    seed = args["seed"] + rank
    # init env
    env = gym.make(args["env_name"])
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    # init agent
    p_z = np.full(args["n_skills"], 1 / args["n_skills"])
    ag = SACAgent(p_z=p_z, **args)
    ag = set_model_params(ag, model_params)

    torch.manual_seed(seed)
    # the np.random.seed will fork the main process, so we need reset again
    np.random.seed(seed)

    temp = 0
    max_n_steps = min(args["max_episode_len"], env.spec.max_episode_steps)

    while True:
        train_samples = []
        done = False
        temp += 1
        z = np.random.choice(args["n_skills"], p=p_z)
        obs = env.reset()
        obs = concat_obs_latent(obs, z, args["n_skills"])
        episode_reward = 0
        one_episode_time = time.time()
        for step in range(1, 1 + max_n_steps):
            wait_update = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                wait_update = True
            if wait_update:
                ag = set_model_params(ag, model_params)
                # train_samples = []
                # obs = env.reset()
                # obs = concat_obs_latent(obs, z, args["n_skills"])
            # simulating to get a trajectory
            with torch.no_grad():
                action = ag.choose_action(obs) # clip if needed
                next_obs, reward, done, _ = env.step(action)
                next_obs = concat_obs_latent(next_obs, z, args["n_skills"])

                # pack tha variable
                # obs_ = torch.from_numpy(obs).float().to("cpu")
                # z_ = torch.ByteTensor([z]).to("cpu")
                # done_ = torch.BoolTensor([done]).to("cpu")
                # action_ = torch.Tensor([action]).to("cpu")
                # next_obs_ = torch.from_numpy(next_obs).float().to("cpu")

                train_samples.append(Transition(obs, z, done, action, next_obs))

                episode_reward += reward
                obs = next_obs
            if done or step >= max_n_steps:
                episode_time = time.time() - one_episode_time
                simlog = (rank, temp, episode_reward, step, episode_time, z)
                data_queue.put((train_samples, simlog))
                time.sleep(1)
                break
        if signal_queue.qsize() > 1:
            break

def evaluate(network, args, signal_queue, evalsignal_queue):
    pass
