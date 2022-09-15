import gym
from agent import SACAgent
from model import load_weights
from config import get_config
import numpy as np
# from mujoco_py import GlfwContext
import cv2
import numpy as np
import os

# GlfwContext(offscreen=True)

class Play:
    def __init__(self, config, env, agent, video_path):
        self.config = config
        self.env = env
        self.agent = agent
        self.video_path = video_path
        self.n_skills = configs["n_skills"]
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID') #  'XVID'输出avi 
        # self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') #  'mp4v'输出mp4

        if not os.path.exists(self.video_path):
            os.mkdir(self.video_path)

    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def evaluate(self):

        for z in range(self.n_skills):
            video_writer = cv2.VideoWriter(self.video_path + f"/skill{z}" + ".avi", self.fourcc, 50.0, (250, 250))
            s = self.env.reset()
            s = self.concat_state_latent(s, z, self.n_skills)
            episode_reward = 0
            for _ in range(self.env.spec.max_episode_steps):
                action = self.agent.choose_action(s)
                s_, r, done, _ = self.env.step(action)
                s_ = self.concat_state_latent(s_, z, self.n_skills)
                episode_reward += r
                if done:
                    break
                s = s_
                I = self.env.render(mode='rgb_array')
                I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                I = cv2.resize(I, (250, 250))
                video_writer.write(I)
            print(f"skill: {z}, episode reward:{episode_reward:.1f}")
            video_writer.release()
        self.env.close()
        cv2.destroyAllWindows()


def main(configs):
    test_env = gym.make(configs["env_name"])
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    configs.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", configs)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    env = gym.make(configs["env_name"])
    model_path = "./checkpoints/HalfCheetah-v3/09-15 09-02-20/model/params.pth"
    video_path = "./checkpoints/HalfCheetah-v3/09-15 09-02-20/video"

    p_z = np.full(configs["n_skills"], 1 / configs["n_skills"])
    agent = SACAgent(p_z=p_z, **configs)
    load_weights(model_path, agent)
    player = Play(configs, env, agent, video_path)
    player.evaluate()



if __name__ == "__main__":
    configs = get_config()
    main(configs)

