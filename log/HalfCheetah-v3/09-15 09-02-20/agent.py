import numpy as np
from model import PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator
import torch
import random
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('obs', 'z', 'done', 'action', 'next_obs'))

def get_model_params(agent):
    p_parms = agent.policy_network
    q1_params = agent.q_value_network1
    q2_params = agent.q_value_network2
    v_params = agent.value_network
    vt_params = agent.value_target_network
    d_params = agent.discriminator
    params_list = [p_parms, q1_params, q2_params, v_params, vt_params, d_params]
    return params_list

def set_model_params(agent, model_params):
    p_parms, q1_params, q2_params, v_params, vt_params, d_params = model_params
    agent.policy_network.load_state_dict(p_parms.state_dict())
    agent.q_value_network1.load_state_dict(q1_params.state_dict())
    agent.q_value_network2.load_state_dict(q2_params.state_dict())
    agent.value_network.load_state_dict(v_params.state_dict())
    agent.value_target_network.load_state_dict(vt_params.state_dict())
    agent.discriminator.load_state_dict(d_params.state_dict())
    return agent

def update_model_params(current_model_params, target_model_params):
    for i,p in enumerate(target_model_params):
        current_model_params[i].load_state_dict(p.state_dict())

def concat_obs_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

class SACAgent:
    def __init__(self,
                 p_z,
                 **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.memory = deque(maxlen=self.config["mem_size"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu" 

        torch.manual_seed(self.config["seed"])
        self.policy_network = PolicyNetwork(n_states=self.n_states + self.n_skills,
                                            n_actions=self.config["n_actions"],
                                            action_bounds=self.config["action_bounds"],
                                            n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                          n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_target_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                                 n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        self.hard_update_target_network()

        self.discriminator = Discriminator(n_states=self.n_states, n_skills=self.n_skills,
                                           n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

    def choose_action(self, states):
        # print(states)
        # print(states.shape)
        # self.set_policy_net_to_cpu_mode()
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    def store(self, state, z, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, z, done, action, next_state)

    def unpack(self, batch, train_device):
        batch = Transition(*zip(*batch))

        states = torch.Tensor(batch.obs).to(train_device)
        zs = torch.ByteTensor(batch.z).unsqueeze(1).long().to(train_device)
        dones = torch.BoolTensor(batch.done).unsqueeze(1).to(train_device)
        actions = torch.Tensor(batch.action).to(train_device)
        next_states = torch.Tensor(batch.next_obs).to(train_device)

        # states = torch.cat(batch.obs).view(self.batch_size, self.n_states + self.n_skills).to(train_device)
        # zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(train_device)
        # dones = torch.cat(batch.done).view(self.batch_size, 1).to(train_device)
        # actions = torch.cat(batch.action).view(-1, self.config["n_actions"]).to(train_device)
        # next_states = torch.cat(batch.next_obs).view(self.batch_size, self.n_states + self.n_skills).to(train_device)
        return states, zs, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        else:
            # use GPU if available
            # self.set_policy_net_to_gpu_mode()
            
            batch = random.sample(self.memory,self.batch_size)
            states, zs, dones, actions, next_states = self.unpack(batch, self.device)
            p_z = from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states)
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.config["alpha"] * log_probs.detach()

            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)

            logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
            p_z = p_z.gather(-1, zs)
            logq_z_ns = log_softmax(logits, dim=-1)
            rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = self.config["reward_scale"] * rewards.float() + \
                           self.config["gamma"] * self.value_target_network(next_states) * (~dones)
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.mse_loss(q1, target_q)
            q2_loss = self.mse_loss(q2, target_q)

            policy_loss = (self.config["alpha"] * log_probs - q).mean()
            logits = self.discriminator(torch.split(states, [self.n_states, self.n_skills], dim=-1)[0])
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            self.soft_update_target_network(self.value_network, self.value_target_network)

            return -discriminator_loss.item()

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data +
                                    (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def get_rng_states(self):
        return torch.get_rng_state(), self.memory.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        self.memory.set_rng_state(random_rng_state)

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        device = torch.device("cpu")
        self.policy_network.to(device)

    def set_policy_net_to_gpu_mode(self):
        device = torch.device("cuda")
        self.policy_network.to(device)
