from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)

def save_weights(weight_path, episode, agent, max_episode_reward, running_logq_zs):
    torch.save({"policy_network_state_dict": agent.policy_network.state_dict(),
                "q_value_network1_state_dict": agent.q_value_network1.state_dict(),
                "q_value_network2_state_dict": agent.q_value_network2.state_dict(),
                "value_network_state_dict": agent.value_network.state_dict(),
                "discriminator_state_dict": agent.discriminator.state_dict(),
                "q_value1_opt_state_dict": agent.q_value1_opt.state_dict(),
                "q_value2_opt_state_dict": agent.q_value2_opt.state_dict(),
                "policy_opt_state_dict": agent.policy_opt.state_dict(),
                "value_opt_state_dict": agent.value_opt.state_dict(),
                "discriminator_opt_state_dict": agent.discriminator_opt.state_dict(),
                "episode": episode,
                "max_episode_reward": max_episode_reward,
                "running_logq_zs": running_logq_zs
                },
                weight_path + "/params.pth")

# def load_weights():
#     # TODO: change path of checkpoint
#     model_dir = glob.glob("Checkpoints/" + self.config["env_name"][:-3] + "/" + "2022-09-14-02-48-04")
#     model_dir.sort()
#     checkpoint = torch.load(model_dir[-1] + "/params.pth")
#     self.log_dir = model_dir[-1].split(os.sep)[-1]
#     self.agent.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
#     self.agent.q_value_network1.load_state_dict(checkpoint["q_value_network1_state_dict"])
#     self.agent.q_value_network2.load_state_dict(checkpoint["q_value_network2_state_dict"])
#     self.agent.value_network.load_state_dict(checkpoint["value_network_state_dict"])
#     self.agent.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
#     self.agent.q_value1_opt.load_state_dict(checkpoint["q_value1_opt_state_dict"])
#     self.agent.q_value2_opt.load_state_dict(checkpoint["q_value2_opt_state_dict"])
#     self.agent.policy_opt.load_state_dict(checkpoint["policy_opt_state_dict"])
#     self.agent.value_opt.load_state_dict(checkpoint["value_opt_state_dict"])
#     self.agent.discriminator_opt.load_state_dict(checkpoint["discriminator_opt_state_dict"])

#     self.max_episode_reward = checkpoint["max_episode_reward"]
#     self.running_logq_zs = checkpoint["running_logq_zs"]

#     return checkpoint["episode"], self.running_logq_zs, (*checkpoint["rng_states"],)



class Discriminator(nn.Module, ABC):
    def __init__(self, n_states, n_skills, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits


class ValueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        return self.value(x)


class QvalueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        self.hidden1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.q_value(x)


class PolicyNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions
        self.action_bounds = action_bounds

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

    def sample_or_likelihood(self, states):
        dist = self(states)
        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1]), log_prob
