import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network import Network, ValueNet
import wandb
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENTROPY_COEFF = 0.001
# STATE_DIM = 
# POLICY_ARGS = {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM}


class A2C(object):
    def __init__(self, args):
        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.batch_size = args['batch_size']
        self.policy_args = {'state_dim': self.state_dim, 'action_dim': self.action_dim}
        # network_args = {'state_dim': self.state_dim, 'action_dim': self.action_dim}
        self.policy = Network(self.policy_args).to(DEVICE)
        self.value_net = ValueNet(self.policy_args).to(DEVICE)
        self.mse = nn.MSELoss()
        self.policy_optimizer = optim.SGD(self.policy.parameters(), lr=args['learning_rate'])
        self.value_net_optimizer = optim.SGD(self.value_net.parameters(), lr=args['learning_rate'])
    def update(self, replay_buffer):
        state_array, action_array, reward_array, true_state_value_array, advantage_array = replay_buffer.sample(self.batch_size)
        state_batch = torch.tensor(state_array).float().to(DEVICE)
        action_batch = torch.tensor(action_array).float().to(DEVICE)
        reward_batch = torch.tensor(reward_array).float().to(DEVICE)
        # logp_old_batch = torch.tensor(logp_old_array).float().to(DEVICE)
        true_state_value_batch = torch.tensor(true_state_value_array).float().to(DEVICE)
        advantage_batch = torch.tensor(advantage_array).float().to(DEVICE)

        true_state_value_batch = true_state_value_batch.unsqueeze(1)
        advantage_batch = advantage_batch.unsqueeze(1)
        # print(true_state_value_batch)
        # print(true_state_value_batch.shape)
        
        state_value_batch = self.value_net(state_batch)
        # print(true_state_value_batch.shape)
        # print(state_value_batch.shape)
        logp_batch = self.policy.logp(state_batch, action_batch)
        # print(logp_batch)
        state_entropy_batch = -logp_batch*torch.exp(logp_batch)
        entropy_loss = -torch.mean(state_entropy_batch)
        
        critic_loss = self.mse(state_value_batch, true_state_value_batch)
        # print(logp_batch.shape)
        # print(advantage_batch.shape)
        action_loss = -torch.mean(logp_batch * advantage_batch)
        actor_loss = action_loss + ENTROPY_COEFF*entropy_loss
        # print(actor_loss)
        # print(actor_loss.shape)
        # exit()
        wandb.log({'action_loss': action_loss.item(), 'entropy_loss': ENTROPY_COEFF*entropy_loss.item()})
        
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        # nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
        self.policy_optimizer.step()

        self.value_net_optimizer.zero_grad()
        critic_loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        # nn.utils.clip_grad_norm_(self.value_net.parameters(), 5)
        self.value_net_optimizer.step()
        return actor_loss, critic_loss