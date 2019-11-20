import numpy as np
import random
import wandb
DISCOUNT_FACTOR = 0.99
# BATCH_SIZE = 256
# ENV_SIZE = 10

class ReplayBuffer(object):
    def __init__(self, args):
        self.size = args['buffer_size']
        self.batch_size = args['batch_size']
        self.env_size = args['env_size']
        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.buffer_extension = args['buffer_size'] * args['env_size']
        self.total_legal_length = 0
        self.current_index = 0
        self.state_buffer = np.zeros(shape=(self.env_size, self.size, self.state_dim))
        self.action_buffer = np.zeros(shape=(self.env_size, self.size, self.action_dim))
        self.reward_buffer = np.zeros(shape=(self.env_size, self.size))
        self.state_value_buffer = np.zeros(shape=(self.env_size, self.size))
        self.true_state_value_buffer = np.zeros(shape=(self.env_size, self.size))
        self.advantage_buffer = np.zeros(shape=(self.env_size, self.size))
        # self.logp_buffer = np.zeros(shape=(self.env_size, self.size)) # for important sampling
        self.terminate_buffer = np.zeros(shape=(self.env_size, self.size))
        # self.index_array = [x for x in range(self.size)]

        self.state_sample_batch_tmp = np.zeros(shape=(self.buffer_extension, self.state_dim))
        self.action_sample_batch_tmp = np.zeros(shape=(self.buffer_extension, self.action_dim))
        self.reward_sample_batch_tmp = np.zeros(shape=(self.buffer_extension))
        # self.logp_sample_batch_tmp = np.zeros(shape=(self.buffer_extension))
        self.true_state_value_sample_batch_tmp = np.zeros(shape=(self.buffer_extension))
        self.advantage_sample_batch_tmp = np.zeros(shape=(self.buffer_extension))
        self.terminate_sample_batch_tmp = np.zeros(shape=(self.buffer_extension))

        self.state_sample_batch = np.zeros(shape=(self.batch_size, self.state_dim))
        self.action_sample_batch = np.zeros(shape=(self.batch_size, self.action_dim))
        self.reward_sample_batch = np.zeros(shape=(self.batch_size))
        # self.logp_sample_batch = np.zeros(shape=(self.batch_size))
        self.true_state_value_sample_batch = np.zeros(shape=(self.batch_size))
        self.advantage_sample_batch = np.zeros(shape=(self.batch_size))

    def store(self, state, action, reward, state_value, isTerminate):
        for i in range(self.env_size):
            self.state_buffer[i][self.current_index] = state[i]
            self.action_buffer[i][self.current_index] = action[i]
            # self.logp_buffer[i][self.current_index] = logp[i]
            self.reward_buffer[i][self.current_index] = reward[i]
            # print(state_value[i])
            self.state_value_buffer[i][self.current_index] = state_value[i]
            # print(state_value[i])
            # print(self.state_value_buffer[i][self.current_index])
            
            # print('----------------')
            self.terminate_buffer[i][self.current_index] = 1 if isTerminate[i]==True else 0

        self.current_index = (self.current_index + 1)%self.size
    def update_true_state_value(self):
        # print(self.current_index)
        discount_factor = DISCOUNT_FACTOR
        # print(self.current_index)
        # discount_time = 0
        # value = 0
        # self.true_state_value_buffer = np.zeros(shape=(1, self.size))
        
        # print(update_index)
        # update_index_next = (update_index + 1) % self.size
        for i in range(self.env_size):
            update_index = self.size-1
            for j in range(self.size):
                # print(update_index)
                # exit()
                if self.terminate_buffer[i][update_index] == 0 and update_index != self.size-1: 
                    self.true_state_value_buffer[i][update_index] = self.reward_buffer[i][update_index] + discount_factor * (self.true_state_value_buffer[i][update_index_next])
                else:
                    self.true_state_value_buffer[i][update_index] = self.reward_buffer[i][update_index]
                    # print('qq')
                # print(self.true_state_value_buffer[0][update_index])
                update_index -= 1
                update_index_next = update_index + 1
        # print(self.true_state_value_buffer)
    def update_advantage(self):
        # A(S(t), A(t)) = Q(S(t), A(t)) - V(S(t)) = R(t) + gamma*V(S(t+1)) - V(S(t))
        discount_factor = DISCOUNT_FACTOR
        
        for i in range(self.env_size):
            update_index = self.size-1
            for j in range(self.size):
                if self.terminate_buffer[i][update_index] == 0 and update_index != self.size-1: 
                    delta = self.reward_buffer[i][update_index] + discount_factor*self.state_value_buffer[i][update_index_next] - self.state_value_buffer[i][update_index]
                    self.advantage_buffer[i][update_index] = delta
                else:
                    delta = self.reward_buffer[i][update_index] - self.state_value_buffer[i][update_index]
                    self.advantage_buffer[i][update_index] = delta
                update_index -= 1
                update_index_next = update_index + 1
    def merge_trajectory(self):
        self.total_legal_length = 0
        for i in range(self.env_size):
            legal_length = self.size
            for j in range(self.size):
                if self.terminate_buffer[i][self.size-1-j] == 0:
                    legal_length -= 1
                else:
                    self.total_legal_length += legal_length
                    self.state_sample_batch_tmp[self.total_legal_length-legal_length:self.total_legal_length] = self.state_buffer[i][0:legal_length]
                    self.action_sample_batch_tmp[self.total_legal_length-legal_length:self.total_legal_length] = self.action_buffer[i][0:legal_length]
                    self.reward_sample_batch_tmp[self.total_legal_length-legal_length:self.total_legal_length] = self.reward_buffer[i][0:legal_length]
                    # self.logp_sample_batch_tmp[self.total_legal_length-legal_length:self.total_legal_length] = self.logp_buffer[i][0:legal_length]
                    self.true_state_value_sample_batch_tmp[self.total_legal_length-legal_length:self.total_legal_length] = self.true_state_value_buffer[i][0:legal_length]
                    self.advantage_sample_batch_tmp[self.total_legal_length-legal_length:self.total_legal_length] = self.advantage_buffer[i][0:legal_length]
                    self.terminate_sample_batch_tmp[self.total_legal_length-legal_length:self.total_legal_length] = self.terminate_buffer[i][0:legal_length]
                    break
            # print(legal_length)
        tra = 0
        for i in range(self.total_legal_length):
            if self.terminate_sample_batch_tmp[i] == 1:
                tra += 1
        average_step = self.total_legal_length / tra
        print('average_step: {average_step}'.format(average_step = average_step))
        wandb.log({'average_step': average_step})


        
        # print(self.total_legal_length)
        # if total_legal_length > 3000:
            # print(total_legal_length)
        if self.total_legal_length < 256:
            print(self.total_legal_length)
            print('gg')
            exit()
    def sample(self, batch_size):
        # [state, action, reward, logp, true_state_value, advantage]
        
        # sample_batch = np.zeros(shape=(batch_size, 6))
        index = batch_size
        index_array = [x for x in range(self.total_legal_length)]
        sample_index = random.sample(index_array, index)

        # for j in range(self.env_size):
            # for i, element in enumerate(sample_index):
                # sample_batch[i] = [self.state_buffer[0][element], self.action_buffer[0][element], self.reward_buffer[0][element], self.logp_buffer[0][element], self.true_state_value_buffer[0][element], self.advantage_buffer[0][element]]
                # self.state_sample_batch[i+j*index] = self.state_buffer[j][element]
                # self.action_sample_batch[i+j*index] = self.action_buffer[j][element]
                # self.reward_sample_batch[i+j*index] = self.reward_buffer[j][element]
                # self.logp_sample_batch[i+j*index] = self.logp_buffer[j][element]
                # self.true_state_value_sample_batch[i+j*index] = self.true_state_value_buffer[j][element]
                # self.advantage_sample_batch[i+j*index] = self.advantage_buffer[j][element]
                
        # print(self.state_sample_batch.dtype)
        for i, element in enumerate(sample_index):
            self.state_sample_batch[i] = self.state_sample_batch_tmp[element]
            self.action_sample_batch[i] = self.action_sample_batch_tmp[element]
            self.reward_sample_batch[i] = self.reward_sample_batch_tmp[element]
            # self.logp_sample_batch[i] = self.logp_sample_batch_tmp[element]
            self.true_state_value_sample_batch[i] = self.true_state_value_sample_batch_tmp[element]
            self.advantage_sample_batch[i] = self.advantage_sample_batch_tmp[element]
        return self.state_sample_batch, self.action_sample_batch, self.reward_sample_batch, self.true_state_value_sample_batch, self.advantage_sample_batch