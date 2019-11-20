from gym_unity.envs import UnityEnv
from replay_buffer import ReplayBuffer
import torch
import numpy as np 
from a2c import A2C
import wandb
wandb.init(project='a2c-unity-crawler')
torch.cuda.empty_cache()
STATE_DIM = 126
ACTION_DIM = 20
BATCH_SIZE = 1024
LEARNING_RATE = 2e-4
EPISODE = 10000
HORIZON = 1024
ENV_SIZE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = './a2c_checkpoint'
a2c_args = {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'batch_size': BATCH_SIZE, 'learning_rate':LEARNING_RATE}
replay_buffer_args = {'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'buffer_size': HORIZON, 'env_size': ENV_SIZE, 'batch_size': BATCH_SIZE}

replay_buffer = ReplayBuffer(args=replay_buffer_args)
a2c = A2C(args=a2c_args)
policy = a2c.policy
value_net = a2c.value_net
env = UnityEnv('crawler', multiagent=True, worker_id = 11).unwrapped

for current_episode in range(EPISODE):
    # env.render()
    env.reset()
    state_list = env.reset()
    average_reward = 0
    total_game_step = 0
    while True:
        if total_game_step >= HORIZON:
            break
        state_tensor = torch.tensor(state_list).float().to(DEVICE)
        mu_vector, sigma_vector = policy(state_tensor)
        action_tensor = policy.act(mu_vector, sigma_vector)
        action_list = []
        # action_tensor_clamp = torch.clamp(action_tensor, -1., 1.)
        for action in action_tensor.cpu().numpy():
            action_list.append(action)
        # print(action_list)
        # print(env.action_space.low)
        # print(env.action_space.high)
        next_state_list, reward_list, done_list, _ = env.step(action_list)
        # print(action)
        # print(type(state))
        # print(type(reward))
        # print(type(done))
        # print('------------------')
        # exit()
        # print(state.shape)
        # print(type(reward_list[2]))
        state_value_tensor = value_net(state_tensor).detach()
        # print(state_tensor.cpu().numpy().shape)
        # print(action_tensor.cpu().numpy().shape)
        # print(state_value_tensor.cpu().numpy().shape)
        # print('--------------')
        # for i in done_list:
        #     if i == True and total_game_step != 999:
        #         print('wow')
        #         print(total_game_step)
        replay_buffer.store(state=state_tensor.cpu().numpy(), action=action_tensor.cpu().numpy(), reward=reward_list, state_value=state_value_tensor.cpu().numpy(), isTerminate=done_list)
        state_list = next_state_list
        total_game_step += 1
        average_reward += sum(reward_list) / len(reward_list)
        # if done_list[0] == True or done_list[1] == True or done_list[2] == True:
        #     print(done_list)
        # print(total_game_step)
            # break
        # print(reward)
    replay_buffer.update_true_state_value()
    replay_buffer.update_advantage()
    replay_buffer.merge_trajectory()
    actor_loss, critic_loss =  a2c.update(replay_buffer)
    wandb.log({'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(), 'average_reward': average_reward})
    if current_episode % 100 == 0:
        print('-------------------------')
        print('episode: {episode}, actor_loss: {actor_loss}, critic_losss: {critic_losss}' \
            .format(episode=current_episode, actor_loss=actor_loss, critic_losss=critic_loss))
        torch.save(a2c.policy.state_dict(), PATH + '_policy')
        torch.save(a2c.value_net.state_dict(), PATH + '_valueNet')
    print(average_reward)

env.close()