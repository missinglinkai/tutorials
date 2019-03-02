import argparse
import gym
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

import missinglink

OWNER_ID = 'your_owner_id'
PROJECT_TOKEN = 'your_project_token'

missinglink_project = missinglink.PyTorchProject(owner_id=OWNER_ID, project_token=PROJECT_TOKEN)


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=1), state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = model(Variable(state))
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.data[0]


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0, 0]
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    loss = torch.cat(policy_losses).sum() + torch.cat(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10
    with missinglink_project.create_experiment(
        model,
        display_name='Actor Critic PyTorch',
        optimizer=optimizer,
        metrics={
            'Length': lambda time: time,
            'Average Length': lambda time: running_reward * 0.99 + time * 0.01
        }
    ) as experiment:
        wrapped_t = experiment.metrics['Length']
        wrapped_running_reward = experiment.metrics['Average Length']
        for i_episode in experiment.loop(condition=lambda _: running_reward <= env.spec.reward_threshold):
            state = env.reset()
            for t in range(10000):  # Don't infinite loop while learning
                action = select_action(state)
                state, reward, done, _ = env.step(action)
                if args.render:
                    env.render()
                model.rewards.append(reward)
                if done:
                    break

            wrapped_t(t)
            running_reward = wrapped_running_reward(t)
            finish_episode()
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                    i_episode, t, running_reward))

        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))


if __name__ == '__main__':
    main()
