import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing

#part1 building ai

# brain
class CNN(nn.Module):
    def __init__(self,number_actions):
        super(CNN, self).__init__()
        #self.number_actions = number_actions
        # we'll use 3 conv layers and 2 full connections
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        # in_channels = no of input channels a image has
        # out_channels = no of filters to apply on the image ie 1st layer will give 32 outputs
        # no. of features of detect
        # kernel_size = 5 to detect the large features of the image and in
        # deeper layers this value in decreased to detect smaller featueresq
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 80)
        # full connection 1 feeded with maxpooled and flettend conv2d(3)'s output
        self.fc2 = nn.Linear(in_features = 80, out_features = number_actions)
        # number_nuerons = no of input neurons which is obtained after flattening

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        # making a random init of a image to get a sample which represents a real image
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        #                      (feed x in conv1)
        #         (maxpooling(2d) done on conv1 output , kernel_size = 3 , sturd = 2)
        #   activation function relu
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        # forward prop the conv net
        return x.data.view(1, -1).size(1) # return no. of expected elements after flattening

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1) # flattening layers
        x = F.relu(self.fc1(x)) # feed into fullyconnected nn
        x = self.fc2(x)
        return x
# body
class SoftmaxBody(nn.Module):
    def __init__(self,T):
        super(SoftmaxBody, self).__init__()
        self.T = T
    def forward(self, outputs):
        probs = F.softmax((outputs)*self.T)
        actions = probs.multinomial()
        return actions
# Make ai
class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs,dtype = np.float32)))
        brain_out = self.brain.forward(input)
        actions = self.body.forward(brain_out)
        return actions.data.numpy()

# train deep q conv
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomDefendLine-v0"))),height = 80, width = 80,grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env,"videos",force=True)
number_actions = doom_env.action_space.n

cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

# setting up exp Replay
n_steps = experience_replay.NStepProgress(doom_env,ai,10)
memory = experience_replay.ReplayMemory(n_steps = n_steps,capacity = 10000)

#eligibility retrace ...  n-step Q-Learning

def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state,series[-1].state],dtype = np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = (cumul_reward * gamma) + step.reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs,dtype=np.float32)) , torch.stack(targets)

class MA: # moving average
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size

    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)

        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]

    def average(self):
        return np.mean(self.list_of_rewards)

ma = MA(100)

loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epoch = 100
for e in range(1,nb_epoch+1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs , targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs) , Variable(targets)

        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward =  ma.average()
    print ("Epoch: %s, Average Reward: %s" % (str(e),str(avg_reward)))
    if(avg_reward >= 15):
        print("AI wins")
        break

doom_env.close()
