import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

"""Here we create our NN
It has 3 hidden convolutional layers:
- First layer has the input dimensions of our output of the environment 
- Second layer has dimensons of fc1_dims
- Third layer has dimensos of fc2_dims
- Fourth layer has the dimesnions of our inputs in the environment
"""
class DeepQNetwork(nn.Module):
    # lr stands for learningrate
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions, fc3_dims=64):
        super(DeepQNetwork, self).__init__()
        # fc stands for fully conected layer
        # We define the variables for our NN
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        # we define our layers of our neural network
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc4 = nn.Linear(self.fc2_dims, self.n_actions)
        # now we devine our optimizer, self.parameters() are all the parameters
        # that can be adjusted, we could also say adjust only thes and these
        # layers
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # Here we devine our loss, it ist the means squared error
        self.loss = nn.MSELoss()
        # --------------The folowing two lines do not work in Google Coolab---------
        self.device = T.device('cuda:0' if T.cuda.is_available() else "cpu")
        self.to(self.device)
        # Sadly in google Colabs we have to use the cpu to train our model
        # self.device=T.device("cpu")
        # self.to(self.device)

    # Here we define how our data flows through our layer
    # state is our input data, the input of the neural network
    # when we call this function we input our output from the environment and
    # get the rewards for each action.
    def forward(self, state):
        # relu is a very popular activation function that sets all negative values
        # to zero, the activation function defines when a neuron is firing, and
        # when an neuron is not firing
        state=state.type(T.cuda.FloatTensor)
        fc1_out = F.relu(self.fc1(state))  # output of our first hidden layer
        fc2_out = F.relu(self.fc2(fc1_out))  # output of our second hidden layer
        fc3_out = F.relu(self.fc3(fc2_out))  # output of our third hidden layer
        # In our output function we do not want to use relu, we want a function,
        # that approximates the reward for each action
        actions = self.fc4(fc3_out)  # gives back the reward for each action

        return actions