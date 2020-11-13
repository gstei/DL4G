import numpy as np
from NeurealNetwork import DeepQNetwork
class Agent():
  """
  -gamma gewichted den reward
  -epsilon gives the ratio between exploring and taking a well nown action
  -learning rate to pass into our deep learning network
  -input dims dimebsion of our input
  -batch size, because we are learning form a batch of memories
  -maximal memory size for the memory
  -eps_end with what factor do we want do decrement epsilon each timestep
  """
  def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=1000000, eps_end=0.05, eps_dec=5e-4, nn_size=64):
    #We save all the variables from the input
    self.gamma = gamma
    self.epsilon = epsilon
    self.eps_min = eps_end
    self.eps_dec = eps_dec
    self.lr = lr
    self.nn_size=nn_size
    #integer repesentation of the available actions
    self.action_space = [i for i in range(n_actions)]
    self.mem_size = max_mem_size
    self.batch_size = batch_size
    #memory counter to ceep track of the first availabe memory
    self.mem_cntr = 0
    self.iter_cntr = 0
    self.replace_target = 500

    # We neeed an evaluation network as can be seen in the image above
    # The evaluation model gets trained every step
    #normaly fc1=256 and fc2=256
    self.Q_eval = NeurealNetwork.DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                fc1_dims=nn_size, fc2_dims=nn_size, fc3_dims=nn_size)
    # The Q_next model is what we predict
    self.Q_next = NeurealNetwork.DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                fc1_dims=nn_size, fc2_dims=nn_size, fc3_dims=nn_size)
    #Is like a list (named array)
    self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
    #We have a memory that ceeps track of the new states the agent accounter
    self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
    #we also have a memory of the staty
    self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
    self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
    #the terminal state is always zero, when we finish the game, the thermi
    #nal state is one and the game is over
    self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    #Function to fill our memory
  def store_transition(self, state, action, reward, state_, terminal):
    #What is the position of the first unoccupied memory
    #The modulo oberater makes sure that we wrap arround, when the storage
    #is full we bgegin again from the beginning
    index = self.mem_cntr % self.mem_size
    #Now when we now where to store we store the data
    self.state_memory[index] = state
    self.new_state_memory[index] = state_
    self.reward_memory[index] = reward
    self.action_memory[index] = action
    self.terminal_memory[index] = terminal
    #We increase the memory counter by one, so that we don't overwrite
    #it the next time
    self.mem_cntr += 1
    # the agent needs a function to choose an action
  def choose_action(self, observation):
    #We want to take an random action if the value is greater than epsilon
    if np.random.random() > self.epsilon:
        #We convert our observation space into an tensor an put the tensor on our gpu
        #where our neural network q_eval ist
        state = T.tensor([observation]).to(self.Q_eval.device)
        #we give the stat in our neural network
        actions = self.Q_eval.forward(state)
        #print('--------------actions:-----------------',actions)
        #We give back the action
        action = T.argmax(actions).item()
    else:
        action = np.random.choice(self.action_space)
        #print('--------------actions:-----------------',action)

    return action
    """
    One possibility is to first let the agent play randomly 
    and fill up the meomeory
    The other possiblity is to start learning as soon as the batch size is
    reached
    """
  def learn(self):
      #if we havent reached the batch size we can't learn anything
      if self.mem_cntr < self.batch_size:
          return

      #first we zero our the gradient on our optimizer
      self.Q_eval.optimizer.zero_grad()
      #we only want to learn up to the filled memory
      max_mem = min(self.mem_cntr, self.mem_size)
      #we choose some random data from the batch
      #False because we want to select one element only once, take it out
      #of the pool
      batch = np.random.choice(max_mem, self.batch_size, replace=False)
      #we need something for bookkeeping
      batch_index = np.arange(self.batch_size, dtype=np.int32)
      #We convert the nuppy array into an pytorch tensor
      #we store the free variables in our tuble in our new variables
      state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
      #The same we do for our new state
      new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
      #we also convert the nuppy array into the pytorch tensor in the next to
      #lines
      #for the action batch we don't need an tensor
      action_batch = self.action_memory[batch]
      #we also convert the nuppy array into the pytorch tensor in the next to
      #lines
      reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
      terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
      #Now we want to train our agent towards the maximum
      #We only want the specific output of our neural network [batch_index, action_batch]
      #The NN would output all action but we only want one
      q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
      #The above funcion returns the reward for the specific state and action
      #Here we do not need to do an dereferencing, because we want to take after that the max action
      q_next = self.Q_next.forward(new_state_batch)
      #-------------------commentet this out------------------------------------
      #print('before:',q_next[terminal_batch])
      #q_next[terminal_batch] = 0.0
      #print('after:',q_next[terminal_batch])
      #Now we implement our formula
      #[0] because the max function returns the value, as well as the index
      q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]

      loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
      #We backpropagate it
      loss.backward()
      self.Q_eval.optimizer.step()

      self.iter_cntr += 1
      #we decrease our epsilon ervery step
      self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                      else self.eps_min

      if self.iter_cntr % self.replace_target == 0:
         self.Q_next.load_state_dict(self.Q_eval.state_dict())
      return loss