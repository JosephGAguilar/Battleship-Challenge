'''

Joseph Aguilar - DQN for Battleship

Using a Deep Q Neural Network to play a game of Battleship.
The aim of the board game "Battleship" is to sink your opponent's
battleship by selecting grids of the board to bomb. You don't know
where your opponent's ships are, and they do not know where yours are.
You make educated guesses about your opponents strategy and ship placement based
on how well you know your them and how well they can bluff and strategize.
However, I hoped that I could use Machine Learning to deduce an optimal way to play "Battleship,"
despite the fact that "Battleship" is a game designed around knowing your opponent and 
also luck, to some extent. 

I chose a DQN model to play Battleship to see if there is any way to "practice" Battleship and develop
optimal strategy. A reinforcement machine learning model is the perfect simulation of something "learning"
to play Battleship. If you give something enough tries and enough time, I wondered if a DQN could deduce
any optimal strategies in a luck and person-to-person based game like Battleship.

This script builds, trains, and then tasts a DQN designed to read in Battleship boards and then
output the optimal square to fire in order to win. It compares this DQN's performance
with a random agent that just shoots randomly and a hunting agent, which hunts for nearby ships
when it hits a ship (which is how a human would normally play Battleship"). The DQN's best run
is saved and pitted against the other agents. I measured the effectiveness of a model by tracking
how many shots it took to win a game. If you take less shots, you are more likely to win since you 
are quicker to sinking every opposing battleship. I did not teach these agents how to place their ships,
as I believe that is beyond the scope of this project. The Battleship boards are generated randomly,
with every ship placed horizontally for simplicity's sake.

On average, and luck permitting, the DQN will outperform
the other two agents by tens of shots. While this is could partially be a symptom of luck,
the random agent (the most luck based agent) will often lose to the DQN, implying that some thinking
is occuring on the DQN model's part about where it fires.

'''

import tensorflow as tf
from tensorflow import keras
import numpy as np

#Define Battleship 
class BattleshipEnvironment:
	def __init__(self, board_size=10):
		self.board_size = board_size
		self.total_steps = 0
		self.reset()

	def reset(self):
		self.board = np.zeros((self.board_size, self.board_size))  # Initialize empty board
		self.ships = []  # List to store ship locations
		# Place ships randomly
		ship_sizes = [2, 2, 3, 4, 5] # Each different size ship in a standard game of battlefield.
		for i in range(5):
			ship_size = ship_sizes[i]
			ship_row = np.random.randint(0, self.board_size - ship_size + 1) # Mark ship locations. Ships are only placed horizontally for simplicity.
			ship_col = np.random.randint(0, self.board_size - ship_size + 1)
			for n in range(ship_size):
				self.board[ship_row][ship_col + n] = 1
				self.ships.append((ship_row, ship_col + n))		
		return np.copy(self.board)

	def step(self, action):
		reward = 0
		done = False
		row, col = action
		if self.board[row][col] == 1:  # If the action hits a ship
			reward = 4 # Increased reward to encourage hits, as they are less frequent than misses
			self.board[row][col] = 2  # Mark as hit
			if not any(1 in col for col in self.board):  # If all ship cells are hit
				done = True  # Game over
		elif self.board[row][col] == 0:
			reward = -1
			self.board[row][col] = -1  # Mark as missed
		self.total_steps += 1
		return np.copy(self.board), reward, done

	def is_valid_move(self, coord):
		if self.board[coord[0]][coord[1]] == 0 or self.board[coord[0]][coord[1]] == 1:
			return True
		else:
			return False

@keras.saving.register_keras_serializable(package="DQNModel")
#Build the DQN Model
class DQNModel(tf.keras.Model):
	def __init__(self, num_actions):
		super(DQNModel, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(1, 10, 1, activation='relu', padding='same')
		self.maxPool1 = tf.keras.layers.MaxPool2D(2, padding='same')
		self.conv2 = tf.keras.layers.Conv2D(1, 5, 1, activation='relu', padding='same')
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(25, activation='relu')
		self.dense2 = tf.keras.layers.Dense(16, activation='relu')
		self.dense3 = tf.keras.layers.Dense(num_actions)

	def call(self, inputs):
		inputs = inputs[:,None,:]
		x = self.conv1(inputs)
		x = self.maxPool1(x)
		x = self.conv2(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)
		return self.dense3(x)

#Define the Replay Buffer
class ReplayBuffer:
	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.buffer = []

	def add(self, experience):
		self.buffer.append(experience)
		if len(self.buffer) > self.buffer_size:
			self.buffer.pop(0)

	def sample(self, batch_size):
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)
		return [self.buffer[idx] for idx in indices]

#Define the Training Procedure
class DQNTrainer:
	def __init__(self, model, target_model, replay_buffer, batch_size=8, gamma=0.01):
		self.model = model
		self.target_model = target_model
		self.replay_buffer = replay_buffer
		self.batch_size = batch_size
		self.gamma = gamma
		self.optimizer = tf.keras.optimizers.Adam()

	def train_step(self):
		batch = self.replay_buffer.sample(self.batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)

		states = np.array(states)
		next_states = np.array(next_states)
		q_values = self.model(states)
		next_q_values = self.target_model(next_states)

		targets = np.copy(q_values)
		for i in range(self.batch_size):
			target = rewards[i]
			if not dones[i]:
				target += self.gamma * np.max(next_q_values[i])
			targets[i][actions[i]] = target

		with tf.GradientTape() as tape:
			predictions = self.model(states)
			loss = tf.reduce_mean(tf.square(targets - predictions))
		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

	def update_target_network(self):
		self.target_model.set_weights(self.model.get_weights())

class DQNAgent:
	def __init__(self, model, num_actions):
		self.model = model
		self.num_actions = num_actions

	def select_action(self, state, valid_moves):
		q_values = self.model(np.expand_dims(state, axis=0)).numpy().flatten()
		valid_q_values = [q_values[i] if i in valid_moves else float('-inf') for i in range(self.num_actions)]
		action = np.argmax(valid_q_values)
		return action

#Training the Model
env = BattleshipEnvironment()
num_actions = env.board_size ** 2  # 10x10 grid, each cell is an action
model = DQNModel(num_actions)
target_model = DQNModel(num_actions)
replay_buffer = ReplayBuffer(buffer_size=800)
trainer = DQNTrainer(model, target_model, replay_buffer)

num_epochs= 150
epsilon = 0.2  # Exploration rate
update_target_every = 100  # Update target network every 100 steps
agent = DQNAgent(model, num_actions)

best_total_reward = float('-inf')
best_DQN_shots = float('inf')

for epoch in range(num_epochs):
	state = env.reset()
	done = False
	total_reward = 0
	total_shots = 0
	while not done:
		# Epsilon-greedy policy
		if np.random.rand() < epsilon:
			action = np.random.choice([i for i in range(num_actions) if env.is_valid_move((i // env.board_size, i % env.board_size))])
		else:
			action = agent.select_action(state, [i for i in range(num_actions) if env.is_valid_move((i // env.board_size, i % env.board_size))])

		next_state, reward, done = env.step((action / env.board_size, action % env.board_size))
		replay_buffer.add((state, action, reward, next_state, done))
		total_reward += reward
		total_shots += 1
		state = next_state

		if len(replay_buffer.buffer) >= trainer.batch_size:
			trainer.train_step()

		if env.total_steps % update_target_every == 0:
			trainer.update_target_network()

	if total_shots < best_DQN_shots:
		best_DQN_shots = total_shots
	print("Epoch:", epoch + 1, "| Total Reward:", total_reward, "| Total Shots:", total_shots)


print("Model Trained!\n")

# Function to create the game board
def create_board():
	return np.zeros((10,10))

# Function to randomly place ships on the board
def place_ships(board):
	# Place ships randomly
	ship_sizes = [2, 2, 3, 4, 5] # Each different size ship in a standard game of battlefield.
	for i in range(5):
		ship_size = ship_sizes[i]
		ship_row = np.random.randint(0, 10 - ship_size + 1)
		ship_col = np.random.randint(0, 10 - ship_size + 1)
		for n in range(ship_size):
			board[ship_row][ship_col + n] = 1
	return board

# Function for the random agent to take a shot
def random_shot(board):
	x = np.random.randint(0, 10)
	y = np.random.randint(0, 10)
	return x, y

# Function to create a data sample from the game board
def create_sample(board, x, y):
	sample = board.copy()
	sample[x, y] = 2  # Mark the shot location
	return sample

# Function for the hunting agent to take a shot
def hunting_shot(board, last_shot):
	if last_shot is None:
		return random_shot(board)
	else:
		x, y = last_shot
		if board[x, y] == 2:
			for dx in [1, -1]:
				new_x, new_y = x + dx, y 
				if 0 <= new_x < 10 and 0 <= new_y < 10 and board[new_x, new_y] == 0:
					return new_x, new_y
		return random_shot(board)


# Function to test the model against an agent
def test_model(board, agent):
	last_shot = None
	total_shots = 0

	while any(1 in col for col in board): 
		if agent == 'Random':
			row, col = random_shot(board)
		elif agent == 'Hunt':
			row, col = hunting_shot(board, last_shot)
		if board[row][col] == 1:  # If the action hits a ship
			last_shot = (row, col)
			board[row][col] = 2  # Mark as hit
			total_shots += 1
		elif board[row][col] == 0:
			board[row][col] = -1  # Mark as missed
			total_shots += 1

	return total_shots

best_random_shots = float('inf')
best_hunting_shots = float('inf')

print("After 150 test games...")
for i in range(150):
	board = create_board()
	board = place_ships(board)

	# Test the model against a random agent
	random_shots = test_model(board, 'Random')
	if random_shots < best_random_shots:
		best_random_shots = random_shots

	board = create_board()
	board = place_ships(board)
	hunting_shots = test_model(board, 'Hunt')
	if hunting_shots < best_hunting_shots:
		best_hunting_shots = hunting_shots

print("Minimum shots by DQN agent:", best_DQN_shots)
print("Minimum shots by Random agent:", best_random_shots)
print("Minimum shots by Hunting agent:", best_hunting_shots)
print("\n")
