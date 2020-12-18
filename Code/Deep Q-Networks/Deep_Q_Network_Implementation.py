#!/usr/bin/env
# coding: utf-8
import collections
import random

import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
import os
import matplotlib.pyplot as plt

class QNetwork():

	# This class defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, environment_name):

		print('Initializing QNetwork for %s ...' % environment_name)
		print('Making the Neural Network Model ...')

		self.environment_name = environment_name
		if self.environment_name == 'CartPole-v0':
			self.input_layer_size = 4
			self.output_layer_size = 2
			self.hidden_layer_size = 30
			self.learning_rate = 0.001
			self.gamma = 1.0

			model = keras.Sequential()
			model.add(keras.layers.Dense(self.hidden_layer_size, activation='tanh', input_shape=(self.input_layer_size,)))
			model.add(keras.layers.Dense(self.output_layer_size, activation='linear'))

		else:  # environment_name == 'MountainCar-v0'
			self.input_layer_size = 2
			self.output_layer_size = 3
			self.hidden_layer_1_size = 96
			self.hidden_layer_2_size = 96
			self.hidden_layer_3_size = 64
			self.learning_rate = 0.0001
			self.gamma = 0.99

			model = keras.Sequential()
			model.add(keras.layers.Dense(self.hidden_layer_1_size, activation='relu', input_shape=(self.input_layer_size,)))
			model.add(keras.layers.Dense(self.hidden_layer_2_size, activation='relu'))
			model.add(keras.layers.Dense(self.hidden_layer_3_size, activation='tanh'))
			model.add(keras.layers.Dense(self.output_layer_size, activation='linear'))

		model.compile(loss='mean_squared_error', optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate), metrics=['accuracy'])

		# Parameters
		self.model = model
		print('Neural Network Initialization Completed.')
		print('QNetwork Initialization Completed.')

	def save_model(self, suffix):
		# Helper function to save model / weights.
		# print('Saving Model as: %s' % 'QNetwork_model_' + suffix + '.h5')
		self.model.save('QNetwork_model_'+suffix+'.h5')

	def save_model_weights(self, suffix):
		# Helper function to save model / weights.
		# print('Saving Model Weights as: %s' % 'QNetwork_weights_'+suffix+'.h5')
		self.model.save_weights('QNetwork_weights_'+suffix+'.h5')

	def load_model(self, model_file):
		# Helper function to load an existing model.
		# print('Loading Model as: %s' % model_file)
		self.model = keras.models.load_model(model_file)

	def load_model_weights(self, weight_file):
		# Helper function to load model weights.
		# print('Loading Model Weights as: %s' % weight_file)
		self.model.load_weights(weight_file)


class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):
		# The memory stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
		
		print('Initializing Replay Memory (mem_size=%d , burn_in=%d) ...' % (memory_size, burn_in))
		self.memory_size = memory_size
		self.burn_in = burn_in  # Only burn in once in training
		self.replay_buffer = collections.deque(maxlen=self.memory_size)
		print('Replay Memory Initialization Completed.')

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
		sampled_transitions_batch = []
		for i in range(batch_size):
			sampled_transitions_batch.append(self.replay_buffer[random.randrange(0, len(self.replay_buffer))])

		return np.asarray(sampled_transitions_batch)  # keras requires Numpy array

	def append(self, transition):
		# Appends transition to the memory.
		self.replay_buffer.append(transition)


class DQN_Agent():

	# Functionalities of this class:
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy.
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.

	def __init__(self, environment_name, render=False):
		# Create an instance of the network itself, as well as the memory.

		print('Initializing QAgent for %s ...' % environment_name)

		self.environment_name = environment_name
		self.render = render
		self.env = gym.make(self.environment_name)
		self.env_test = gym.make(self.environment_name)

		if self.environment_name == 'CartPole-v0':
			self.action_space_size = 2
		else:  # environment_name == 'MountainCar-v0'
			self.action_space_size = 3

		self.QNN = QNetwork(self.environment_name)

		self.QNN_target = QNetwork(self.environment_name)
		self.QNN_target.model.set_weights(self.QNN.model.get_weights())

		self.replay_memory = Replay_Memory()
		self.STATE_IDX = 0
		self.ACTION_IDX = 1
		self.REWARD_IDX = 2
		self.NEXT_STATE_IDX = 3
		self.IS_DONE_IDX = 4

		self.epsilon_start = 0.5
		self.epsilon_end = 0.05
		self.epsilon_decay_iterations = 100000
		self.epsilon_change = (self.epsilon_end - self.epsilon_start)/(self.epsilon_decay_iterations-1)
		self.epsilon_current = self.epsilon_start  # Initialized Current

		self.train_counter = 0
		self.batch_size = 32

		if self.environment_name == 'CartPole-v0':
			self.num_episodes = 2000
			self.train_period = 1
			self.train_target_period = 200
		else:  # environment_name == 'MountainCar-v0'
			self.num_episodes = 2000
			self.train_period = 1
			self.train_target_period = 200

		self.eval_episode_period = 100
		self.eval_episodes_length = 100

		self.loss_thru_training = []
		self.total_reward_test = []

		# CPU vs. GPU
		print('QAgent Initialization Completed.')

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.
		do_explore = np.random.choice(2, p=[1-self.epsilon_current, self.epsilon_current])
		# Returning Action
		if do_explore:
			return np.random.randint(self.action_space_size)
		else:
			return np.argmax(q_values)

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time.
		return np.argmax(q_values)

	def train(self, num_episodes=1000):
		# Train the network.
		# If training without experience replay_memory, then interact with the environment
		# while also updating the network parameters.

		self.burn_in_memory()

		for episode in range(num_episodes):
			reward_net = 0

			state = self.env.reset()

			# Select an action using the DQA
			is_done = False

			time_step = 0
			while not is_done:

				# epsilon action
				action = self.epsilon_greedy_policy(self.QNN.model.predict(np.array([state]), batch_size=1))
				next_state, reward, is_done, info = self.env.step(action)

				reward_clipped = np.clip(reward, -1, 1)

				self.replay_memory.replay_buffer.append((state, action, reward_clipped, next_state, is_done))

				if time_step % self.train_period == 0:

					self.train_counter += 1

					# print('Episode: %d' % episode)
					# print('Train Count of Model: '+str(self.train_counter))
					sampled_transitions_batch = self.replay_memory.sample_batch(batch_size=self.batch_size)  # (St, A, R, St+1, is_done)

					states = []
					q_vals = []
					for data in sampled_transitions_batch:
						states.append(data[self.STATE_IDX])  # list of list

						q_next = self.QNN.model.predict(np.array([data[self.STATE_IDX]]), batch_size=1)[0]
						q_pred = q_next[data[self.ACTION_IDX]]
						# Might need to make it list of list
						q_next[data[self.ACTION_IDX]] = data[self.REWARD_IDX]
						if not data[self.IS_DONE_IDX]:
							q_next[data[self.ACTION_IDX]] += \
								self.QNN_target.gamma \
								* np.max(self.QNN_target.model.predict(np.array([data[self.NEXT_STATE_IDX]]), batch_size=1)[0])

						q_vals.append(q_next)

						history = self.QNN.model.fit(np.array(states),
													 np.array(q_vals),
													 epochs=1, batch_size=self.batch_size, verbose=0)

						# self.QNN.save_model_weights(str(self.train_counter))
						self.loss_thru_training.append(q_next[data[self.ACTION_IDX]]-q_pred)
						# self.loss_thru_training.append(history.history['loss'][-1])

						# print('Loss %.4f ' % history.history['loss'][-1])
						# print('Acc %.4f \n' % history.history['acc'][-1])

					if self.train_counter % self.train_target_period == 0:
						self.QNN_target.model.set_weights(self.QNN.model.get_weights())

				self.epsilon_current += self.epsilon_change  # epsilon_change is negative
				self.epsilon_current = np.clip(self.epsilon_current, a_max=self.epsilon_start, a_min=self.epsilon_end)

				reward_net += reward
				state = next_state

				if is_done:
					break

				time_step += 1

			# end_while

			print('Episode in progress: %d \t - \t Reward: %.2f' % (episode, reward_net))

			if episode % self.eval_episode_period == 0:
				self.QNN.save_model_weights(str(episode))
				self.QNN.save_model(str(episode))
				test_reward_curr = self.test()
				print('Test-Reward at Episode %d: \t %.2f' % (episode, float(test_reward_curr)))
				self.total_reward_test.append(test_reward_curr)
				# Saving for convenience every 5 times
				if episode % (5 * self.eval_episode_period) == 0:
					np.save('TD_Loss_CartPole_double_tmp', np.array(self.loss_thru_training))
					np.save('Total_Test_Reward_CartPole_double_tmp', np.array(self.total_reward_test))
		# end_for

	def test(self, model_file=None):
		# Evaluate the performance of agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Interact with the environment, irrespective of whether using a memory.
		if model_file:
			self.QNN.load_model_weights(model_file)

		rewards_total = []
		for episode in range(self.eval_episodes_length):
			reward_net = 0
			state = self.env_test.reset()
			is_done = False
			time_step = 0
			while not is_done:
				# epsilon action
				action = self.greedy_policy(self.QNN.model.predict(np.array([state]), batch_size=1))
				next_state, reward, is_done, info = self.env_test.step(action)
				reward_net += reward
				if is_done:
					break
				state = next_state
			# end_while
			rewards_total.append(reward_net)
		# end_for

		return np.mean(rewards_total)

	def burn_in_memory(self):
		# Initialize replay memory with a burn_in number of episodes / transitions.
		print('Starting Burn-In ...')
		state = self.env.reset()
		for transition_idx in range(self.replay_memory.burn_in):
			# action = np.random.randint(self.env.action_space.shape[0])
			action = np.random.randint(self.action_space_size)
			next_state, reward, is_done, info = self.env.step(action)
			transition = (state, action, reward, next_state, is_done)
			self.replay_memory.append(transition)
			if is_done:
				state = self.env.reset()
			else:
				state = next_state
		print('Burn-In Completed')




def test_video(agent, env, epi):
	save_path = "./videos-%s-%s" % (env, epi)
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	# To create video
	env = gym.wrappers.Monitor(agent.env, save_path, force=True)
	reward_total = []
	state = env.reset()
	done = False
	while not done:
		env.render()
		action = agent.epsilon_greedy_policy(state, 0.05)
		next_state, reward, done, info = env.step(action)
		state = next_state
		reward_total.append(reward)
	print("reward_total: {}".format(np.sum(reward_total)))
	agent.env.close()


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env', dest='env', type=str)
	parser.add_argument('--render', dest='render', type=int, default=0)
	parser.add_argument('--train', dest='train', type=int, default=1)
	parser.add_argument('--model', dest='model_file', type=str)
	return parser.parse_args()


def main(args):
	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory.
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session.
	keras.backend.tensorflow_backend.set_session(sess)

	# Create an instance of the DQN_Agent class here, and then train / test it.
	agent = DQN_Agent(environment_name)

	if args.train:
		agent.train(num_episodes=10000)

		np.save('Total_Test_Reward_double', np.array(agent.total_reward_test))
		np.save('TD_Loss_double', np.array(agent.loss_thru_training))

		plt.figure(1)
		plt.plot(agent.total_reward_test)
		plt.title('Total Test Reward Throughout Training')
		plt.xlabel('Iterations')
		plt.ylabel('Average Test Reward')
		plt.show()

		plt.figure(2)
		plt.plot(agent.loss_thru_training)
		plt.title('TD-Loss Throughout Training')
		plt.xlabel('Iterations')
		plt.ylabel('TD Loss')
		plt.show()


if __name__ == '__main__':
	main(sys.argv)
