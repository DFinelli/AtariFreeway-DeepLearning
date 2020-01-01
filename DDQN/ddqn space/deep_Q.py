import gym
import numpy as np
import random
import keras
import cv2
from replay_buffer import ReplayBuffer
from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense

# List of hyper-parameters and constants

# GAMMA, discount rate ???
DECAY_RATE = 0.99
BUFFER_SIZE = 40000
MINIBATCH_SIZE = 64
TOT_FRAME = 3000000
EPSILON_DECAY = 1000000
MIN_OBSERVATION = 5000
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 0.1
NUM_ACTIONS = 3
# learning rate ALPHA ??
TAU = 0.01
# Number of frames to throw into network
NUM_FRAMES = 3


class DeepQ(object):
    """Constructs the desired deep q learning network"""
    def __init__(self):
        self.construct_q_network()

    def construct_q_network(self):
        # Uses the network architecture found in DeepMind paper
        self.model = Sequential()
        # self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(NUM_FRAMES, 84, 84)))
        self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(NUM_ACTIONS))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.00001))

        # Creates a target network as described in DeepMind paper
        self.target_model = Sequential()
        self.target_model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 3, 3))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Flatten())
        self.target_model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.target_model.add(Dense(NUM_ACTIONS))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.00001))
        self.target_model.set_weights(self.model.get_weights())

        #print("model.get_weights", self.model.get_weights())
        # len = 10
        # len[0] = 8

        print("Successfully constructed networks.")

    def predict_movement(self, data, epsilon):
        """Predict movement of game controller where is epsilon
        probability randomly move."""
        q_actions = self.model.predict(data.reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
        #print("len q_actions=", len(q_actions[0]))
        #print("q_actions=", q_actions)
        QQ_actions = np.array([[q_actions[0][0], q_actions[0][1], q_actions[0][2]]])
        opt_policy = np.argmax(QQ_actions)
        print("QQ_actions =", QQ_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            #print("rand < epsilon", rand_val, epsilon)
            opt_policy = np.random.randint(0, NUM_ACTIONS)
        return opt_policy, QQ_actions[0, opt_policy]

    #        """Trains network to fit given parameters"""
    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):

        # batches are sample from the experience replay

        # The shape attribute for numpy arrays returns the dimensions of the array. If Y has n rows and m columns,
        # then Y.shape is (n,m). So Y.shape[0] is n.
        batch_size = s_batch.shape[0]

        # making a mxn matrix       m (rows) = all the experiences
        #                           n (cols) = action space
        # np.zeros: Return a new array of given shape (mxn), filled with zeros.
        #                 n, u, d
        #   i=0     [   [ 0, 0, 0]
        #   .          [ 0, 0, 0]
        #   .          [ 0, 0, 0]
        #   .         [0,  0, 0]
        # i=batchsize [0, 0, 0]
        targets = np.zeros((batch_size, NUM_ACTIONS))

        # for all experiences' associated q values for each action
        for i in range(batch_size):
            # a[i] is indexing some experience (s,a,d,r,s') related concept

            # CHOOSE ACTION FROM THE MODEL
            # for the s given, model predicts and returns (pt_policy, QQ_actions)
            # q values for each action and saves them as the targets
            targets[i] = self.model.predict(s_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)

            # EVALUATE ACTION FROM TARGET MODEL
            # for the s' given, model predicts and returns returns (pt_policy, QQ_actions)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)

            # example: 10th experience = (s=RGB10, a=1, r=1, d=0, s'= RGB11)
            # action[10] = 1  aka up
            # reward[10] = 1 aka crossed road
            #
            # targets[10, action[10]] = reward[10]
            # targets[10, 1] = 1
            #
            #                 n, u, d
            #   i=0     [    [ 0, 0, 0]
            #   .           [ 0, 0, 0]
            #   i=10       [ 0, 1, 0]       # targets[10, 1] = 1
            #   .          [0, 0, 0]
            # i=batchsize [0, 0, 0]
            #
            targets[i, a_batch[i]] = r_batch[i]

            # THIS IS UPDATING MODEL WEIGHTS
            # if experience didn't end the game
            if d_batch[i] == False:
                # reward + discounted q val of future state
                targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)

            #                 n, u, d
            #   i=0     [    [ 0, 0, 0]
            #   .           [ 0, 0, 0]      #               reward + discounted future Q val
            #   i=10       [ 0, 1, 0]       # targets[10, 1] = 1 + DECAY_RATE * fut_action
            #   .          [0, 0, 0]
            # i=batchsize [0, 0, 0]

        # so the model runs s_batch through and gets an associated qval for each action
        # then calculates the avg loss amount all neurons
        loss = self.model.train_on_batch(s_batch, targets)

        # Print the loss every 10 iterations.
        #if observation_num % 10 == 0:
            #print("We had a loss equal to ", loss)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model = load_model(path)
        print("Succesfully loaded network.")

    # THIS IS UPDATING TARGET MODEL WEIGHTS
    # this is done subsequently after model weights have been updated
    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        # update target model weights by applying a learning rate * model weight + exist targetmodel weight
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

if __name__ == "__main__":
    from space_invaders import SpaceInvader
    print("Haven't finished implementing yet...'")
    space_invader = SpaceInvader()
    space_invader.load_network("saved.h5")
    # print space_invader.calculate_mean()
    # space_invader.simulate("deep_q_video", True)
    space_invader.train(TOT_FRAME)
