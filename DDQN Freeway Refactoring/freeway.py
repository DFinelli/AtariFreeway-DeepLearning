import gym
import cv2
from replay_buffer import ReplayBuffer
import numpy as np
from duel_Q import DuelQ
from deep_Q import DeepQ

# List of hyper-parameters and constants
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 32
TOT_FRAME = 1000000
EPSILON_DECAY = 300000
MIN_OBSERVATION = 5000
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 1.0
# Number of frames to throw into network
NUM_FRAMES = 3

class Freeway(object):

    def __init__(self, mode):
        #self.env = gym.make('SpaceInvaders-v0')
        self.env = gym.make('Freeway-v0')
        self.env.reset()
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # Construct appropriate network based on flags
        if mode == "DDQN":
            self.deep_q = DeepQ()
        elif mode == "DQN":
            self.deep_q = DuelQ()

        # PROCESS BUFFER
        # A buffer that keeps the last 3 images -- SAVING 3 FRAMES as 1 STATE
        self.process_buffer = []
        # Initialize buffer with the first frame
        s1, r1, _, _ = self.env.step(0)
        s2, r2, _, _ = self.env.step(0)
        s3, r3, _, _ = self.env.step(0)
        self.process_buffer = [s1, s2, s3]

    def load_network(self, path):
        self.deep_q.load_network(path)

    def convert_process_buffer(self):
        """Converts the list of NUM_FRAMES images in the process buffer
        into one training sample"""
        black_buffer = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (84, 90)) for x in self.process_buffer]
        black_buffer = [x[1:85, :, np.newaxis] for x in black_buffer]
        return np.concatenate(black_buffer, axis=2)


    def train(self, num_frames):
        #initial num_frames to 1,000,000

        observation_num = 0

        # the curr state (= 3sequences of states) living in the self.process_buffer is turned into 1 training sample
        curr_state = self.convert_process_buffer()

        epsilon = INITIAL_EPSILON
        alive_frame = 0
        total_reward = 0

        #num_frames set to = 1,000,000 originally
        while observation_num < num_frames:
            if observation_num % 1000 == 999:
                print(("Executing loop %d" %observation_num))

            # Slowly decay the learning rate
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

            # the initial state (= 3sequences of states) living in self.process_buffer is turned into 1 training sample
            # we save this because we will need the OG s after taking action (for experience buffer s,a,d,r,s')
            initial_state = self.convert_process_buffer()

            # reset process_buffer because we will add the subsequent state after taking predicted action
            self.process_buffer = []



            # input the curr_state into the "Q-Network Model" and generate Q values for each action
            predict_movement, predict_q_value = self.deep_q.predict_movement(curr_state, epsilon)

            #dan only OG line is-> predict_movement, predict_q_value = self.deep_q.predict_movement(curr_state, epsilon)
            #if observation_num < 3000:
                #predict_movement = 1

            if observation_num % 1000 == 999:
                print("QQ_actions", predict_q_value)
                print("the_actions =", predict_movement)


            reward, done = 0, False
            # all caps NUM_FRAMES = 3
            for i in range(NUM_FRAMES):
                temp_observation, temp_reward, temp_done, _ = self.env.step(predict_movement)
                reward += temp_reward
                # add this new temp state observation to process_buffer so it can be processed soon
                self.process_buffer.append(temp_observation)
                done = done | temp_done

            #observation number is for total training frames
            #if observation_num % 10 == 0:
                #print("We predicted a q value of ", predict_q_value)

            # if game ended, reset env and round specific counters
            if done:
                #print("Lived with maximum time ", alive_frame)
                #print("Earned a total of reward equal to ", total_reward)
                self.env.reset()
                alive_frame = 0
                total_reward = 0

            # process the new state after taking predicted action (already added the state earlier to process_buffer)
            new_state = self.convert_process_buffer()

            # now add this s,a,r,d,s' experience to replay_buffer
            self.replay_buffer.add(initial_state, predict_movement, reward, done, new_state)

            total_reward += reward

            # once reached the quota for minimum number of observations (OG MIN_OBSERVATION = 5000)
            if self.replay_buffer.size() > MIN_OBSERVATION:
                # sample from replay_buffers
                s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(MINIBATCH_SIZE)
                # this trains the Q Network model
                self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
                # this trains the Target Q Network model?
                self.deep_q.target_train()

            # Save the network every 100000 iterations
            #if observation_num % 10000 == 9999:
            if observation_num % 1000 == 999:
                print("Saving Network")
                self.deep_q.save_network("saved.h5")

            alive_frame += 1
            observation_num += 1
    #
    def simulate(self, path = "", save = False):
        """Simulates game"""
        done = False
        tot_award = 0
        if save:
            self.env.monitor.start(path, force=True)
        self.env.reset()
        self.env.render()
        while not done:
            state = self.convert_process_buffer()
            # predict_movement() .... returns --> tuple-> opt_policy, q_actions[0, opt_policy]
            predict_movement = self.deep_q.predict_movement(state, 0)[0]
            #print("predict_movement=", predict_movement)
            self.env.render()
            observation, reward, done, _ = self.env.step(predict_movement)
            tot_award += reward
            self.process_buffer.append(observation)
            self.process_buffer = self.process_buffer[1:]
        if save:
            self.env.monitor.close()

    def calculate_mean(self, num_samples = 100):
        reward_list = []
        print("Printing scores of each trial")
        for i in range(num_samples):
            done = False
            tot_award = 0
            self.env.reset()
            while not done:
                state = self.convert_process_buffer()
                predict_movement = self.deep_q.predict_movement(state, 0.0)[0]
                observation, reward, done, _ = self.env.step(predict_movement)
                tot_award += reward
                self.process_buffer.append(observation)
                self.process_buffer = self.process_buffer[1:]
            print(tot_award)
            reward_list.append(tot_award)
        return np.mean(reward_list), np.std(reward_list)

