import numpy as np
import gym
import ContinuousCartPole
import network
import matplotlib.pyplot as plt
from datetime import datetime
import os

class Agent:
    def __init__(self, options):

        # Set variables from options

        if(options['performance']):
            print('Performance run')
        else:
            #create new folder for saving data
            print('Training run')
            self.session_name = options['session_name']
            self.folder_path = self.__create_folder()

        self.env_name = options['env']
        self.env = gym.make(self.env_name)

        # Check which type of action space from
        if self.env.action_space.__class__ == gym.spaces.discrete.Discrete:
            options['discrete_actions'] = True
            self.discrete_actions = True
            options['n_outputs'] = self.env.action_space.n
        else:
            options['discrete_actions'] = False
            self.discrete_actions = False
            options['n_outputs'] = self.env.action_space.shape[0]

        options['n_inputs'] = self.env.observation_space.shape[0]

        self.n_epochs = options['n_max_epochs']
        n_last_epochs_visual = 0    # Display the last n epochs of the training
        self.visual_epoch_limit = self.n_epochs - n_last_epochs_visual
        self.discount_rate = options['discount_rate']
        self.n_games_pr_epoch = options['n_games_pr_epoch']
        
        self.policy = network.PolicyGradientModel(options)
        self.score_log = np.array([])
        self.show_sim = options['show_sim']
        self.max_env_timesteps = options['max_env_timesteps']

    def __enter__(self):
        return self

    def run_training(self):
        self.env._max_episode_steps = self.max_env_timesteps
        stddiv = 0.8*2
        for epoch in range(self.n_epochs):
            all_rewards = []
            all_gradients = []
            self.print_satus(epoch)
            temp_score = 0

            for game in range(self.n_games_pr_epoch):
                current_rewards = []
                current_gradients = []
                done = False
                obs = self.env.reset()

                while not done:
                    self.__render_env(epoch, self.env)  # Renders only when above limit or show_sim = True

                    if self.discrete_actions:
                        action, gradients = self.policy.run_model(obs)
                        obs, reward, done, _ = self.env.step(action)
                    else:
                        action, gradients = self.policy.run_model(obs, np.array([stddiv]))
                        obs, reward, done, _ = self.env.step(np.array([action]))
                    # if reward > 0:
                    #     print("You car made it!!!!!")

                    current_rewards.append(reward)
                    current_gradients.append(gradients)
                    if game == self.n_games_pr_epoch - 1:
                        print("Action: {}".format(action))

                rewardSum = sum(current_rewards)
                temp_score += rewardSum
                if self.env_name != 'MountainCarContinuous-v0' or reward > 0:
                    all_rewards.append(current_rewards)
                    all_gradients.append(current_gradients)

            if epoch % 100 == 0:
                stddiv *= 0.5
            # stddiv *= 0.996
            print("Stddiv: {}".format(stddiv))
            mean_epoch_score = temp_score/float(self.n_games_pr_epoch)
            print("Score: {}".format(mean_epoch_score))
            self.score_log = np.append(self.score_log, mean_epoch_score)
            # if not all_rewards:
            #     continue
            all_rewards = self.__discount_and_normalize_rewards(all_rewards, self.discount_rate)
            feed_dict = self.__compute_mean_gradients(all_gradients, all_rewards)
            self.policy.fit_model(feed_dict)


    def _noise(self,  epoch):
        self.sigma*0.9977
        return np.random.normal(0, np.sqrt(self.sigma))


    def run_performance(self):
        self.env._max_episode_steps = self.max_env_timesteps
        for epoch in range(self.n_epochs):
            current_rewards = []
            score = 0
            done = False
            obs = self.env.reset()
            frame_count = 0
            while not done:
                self.__render_env(epoch, self.env)  # Renders only when above limit or show_sim = True
                if self.discrete_actions:
                    action = self.policy.run_model_performance(obs)
                    obs, reward, done, _ = self.env.step(action)
                else:
                    action = self.policy.run_model_performance(obs, np.array([0]))
                    obs, reward, done, _ = self.env.step(np.array([action]))
                print("frame_count: {}, done: {}".format(frame_count, done))
                frame_count += 1
                current_rewards.append(reward)

            score += sum(current_rewards)
            print(score)

    def plot_and_save_scores_and_model(self):
        plt.ioff()
        plt.figure(figsize=(12, 6))
        plt.plot(self.score_log)
        info = 'epochs:{}, games_pr_epoch:{} discount:{}, learning rate:{},layers:[d:{},w:{}] {}'.format(self.n_epochs,
                                                                           self.n_games_pr_epoch,
                                                                           self.discount_rate,
                                                                           self.policy.learning_rate,
                                                                           self.policy.n_hidden_layers,
                                                                           self.policy.n_hidden_width,
                                                                           datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
        plt.title('Rewards')
        plt.annotate(info, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top')
        plt.xlabel('epoch [{} games]'.format(self.n_games_pr_epoch))
        plt.ylabel('reward')

        plt.savefig(self.folder_path + r'/result_plot')
        self.policy.save_model(self.folder_path + r'/result_model')
        plt.clf()

    def print_satus(self, epoch):
        print("Epoch: {} / {}\t[ lr: {}, layers:{}, width:{} ]".format(epoch, self.n_epochs,
                                                                    self.policy.learning_rate,
                                                                    self.policy.n_hidden_layers,
                                                                    self.policy.n_hidden_width))
    def __render_env(self, epoch, env):
        if epoch > self.visual_epoch_limit or self.show_sim:
            env.render()

    # Taken from "Hands on machine learning"
    def __discount_rewards(self, rewards, discount_rate):
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards

    # Taken from "Hands on machine learning"
    def __discount_and_normalize_rewards(self, all_rewards, discount_rate):
        all_discounted_rewards = [self.__discount_rewards(rewards, discount_rate)
                                  for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean)/reward_std
                for discounted_rewards in all_discounted_rewards]

    def __compute_mean_gradients(self, all_gradients, all_rewards):
        feed_dict = {}

        for var_index, grad_placeholder in enumerate(self.policy.gradient_placeholders):

            temp_all_gradients = [reward * all_gradients[game_index][step][var_index]
                                for game_index, rewards in enumerate(all_rewards)
                                for step, reward in enumerate(rewards)
                                if reward > 0]  # Only keep the gradients yielding the best rewards, > 0 since normalized
            mean_gradients = np.mean(temp_all_gradients, axis=0)
            feed_dict[grad_placeholder] = mean_gradients
        return feed_dict

    def __create_folder(self):
        folder_path = r'./training_log/{}/{}'.format(self.session_name, datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path

    def __exit__(self, *a):
        self.policy.close();
        print("In exit.")