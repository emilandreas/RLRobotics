import agent
import tensorflow as tf
import time

##################################################
# NAME THE SESSION
session = 'mountaincar_test'

# Set parameters
env = 'MountainCarContinuous-v0'
# env = 'CartPole-v0'
n_max_epochs = 1500
n_games_pr_epoch = 1
discount_rate_grid = [0.95, 0.95, 0.95, 0.99, 0.99, 0.99]
max_env_timesteps = 1000

learning_rates_grid = [0.01]
n_hidden_layers_grid = [1]
n_hidden_width_grid = [4]
##################################################

#  Run training session
for discount_rate in discount_rate_grid:
    for n_hidden_layers in n_hidden_layers_grid:
        for n_hidden_width in n_hidden_width_grid:
            for learning_rate in learning_rates_grid:

                options = {'env': env,
                           'n_max_epochs': n_max_epochs,
                           'n_games_pr_epoch': n_games_pr_epoch,
                           'discount_rate': discount_rate,
                           'show_sim': False,
                           'n_hidden_layers': n_hidden_layers,
                           'n_hidden_width': n_hidden_width,
                           'learning_rate': learning_rate,
                           'max_env_timesteps': max_env_timesteps,
                           'session_name': session,
                           'performance': False}

                tf.reset_default_graph()
                with agent.Agent(options) as cart_agent:
                    try:

                        cart_agent.run_training()

                    except KeyboardInterrupt:
                        pass
                    cart_agent.plot_and_save_scores_and_model()


print('The end.')

