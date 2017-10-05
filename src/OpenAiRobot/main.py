import agent
import tensorflow as tf
import time

learning_rates_grid = [0.1, 0.01, 0.001]
n_hidden_layers_grid = [1, 2]
n_hidden_width_grid = [4, 16, 128]

# NAME THE SESSION
session = 'monday_deletable'

for n_hidden_layers in n_hidden_layers_grid:
    for n_hidden_width in n_hidden_width_grid:
        for learning_rate in learning_rates_grid:

            options = {'n_max_epochs': 3,
                       'n_games_pr_epoch': 10,
                       'discount_rate': 0.95,
                       'show_sim': False,
                       'n_hidden_layers': n_hidden_layers,
                       'n_hidden_width': n_hidden_width,
                       'learning_rate': learning_rate,
                       'max_env_timesteps': 1000,
                       'session_name': session,
                       'performance': False}

            tf.reset_default_graph()
            with agent.Agent(options) as cart_agent:
                #cart_agent = agent.Agent(options)
                try:
                    cart_agent.run_training()
                except KeyboardInterrupt:
                    pass
                cart_agent.plot_and_save_scores_and_model()


print('The end.')

