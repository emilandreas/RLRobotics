import agent

env = 'ContinuousCartPole-v0'
# env = 'MountainCarContinuous-v0'
# env = 'CartPole-v0'
options = {'env': env,
           'n_max_epochs': 3000,
           'n_games_pr_epoch': 10,
           'discount_rate': 0.95,
           'show_sim': True,
           'n_hidden_layers': 1,
           'n_hidden_width': 1,
           'learning_rate': 1,
           'max_env_timesteps': 1000,
           'session_name': 'performance',
           'dropout': False,
           'performance': True}

with agent.Agent(options) as cart_agent:
    #cart_agent = agent.Agent(options)
    try:
        path = 'training_log/ContinuousCartRunWstddiv/2017-11-02_10_13_30'
        # path = 'training_log/ContinuousCartRun/2017-10-30_15_14_52'
        # path = 'training_log/over_weekend_1/2017-09-15_20_33_01'
        # path = 'training_log/testytest/2017-10-23_14_12_13'
        meta_path = path + '/' + 'result_model.meta'
        cart_agent.policy.restore_model(meta_path, path)
        cart_agent.run_performance()
    except KeyboardInterrupt:
        pass
    cart_agent.plot_and_save_scores_and_model()

