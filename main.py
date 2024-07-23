if __name__ == '__main__':
    from WarehouseEnv import WarehouseEnvironment
    from DQN import Agent
    from cnn import CNNLSTMModel
    from model_summary import print_model_summary

    env = WarehouseEnvironment()
    agent = Agent(env, CNNLSTMModel(30,30,4,4))

    batch_size = 3
    num_of_episodes = 420
    # Maximum number of steps
    timesteps_per_episode = 1000
    # Output model summary information
    print_model_summary(agent.q_network, (batch_size, 1, 30, 30, 4), batch_size)

    training_starts = False 
    train_index = 5
    steps = 1

    import numpy as np
    import progressbar

    for e in range(0, num_of_episodes):
        # Reset the enviroment
        _, state = env.reset()
        
        # Initialize variables
        reward = 0
        terminated = False
        
        bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=
                                      [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        for timestep in range(timesteps_per_episode):
            # Run Action
            action = agent.act(state)
            
            # Take action
            next_state, _, reward, terminated = env.step(action) 
            agent.store(state, action, reward, next_state, terminated)
            
            state = next_state

            env.render()
            env.render_forvideo(train_index, steps)
            steps = steps + 1
            
            if terminated:
                agent.alighn_target_model()
                break
                
            if len(agent.expirience_replay) > batch_size:
                agent.retrain(batch_size)
            
            if timestep % 10 == 0:
                bar.update(timestep/10 + 1)
        
        bar.finish()
        if (e + 1) % 1 == 0:
            print("Episode: {}".format(e + 1))
            