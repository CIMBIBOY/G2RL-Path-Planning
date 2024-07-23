import numpy as np
from WarehouseEnv import WarehouseEnvironment
from eval import evaluate_performance
from DQN import Agent
from cnn import CNNLSTMModel

if __name__ == '__main__':
    env = WarehouseEnvironment()
    
    # Use a dummy agent or use the q_table for evaluation
    agent = Agent(env, CNNLSTMModel(30, 30, 4, 4))
    # or
    q_table = np.zeros([env.n_states, env.n_actions])

    print(" ---------- Evaluating Performance ----------")
    performance_metrics = evaluate_performance(env, agent, num_episodes=100)
    print(performance_metrics)