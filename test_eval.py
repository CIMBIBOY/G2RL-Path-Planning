import numpy as np
from WarehouseEnv import WarehouseEnvironment
from eval import evaluate_performance
from DQN import Agent
from cnn import CNNLSTMModel
from maskPPO import MaskPPOAgent
import torch

def get_dimensions(nested_list):
    if isinstance(nested_list, list):
        return len(nested_list), len(nested_list[0]) if nested_list else 0
    elif isinstance(nested_list, np.ndarray):
        return nested_list.shape
    else:
        raise ValueError("Input must be a list or numpy array")

def test_agent(env, agent, num_episodes=100):
    try:
        print(" ---------- Evaluating Performance ----------")
        performance_metrics = evaluate_performance(env, agent, num_episodes=num_episodes)
        print("\nPerformance Metrics:")
        for key, value in performance_metrics.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

if __name__ == '__main__':
    try:
        # Set up the environment
        env = WarehouseEnvironment(pygame_render = True)
        _, state = env.reset() # image of first reset

        print(state.shape)
        # env.render_video(0,420)

        if env.init_arr is None:
            raise ValueError("env.init_arr is None after initialization")

        dimensions = get_dimensions(env.init_arr)
        print(f"Environment created with dimensions: {dimensions}")

        if dimensions == (0, 0):
            print("Warning: Environment dimensions are (0, 0). There might be an issue with environment initialization.")
        
        # Set up the agent
        state_size = env.n_states
        action_size = env.n_actions
        if torch.cuda.is_available(): metal = 'cuda'
        else: metal = 'cpu'
        # Init agent with network
        agent = MaskPPOAgent(env, CNNLSTMModel(30,30,4,3).to(metal), device= metal, batch_size= 256)
        model_weights_path = './weights/ppo_model_cuda_czm_ppo.pth'
        # Load model weights if provided
        if model_weights_path:
            agent.model.load_state_dict(torch.load(model_weights_path, map_location=metal, weights_only=True))
            print(f"Loaded model weights from: {model_weights_path}")
            print("Agent created")

        # Optionally, load a pre-trained model
        # model_path = "path_to_your_model.pth"
        # agent.q_network.load_state_dict(torch.load(model_path))
        # print(f"Loaded pre-trained model from {model_path}")

        # Run the evaluation
        test_agent(env, agent)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()