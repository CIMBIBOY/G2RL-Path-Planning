import numpy as np
import time
from map_generator import map_to_value
from global_mapper import find_path, return_path
import os

import numpy as np
import matplotlib.pyplot as plt
import os

def save_evaluation_image(init_arr, start, end, agent_path, a_star_path, episode, eval_folder='G2RL-Path-Planning/eval_images'):
    """
    Saves an evaluation image showing the environment, start and end positions, agent's path, and optimal path.

    Args:
        init_arr (numpy.ndarray): The initial environment array (48x48).
        start (tuple): The start position (x, y).
        end (tuple): The end position (x, y).
        agent_path (list): List of coordinates representing the agent's path.
        a_star_path (list): List of coordinates representing the optimal path.
        episode (int): The episode number.
        eval_folder (str): The folder where evaluation images will be saved.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(init_arr, cmap='binary')

    # Plot start and end positions
    plt.plot(start[1], start[0], 'ro', markersize=10, label='Start')
    plt.plot(end[1], end[0], 'go', markersize=10, label='End')

    # Plot agent's path
    if agent_path:
        agent_path = np.array(agent_path)
        plt.plot(agent_path[:, 1], agent_path[:, 0], 'gray', linestyle=':', linewidth=2, label="Agent's Path")

    # Plot optimal path
    if a_star_path:
        a_star_path = np.array(a_star_path)
        plt.plot(a_star_path[:, 1], a_star_path[:, 0], 'blue', linestyle=':', linewidth=2, label='Optimal Path')

    plt.title(f'Episode {episode + 1} Evaluation')
    plt.legend()
    plt.grid(False)

    # Ensure the evaluation folder exists
    os.makedirs(eval_folder, exist_ok=True)

    # Save the figure
    plt.savefig(os.path.join(eval_folder, f'episode_{episode + 1}_eval.png'))
    plt.close()

def evaluate_performance(env, agent, num_episodes=100, eval_folder="eval_images"):
    """
    Evaluates the performance of the agent in the WarehouseEnvironment.

    Args:
        env (WarehouseEnvironment): The environment instance.
        agent (Agent): The agent instance.
        num_episodes (int): The number of episodes to run for evaluation.
        eval_folder (str): The folder where evaluation images will be saved.

    Returns:
        dict: A dictionary containing the performance metrics.
    """
    # Ensure the evaluation folder exists
    os.makedirs(eval_folder, exist_ok=True)

    total_moving_cost = 0
    total_detour_percentage = 0
    total_computing_time = 0
    failed_paths = 0

    for episode in range(num_episodes):
        _, state = env.reset()

        start_cell = env.dynamic_coords[env.agent_idx][0]
        end_cell = env.dynamic_coords[env.agent_idx][-1]
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Start cell: {start_cell}, End cell: {end_cell}")

        value_map = map_to_value(env.init_arr.squeeze())
        optimal_path, _ = find_path(value_map, start_cell, end_cell)
        
        if optimal_path == 'fail':
            print("Optimal pathfinding failed before episode start.")
            optimal_path_coords = []
            optimal_path_length = 0
        else:
            optimal_path_coords = return_path(optimal_path)
            optimal_path_length = len(optimal_path_coords)
            print(f"Optimal path length: {optimal_path_length}")

        done = False
        steps = 0
        path = []
        start_time = time.time()

        while not done:
            action = agent.act(state)
            step_result = env.step(action)

            # Ensure step_result has 4 values
            if len(step_result) != 4:
                raise ValueError(f"Unexpected step result: {step_result}")
            
            next_state, cell, reward, done = step_result

            # Handle cases where any of the step results are None
            if next_state is None or cell is None or reward is None:
                print(f"Warning: Step result contains None. Ending episode.")
                break

            agent.store(state, action, reward, next_state, done)
            state = next_state
            steps += 1

            cell = env.agent_prev_coord
            path.append(cell)

        end_time = time.time()
        computing_time = (end_time - start_time) / steps

        # Calculate Moving Cost
        if not isinstance(start_cell, list):
            print(f"Warning: start_cell: {start_cell} is not a list")   

        if isinstance(cell, (list, tuple)):
            manhattan_distance = abs(cell[0] - start_cell[0]) + abs(cell[1] - start_cell[1])
        else:
            manhattan_distance = 0

        if manhattan_distance != 0:
            moving_cost = steps / (manhattan_distance + 1e-7)
        else:
            moving_cost = 0

        # Calculate Detour Percentage
        if optimal_path_length > 0:
            detour_percentage = ((steps - optimal_path_length) / optimal_path_length * 100)
        else:
            detour_percentage = 0
            failed_paths += 1

        total_moving_cost += moving_cost
        total_detour_percentage += detour_percentage
        total_computing_time += computing_time

        print(f"Episode {episode + 1} completed:")
        print(f"  Steps taken: {steps}")
        print(f"  Optimal path length: {optimal_path_length}")
        print(f"  Moving cost: {moving_cost:.4f}")
        print(f"  Detour percentage: {detour_percentage:.2f}%")
        print(f"  Computing time: {computing_time:.4f} s/step")
        print(f"  Failed paths so far: {failed_paths}")
        print(f"  Reached goals so far: {env.arrived}")

        # Save the evaluation image
        save_evaluation_image(env.init_arr, start_cell, end_cell, path, optimal_path_coords, episode, eval_folder)

    return {
        "moving_cost": total_moving_cost / num_episodes,
        "detour_percentage": total_detour_percentage / num_episodes,
        "computing_time": total_computing_time / num_episodes,
        "failed_paths": failed_paths,
        "agent_reached_goal": env.arrived
    }