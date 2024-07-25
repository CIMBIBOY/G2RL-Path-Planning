import numpy as np
import time
import matplotlib.pyplot as plt
from map_generator import heuristic_generator, map_to_value
from global_mapper import find_path, return_path

def visualize_map_and_paths(value_map, start, end, agent_path, a_star_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(value_map, cmap='binary')
    plt.plot([p[1] for p in agent_path], [p[0] for p in agent_path], 'r-', label='Agent Path')
    if a_star_path:
        plt.plot([p[1] for p in a_star_path], [p[0] for p in a_star_path], 'g-', label='A* Path')
    plt.plot(start[1], start[0], 'bo', label='Start')
    plt.plot(end[1], end[0], 'yo', label='End')
    plt.legend()
    plt.title('Value Map with Paths')
    plt.show()

def evaluate_performance(env, agent, num_episodes=100):
    """
    Evaluates the performance of the agent in the WarehouseEnvironment.

    Args:
        env (WarehouseEnvironment): The environment instance.
        agent (Agent): The agent instance.
        num_episodes (int): The number of episodes to run for evaluation.

    Returns:
        dict: A dictionary containing the performance metrics.
    """
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
        else:
            optimal_path_length = len(optimal_path)
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

            # Render the environment
            if env.pygame_render:
                env.render()
            
            agent.store(state, action, reward, next_state, done)
            state = next_state
            steps += 1

            cell = env.agent_prev_coord
            path.append(cell)

        end_time = time.time()
        computing_time = (end_time - start_time) / steps

        # Calculate Moving Cost
        if start_cell is None:
            print(f"Warning: start_cell: {start_cell} is NONE")
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
        value_map = map_to_value(env.init_arr.squeeze())
        if isinstance(start_cell, (list, tuple)) and isinstance(cell, (list, tuple)):
            a_star_path, _ = find_path(value_map, start_cell, cell)
        else:
            print(f"Warning: Invalid start or cell value. Start: {start_cell}, Cell: {cell}")
            a_star_path = 'fail'

        print(f"A* path: {a_star_path}")

        if a_star_path == 'fail':
            failed_paths += 1
            detour_percentage = 0
        else:
            a_star_path_length = len(a_star_path)
            detour_percentage = ((steps - a_star_path_length) / a_star_path_length * 100) if a_star_path_length > 0 else 0

        total_moving_cost += moving_cost
        total_detour_percentage += detour_percentage
        total_computing_time += computing_time

        print(f"Episode {episode + 1} completed:")
        print(f"  Steps taken: {steps}")
        print(f"  Moving cost: {moving_cost:.4f}")
        print(f"  Detour percentage: {detour_percentage:.2f}%")
        print(f"  Computing time: {computing_time:.4f} s/step")
        print(f"  Failed paths so far: {failed_paths}")

        # Visualize the map and paths every 10th episode
        if episode % 10 == 0 or episode == num_episodes - 1:
            plt.clf()  # Clear the current figure
            visualize_map_and_paths(value_map, start_cell, end_cell, path, a_star_path)

    return {
        "moving_cost": total_moving_cost / num_episodes,
        "detour_percentage": total_detour_percentage / num_episodes,
        "computing_time": total_computing_time / num_episodes,
        "failed_paths": failed_paths
    }