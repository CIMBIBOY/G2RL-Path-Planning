import numpy as np
import time
from map_generator import heuristic_generator, map_to_value
from global_mapper import find_path

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

    for _ in range(num_episodes):
        start_cell, state = env.reset()
        done = False
        steps = 0
        start_time = time.time()

        while not done:
            action = agent.act(state)
            next_state, cell, reward, done = env.step(action)
            env.render()
            agent.store(state, action, reward, next_state, done)
            state = next_state
            steps += 1

        end_time = time.time()
        computing_time = (end_time - start_time) / steps

        # Calculate Moving Cost
        if isinstance(cell, (list, tuple)):
            manhattan_distance = abs(cell[0] - start_cell[0]) + abs(cell[1] - start_cell[1])
        else:
            # If cell is an integer, assume it represents the index of the agent's final position
            manhattan_distance = abs(cell // env.init_arr.shape[1] - start_cell // env.init_arr.shape[1]) + abs(cell % env.init_arr.shape[1] - start_cell % env.init_arr.shape[1])

        if manhattan_distance != 0:
            moving_cost = steps / manhattan_distance + 1e-7
        else:
            moving_cost = 0

        # Calculate Detour Percentage
        value_map = map_to_value(env.init_arr.squeeze())
        if isinstance(start_cell, (list, tuple)):
            h_map = heuristic_generator(value_map, (start_cell[0], start_cell[1]))
            a_star_path, _ = find_path(value_map, (start_cell[0], start_cell[1]), (cell[0], cell[1]) if isinstance(cell, (list, tuple)) else (cell // env.init_arr.shape[1], cell % env.init_arr.shape[1]))
        else:
            h_map = heuristic_generator(value_map, (start_cell // env.init_arr.shape[1], start_cell % env.init_arr.shape[1]))
            a_star_path, _ = find_path(value_map, (start_cell // env.init_arr.shape[1], start_cell % env.init_arr.shape[1]), (cell[0], cell[1]) if isinstance(cell, (list, tuple)) else (cell // env.init_arr.shape[1], cell % env.init_arr.shape[1]))
        a_star_path_length = len(a_star_path) if a_star_path != 'fail' else 0
        detour_percentage = (steps - a_star_path_length) / a_star_path_length * 100 if a_star_path_length > 0 else 0

        total_moving_cost += moving_cost
        total_detour_percentage += detour_percentage
        total_computing_time += computing_time

    return {
        "moving_cost": total_moving_cost / num_episodes,
        "detour_percentage": total_detour_percentage / num_episodes,
        "computing_time": total_computing_time / num_episodes
    }