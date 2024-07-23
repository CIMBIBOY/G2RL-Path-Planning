def update_coords(coords, inst_arr, agent, time_idx, width, global_map, direction, agent_old_coordinates, cells_skipped, dist):
    h, w = inst_arr.shape[:2]
    local_obs = np.array([])
    local_map = np.array([])
    agent_reward = 0
    agent_path = coords[agent]
    agent_done = False
    h_old, w_old = agent_old_coordinates[0], agent_old_coordinates[1]
    h_new, w_new = h_old + direction[1], w_old + direction[0]

    if (h_new == agent_path[-1][0] and w_new == agent_path[-1][1]):
        agent_done = True

    if (h_new >= h or w_new >= w) or (h_new < 0 or w_new < 0) or \
       (inst_arr[h_new, w_new][0] == 255 and inst_arr[h_new, w_new][1] == 165 and inst_arr[h_new, w_new][2] == 0) or \
       (inst_arr[h_new, w_new][0] == 0 and inst_arr[h_new, w_new][1] == 0 and inst_arr[h_new, w_new][2] == 0):
        agent_reward += rewards_dict('1')
        agent_done = True
        h_new, w_new = h_old, w_old
    else:
        if (global_map[h_new, w_new] == 255):
            agent_reward += rewards_dict('0')
            cells_skipped += 1

        if (global_map[h_new, w_new] != 255 and cells_skipped >= 0):
            agent_reward += rewards_dict('2', cells_skipped)
            cells_skipped = 0

    new_dist = manhattan_distance(h_new, w_new, agent_path[-1][0], agent_path[-1][1])
    if new_dist < dist:
        dist = new_dist

    if np.array_equal(inst_arr[h_old, w_old], [255, 0, 0]):
        inst_arr[h_old, w_old] = [255, 255, 255]

    for idx, path in enumerate(coords):
        if idx == agent:
            continue
        
        if time_idx < len(path):
            h_old, w_old = path[time_idx - 1]
            h_new, w_new = path[time_idx]
        else:
            continue

        if h_new >= h or w_new >= w or h_new < 0 or w_new < 0:
            continue

        cell_coord = inst_arr[h_new, w_new]
        
        if np.array_equal(cell_coord, [255, 255, 255]):
            inst_arr[h_new, w_new] = [255, 165, 0]
            inst_arr[h_old, w_old] = [255, 255, 255]
        else:
            h_new, w_new = h_old, w_old

        coords[idx] = path[:time_idx] + [[h_new, w_new]] + path[time_idx + 1:]

    if not agent_done:
        inst_arr[h_new, w_new] = [255, 0, 0]

    coords[agent] = coords[agent][:time_idx] + [[h_new, w_new]] + coords[agent][time_idx + 1:]

    local_obs = inst_arr[max(0, h_new - width):min(h - 1, h_new + width), max(0, w_new - width):min(w - 1, w_new + width)]
    global_map[h_old, w_old] = 255
    local_map = global_map[max(0, h_new - width):min(h - 1, h_new + width), max(0, w_new - width):min(w - 1, w_new + width)]

    return np.array(local_obs), np.array(local_map), global_map, agent_done, agent_reward, cells_skipped, inst_arr, [h_new, w_new], dist, coords