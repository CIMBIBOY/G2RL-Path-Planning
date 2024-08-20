import numpy as np
import random
from map_generator import start_end_points

def test_start_end_points():
    # Test case 1: Basic functionality
    def test_basic():
        arr = np.zeros((10, 10))
        obs_coords = [[1, 1], [3, 3], [7, 7]]
        result = start_end_points(obs_coords, arr)
        assert result is not None, "Function should return a valid result"
        assert len(result) == len(obs_coords), "Should return an end point for each obstacle"

    # Test case 2: Handling static obstacles
    def test_static_obstacles():
        arr = np.zeros((10, 10))
        arr[5, 5] = 1  # Static obstacle
        obs_coords = [[1, 1], [3, 3], [7, 7]]
        result = start_end_points(obs_coords, arr)
        assert result is not None, "Function should return a valid result"
        for _, coords in result:
            assert arr[coords[2], coords[3]] == 0, "End point should not be on a static obstacle"

    # Test case 3: Start and end points are different
    def test_different_start_end():
        arr = np.zeros((10, 10))
        obs_coords = [[1, 1], [3, 3], [7, 7]]
        result = start_end_points(obs_coords, arr)
        assert result is not None, "Function should return a valid result"
        for _, coords in result:
            assert coords[:2] != coords[2:], "Start and end points should be different"

    # Test case 4: No duplicate end points
    def test_no_duplicate_ends():
        arr = np.zeros((10, 10))
        obs_coords = [[1, 1], [3, 3], [7, 7]]
        result = start_end_points(obs_coords, arr)
        assert result is not None, "Function should return a valid result"
        end_points = [tuple(coords[2:]) for _, coords in result]
        assert len(end_points) == len(set(end_points)), "No duplicate end points allowed"

    # Test case 5: End points are not start points
    def test_end_not_start():
        arr = np.zeros((10, 10))
        obs_coords = [[1, 1], [3, 3], [7, 7]]
        result = start_end_points(obs_coords, arr)
        assert result is not None, "Function should return a valid result"
        start_points = set(tuple(c) for c in obs_coords)
        for _, coords in result:
            assert tuple(coords[2:]) not in start_points, "End point should not be a start point"

    # Test case 6: Handling a full grid
    def test_full_grid():
        arr = np.ones((5, 5))  # All cells are obstacles
        arr[0, 0] = 0  # Only one free cell
        obs_coords = [[0, 0]]
        result = start_end_points(obs_coords, arr)
        assert result is None, "Function should return None when no valid end points are possible"

    # Test case 7: Large number of obstacles
    def test_many_obstacles():
        arr = np.zeros((20, 20))
        obs_coords = [[random.randint(0, 19), random.randint(0, 19)] for _ in range(50)]
        result = start_end_points(obs_coords, arr)
        if result is not None:
            assert len(result) == len(obs_coords), "Should handle large number of obstacles"

    # Test case 8: Consistency across multiple runs
    def test_consistency():
        arr = np.zeros((10, 10))
        obs_coords = [[1, 1], [3, 3], [7, 7]]
        results = [start_end_points(obs_coords, arr) for _ in range(10)]
        assert all(result is not None for result in results), "All runs should produce valid results"
        assert all(len(result) == len(obs_coords) for result in results), "All runs should produce the same number of end points"
        
        # Check that start points are consistent across all runs
        for i in range(len(obs_coords)):
            start_points = [result[i][1][:2] for result in results]
            assert all(start == obs_coords[i] for start in start_points), f"Start point for obstacle {i} should be consistent"
        
        # Check that end points may vary (randomness check)
        end_points_sets = [set(tuple(result[i][1][2:]) for i in range(len(obs_coords))) for result in results]
        assert len(set.union(*end_points_sets)) > len(obs_coords), "End points should vary across runs due to randomness"

    # Run all test cases
    test_basic()
    test_static_obstacles()
    test_different_start_end()
    test_no_duplicate_ends()
    test_end_not_start()
    test_full_grid()
    test_many_obstacles()
    test_consistency()
    
    print("All tests passed successfully!")

# Run the test function
if __name__ == "__main__":
    test_start_end_points()