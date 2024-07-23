import sys
import os

# Get the directory containing the current file (__init__.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (which should be G2RL-Path-Planning) to the Python path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)