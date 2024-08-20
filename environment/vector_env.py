from environment.WarehouseEnv import WarehouseEnvironment

def make_custom_env(seed, idx, **kwargs):
    def thunk():
        # Create an instance of your custom environment
        env = WarehouseEnvironment(**kwargs, seed=seed + idx)
        
        # Apply the seed to ensure reproducibility
        env.seed(seed + idx)
        
        return env
    
    return thunk