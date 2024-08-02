def init_parser(self):
   
    self.add_argument('--method', type=str, choices=['dqn', 'qnet'], default='dqn',
                        help='Choose the training method: deep Q-network or traditional Q-network')
    self.add_argument('--render', type=str, choices=['pygame', 'all', 'off'], default='off',
                        help='Choose to visualize the training in pygame? Options: --render pygame, or --render off, or --render all for video and pygame rendering')
   # Add an argument for the number of episodes
    self.add_argument('--epochs', type=int, default=1000,
                        help='Number of episodes for training.')
    self.add_argument('--timesteps', type=int, default=33,
                        help='Number of timesteps for a single episode.')
    self.add_argument('--metal', type=str, choices=['cpu', 'cuda'], default='cpu',
                    help='Choose CUDA or CPU accelaration')
    self.add_argument('--seed', type=int, default=42, 
                    help='Random seed for reproducibility.')
    self.add_argument('--train', type=str, choices=['scratch', 'retrain'], default='scratch',
                        help='Choose whether to train from scratch or retrain an existing model.')
    self.add_argument('--model_weights', type=str, default=None,
                        help='Path to the model weights file (required if --train is set to retrain).')