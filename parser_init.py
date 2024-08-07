def init_parser(self):
   
    self.add_argument('--method', type=str, choices=['dqn', 'qnet', 'mppo'], default='dqn',
                        help='Choose the training method: deep Q-network, traditional Q-network or masked PPO agent')
    self.add_argument('--render', type=str, choices=['pygame', 'video', 'off'], default='off',
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
    self.add_argument('--batch', type=int, default=32, 
                    help='Batch size for training process')
    self.add_argument('--train_name', type=str, default='train', 
                    help='Specifiy the name of the current train')
    self.add_argument('--cmd_log', type=int, default=5, 
                    help='Set command line log frequency')
    self.add_argument('--explore', type=int, default=200000, 
                    help='Set exploration steps for e-greedy decay')