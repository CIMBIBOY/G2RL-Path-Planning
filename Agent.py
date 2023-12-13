import numpy as np
import random
from collections import deque
from network import MLP, CNN

class MLP_Agent:
    def __init__(self, enviroment, model):
        
        # 状态数是环境的格子数
        self._state_size = enviroment.n_states
        self._action_space = enviroment.action_space()
        self._action_size = enviroment.n_actions
        # 经验回放池
        self.expirience_replay = deque(maxlen=4)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        
        # Build networks
        self.q_network = model
        self.target_network = model

    def store(self, state, action, reward, next_state, terminated):
        # 放入经验回放池
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        # 构建初始化网络模型
        model = MLP(self._state_size, self._action_size)
        return model

    def alighn_target_model(self):
        # 更新目标网络
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        # 采取动作
        if np.random.rand() <= self.epsilon:
            return random.choice(self._action_space)
        
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        # 利用经验池训练
        minibatch = random.sample(self.expirience_replay, batch_size)
        
        for state, action, reward, next_state, terminated in minibatch:
            
            target = self.q_network.predict(state)
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.q_network.fit(state, target, epochs=1, verbose=0)