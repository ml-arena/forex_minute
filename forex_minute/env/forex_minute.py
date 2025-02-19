import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from .renderer import ForexMinuteRenderer

class ForexMinuteEnv(gym.Env):
    """
    A Forex trading environment for minute-level data with three possible actions:
    sell (0), hold (1), and buy (2).
    """
    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 4}

    def __init__(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        initial_info: Optional[Dict[str, Any]] = None,
        history_length: int = 100,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Process input data
        if data:
            self.closes = [dt['close_price'] for dt in data]
            self.timestamps = [
                datetime.strptime(dt["timestamp"], '%Y-%m-%dT%H:%M:%S') 
                for dt in data
            ]
        else:
            # Use sample data if none provided
            self._load_sample_data()
            
        self.tmax = len(self.closes)
        
        # Set initial trading state
        if initial_info:
            self._process_initial_info(initial_info)
        else:
            self.initial_t = min(120, self.tmax - 1)
            self.initial_info = {
                "initial_t": self.initial_t,
                "position_type": 1,  # Start with no position
                "position_value": 0.0
            }

        # Environment configuration
        self.history_length = history_length
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),  # position_type, position_value, current_price
            high=np.array([2, np.inf, np.inf]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # sell (0), hold (1), buy (2)
        
        # Rendering setup
        self.render_mode = render_mode
        self.renderer = ForexMinuteRenderer() if render_mode in ['human', 'rgb_array'] else None

    def _load_sample_data(self):
        """Load sample data for testing purposes."""
        self.closes = [np.random.uniform(0.0, 1.0) for _ in range(2880)]  # Sample static price for 48 hours
        base_time = datetime.strptime("2024-04-22T00:00:00", '%Y-%m-%dT%H:%M:%S')
        self.timestamps = [
            base_time + timedelta(minutes=i)
            for i in range(2880)
        ]

    def _process_initial_info(self, initial_info: Dict[str, Any]):
        """Process initial trading state information."""
        self.initial_t = 0
        initial_timestamp = datetime.strptime(
            initial_info["timestamp"],
            '%Y-%m-%dT%H:%M:%S'
        )
        while (self.initial_t < len(self.timestamps) and 
               self.timestamps[self.initial_t] < initial_timestamp):
            self.initial_t += 1
            
        self.initial_info = {
            "initial_t": self.initial_t,
            "position_type": initial_info["position_type"],
            "position_value": initial_info["position_value"]
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset trading state
        self.t = self.initial_info["initial_t"]
        self.position_type = self.initial_info["position_type"]
        self.position_value = self.initial_info["position_value"]
        
        # Create observation
        observation = np.array([
            self.position_type,
            self.position_value,
            self.closes[self.t]
        ])
        
        info = self._get_info()
        return observation, info

    def _get_info(self, done: bool = False) -> Dict[str, Any]:
        """Get current state information."""
        if done:
            return {
                "timestamp": self.timestamps[self.t],
                "position_type": self.position_type,
                "position_value": self.position_value
            }
        return {}

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on action and position."""
        reward = 0.0
        
        if action == 0:  # Sell
            if self.position_type == 1:  # From hold to sell
                self.position_type = 0
                self.position_value = self.closes[self.t]
            elif self.position_type == 2:  # From buy to sell
                reward = self.closes[self.t] - self.position_value
                self.position_type = 0
                self.position_value = self.closes[self.t]
                
        elif action == 1:  # Hold/Release
            if self.position_type == 0:  # Close sell
                reward = self.position_value - self.closes[self.t]
                self.position_type = 1
                self.position_value = 0.0
            elif self.position_type == 2:  # Close buy
                reward = self.closes[self.t] - self.position_value
                self.position_type = 1
                self.position_value = 0.0
                
        elif action == 2:  # Buy
            if self.position_type == 1:  # From hold to buy
                self.position_type = 2
                self.position_value = self.closes[self.t]
            elif self.position_type == 0:  # From sell to buy
                reward = self.position_value - self.closes[self.t]
                self.position_type = 2
                self.position_value = self.closes[self.t]
                
        return reward

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        # Calculate reward and update position
        reward = self._calculate_reward(action)
        
        # Move to next timestep
        self.t += 1
        done = self.t >= self.tmax - 1
        
        # Create observation
        observation = np.array([
            self.position_type,
            self.position_value,
            self.closes[self.t]
        ])
        
        # Get additional info
        info = self._get_info(done)
        
        # Render if needed
        if self.render_mode:
            self.render()
            
        return observation, reward, done, False, info

    def render(self) -> Optional[np.ndarray]:
        """Render the current state."""
        if not self.renderer:
            return None
            
        # Prepare state for rendering
        state = {
            'timestamps': self.timestamps[max(0, self.t - self.history_length):self.t + 1],
            'closes': self.closes[max(0, self.t - self.history_length):self.t + 1],
            'position_type': self.position_type,
            'position_value': self.position_value,
            'current_price': self.closes[self.t]
        }
        
        return self.renderer.render(state)

    def close(self):
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()