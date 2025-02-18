"""
ForexMinute environment
"""
from gymnasium.envs.registration import register
import os

# Get the directory containing this file
PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# Register the environment
register(
    id='ForexMinute-v0',
    entry_point='forex_minute.env.forex_minute:ForexMinuteEnv',
    max_episode_steps=100,
)