from importlib.metadata import version

__version__ = version("dynamicauc")

from gymnasium.envs.registration import register

register(
    id="DynamicAuction-v0",
    entry_point="dynamicauc.env:AuctionEnv",
    max_episode_steps=None, 
)