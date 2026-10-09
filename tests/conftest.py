from pathlib import Path

from jaxmodal import config

collect_ignore = ["multidevice"]

config.enable_x64()
config.enable_cache(Path(__file__).parents[1] / ".jax_cache")
