"""Exposes leafsim library to user."""

from importlib.metadata import version

from .leafsim import SUPPORTED_MODELS, LeafSim

__all__ = ["LeafSim", "SUPPORTED_MODELS"]

__version__ = version("leafsim")
