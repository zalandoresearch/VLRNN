"""
vlrnn.

A PyTorch library for Very Long Recurrent Neural Networks.
"""

__version__ = "0.0.1"
__author__ = 'Roland Vollgraf'

from .modules import OutputModule, RNNModule, BaseRNN, PlainRNN, VLRNN
from .utilities import *