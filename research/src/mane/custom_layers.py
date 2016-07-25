"""Custom keras layers
"""
# Coding: utf-8
# File name: custom_layer.py
# Created: 2016-07-24
# Description:
## v0.0: File created. MergeRowDot layer.

from __future__ import division
from __future__ import print_function

__author__ = "Hoang Nguyen"
__email__ = "hoangnt@ai.cs.titech.ac.jp"

from keras import backend as K
from keras.engine.topology import Merge
import numpy as np

# >>> BEGIN CLASS RowDot <<<

class RowDot(Merge):
  """
  Layer for element wise merge mul and take sum along
  the second axis.
  """
  
  ##################################################################### __init__
  def __init__(self, **kwargs):
    """
    Init function.
    """
    super(RowDot, self).__init__(**kwargs)

  ######################################################################### call
  def call(self, inputs):
    """
    Layer logic.
    """
    l1 = inputs[0]
    l2 = inputs[1]
    output = K.sum(inputs[0], inputs[1], axis=[1,1])
    return output
  
# === End CLASS MergeRowDot <<<      
