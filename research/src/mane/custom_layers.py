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
  def __init__(self, layers=None, **kwargs):
    """
    Init function.
    """
    super(RowDot, self).__init__(layers=None, **kwargs)

  ######################################################################### call
  def call(self, inputs, **kwargs):
    """
    Layer logic.
    """
    print('Inputs 0 shape: %s' % str(inputs[0].shape))
    print('Inputs 1 shape: %s' % str(inputs[1].shape))
    l1 = inputs[0]
    l2 = inputs[1]
    output = K.batch_dot(inputs[0], inputs[1], axes=[1,1])
    return output
  
# === End CLASS MergeRowDot <<<      

# >>> BEGIN HELPER FUNCTIONS <<<

############################################################################ dot
