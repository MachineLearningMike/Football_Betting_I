import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

from M_Config import *

from Trans_Encoder_No_Emb import *


class Layer_Frozer() :
     
    def __init__( self, reference_layer, **kwargs ) :
        super().__init__( **kwargs )

        self.reference_layer = reference_layer


    def call( self, inputs, training ) :

        outputs = self.reference_layer.call( inputs, training = False )

        return outputs
