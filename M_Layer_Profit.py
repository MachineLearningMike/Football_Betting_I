import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

from M_Config import *

from M_BettingMath import *

class Layer_Profit( Layer ) :
     
    def __init__( self, queries, **kwargs ) :
        super().__init__( **kwargs )  # Do NOT put 'dynamic = True' here. It leads to 'NotImplementedError'. DO NOT put trainable = ...

        self.queries = queries

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'queries': self.queries,
        })

        return config


    def build( self, input_shape ) : # input_shape = [ ( None, len( self.queries ) ) ] * 3 

        super().build(input_shape)

        return


    def call( self, inputs, training ) : # input_shape = [ ( None, len( self.queries ) ) ] * 3 : stake, odds, happened

        odds, happened_true, happened_pred = inputs

        stake = Get_Stake_On_Hap( odds, happened_pred )
        profit_games_queries = Get_Pofit_On_Hap( odds, happened_true, stake )

        return profit_games_queries
    
    
    def compute_output_shape( self, input_shape ) : # input_shape = [ ( None, len( self.queries ) ) ] * 3 : stake, odds, happened
         output_shape = input_shape[0]

         return output_shape


