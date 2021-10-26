import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

from M_Config import *

class Layer_One_Sum( Layer ) :
     
    def __init__( self, queries, **kwargs ) :
        super().__init__( **kwargs ) # Do NOT put 'dynamic = True' here. It leads to 'NotImplementedError'. DO NOT put trainable = ...

        self.queries = queries

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'queries': self.queries
        })
        return config

    def build( self, input_shape ) : # input_shape = ( None, self.nQueries )

        super().build(input_shape)
        return 

    def call( self, inputs, training ) : # inputs.shape = ( None, self.nQueries )

        bet_worth = inputs

        bp = tf.split( bet_worth, inputs.shape[-1], axis = -1 ) # bp[k].shape = (None, 1)
        bp_one_sum = []

        # Be careful: queries might be inconsistent.

        #if bp.count(0) * bp.count(1) * bp.count(2) > 0 : # "Home__Win", "Game_Draw", and "Away__Win" queries are found.
        bp_one_sum+= self.Force_One_Sum( [ bp[0], bp[1], bp[2] ] )
        #if bp.count(3) * bp.count(4) > 0 : # "Both_Sc_Y" and "Both_Sc_N" queries are found.
        bp_one_sum += self.Force_One_Sum( [ bp[3], bp[4] ] )
        #if bp.count(5) * bp.count(6) > 0 : # "Over__1.5" and "Under_1.5" queries are found.
        bp_one_sum += self.Force_One_Sum( [ bp[5], bp[6] ] ) #================================================= Work?
        #if bp.count(7) * bp.count(8) > 0 : # "Over__2.5" and "Under_2.5" queries are found.
        bp_one_sum += self.Force_One_Sum( [ bp[7], bp[8] ] )

        # "Score_1_0", "Score_2_1"and "Score_1_1" queries don't make a complet one, actually.
        #if bp.count(9) * bp.count(10) * bp.count(11) > 0 : # "Score_1_0", "Score_2_1"and "Score_1_1" queries are found.
        bp_one_sum += self.Force_One_Sum( [ bp[9], bp[10], bp[11] ] )

        ret = tf.concat( bp_chosen, axis = -1 )

        return ret
    

    def compute_output_shape( self, input_shape ) :
        output_shape = input_shape

        return output_shape


    def Force_One_Sum( self, bp_list ) :
        bp_a = tf.concat( bp_list[ 0: -1 ], axis = -1 ) # bp_tensor.shape = (None, N)
        ones = tf.ones_like( bp_list[-1], dtype = bp_b.dtype )
        bp_b = ones - tf.reduce_sum( bp_a, axis = -1, keepdims = True )
        bps = tf.concat( [ bp_a, bp_b ], axis = -1 )
        bp_list_return = tf.split( bps, bps.shape[-1], axis = -1 )

        return  bp_list_return #
    


    



