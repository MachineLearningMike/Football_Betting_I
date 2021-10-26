import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

from M_Config import *

class Layer_Rgularizer( Layer ) :
     
    def __init__( self, queries, **kwargs ) :
        super().__init__( **kwargs ) # Do NOT put 'dynamic = True' here. It leads to 'NotImplementedError'. DO NOT put trainable = ...

        self.queries = queries

        self.SoftMax = tf.math.softmax


        input_shrink_score = tf.keras.layers.Input( shape = ( len( queries ), ), dtype = tf.float32 )
        x = tf.keras.layers.Dense( units = 3, activation = 'sigmoid' ) ( input_shrink_score )
        self.shrink_score_q_group = tf.keras.models.Model( inputs = input_shrink_score, outputs = x, name = 'shrink_factor' )


        input_over_under = tf.keras.layers.Input( shape = ( len( queries ), ), dtype = tf.float32 )
        x = tf.keras.layers.Dense( units = 1, activation = 'sigmoid' ) ( input_over_under )
        self.over_under_factor = tf.keras.models.Model( inputs = input_over_under, outputs = x, name = 'shrink_factor' )


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

        happended_logit, odds = inputs

        pred = tf.split( happended_logit, happended_logit.shape[-1], axis = -1 ) # pred[k].shape = (None, 1)
        pred_chosen = []

        # Be careful: queries might be inconsistent.

        win_q_group_found = 0 in self.queries and 1 in self.queries and 2 in self.queries # "Home__Win", "Game_Draw", and "Away__Win" queries are found.
        pred_chosen += tf.cond( win_q_group_found, lambda: self.Get_Softmax( [ pred[0], pred[1], pred[2] ] ), lambda: [] )

        both_q_group_found = 3 in self.queries and 4 in self.queries # "Both_Sc_Y" and "Both_Sc_N" queries are found.
        pred_chosen += tf.cond( both_q_group_found, lambda: self.Get_Softmax( [ pred[3], pred[4] ] ), lambda: [] )

        s15_q_group_found = 5 in self.queries and 6 in self.queries # "Over__1.5" and "Under_1.5" queries are found.
        pred_chosen += tf.cond( s15_q_group_found, lambda: self.Get_Softmax( [ pred[5], pred[6] ] ), lambda: [] ) #================================================= Work?

        s25_q_group_found = 7 in self.queries and 8 in self.queries # "Over__2.5" and "Under_2.5" queries are found.
        pred_chosen += tf.cond( s25_q_group_found, lambda: self.Get_Softmax( [ pred[7], pred[8] ] ), lambda: [] )

        # The only query group that is incomplete, in the meaning that the probabilties don't sum up to 1.
        score_q_group_found = 9 in self.queries and 10 in self.queries and 11 in self.queries # "Score_1_0", "Score_2_1"and "Score_1_1" queries are found.
        temp_pred = tf.cond( score_q_group_found, lambda: self.Get_Softmax( [ pred[9], pred[10], pred[11] ] ), lambda: [] )
        pred_chosen += tf.cond( score_q_group_found, lambda: self.Shrink_Score_Q_Group( temp_pred, odds ), lambda: [] )

        #tf.cond( s15_q_group_found and s25_q_group_found, \
        #    lambda : Align_Over_Under_Groups( self, [ pred_chosen[5], pred_chosen[6] ], [ pred_chosen[7], pred_chosen[8] ], odds ), \
        #    lambda : None  ) # Complete it later.

        ret = tf.concat( pred_chosen, axis = -1 )

        return ret
    

    def compute_output_shape( self, input_shape ) :
        output_shape = input_shape

        return output_shape


    def Get_Softmax( self, pred_list ) :
        pred_tensor = tf.concat( pred_list, axis = -1 ) # pred_tensor.shape = (None, N)
        pred_tensor = self.SoftMax( pred_tensor )
        pred_list_return = tf.split( pred_tensor, pred_tensor.shape[-1], axis = -1 )

        return  pred_list_return


    def Shrink_Score_Q_Group( self, pred_list, odds ) : # Give a chance to learn how to shrink the incomplete score group probabilities.
        pred_tensor = tf.concat( pred_list, axis = -1 ) # pred_tensor.shape = (None, N)
        shrink_factor = self.shrink_score_q_group( odds )
        pred_tensor = pred_tensor * shrink_factor
        pred_list_return = tf.split( pred_tensor, pred_tensor.shape[-1], axis = -1 )

        return  pred_list_return


    def Align_Over_Under_Groups( self, pred_list_15, pred_list_25, odds ) : # Give a chance to learn how to shrink the incomplete score group probabilities.

        pu15 = pred_list_15[1]
        po15 = 1.0 - pu15

        alpha = self.over_under_factor( odds )

        pu25 = alpha + ( 1.0 - alpha ) * pu15
        po25 = ( 1.0 - alpha ) * ( 1.0 - pu15 )

        return  [ po15, pu15 ], [po25, pu25 ]


    def ChooseMaxExclusive( self, pred_list ) :
        pred_tensor = tf.concat( pred_list, axis = -1 ) # pred_tensor.shape = (None, N)
        pred_shape = tf.shape(pred_tensor, out_type = tf.int64 ) # Pls, avoid using pred_tenosr.shape, which is partially know.
        range = tf.expand_dims( tf.range(0, pred_shape[0], dtype = tf.int64 ), axis = -1 )
        argmax = tf.expand_dims( tf.argmax( pred_tensor, axis = -1 ), axis = -1 )
        indices = tf.concat( [range, argmax], axis = -1 )
        max_tensor = tf.reduce_max( pred_tensor, axis = -1 )
        pred_tensor = tf.scatter_nd( indices, max_tensor, pred_shape )
        pred_list_return = tf.split( pred_tensor, pred_tensor.shape[-1], axis = -1 )

        return  pred_list_return #



    



