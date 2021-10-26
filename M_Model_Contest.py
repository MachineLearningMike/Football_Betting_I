import numpy as np
import tensorflow as tf

from M_Config import *

from M_Football import *
from M_BettingMath import *

from M_EarlyStop_0 import *
from M_Loss_Monitor import *

from M_Model_Chained import *

"""
from Trans_Encoder_No_Emb import *
from M_Layer_Regularizer import *
from M_Layer_One_Sum import *
from M_Layer_Softmax import *
from M_Layer_Freezer import *
"""

class Model_Contest( Model_Chained ) :

    def __init__( self, task ) :

        print( 'Constructing a Model_Contest.')

        super().__init__( task )

        self.model_name = 'Model_Contest'

        """
        self.regularizer = Layer_Rgularizer( self.task.queries )
        self.one_sum = Layer_One_Sum( self.task.queries )
        self.softmax = Layer_Softmax( self.task.queries )
        """
        
        self.root_queries = None
        rt = self.config.get( 'RootQueries', None )
        if rt is not None and rt is True :
            self.root_queries = super_q_group.Root_Queries()
        else :
            self.root_queries = self.task.queries


    def Get_Custom_Objects ( self ) :
        
        return {} # keys must be a class name, not a layer name.


    def Get_Outcomes( self, modelOutput, odds, happened_true = None ) :

        stake_pred_onehot = None; profit_pred_onehot = None

        odds = tf.convert_to_tensor( odds )

        size_splits = super_q_group.Size_Splits()

        stake_pred_onehot = []; profit_pred_onehot = []

        for output_id in range( len( modelOutput ) ) :
            # group was always pushed to happened_true by fitting NO MATTER what derivative loss functions we use. 
            # And, happened_true is all of 1s and 0s. We must find at which place the value is most pushed to 1.

            loss, stake_true_onehot_group, profit_true_onehot_group, stake_pred_onehot_group, profit_pred_onehot_group = \
                Analyize_Output2( output_id, odds, happened_true, size_splits, modelOutput[ output_id ] )

            stake_pred_onehot.append( stake_true_onehot_group )
            profit_pred_onehot.append( profit_pred_onehot_group )

        stake_pred_onehot = tf.concat( stake_pred_onehot, axis = -1 ) # tf will do even for a 1-length array.
        profit_pred_onehot = tf.concat( profit_pred_onehot, axis = -1 ) # tf will do even for a 1-length array.

        return stake_pred_onehot, profit_pred_onehot

