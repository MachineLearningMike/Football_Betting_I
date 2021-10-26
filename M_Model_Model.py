import numpy as np
import tensorflow as tf

from M_Config import *

from M_Football import *
from M_BettingMath import *

from M_Model_Chained import *

class Model_Model( Model_Chained ) :

    def __init__( self, task ) :

        print( 'Constructing a Model_Model.')

        super().__init__( task )

        self.model_name = 'Model'

        self.root_queries = None
        rt = self.config.get( 'RootQueries', None )
        if rt is not None and rt is True :
            self.root_queries = super_q_group.Root_Queries()
        else :
            self.root_queries = self.task.queries


    def Get_Custom_Objects ( self ) :

        return {}
        # return { 'Encoder' : Encoder, 'Layer_Regularizer' : Layer_Rgularizer, 'Layer_One_Sum' : Layer_One_Sum, 'Layer_Softmax' : Layer_Softmax } # keys must be a class name, not a layer name.


    def Get_Outcomes( self, modelOutput, odds, happened_true = None, debugging = False, debugging_arg = None ) :

        stake_0_or_1_hot, profit_0_or_1_hot, stake_place_0_or_1_hot = None, None, None

        odds = tf.convert_to_tensor( odds )

        size_splits = super_q_group.Size_Splits()
        dummies = super_q_group.DummyQueries_GroupWise()
        
        stake_0_or_1_hot = []; profit_0_or_1_hot = []; stake_place_0_or_1_hot = [] 

        for output_id in range( len( modelOutput ) ) :

            happened_pred_1_hot_group, stake_0_or_1_hot_group, stake_palce_0_or_1_hot_group = \
                BettingMath.Decide_Stake_From_Model_Output( output_id, odds, size_splits, modelOutput[ output_id ], dummies[ output_id ], debugging, debugging_arg )

            stake_0_or_1_hot.append( stake_0_or_1_hot_group )
            #stake_place_0_or_1_hot.append( stake_palce_0_or_1_hot_group )

            if happened_true is not None :
                happened_true_1_hot_group, profit_0_or_1_hot_group = \
                    BettingMath.Find_BackProfit_From_Stake( output_id, odds, happened_true, size_splits, stake_0_or_1_hot_group, dummies[ output_id ], debugging )
                profit_0_or_1_hot.append( profit_0_or_1_hot_group )
       
        stake_0_or_1_hot = tf.concat( stake_0_or_1_hot, axis = -1 ) # tf will do even for a 1-length array.
        #stake_place_0_or_1_hot = tf.concat( stake_place_0_or_1_hot, axis = -1 )

        if happened_true is not None :
            profit_0_or_1_hot = tf.concat( profit_0_or_1_hot, axis = -1 ) # tf will do even for a 1-length array.

        return stake_0_or_1_hot, profit_0_or_1_hot
