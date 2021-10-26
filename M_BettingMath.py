import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

from M_Config import *


class BettingMath() :

    def Add( a, b ) :
        return a + b

    def Convert_To_OneHot( array, axis ) :
        pos = tf.math.argmax( array, axis = axis )
        one_hot = tf.one_hot( pos, depth = array.shape[axis] )

        return one_hot

    def Onehot_For_1_Proximity( array, axis ) :
        one_proximity = 1.0 / ( tf.pow( ( array - 1.0 ) * ( array - 1.0 ), 0.5 ) + 1e-7 )  # Whether greater or less, a number closest to 1.0 must be promoted.
        maxpos = tf.math.argmax( one_proximity, axis = axis )
        one_hot = tf.one_hot( maxpos, depth = array.shape[axis] )
        return one_hot

    def Softmax_For_1_Proximity( array, axis ) :
        one_proximyty = 1.0 / ( tf.pow( ( array - 1.0 ) * ( array - 1.0 ), 0.5 ) + 1e-7 )  # Whether greater or less, a number closest to 1.0 must be promoted.
        softmax = tf.math.softmax( one_proximyty, axis = axis )

        return softmax



    def Get_MaxDrop( profit, axis ) : # profit.shape = [ None, len(queries) ]
        #return tf.Variable( [ 33.3] * profit.shape[-1] )

        y = profit.numpy()

        max_drop = []; max_up = []

        for query_id in range( y.shape[1] ) :

            min1 = min2 = min11 = max1 = max2 = max21 = maxnum = minnum = 0

            for i in range( 1, y.shape[0] ) :
                if y[ i, query_id ] - y[ i-1, query_id ] > 0 :
                    if y[ i, query_id ] - y[ min11, query_id ] > maxnum:
                        max1 = i ; min1 = min11 ; maxnum = y[ max1, query_id ] - y[ min1, query_id ]
                    if y[ i, query_id ] > y[ max21, query_id ] : max21 = i
                else:
                    if y[ i, query_id ] < y[ min11, query_id ] : min11 = i
                    if y[ i, query_id ] - y[ max21, query_id ] < minnum :
                        min2 = i; max2 = max21; minnum = y[ min2, query_id ] - y[ max2, query_id ]

            max_drop.append( y[ max2, query_id ] - y[ min2, query_id ] )
            max_up.append( y[ max1, query_id ] - y[ min1, query_id ] )

        return tf.Variable( max_drop, dtype = profit.dtype )



    def Onehot_For_1_Proximity_ModelOutput( modelOutput ) :

        happened_pred_by_group = modelOutput

        happened_pred = []
        for group_id in range( len( happened_pred_by_group ) ) :

            #------------------------------------- Focus Principle ----------------------------------------------------------------
            #
            #   happened_pred was always forced to converge to heppened_true, NO MATTER what loss functions we adopt.
            #   Specific loss functions control what/which values of what/which training examples happened_pred will converge to and how fast/slow.
            #   A loss function, for example, tries to force happened_pred to converge to happened_true where happened_true is 1, ignoring 0s.
            #   This function tries to reversely find out at what places/queries happened_pred converges to 1.

            happened_group = happened_pred_by_group[ group_id ]

            onehot_for_1_proximity = Onehot_For_1_Proximity( happened_group, axis = -1 )
            happened_pred.append( onehot_for_1_proximity )
        
        happened_pred = np.concatenate( happened_pred, axis = - 1 )

        return happened_pred

    #============================================================================================== An Analysis class that encapsulates 1_Proximity strategy 

    # happened_anyhot_anyvalue_group : freely distributed value.

    def Analyize_Output( output_id, odds, happened_true, size_splits, happened_pred_anyvalue_allhot_group ) :

        odds_group = tf.split( odds, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]

        happened_true_one_onehot_group = tf.split( happened_true, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]
        stake_true_onehot_group = Get_Stake_Onehot_QGroup( odds_group, happened_true_one_onehot_group )
        profit_true_onehot_group = Get_Pofit_On_Hap( odds_group, happened_true_one_onehot_group, stake_true_onehot_group )

        happened_pred_one_onehot_group = Get_Happened_Onehot_QGroup( happened_pred_anyvalue_allhot_group )
        stake_pred_onehot_group = Get_Stake_Onehot_QGroup( odds_group, happened_pred_anyvalue_allhot_group )
        profit_pred_onehot_group = Get_Pofit_On_Hap( odds_group, happened_true_one_onehot_group, stake_pred_onehot_group )

        loss = tf.reduce_sum( profit_pred_onehot_group, axis = -1 )
        loss = tf.reduce_mean( loss, axis = -1 )

        return loss, stake_true_onehot_group, profit_true_onehot_group, stake_pred_onehot_group, profit_pred_onehot_group

    #-------------------------------------------------------------------------------- Underlying modules.

    def Get_Happened_Onehot_QGroup( happened_anyhot_anyvalue_group ) :
        happened_pred_onehot = Onehot_For_1_Proximity( happened_anyhot_anyvalue_group, axis = -1 )

        return happened_pred_onehot


    def Get_Stake_Onehot_QGroup( odds_group, happened_anyhot_anyvalue_group ) :

        happened_pred_softmax = Softmax_For_1_Proximity( happened_anyhot_anyvalue_group, axis = -1 ) # Note: softmax gives a constraint of 1 freedom degree.
        happened_pred_onehot = Get_Happened_Onehot_QGroup( happened_pred_softmax ) # Here is a bug. happened_pred_onehot will not comply with stake_pred, which is odds based.

        stake_pred_zero_if_happ_is_not_enough = Get_Stake_On_Hap( odds_group, happened_pred_softmax )
        stake_pred_onehot_or_zerohot = stake_pred_zero_if_happ_is_not_enough * happened_pred_onehot  # We will bet with this value. In loss functions, happened_pred_onehot is replaced with happended_true.

        return stake_pred_onehot_or_zerohot

    #============================================================================================== An Analysis class that encapsulates sigmoid stragegy.

    # happened_anyhot_anyvalue_group : activated by sigmoid.

    def Analyize_Output2( output_id, odds, happened_true, size_splits, happened_between_01_group, dummyQueries = None ) :

        odds_group = tf.split( odds, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]

        happened_true_one_onehot_group = tf.split( happened_true, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]
        stake_true_onehot_group = Get_Stake_Onehot_QGroup2( odds_group, happened_true_one_onehot_group, dummyQueries )
        profit_true_onehot_group = Get_Pofit_On_Hap( odds_group, happened_true_one_onehot_group, stake_true_onehot_group )

        happened_pred_one_onehot_group = Get_Happened_Onehot_QGroup2( happened_between_01_group ) # No use. :)
        stake_pred_onehot_group = Get_Stake_Onehot_QGroup2( odds_group, happened_between_01_group, dummyQueries )
        profit_pred_onehot_group = Get_Pofit_On_Hap( odds_group, happened_true_one_onehot_group, stake_pred_onehot_group )

        loss = tf.reduce_sum( profit_pred_onehot_group, axis = -1 )
        loss = - tf.reduce_mean( loss, axis = -1 )

        return loss, stake_true_onehot_group, profit_true_onehot_group, stake_pred_onehot_group, profit_pred_onehot_group

    #-------------------------------------------------------------------------------- Underlying modules.

    def Get_Happened_Onehot_QGroup2( happened_between_01_group ) :

        happened_softmax = tf.math.softmax( happened_between_01_group, axis = -1 )
        
        onehot = tf.one_hot( tf.math.argmax( happened_softmax, axis = -1 ), depth = happened_softmax.shape[-1] )

        return onehot


    def Get_Stake_Onehot_QGroup2( odds_group, happened_between_01_group, dummyQueries = None ) :

        happened_softmax = tf.math.softmax( happened_between_01_group, axis = -1 )
        stake_zero_if_happ_is_not_enough = Get_Stake_On_Hap( odds_group, happened_softmax, dummyQueries )

        stake_softmax = tf.math.softmax( stake_zero_if_happ_is_not_enough, axis = -1 )
        stake_onehot = tf.one_hot( tf.math.argmax( stake_softmax, axis = -1 ), depth = stake_softmax.shape[-1] ) # hot at index = 0 if all is the same.

        stake_onehot_or_zerohot = stake_zero_if_happ_is_not_enough * stake_onehot  # We will bet with this value. In loss functions, happened_pred_onehot is replaced with happended_true.

        return stake_onehot_or_zerohot

    #============================================================================================== An Analysis class that encapsulates sigmoid stragegy, taking into consideration the stability.

    # happened_anyhot_anyvalue_group : activated by sigmoid.

    def Analyize_Output3( output_id, odds, happened_true, size_splits, happened_between_01_group, dummyQueries = None ) :

        odds_group = tf.split( odds, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]

        happened_true_one_onehot_group = tf.split( happened_true, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]
        stake_true_onehot_group, _ = Get_Stake_Onehot_QGroup3( odds_group, happened_true_one_onehot_group, dummyQueries )
        profit_true_onehot_group = Get_Pofit_On_Hap( odds_group, happened_true_one_onehot_group, stake_true_onehot_group )

        happened_pred_one_onehot_group = Get_Happened_Onehot_QGroup3( happened_between_01_group ) # No use. :)
        stake_pred_onehot_group, stake_pred_onehot_group_binary = Get_Stake_Onehot_QGroup3( odds_group, happened_between_01_group, dummyQueries )
        profit_pred_onehot_group = Get_Pofit_On_Hap( odds_group, happened_true_one_onehot_group, stake_pred_onehot_group )

        loss = tf.reduce_sum( profit_pred_onehot_group, axis = -1 )
        std = tf.math.reduce_std( loss, axis = -1 )
        loss = - tf.reduce_mean( loss, axis = -1 ) / ( 1.0 + std ) #------------------ offset = 1.0 both for 0-divide safety and magnification of std.

        return loss, stake_true_onehot_group, profit_true_onehot_group, stake_pred_onehot_group, profit_pred_onehot_group, stake_pred_onehot_group_binary


    #-------------------------------------------------------------------------------- Underlying modules.

    def Get_Happened_Onehot_QGroup3( happened_between_01_group ) :

        happened_softmax = tf.math.softmax( happened_between_01_group, axis = -1 )
        
        onehot = tf.one_hot( tf.math.argmax( happened_softmax, axis = -1 ), depth = happened_softmax.shape[-1] )

        return onehot


    def Get_Stake_Onehot_QGroup3( odds_group, happened_between_01_group, dummyQueries = None ) :

        happened_softmax = tf.math.softmax( happened_between_01_group, axis = -1 )
        stake_zero_if_happ_is_not_enough = Get_Stake_On_Hap( odds_group, happened_softmax, dummyQueries )

        stake_softmax = tf.math.softmax( stake_zero_if_happ_is_not_enough, axis = -1 )
        stake_onehot = tf.one_hot( tf.math.argmax( stake_softmax, axis = -1 ), depth = stake_softmax.shape[-1] ) # hot at index = 0 if all is the same.

        stake_onehot_or_zerohot = stake_zero_if_happ_is_not_enough * stake_onehot  # We will bet with this value. In loss functions, happened_pred_onehot is replaced with happended_true.

        return stake_onehot_or_zerohot, stake_onehot


    #--------------------------------------------------------------------------------------------------------------------------------------------------
    #================================= An Analysis class that encapsulates sigmoid stragegy, taking into consideration the stability. AND False Negative penalty!!!

    # happened_anyhot_anyvalue_group : activated by sigmoid.

    def Analyize_Output4( output_id, odds, happened_true, size_splits, happened_group, dummyQueries = None ) :

        odds_group = tf.split( odds, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]

        happened_true_one_onehot_group = tf.split( happened_true, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]
        stake_true_onehot_group, _ = Get_Stake_Onehot_QGroup4_Loyal( odds_group, happened_true_one_onehot_group, dummyQueries )
        profit_true_onehot_group = Get_Pofit_On_Hap( odds_group, happened_true_one_onehot_group, stake_true_onehot_group )

        happened_pred_one_onehot_group = Get_Happened_Onehot_QGroup4( happened_group ) # No use. :)
        stake_pred_onehot_group, stake_pred_onehot_group_binary = Get_Stake_Onehot_QGroup4( odds_group, happened_group, dummyQueries )
        profit_pred_onehot_group = Get_Pofit_On_Hap( odds_group, happened_true_one_onehot_group, stake_pred_onehot_group )
        #FalseNegative = tf.cast( tf.logical_and( tf.equal( happened_pred_one_onehot_group, 1.0 ), tf.equal( stake_pred_onehot_group_binary, 0.0 ) ), dtype = tf.float32 )
        falseNegative = ( 1.0 - tf.nn.relu( stake_pred_onehot_group_binary - 1e-9 ) ) * happened_true_one_onehot_group
        missing_stake_FN_group, _ = Get_Stake_Onehot_QGroup4_Loyal( odds_group, falseNegative, dummyQueries )
        missing_profit_group = Get_Pofit_On_Hap( odds_group, happened_true_one_onehot_group, missing_stake_FN_group )

        if super_config['exacerbate'] :
            profit = profit_pred_onehot_group - missing_profit_group
        else :
            profit = profit_pred_onehot_group

        loss = tf.reduce_sum( profit, axis = -1 )
        std = tf.math.reduce_std( loss, axis = -1 )
        loss = - tf.reduce_mean( loss, axis = -1 ) / ( 1.0 + std ) #------------------ offset = 1.0 both for 0-divide safety and magnification of std.

        return loss, stake_true_onehot_group, profit_true_onehot_group, stake_pred_onehot_group, profit_pred_onehot_group, stake_pred_onehot_group_binary


    #-------------------------------------------------------------------------------- Underlying modules.

    def Get_Happened_Onehot_QGroup4( happened_group ) :

        #happened_softmax = tf.math.softmax( happened_group, axis = -1 )
        
        onehot = tf.one_hot( tf.math.argmax( happened_group, axis = -1 ), depth = happened_group.shape[-1] )

        return onehot


    def Get_Stake_Onehot_QGroup4( odds_group, happened_group, dummyQueries = None ) :

        #----------------------------!!!!!! Critically sensitive
        #     softmax( 0, 1 ) = ( 0.27, 0.73 ). Large odds will easily attract stake.
        #
        #     softmax( 0, 3 ) = ( 0.047, 0.952). Odds of 20x ...

        happened_group = happened_group * super_config['ignore_odds']

        happened_softmax = tf.math.softmax( happened_group, axis = -1 ) # NOTE : softmax( 0, 1 ) = ( 0.27, 0.73 )
        stake_zero_if_happ_is_not_enough = Get_Stake_On_Hap( odds_group, happened_softmax, dummyQueries )

        #stake_softmax = tf.math.softmax( stake_zero_if_happ_is_not_enough, axis = -1 )
        stake_onehot = tf.one_hot( tf.math.argmax( stake_zero_if_happ_is_not_enough, axis = -1 ), depth = stake_zero_if_happ_is_not_enough.shape[-1] ) # hot at index = 0 if all is the same.

        stake_onehot_or_zerohot = stake_zero_if_happ_is_not_enough * stake_onehot  # We will bet with this value. In loss functions, happened_pred_onehot is replaced with happended_true.

        return stake_onehot_or_zerohot, stake_onehot


    def Get_Stake_Onehot_QGroup4_Loyal( odds_group, happened_group, dummyQueries = None ) :

        happened_group = happened_group * 99.9

        happened_softmax = tf.math.softmax( happened_group, axis = -1 )
        stake_zero_if_happ_is_not_enough = Get_Stake_On_Hap( odds_group, happened_softmax, dummyQueries )

        #stake_softmax = tf.math.softmax( stake_zero_if_happ_is_not_enough, axis = -1 )
        stake_onehot = tf.one_hot( tf.math.argmax( stake_zero_if_happ_is_not_enough, axis = -1 ), depth = stake_zero_if_happ_is_not_enough.shape[-1] ) # hot at index = 0 if all is the same.

        stake_onehot_or_zerohot = stake_zero_if_happ_is_not_enough * stake_onehot  # We will bet with this value. In loss functions, happened_pred_onehot is replaced with happended_true.

        return stake_onehot_or_zerohot, stake_onehot


    #--------------------------------------------------------------------------------------------------------------------------------------------------

    #================================= An Analysis class that encapsulates sigmoid stragegy, taking into consideration the stability. 
    # False Negative penalty!!!
    # Softmax Normalization

    # happened_anyhot_anyvalue_group : activated by sigmoid.

    def Analyize_Output5( output_id, odds, happened_true, size_splits, happened_group, dummyQueries = None, debugging = False, debugging_arg = None ) :

        odds_group = tf.split( odds, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]

        happened_group = Softmax_Normalization( happened_group, odds_group ) #-----------------

        happened_true_1_hot_group = tf.split( happened_true, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]
        stake_true_1_hot_group, _ = Get_Stake_Onehot_QGroup5_Loyal( odds_group, happened_true_1_hot_group, dummyQueries )
        profit_true_1_hot_group = Get_Pofit_On_Hap( odds_group, happened_true_1_hot_group, stake_true_1_hot_group )

        if debugging : # Pretent the accuracy of 100%. #--------------------------------------
            """
            # Scenario - 1 : Assumes 100% accuracy.
            happened_pred_1_hot_group = happened_true_1_hot_group
            stake_0_or_1_hot_group, stake_palce_0_or_1_hot_group = stake_true_1_hot_group, happened_true_1_hot_group
            profit_0_or_1_hot_group = profit_true_1_hot_group
            """
            # Scenario - 2 : Assumes predicted_happened.
            p_happened = debugging_arg
            happened_pred_1_hot_group = tf.split( p_happened, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]
            stake_0_or_1_hot_group, stake_palce_0_or_1_hot_group = Get_Stake_Onehot_QGroup5( odds_group, happened_pred_1_hot_group, dummyQueries )
            profit_0_or_1_hot_group = Get_Pofit_On_Hap( odds_group, happened_true_1_hot_group, stake_0_or_1_hot_group )
        else :
            happened_pred_1_hot_group = Get_Happened_Onehot_QGroup5( happened_group ) # No use. :)
            stake_0_or_1_hot_group, stake_palce_0_or_1_hot_group = Get_Stake_Onehot_QGroup5( odds_group, happened_group, dummyQueries )
            profit_0_or_1_hot_group = Get_Pofit_On_Hap( odds_group, happened_true_1_hot_group, stake_0_or_1_hot_group )

        #FalseNegative = tf.cast( tf.logical_and( tf.equal( happened_pred_1_hot_group, 1.0 ), tf.equal( stake_palce_0_or_1_hot_group, 0.0 ) ), dtype = tf.float32 )
        falseNegative = tf.math.round( ( 1.0 - tf.nn.relu( stake_palce_0_or_1_hot_group - 1e-9 ) ) ) * happened_true_1_hot_group
        missing_stake_FN_group, _ = Get_Stake_Onehot_QGroup5_Loyal( odds_group, falseNegative, dummyQueries )
        missing_profit_group = Get_Pofit_On_Hap( odds_group, happened_true_1_hot_group, missing_stake_FN_group )

        if super_config['exacerbate'] :
            profit = profit_0_or_1_hot_group - missing_profit_group
        else :
            profit = profit_0_or_1_hot_group

        loss = tf.reduce_sum( profit, axis = -1 )
        std = tf.math.reduce_std( loss, axis = -1 )
        if super_config['std'] :
            loss = - tf.reduce_mean( loss, axis = -1 ) / ( 1.0 + std ) #------------------ offset = 1.0 both for 0-divide safety and magnification of std.
        else :
            loss = - tf.reduce_mean( loss, axis = -1 )

        return loss, stake_true_1_hot_group, profit_true_1_hot_group, stake_0_or_1_hot_group, profit_0_or_1_hot_group, stake_palce_0_or_1_hot_group

    #-------------------------------------------------------------------------------- Softmax Normalization

    def Softmax_Normalization( happened_group, odds_group ) :
        alpha = 1.0
        beta = 2.1 * ( tf.math.log( tf.reduce_max( odds_group, axis = -1 ) ) - tf.math.log( tf.reduce_min( odds_group, axis = -1 ) )  )
        happened_group = alpha + happened_group * tf.stack( [ beta ] * happened_group.shape[-1], axis = -1 )

        return happened_group
    #-------------------------------------------------------------------------------- Underlying modules.

    def Get_Happened_Onehot_QGroup5( happened_group ) :

        #happened_softmax = tf.math.softmax( happened_group, axis = -1 )
        
        onehot = tf.one_hot( tf.math.argmax( happened_group, axis = -1 ), depth = happened_group.shape[-1] )

        return onehot


    def Get_Stake_Onehot_QGroup5( odds_group, happened_group, dummyQueries = None ) :

        #----------------------------!!!!!! Critically sensitive. 
        #     softmax( 0, 1 ) = ( 0.27, 0.73 ). Large odds will easily attract stake.
        #
        #     softmax( 0, 3 ) = ( 0.047, 0.952). Odds of 20x ...
        #
        #     Softmax_Normalization is thert to prevent the distorton.

        happened_softmax = tf.math.softmax( happened_group, axis = -1 ) # NOTE : softmax( 0, 1 ) = ( 0.27, 0.73 ).
        stake_0_or_more_hot = Get_Stake_On_Hap( odds_group, happened_softmax, dummyQueries )

        #stake_softmax = tf.math.softmax( stake_zero_if_happ_is_not_enough, axis = -1 )
        stake_palce_1_hot_possibly_dummy = tf.one_hot( tf.math.argmax( stake_0_or_more_hot, axis = -1 ), depth = stake_0_or_more_hot.shape[-1] ) # hot at index = 0 if all is the same. A dummy 1.
        stake_place_0_or_more_hot = tf.math.round( stake_0_or_more_hot / ( stake_0_or_more_hot + 1e-7 )  ) # Now, all 1s represent a positive stake. Some games have not 1s.
        stake_place_0_or_1_hot = stake_palce_1_hot_possibly_dummy * stake_place_0_or_more_hot # Now, all is true 1, with no dummy 1s.

        stake_0_or_1_hot = stake_0_or_more_hot * stake_place_0_or_1_hot  # We will bet with this value. In loss functions, happened_pred_onehot is replaced with happended_true.

        return stake_0_or_1_hot, stake_place_0_or_1_hot


    def Get_Stake_Onehot_QGroup5_Loyal( odds_group, happened_group, dummyQueries = None ) :

        happened_softmax = tf.math.softmax( happened_group, axis = -1 )
        stake_zero_if_happ_is_not_enough = Get_Stake_On_Hap( odds_group, happened_softmax, dummyQueries )

        #stake_softmax = tf.math.softmax( stake_zero_if_happ_is_not_enough, axis = -1 )
        stake_onehot = tf.one_hot( tf.math.argmax( stake_zero_if_happ_is_not_enough, axis = -1 ), depth = stake_zero_if_happ_is_not_enough.shape[-1] ) # hot at index = 0 if all is the same.

        stake_onehot_or_zerohot = stake_zero_if_happ_is_not_enough * stake_onehot  # We will bet with this value. In loss functions, happened_pred_onehot is replaced with happended_true.

        return stake_onehot_or_zerohot, stake_onehot


    #--------------------------------------------------------------------------------------------------------------------------------------------------


    #--------------------------------------------------------------------------------------------------------------------------------------------------

    #================================= An Analysis class that encapsulates sigmoid stragegy, taking into consideration the stability. 
    # False Negative penalty!!!
    # Softmax Normalization

    # happened_anyhot_anyvalue_group : activated by sigmoid.

    def Get_Loss6( output_id, odds, happened_true, size_splits, happened_group, dummyQueries = None, debugging = False, debugging_arg = None ) :

        _, stake_0_or_1_hot_group, stake_palce_0_or_1_hot_group = \
            BettingMath.Decide_Stake_From_Model_Output( output_id, odds, size_splits, happened_group, dummyQueries, debugging, debugging_arg )

        happened_true_1_hot_group, profit_0_or_1_hot_group = BettingMath.Find_BackProfit_From_Stake( output_id, odds, happened_true, size_splits, stake_0_or_1_hot_group, dummyQueries, debugging )

        odds_group = tf.split( odds, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]
        falseNegative = tf.math.round( ( 1.0 - tf.nn.relu( stake_palce_0_or_1_hot_group - 1e-9 ) ) ) * happened_true_1_hot_group
        missing_stake_FN_group, _ = BettingMath.Get_Stake_Onehot_QGroup6_Loyal( odds_group, falseNegative, dummyQueries )
        missing_profit_group = BettingMath.Get_Pofit_On_Hap( odds_group, happened_true_1_hot_group, missing_stake_FN_group )

        if super_config['exacerbate'] :
            profit = profit_0_or_1_hot_group - missing_profit_group
        else :
            profit = profit_0_or_1_hot_group

        loss = tf.reduce_sum( profit, axis = -1 ) # Sum over queries. loss is then sum_profit : Game -> profit in Real.
        std = tf.math.reduce_std( loss, axis = -1 )
        if super_config['std'] :
            loss = - tf.reduce_mean( loss, axis = -1 ) / ( 1.0 + std ) #------------------ offset = 1.0 both for 0-divide safety and magnification of std.
        else :
            loss = - tf.reduce_mean( loss, axis = -1 ) # Mean over games. loss is then - ( average profit per game ) in Real.

        return loss


    #-------------------------------------------------------------------------------------------------------------

    def Decide_Stake_From_Model_Output( output_id, odds, size_splits, happened_group, dummyQueries = None, debugging = '', debugging_arg = None ) :

        odds_group = tf.split( odds, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]
        
        happened_group = BettingMath.Softmax_Normalization( happened_group, odds_group ) #-----------------

        if debugging == 'follow_pred_happened' : # Pretent the accuracy of 100%. #--------------------------------------
            
            p_happened = debugging_arg
            happened_pred_1_hot_group = tf.split( p_happened, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]
            stake_0_or_1_hot_group, stake_palce_0_or_1_hot_group = BettingMath.Get_Stake_Onehot_QGroup6( odds_group, happened_pred_1_hot_group, dummyQueries )

        elif debugging == 'follow_true_happened' :
            
            happened_true = debugging_arg
            happened_true_1_hot_group = tf.split( happened_true, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]            
            stake_true_1_hot_group, _ = BettingMath.Get_Stake_Onehot_QGroup6_Loyal( odds_group, happened_true_1_hot_group, dummyQueries )

            happened_pred_1_hot_group = happened_true_1_hot_group
            stake_0_or_1_hot_group, stake_palce_0_or_1_hot_group = stake_true_1_hot_group, happened_true_1_hot_group
            
        else :

            happened_pred_1_hot_group = BettingMath.Get_Happened_Onehot_QGroup6( happened_group )
            stake_0_or_1_hot_group, stake_palce_0_or_1_hot_group = BettingMath.Get_Stake_Onehot_QGroup6( odds_group, happened_group, dummyQueries )

        return happened_pred_1_hot_group, stake_0_or_1_hot_group, stake_palce_0_or_1_hot_group


    #--------------------------------------------------------------------------------------------------------------
    
    def Find_BackProfit_From_Stake( output_id, odds, happened_true, size_splits, stake_0_or_1_hot_group, dummyQueries = None, debugging = '' ) :

        odds_group = tf.split( odds, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]

        happened_true_1_hot_group = tf.split( happened_true, num_or_size_splits = size_splits, axis = -1 ) [ output_id ]

        if debugging == 'follow_true_happened' :
            # Replace the argument stake_0_or_1_hot_group with idealistic one.
            stake_true_1_hot_group, _ = BettingMath.Get_Stake_Onehot_QGroup6_Loyal( odds_group, happened_true_1_hot_group, dummyQueries )
            profit_true_1_hot_group = BettingMath.Get_Pofit_On_Hap( odds_group, happened_true_1_hot_group, stake_true_1_hot_group )
            profit_0_or_1_hot_group = profit_true_1_hot_group

        else :
            profit_0_or_1_hot_group = BettingMath.Get_Pofit_On_Hap( odds_group, happened_true_1_hot_group, stake_0_or_1_hot_group )

        return happened_true_1_hot_group, profit_0_or_1_hot_group

    #-------------------------------------------------------------------------------- Softmax Normalization

    def Softmax_Normalization( happened_group, odds_group ) :

        if happened_group.shape[ -1 ] > 1 :
            alpha = 1.0
            beta = 2.1 * ( tf.math.log( tf.reduce_max( odds_group, axis = -1 ) ) - tf.math.log( tf.reduce_min( odds_group, axis = -1 ) )  )
            happened_group = alpha + happened_group * tf.stack( [ beta ] * happened_group.shape[-1], axis = -1 )
        else :
            happened_group = happened_group

        # Note : The underlying theoy to this normalization can be found in my Research Notebook 3.

        return happened_group
    #-------------------------------------------------------------------------------- Underlying modules.

    def Get_Happened_Onehot_QGroup6( happened_group ) :

        #happened_softmax = tf.math.softmax( happened_group, axis = -1 )
        
        onehot = tf.one_hot( tf.math.argmax( happened_group, axis = -1 ), depth = happened_group.shape[-1] )

        return onehot


    def Get_Stake_Onehot_QGroup6( odds_group, happened_group, dummyQueries = None ) :

        #----------------------------!!!!!! Critically sensitive. 
        #     softmax( 0, 1 ) = ( 0.27, 0.73 ). Large odds will easily attract stake.
        #
        #     softmax( 0, 3 ) = ( 0.047, 0.952). Odds of 20x ...
        #
        #     Softmax_Normalization is thert to prevent the distorton.

        if happened_group.shape[ -1 ] > 1 :
            happened_softmax = tf.math.softmax( happened_group, axis = -1 ) # NOTE : softmax( 0, 1 ) = ( 0.27, 0.73 ).
        else :
            happened_softmax = happened_group # Already sigmoid-ed.

        stake_0_or_more_hot = BettingMath.Get_Stake_On_Hap( odds_group, happened_softmax, dummyQueries )

        #stake_softmax = tf.math.softmax( stake_zero_if_happ_is_not_enough, axis = -1 )
        stake_palce_1_hot_possibly_dummy = tf.one_hot( tf.math.argmax( stake_0_or_more_hot, axis = -1 ), depth = stake_0_or_more_hot.shape[-1] ) # hot at index = 0 if all is the same. A dummy 1.
        stake_place_0_or_more_hot = tf.math.round( stake_0_or_more_hot / ( stake_0_or_more_hot + 1e-7 )  ) # Now, all 1s represent a positive stake. Some games have not 1s.
        stake_place_0_or_1_hot = stake_palce_1_hot_possibly_dummy * stake_place_0_or_more_hot # Now, all is true 1, with no dummy 1s.

        stake_0_or_1_hot = stake_0_or_more_hot * stake_place_0_or_1_hot  # We will bet with this value. In loss functions, happened_pred_onehot is replaced with happended_true.

        return stake_0_or_1_hot, stake_place_0_or_1_hot


    def Get_Stake_Onehot_QGroup6_Loyal( odds_group, happened_group, dummyQueries = None ) :

        stake_0_or_more_hot = BettingMath.Get_Stake_On_Hap( odds_group, happened_group, dummyQueries )

        #stake_softmax = tf.math.softmax( stake_zero_if_happ_is_not_enough, axis = -1 )
        stake_palce_1_hot_possibly_dummy = tf.one_hot( tf.math.argmax( stake_0_or_more_hot, axis = -1 ), depth = stake_0_or_more_hot.shape[-1] ) # hot at index = 0 if all is the same. A dummy 1.

        stake_0_or_1_hot = stake_0_or_more_hot * stake_palce_1_hot_possibly_dummy  # We will bet with this value. In loss functions, happened_pred_onehot is replaced with happended_true.

        return stake_0_or_1_hot, stake_palce_1_hot_possibly_dummy


    #--------------------------------------------------------------------------------------------------------------------------------------------------

    def Get_Bet_Prob_Target( r1, beta, gamma, odds, happened ) :
        r0 = - beta * r1 / ( 1.0 - beta )
        slope = ( 1.0 - beta ) / ( r1 + 1e-7 )
        alpha =   - ( gamma - beta ) / ( slope + 1e-7 ) 

        profit_on_bet = ( odds - 1.0 ) * ( happened ) + ( -1.0 ) * ( 1.0 - happened )
        intermediary = tf.keras.activations.relu( x = slope * profit_on_bet , alpha = alpha ) + beta
        bet_worth_target = tf.keras.activations.sigmoid( 3.5 * intermediary )

        return bet_worth_target

    def Get_Stake_On_Hap( odds, happened_true_or_pred, dummyQueries = None ) :

        POUS = odds * happened_true_or_pred - 1.0 # expected_profit_on_1_dollar_stake #---------------------- (-int, inf)

        if dummyQueries is not None and len( dummyQueries ) > 0 :
            POUS = tf.transpose( POUS )
            # If you feel it difficult to understand tf.tensor_scatter_nd_update(.), read this block of code. It shows the essence.
            indices = tf.constant( [ [ query ] for query in dummyQueries ] )
            updates = tf.zeros( shape = [ dim for dim in indices.shape[ : 1 ] ] + [ dim for dim in POUS.shape[ 1 : ] ], dtype = tf.float32 )
            POUS = tf.tensor_scatter_nd_update( POUS, indices, updates )

            POUS = tf.transpose( POUS )

        stake_on_foretest = BettingMath.Stake_Strategy( POUS )
        """
        [ EPOUS = odds * happened_pred - 1.0 ] is the Expected_Profit_On_Unit_Stake, when the probability of winning is 'happened_pred'.
        If we believe in 'happened_pred', then ReLu( EPOUS ) represents how much profit the bookie allows us to win, deliberately or mistakenly.
        [ APOUS = odds * happened_true - 1.0 ] is the Actual_Profit_On_Unit_stake.
        Our stake will be STAKE = Strategy( EPOUS ).
        Once truth comes out, our actual profit will be PROFIT = APUS * STAKE.
        We want to find parameters that generates such 'happened_pred' that PROFIT is maximized.
        """
        return stake_on_foretest


    def OH1( odds, happened ) :
        return odds * happened - 1


    def Stake_Strategy( Expected_POUS ) :
        """
        The most simple strategy is here, which is the same as [ how much profit the bookie allows us to win, deliberately or mistakenly.]
        Enhance it more, later. Simple multiplication by a number is meaningless. Find best non-lineraity.
        """
        pous = Expected_POUS

        if super_config['const.stake'] :
            stake = tf.keras.activations.relu( pous ) / ( pous + 1e-7 ) * super_config['stake']  # PENDING stake == 10.
        else :
            delta = 1e-3
            stake = tf.math.pow( tf.keras.activations.relu( pous ) + delta, super_config['stakePow'] ) * super_config['stake']
            stake = tf.keras.activations.relu( stake - tf.math.pow( delta, super_config['stakePow'] ) * super_config['stake'] ) # Trick to avoid extremly large gradient of pow( x,  < 1 ) at x = 0.
            #stake = tf.minimum( tf.keras.activations.relu( stake ), tf.keras.activations.relu( pous ) )
            # relu( pow( relu(pous)+delta, sp ) * ST - pow( delta, sp ) * ST ) <= 0
            # <-> pow( relu(pous)+delta, sp ) * ST <= pow( delta, sp ) * ST
            # <-> relu(pous) + delta <= delta
            # <-> relu(pous) <= 0
            # <-> pous <= 0

        return stake


    def Get_Pofit_On_Hap( odds, happened_true_or_pred, stake ) :

        POUS = odds * happened_true_or_pred - 1.0
        profit = POUS * stake # [game, query]

        return profit

    #========================================================================================

    def Get_Bet_Worth_Target_V1( min_oh_1, threshold_bet_worth, full_oh_1, full_bet_worth, odds, happened_true ) :
        """
        threshold_bet_worth : threshold_prob in (0.0, 1.0)
        full_bet_worth : full_prob in (threshold_bet_worth, 1.0)
        min_oh_1 : (0.0, inf)
        full_oh_1 : (min_oh_1, inf)
        """
        def logit( x ) : return tf.math.log( x / ( 1.0 - x ) )
        
        alpha = ( logit( full_bet_worth ) - logit ( threshold_bet_worth ) ) / ( full_oh_1 - min_oh_1 )

        off = min_oh_1 - logit( threshold_bet_worth ) / alpha

        x = odds * happened_true - 1.0
        bet_worth_target = tf.math.sigmoid( alpha * ( x - off ) )
        
        return bet_worth_target


    def Get_Stake_On_Bet_Worth_V1( bet_worth, max_stake, threshold_bet_worth ) :

        binary_decision = tf.nn.relu( bet_worth - threshold_bet_worth ) / (bet_worth - threshold_bet_worth) #tf.cast( tf.greater( bet_worth, threshold_bet_worth ), tf.float32 )
        #stake = binary_decision * ( max_stake - 1 ) / ( 1.0 - threshold_bet_worth ) * bet_worth + ( 1.0 - max_stake * threshold_bet_worth ) / ( 1.0 + threshold_bet_worth )
        stake = binary_decision  * ( max_stake - 1 ) / ( 1.0 - threshold_bet_worth ) * bet_worth + ( 1.0 - max_stake * threshold_bet_worth ) / ( 1.0 + threshold_bet_worth )

        return stake


    def Get_Bet_Worth_Target_V2( odds, happened_true ) :

        x = odds * happened_true - 1.0

        bet_worth_target = tf.nn.relu( x )
        
        return bet_worth_target


    def Get_Stake_On_Bet_Worth_V2( bet_worth, max_stake, full_oh_1, full_bet_worth ) :

        def logit( x ) : return tf.math.log( x / ( 1.0 - x ) )

        off = logit( 1.0 / max_stake ) * full_bet_worth / ( logit( 1.0 / max_stake ) - logit( full_bet_worth ) )
        alpha = - logit( 1.0 / max_stake ) / off
        stake = max_stake * tf.math.sigmoid( alpha * ( bet_worth - off ) )

        return stake


    #========================================================================================

    def Get_Bet_Decision( bet_worth, odds ) :
        bet_decision = tf.cast( tf.greater( bet_worth, 0.5 ), tf.float32 )
        maxBetUnits = 5.0 # units
        bet_decision = bet_decision * ( bet_worth * 2.0 * ( maxBetUnits - 1.0 ) + 2.0 - maxBetUnits )

        return bet_decision

    def Get_Profit_Series( odds, happened, bet_decision ) :
        profit_series = ( ( odds - 1.0 ) * ( happened ) + ( -1.0 ) * ( 1.0 - happened ) ) * bet_decision
        return profit_series

    def Get_Profit_Series_Accumulated( profit_series ) :
        a = tf.expand_dims( tf.ones_like(profit_series), -1 );  b = tf.transpose(a)
        lowerTriangle = tf.linalg.band_part(  a * b, -1, 0)
        profit_series_acc = tf.squeeze( tf.matmul( lowerTriangle, tf.expand_dims( profit_series, -1 ) ) )

        """
        ones = tf.ones_like( profit_series_acc )
        zeros = tf.zeros_like( profit_series_acc )
        i = 2
        d = tf.linalg.band_part( ones, 0, i + 1) - tf.linalg.band_part( ones, 0, i )
        r = - tf.reduce_sum( d, -1 )
        v = tf.stack( [zeros] * i + [r] + [zeros] * 3, axis = - 1 )
        delta = tf.matmul( v + d, profit_series_acc )
        """

        return profit_series_acc

    def Get_Profit( profit_series ) :
        profit = tf.reduce_sum( profit_series ) # NOT reduce_mean.
        return profit

    def Get_Profit_From_Threshold( bet_worth, threshold, odds, happened ) :

        bet_decision = tf.cast( tf.greater( bet_worth, threshold ), tf.float32 )
        profit = ( ( odds - 1.0 ) * ( happened ) + ( -1.0 ) * ( 1.0 - happened ) ) * bet_decision
        #penalty = tf.cast( tf.logical_and( tf.less(bet_decision, 0.5), tf.greater(happened, 0.5)  ),  tf.float32 ) * 0.01 # ======================== no_bet while happened, 1 % penalty.
        # profit = profit - penalty
        profit = tf.reduce_sum( profit ) # NOT recude_mean. It will be mean-ed later.

        return bet_decision, profit

    def Get_Better_Profit( best_bet_decision, best_profit, best_threshold, bet_worth, threshold, odds, happened) :

        bet_decision, profit = Get_Profit_From_Threshold( bet_worth, threshold, odds, happened )

        choose_old = tf.cast( tf.greater( best_profit, profit ), tf.float32 )

        best_bet_decision = best_bet_decision * choose_old + bet_decision * ( 1.0 - choose_old )
        best_profit = best_profit * choose_old + profit * ( 1.0 - choose_old )
        best_threshold = best_threshold * choose_old + threshold * ( 1.0 - choose_old )

        return best_bet_decision, best_profit, best_threshold


    def Get_Bet_Mix( y_true, y_pred ):

        rows, odds, happened = tf.split( y_true, num_or_size_splits = [ 1, 1, 1 ], axis = -1 )
        rows = tf.squeeze(rows); odds = tf.squeeze(odds); happened = tf.squeeze(happened)
        y_pred = tf.squeeze(y_pred)
        bet_worth = y_pred

        return happened, bet_worth

    def Get_Best_Profit ( bet_worth, odds, happened ) :

        best_threshold = -0.1
        best_bet_decision, best_profit = Get_Profit_From_Threshold( bet_worth, best_threshold, odds, happened )
        best_bet_decision, best_profit, best_threshold = Get_Better_Profit( best_bet_decision, best_profit, best_threshold, bet_worth, 0.0, odds, happened)
        best_bet_decision, best_profit, best_threshold = Get_Better_Profit( best_bet_decision, best_profit, best_threshold, bet_worth, 0.1, odds, happened)
        best_bet_decision, best_profit, best_threshold = Get_Better_Profit( best_bet_decision, best_profit, best_threshold, bet_worth, 0.2, odds, happened)
        best_bet_decision, best_profit, best_threshold = Get_Better_Profit( best_bet_decision, best_profit, best_threshold, bet_worth, 0.3, odds, happened)
        best_bet_decision, best_profit, best_threshold = Get_Better_Profit( best_bet_decision, best_profit, best_threshold, bet_worth, 0.4, odds, happened)
        best_bet_decision, best_profit, best_threshold = Get_Better_Profit( best_bet_decision, best_profit, best_threshold, bet_worth, 0.5, odds, happened)
        best_bet_decision, best_profit, best_threshold = Get_Better_Profit( best_bet_decision, best_profit, best_threshold, bet_worth, 0.6, odds, happened)
        best_bet_decision, best_profit, best_threshold = Get_Better_Profit( best_bet_decision, best_profit, best_threshold, bet_worth, 0.7, odds, happened)
        best_bet_decision, best_profit, best_threshold = Get_Better_Profit( best_bet_decision, best_profit, best_threshold, bet_worth, 0.8, odds, happened)
        best_bet_decision, best_profit, best_threshold = Get_Better_Profit( best_bet_decision, best_profit, best_threshold, bet_worth, 0.9, odds, happened)
        best_bet_decision, best_profit, best_threshold = Get_Better_Profit( best_bet_decision, best_profit, best_threshold, bet_worth, 1.0, odds, happened)

        """
        for threshold in np.arange( 0.0, 1.0, 0.05 ) :
            bet_decision, profit = Get_Profit( bet_worth, threshold, odds, happened )
            if best_profit < profit :
                best_bet_decision = bet_decision
                best_profit = profit
                best_threshold = threshold
        """

        #tf.print( best_threshold, best_profit )

        return best_bet_decision, best_profit, best_threshold


add = BettingMath.Add( 2, 3 )
print( add )