import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from M_BettingMath import *
from M_Config import *


r1 = 2.0
beta = -0.2
gamma = -0.8


class Loss_Bet_Worth(tf.keras.losses.Loss):
    def __init__(self, query_id, name='LBP'):
        super().__init__(name=name)

        self.query_id = query_id

    

    def call(self, y_true, y_pred):

        rows, odds, happened = tf.split( y_true, num_or_size_splits = [ 1, 1, 1 ], axis = -1 )

        rows = tf.squeeze(rows); odds = tf.squeeze(odds); happened = tf.squeeze(happened)
        y_pred = tf.squeeze(y_pred)
        bet_worth = y_pred

        """
        #happened, bet_worth, bet_decision = Get_Bet_Mix( y_true, y_pred )
        bet = tf.cast( tf.greater( bet_worth, 0.5 ), tf.float32 )
        penalty = tf.cast( tf.logical_and( tf.less(bet, 0.5), tf.greater(happened, 0.5)  ),  tf.float32 ) * 0.01 # no_bet while happened, 1 % penalty to add to losses of profit type.

        profit = ( ( odds - 1.0 ) * ( happened ) + ( -1.0 ) * ( 1.0 - happened ) ) * bet_worth
        loss1 = - tf.reduce_mean( profit, axis = None )

        loss2 = (   (odds - 1.0) * (1.0 - bet_worth)    -    (odds - 1.0) * bet_worth  ) * happened       +      bet_worth * ( 1.0 - happened )
        loss2 = tf.reduce_mean( loss2, axis = None ) # do NOT negate.

        loss3 = tf.keras.losses.binary_crossentropy( happened, bet_worth ) # reduced auto.
        loss3 = tf.reduce_mean( loss3, axis = None )

        profit_happened = ( ( odds - 1.0 ) * ( happened ) + ( -1.0 ) * ( 1.0 - happened ) )
        #profit_happened = ( tf.cast( tf.greater( profit_happened, 0), tf.float32 ) - 0.5 ) * 4.0 # -2 for loss, +2 for profit
        profit_prob = tf.cast( tf.greater( profit_happened, 0), tf.float32 ) # 0 for loss (-1), 1 for profit (odd-1)
        loss4 = tf.keras.losses.binary_crossentropy( profit_prob, bet_worth )
        loss4 = tf.reduce_mean( loss4, axis = None )
        """

        """
        bet_decision = Get_Bet_Decision( bet_worth, odds, self.query_id )
        profit_series = Get_Profit_Series( odds, happened, betting_decision )
        profit_series_acc = Get_Profit_Series_Accumulated( profit_series )
        profit = Get_Profit ( profit_series )
        """

        bet_worth_target = Get_Bet_Worth_Target( r1, beta, gamma, odds, happened )
        loss5 = tf.keras.losses.binary_crossentropy( bet_worth_target, bet_worth )

        #min = tf.reduce_min( bet_worth_target ); max = tf.reduce_max( bet_worth_target ); mean = tf.reduce_mean( bet_worth_target )
        #tf.print( bet_worth_target )

        augment = 1.0 / ( 1.01 - bet_worth_target + 1e-6 )
        loss5 = loss5 * augment

        loss5 = tf.reduce_mean( loss5, axis = None )

        #risk = tf.math.maximum( 1.0, - tf.reduce_min(profit_series_acc) * 1.2 )
        #loss5 = loss5 * risk

        return loss5 # loss3 #loss1 + loss2 + loss3



class Loss_Bet_Decision(tf.keras.losses.Loss):
    def __init__(self, query_id, name='LBD'):
        super().__init__(name=name)

        self.query_id = query_id

    def call(self, y_true, y_pred):

        rows, odds, happened = tf.split( y_true, num_or_size_splits = [ 1, 1, 1 ], axis = -1 )

        rows = tf.squeeze(rows); odds = tf.squeeze(odds); happened = tf.squeeze(happened)
        y_pred = tf.squeeze(y_pred)

        no_bet_worth = y_pred

        bet_worth = 1.0 - no_bet_worth


        """
        bet_decision = Get_Bet_Decision( bet_worth, odds, self.query_id )
        profit_series = Get_Profit_Series( odds, happened, bet_decision )
        profit_series_acc = Get_Profit_Series_Accumulated( profit_series )
        profit = Get_Profit ( profit_series )
        """

        bet_worth_target = Get_Bet_Worth_Target( r1, beta, gamma, odds, happened )
        loss5 = tf.keras.losses.binary_crossentropy( bet_worth_target, bet_worth )

        #min = tf.reduce_min( bet_worth_target ); max = tf.reduce_max( bet_worth_target ); mean = tf.reduce_mean( bet_worth_target )
        #tf.print( bet_worth_target )

        augment = 1.0 / ( 1.01 - bet_worth_target + 1e-6 )
        loss5 = loss5 * augment

        loss5 = tf.reduce_mean( loss5, axis = None )

        #risk = tf.math.maximum( 1.0, - tf.reduce_min(profit_series_acc) * 1.2 )
        #loss5 = loss5 * risk

        return loss5 # loss3 #loss1 + loss2 + loss3

#====================================================================================================================================================

class Loss_MSE_On_Goals(tf.keras.losses.Loss):
    def __init__(self, goalId, name = 'mse.on.goals'):
        super().__init__(name=name)

        self.goalId = goalId
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):

        # self.queries is not accessable. so
        #rows, odds, happened_true = tf.split( y_true, num_or_size_splits = [ 1, len(self.queries), len(self.queries) ], axis = -1 )

        goals_true = [ None, None ]
        rows, goals_true[0], goals_true[1], _ = tf.split( y_true, num_or_size_splits = [ 1, 1, 1, -1 ], axis = -1 )
        #odds, happened_true = tf.split( remainder, num_or_size_splits = 2, axis = -1 )
        #odds = tf.squeeze( odds, axis = -1 ); happened_true = tf.squeeze( happened_true, axis = -1 )

        goals_pred = y_pred
      
        loss = self.loss_function ( goals_true[ self.goalId ], goals_pred )

        return loss

#=============================================================================

class Loss_Sparce_Cat_X_Entropy_On_GLogits(tf.keras.losses.Loss):
    def __init__(self, goalId, name='sparce.cat.x.entropy.on.glogits'):
        super().__init__(name=name)

        self.goalId = goalId
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy( from_logits = True ) # from log of odds.

    def call(self, y_true, y_pred):

        # self.queries is not accessable. so
        #rows, odds, happened_true = tf.split( y_true, num_or_size_splits = [ 1, len(self.queries), len(self.queries) ], axis = -1 )

        goals_true = [ None, None ]
        rows, goals_true[0], goals_true[1], _ = tf.split( y_true, num_or_size_splits = [ 1, 1, 1, -1 ], axis = -1 )
        #odds, happened_true = tf.split( remainder, num_or_size_splits = 2, axis = -1 )
        #odds = tf.squeeze( odds, axis = -1 ); happened_true = tf.squeeze( happened_true, axis = -1 )

        goals_pred = y_pred
      
        loss = self.loss_function ( goals_true[ self.goalId ], goals_pred )

        return loss

#=============================================================================

class Loss_Sparce_Cat_X_Entropy_On_GSoft(tf.keras.losses.Loss):
    def __init__(self, goalId, name='sparce.cat.x.entropy.on.gsoft'):
        super().__init__(name=name)

        self.goalId = goalId
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy( from_logits = False ) # from softmax

    def call(self, y_true, y_pred):

        # self.queries is not accessable. so
        #rows, odds, happened_true = tf.split( y_true, num_or_size_splits = [ 1, len(self.queries), len(self.queries) ], axis = -1 )

        goals_true = [ None, None ]
        rows, goals_true[0], goals_true[1], _ = tf.split( y_true, num_or_size_splits = [ 1, 1, 1, -1 ], axis = -1 )
        #odds, happened_true = tf.split( remainder, num_or_size_splits = 2, axis = -1 )
        #odds = tf.squeeze( odds, axis = -1 ); happened_true = tf.squeeze( happened_true, axis = -1 )

        goals_pred = y_pred
      
        loss = self.loss_function ( goals_true[ self.goalId ], goals_pred )

        return loss


#=============================================================================

class Loss_MSE_On_Happened_Prob(tf.keras.losses.Loss):
    def __init__(self, output_id, name='mse.on.happened.prob'):
        super().__init__(name=name)

        self.output_id = output_id
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):

        # self.queries is not accessable. so
        #rows, odds, happened_true = tf.split( y_true, num_or_size_splits = [ 1, len(self.queries), len(self.queries) ], axis = -1 )
        rows, _, remainder = tf.split( y_true, num_or_size_splits = [ 1, 2, -1 ], axis = -1 )
        odds, happened_true = tf.split( remainder, num_or_size_splits = 2, axis = -1 )
        #odds = tf.squeeze( odds, axis = -1 ); happened_true = tf.squeeze( happened_true, axis = -1 )

        size_splits = super_q_group.Size_Splits()
        happened_true_list = tf.split( happened_true, num_or_size_splits = size_splits, axis = -1 )
        happened_true = happened_true_list[ self.output_id ]

        happened_pred = y_pred # model output is group by group.
      
        mean_squared_error = self.loss_function ( happened_true, happened_pred )
        loss = mean_squared_error

        #tf.cond( tf.math.is_nan(loss), lambda: tf.print('ERROR 4'), lambda: tf.print(end='') )

        return loss

#=============================================================================

class Loss_Distance_To_1_On_Happened_Prob(tf.keras.losses.Loss):
    def __init__(self, output_id, name = 'distance.to.1.on.happened.prob'):
        super().__init__(name=name)

        self.output_id = output_id
        self.size_splits = super_q_group.Size_Splits()

    def call(self, y_true, y_pred):

        # self.queries is not accessable. so
        #rows, odds, happened_true = tf.split( y_true, num_or_size_splits = [ 1, len(self.queries), len(self.queries) ], axis = -1 )
        rows, _, remainder = tf.split( y_true, num_or_size_splits = [ 1, 2, -1 ], axis = -1 )
        _, happened_true = tf.split( remainder, num_or_size_splits = 2, axis = -1 )
        #odds = tf.squeeze( odds, axis = -1 ); happened_true = tf.squeeze( happened_true, axis = -1 )
      
        happened_true = tf.split( happened_true, num_or_size_splits = self.size_splits, axis = -1 ) [ self.output_id ]

        happened_pred = y_pred # model output is group by group.

        #====================================== Focus Principle ==========================================================================================
        #
        #  distance_between_pred_and_true is ignored at those places where happened_true is 0.
        #  Only values of distance_between_pred_and_true on those queries/palces where happened_true is 1, count in loss, forcing happened_pred to converge to happened_true where happened_true is 1.
        #  happened_pred, so the model, is now free at most of queries/places and can focus on learning sparce 1s WHIT ITS LIMITED learning resources.
        #  (Learning resources of the model is limited by its structure and weights.)
        #
        distance_between_pred_and_true = tf.math.pow( ( ( happened_pred - happened_true ) * ( happened_pred - happened_true ) ), 0.5 )
        distance_between_pred_and_true_focused = distance_between_pred_and_true * happened_true # Multiplied by happened_true - focus principle.
        loss =  tf.reduce_sum( distance_between_pred_and_true_focused )

        #tf.cond( tf.math.is_nan(loss), lambda: tf.print('ERROR 4'), lambda: tf.print(end='') )

        return loss

#=============================================================================
#=============================================================================

class Loss_Minus_Profit_On_Happened_Prob(tf.keras.losses.Loss):
    def __init__(self, output_id, name = 'minus.profit.on.happened.prob'):
        super().__init__(name=name)

        self.output_id = output_id
        self.size_splits = super_q_group.Size_Splits()
        self.dummyQueries = super_q_group.DummyQueries_GroupWise()[ output_id ]
        #self.binary_crossentropy = tf.losses.BinaryCrossentropy( from_logits = True, reduction=tf.keras.losses.Reduction.NONE )

    def call(self, y_true, y_pred):

        #tf.print( 'y_pred', y_pred )

        rows, _, remainder = tf.split( y_true, num_or_size_splits = [ 1, 2, -1 ], axis = -1 )
        odds, happened_true = tf.split( remainder, num_or_size_splits = 2, axis = -1 )
        #odds = tf.squeeze( odds, axis = -1 ); happened_true = tf.squeeze( happened_true, axis = -1 )

        # loss, _, _, _, _, for Analyize_Output1 and Analyize_Output2. loss, _, _, _, _, _ for Analyize_Output3
        loss = BettingMath.Get_Loss6( self.output_id, odds, happened_true, self.size_splits, y_pred )

        """
        odds_group = tf.split( odds, num_or_size_splits = self.size_splits, axis = -1 ) [ self.output_id ]

        happened_true_one_onehot_group = tf.split( happened_true, num_or_size_splits = self.size_splits, axis = -1 ) [ self.output_id ]
        happened_pred_anyvalue_allhot_group = y_pred # model output is group by group.
        happened_pred_one_onehot_group = Get_Happened_Onehot_QGroup( happened_pred_anyvalue_allhot_group )

        stake_true_onehot_group = Get_Stake_Onehot_QGroup( odds_group, happened_true_one_onehot_group )
        stake_pred_onehot_group = Get_Stake_Onehot_QGroup( odds_group, happened_pred_anyvalue_allhot_group ) # This is our final betting decision. Loss must give a feedback to the betting decision.

        profit_true_onehot_group = Get_Pofit_On_Hap( odds_group, happened_true_one_onehot_group, stake_true_onehot_group )
        profit_pred_onehot_group = Get_Pofit_On_Hap( odds_group, happened_pred_one_onehot_group, stake_pred_onehot_group )

        #loss = tf.losses.mean_absolute_error( happened_true_one_onehot_group, happened_pred_anyvalue_allhot_group )
        #loss = tf.losses.mean_absolute_error( stake_target_onehot_group, stake_pred_onehot_group )
        loss = tf.losses.mean_absolute_error( profit_true_onehot_group, profit_pred_onehot_group )
        loss = tf.reduce_mean( loss )
        """

        return loss


class Loss_Minus_Profit_On_Happened_Prob_Summary(tf.keras.losses.Loss):
    def __init__(self, name = 'minus.profit.on.happened.prob.summary'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):

        odds, happened_true = tf.split( y_true, num_or_size_splits = [ 1, 1 ], axis = -1 )
        happened_pred = y_pred

        stake = BettingMath.Get_Stake_On_Hap( odds, happened_pred )
        profit = BettingMath.Get_Pofit_On_Hap( odds, happened_true, stake )

        loss = - tf.reduce_sum( profit, axis = -1 )
        std = tf.math.reduce_std( loss, axis = -1 )

        return loss

#=============================================================================

class Loss_MSE_On_Bet_Worth_V1(tf.keras.losses.Loss):
    def __init__(self, queries, name='mse.on.bet.worth'):
        super().__init__(name=name)

        self.queries = queries
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):

        # self.queries is not accessable. so
        #rows, odds, happened_true = tf.split( y_true, num_or_size_splits = [ 1, len(self.queries), len(self.queries) ], axis = -1 )
        rows, _, remainder = tf.split( y_true, num_or_size_splits = [ 1, 2, -1 ], axis = -1 )
        odds, happened_true = tf.split( remainder, num_or_size_splits = 2, axis = -1 )
        #odds = tf.squeeze( odds, axis = -1 ); happened_true = tf.squeeze( happened_true, axis = -1 )

      
        bet_worth_target = Get_Bet_Worth_Target_V1( super_config['min_oh_1'], super_config['thres_bWorth'], super_config['full_oh_1'], super_config['full_bWorth'], odds, happened_true )
        
        bet_worth = y_pred
      
        mean_squared_error = self.loss_function ( bet_worth_target, bet_worth )
        loss = mean_squared_error

        """
        augment = 1.0 / ( 1.01 - betting_prob_target + 1e-6 )
        loss = loss * augment # Exacerbase the loss at high-profitable games.

        loss = tf.reduce_mean( loss, axis = None )
        """

        #risk = tf.math.maximum( 1.0, - tf.reduce_min(profit_series_acc) * 1.2 )
        #loss5 = loss5 * risk

        #tf.cond( tf.math.is_nan(loss), lambda: tf.print('ERROR 4'), lambda: tf.print(end='') )

        return loss

#=============================================================================

class Loss_MSE_On_Bet_Worth_V2(tf.keras.losses.Loss):
    def __init__(self, queries, name='mse.on.bet.worth'):
        super().__init__(name=name)

        self.queries = queries
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):

        # self.queries is not accessable. so
        #rows, odds, happened_true = tf.split( y_true, num_or_size_splits = [ 1, len(self.queries), len(self.queries) ], axis = -1 )
        rows, _, remainder = tf.split( y_true, num_or_size_splits = [ 1, 2, -1 ], axis = -1 )
        odds, happened_true = tf.split( remainder, num_or_size_splits = 2, axis = -1 )
        #odds = tf.squeeze( odds, axis = -1 ); happened_true = tf.squeeze( happened_true, axis = -1 )

      
        bet_worth_target = Get_Bet_Worth_Target_v2( odds, happened_true )

        bet_worth = y_pred
      
        mean_squared_error = self.loss_function ( bet_worth_target, bet_worth )
        loss = mean_squared_error

        """
        augment = 1.0 / ( 1.01 - betting_prob_target + 1e-6 )
        loss = loss * augment # Exacerbase the loss at high-profitable games.

        loss = tf.reduce_mean( loss, axis = None )
        """

        #risk = tf.math.maximum( 1.0, - tf.reduce_min(profit_series_acc) * 1.2 )
        #loss5 = loss5 * risk

        #tf.cond( tf.math.is_nan(loss), lambda: tf.print('ERROR 4'), lambda: tf.print(end='') )

        return loss

#=============================================================================

class Loss_Profit_On_Free_Stake(tf.keras.losses.Loss):
    def __init__(self, queries, name='max.profit.on.free.stake'):
        super().__init__(name=name)

        self.queries = queries
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):

        # self.queries is not accessable. so
        #rows, odds, happened_true = tf.split( y_true, num_or_size_splits = [ 1, len(self.queries), len(self.queries) ], axis = -1 )
        rows, _, remainder = tf.split( y_true, num_or_size_splits = [ 1, 2, -1 ], axis = -1 )
        odds, happened_true = tf.split( remainder, num_or_size_splits = 2, axis = -1 )
        #odds = tf.squeeze( odds, axis = -1 ); happened_true = tf.squeeze( happened_true, axis = -1 )
        
        #bet_worth_target = Get_Bet_Worth_Target_V1( config['min_oh_1'], config['thres_bWorth'], config['full_oh_1'], config['full_bWorth'], odds, happened_true )
        #stake_target = Get_Stake_On_Bet_Worth_V1( bet_worth_target, config['max_stake'], config['thres_bWorth'] )
        
        #profit_series_target = Get_Profit_Series( odds, happened_true, bet_stake_target ) :
        #profit_target = Get_Profit( profit_series_target )

        # What to learn: bet_worth_target, stake_target, profit_series_target, profit_target. All of themm has real score outcome implied in them.
        # Anything is equivalent mathmatically, but not in learning performance.

        stake = y_pred

        profit_series = Get_Profit_Series( odds, happened_true, stake )
        loss = - tf.reduce_sum( profit_series )

        return loss