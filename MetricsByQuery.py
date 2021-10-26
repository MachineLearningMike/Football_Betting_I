import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from M_BettingMath import *
from Config import *


class MetricsClass_TP_TN_FP_FN(tf.keras.metrics.Metric):
    nOutputs = 5
    defaultOutput = { 'seeking' : 'Yes!', 'TP' : 0., 'TN' : 0., 'FP' : 0., 'FN' : 0. } 

    # slope = 4.0 constant, as sigmoind (5) = 1 (nearly) is the earliest 1 intuitively.
    def __init__(self, name="M3", center = 20.0, center1 = 10.0, center2 = 7.5, **kwargs):
        super(MetricsClass_TP_TN_FP_FN, self).__init__(name=name, **kwargs)

        self.TP = self.add_weight(name="TP", initializer="zeros", dtype = tf.float32)
        self.TN = self.add_weight(name="TN", initializer="zeros", dtype = tf.float32)
        self.FP = self.add_weight(name="FP", initializer="zeros", dtype = tf.float32)
        self.FN = self.add_weight(name="FN", initializer="zeros", dtype = tf.float32)

        self.center = center; self.center1 = center1; self.center2 = center2

        self.PerformanceIndices = [ self.GetPerformanceIndex, self.GetPerformanceIndex1, self.GetPerformanceIndex2, self.GetPerformanceIndex3 ]

    def reset_states(self):
        self.TP.assign(0.)
        self.TN.assign(0.)
        self.FP.assign(0.)
        self.FN.assign(0.)

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = {}
        base_config = super(MetricsClass_TP_TN_FP_FN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def update_state(self, y_true, y_pred, sample_weight=None): # list? of y_true for queries. y_ture : float32, shape = [match(batched), 1], value : 1 for FALSE, 0 for TRUE, 0 <= value <= 1

        # No! tf.assert_equal( y_true.shape, y_pred.shape )
        
        true = tf.cast( 1. - y_true, tf.bool )
        pred = tf.less( y_pred, 0.5 )

        TP = K.sum ( tf.cast(             tf.math.logical_and(    true, pred  ),                                 tf.float32 ) )
        TN = K.sum ( tf.cast(             tf.math.logical_not(    tf.math.logical_or( true, pred ) ),            tf.float32 ) )
        FP = K.sum ( tf.cast(             tf.math.logical_and(    tf.math.logical_not( true ), pred ),           tf.float32 ) )
        FN = K.sum ( tf.cast(             tf.math.logical_and(    true,   tf.math.logical_not(pred) ),           tf.float32 ) )

        # No! tf.assert_equal( Total, TP + TN + FP + FN )

        self.TP.assign_add( TP )
        self.TN.assign_add( TN )
        self.FP.assign_add( FP )
        self.FN.assign_add( FN )

    def result(self):
        return  self.TP, self.TN, self.FP, self.FN


    def PackOutputInDict(self, output ) :
        return { 'TP': output[0], 'TN': output[1], 'FP': output[2], 'FN': output[3] }

    def Print (self, query_id, metrics ) :
        total = int( round ( metrics['TP'] + metrics['TN']  +metrics['FP']  + metrics['FN'] ) )

        couples = "Q{:>2} --- Total: {:>4}, TP: {:>4}, TN: {:>4}, FP: {:>4}, FN: {:>4}".format ( \
            query_id, total, int(round(metrics['TP'])), int(round(metrics['TN'])), int(round(metrics['FP'])), int(round(metrics['FN'])) )
        print (couples)

    epsilon = 1e-9

    def PrintModelSummary(self, viewpoint, query_id, query, seeking, modelSummary) :   

        TP = modelSummary['TP']
        TN = modelSummary['TN']
        FP = modelSummary['FP']
        FN = modelSummary['FN']
        Total = TP + TN + FP + FN

        Y_accuracy = int(round(TP*100/(TP+FP+epsilon)))
        Y_coverage = int(round(TP*100/(TP+FN+epsilon)))
        N_accuracy = int(round(TN*100/(TN+FN+epsilon)))
        N_coverage = int(round(TN*100/(TN+FP+epsilon)))
        Positive = int(round((TP+FN)*100/(Total+epsilon)))
        Negative = int(round((TN+FP)*100/(Total+epsilon)))
        Yes_rate = int(round((TP+FP)*100/(Total+epsilon)))
        No_rate = int(round((TN+FN)*100/(Total+epsilon)))

        PL_Y_100G = self.GetPerformanceIndex3(Y_accuracy, Y_coverage, Positive, Yes_rate)
        PL_N_100G = self.GetPerformanceIndex3(N_accuracy, N_coverage, Positive, No_rate)

        text = "Q{:>2} - [ {} ]    Seeking {}   \
            Y - acc cov rel = {:>3} {:>3} {:>3}, PL_100G {:.2f}    \
            N - acc cov rel = {:>3} {:>3} {:>3}, PL_100G {:.2f} ".format ( \
            query_id, query, seeking, \
            #Y_accuracy, Y_coverage, Positive, self.PerformanceIndices[viewpoint](Y_accuracy, Y_coverage, Positive), 1.0*Y_accuracy/100 * (100.0/(Positive+5)-1) - (1.0 - 1.0*Y_accuracy/100), \
            #N_accuracy, N_coverage, Negative, self.PerformanceIndices[viewpoint](N_accuracy, N_coverage, Negative), 1.0*N_accuracy/100 * (100.0/(Negative+5)-1) - (1.0 - 1.0*N_accuracy/100) )
            Y_accuracy, Y_coverage, Positive, PL_Y_100G, \
            N_accuracy, N_coverage, Negative, PL_N_100G )

        print ( text )

    def GetPerformanceIndex(self, accuracy, coverage, relevance, purposeRate100):
        return accuracy * tf.keras.activations.sigmoid( 4.0 * ( ( coverage - self.center ) / self.center ) ) / ( relevance + epsilon )

    def GetPerformanceIndex1(self, accuracy, coverage, relevance, purposeRate100):
        return accuracy * tf.keras.activations.sigmoid( 4.0 * ( ( coverage - self.center1 ) / self.center1 ) ) / ( relevance + epsilon )

    def GetPerformanceIndex2(self, accuracy, coverage, relevance, purposeRate100):
        return accuracy * tf.keras.activations.sigmoid( 4.0 * ( ( coverage - self.center2 ) / self.center2 ) ) / ( relevance + epsilon )

    def GetPerformanceIndex3(self, accuracy, coverage, relevance, purposeRate100):
        """
        Profit for 100 games
        """

        bookieMargin = 0.20
        winningProbPerBetting = 1.0 * accuracy / 100
        fairOdds = 100.0 / ( relevance + epsilon )
        bookieOdds = ( fairOdds - 1.0 ) * ( 1.0 - bookieMargin ) + 1.0
        bettingCounts100 = purposeRate100

        return (  winningProbPerBetting * ( bookieOdds - 1.0 ) + ( 1.0 - winningProbPerBetting ) * (-1.0) ) * bettingCounts100

    def IsBetterModel(self, viewpoint, seeking, metrics_new, metrics ) :
        TP_new = metrics_new['TP']
        TN_new = metrics_new['TN']
        FP_new = metrics_new['FP']
        FN_new = metrics_new['FN']
        Total_new = TP_new + TN_new + FP_new + FN_new

        Y_accuracy_new = TP_new * 100 / (TP_new + FP_new + epsilon)
        Y_coverage_new = TP_new * 100 / (TP_new + FN_new + epsilon)
        N_accuracy_new = TN_new * 100 / (TN_new + FN_new + epsilon)
        N_coverage_new = TN_new * 100 / (TN_new + FP_new + epsilon)
        Positive_new = (TP_new + FN_new) * 100 / (Total_new + epsilon)
        Negative_new = (TN_new + FP_new) * 100 / (Total_new + epsilon)

        TP = metrics['TP']
        TN = metrics['TN']
        FP = metrics['FP']
        FN = metrics['FN']
        Total = TP + TN + FP + FN

        Y_accuracy = TP * 100 / (TP+FP+epsilon)
        Y_coverage = TP * 100 / (TP+FN+epsilon)
        N_accuracy = TN * 100 / (TN+FN+epsilon)
        N_coverage = TN * 100 / (TN+FP+epsilon)
        Positive = (TP+FN) * 100 / (Total+epsilon)
        Negative = (TN+FP) * 100 / (Total+epsilon)

        performanceIndex = self.PerformanceIndices[viewpoint]

        if seeking == 'Yes!' :
            YesRate100_new = 100.0 * (TP_new + FP_new) / ( Total_new + epsilon )
            performance_new = performanceIndex( Y_accuracy_new, Y_coverage_new, Positive_new, YesRate100_new)
            performance_new = tf.cast( performance_new, tf.float32 )

            YesRate100 = 100.0 * (TP + FP) / ( Total + epsilon )
            performance = performanceIndex( Y_accuracy, Y_coverage, Positive, YesRate100)
            performance = tf.cast( performance, tf.float32 )

            return ( performance * (1.0 + 0.001) ) < performance_new, performance, performance_new
 
        elif seeking == 'Noo!' :
            NoRate100_new = 100.0 * (TN_new + FN_new) / ( Total_new + epsilon)
            performance_new = performanceIndex( N_accuracy_new, N_coverage_new, Negative_new, NoRate100_new)
            performance_new = tf.cast( performance_new, tf.float32 )

            NoRate100 = 100.0 * (TN + FN) / ( Total + epsilon)
            performance = performanceIndex( N_accuracy, N_coverage, Negative, NoRate100)
            performance = tf.cast( performance, tf.float32 )

            return ( performance * (1.0 + 0.001) ) < performance_new, performance, performance_new
 
#====================================================================================================================

class MetricsClass_BettingPL(tf.keras.metrics.Metric):
    nOutputs = 5
    defaultOutput = { 'PL' : 0., 'TP' : 0., 'TN' : 0., 'FP' : 0., 'FN' : 0. } 

    # slope = 4.0 constant, as sigmoind (5) = 1 (nearly) is the earliest 1 intuitively.
    def __init__(self, name="PL", bettingThreshold = 0.5, isFixedThreshold = True, center = 20.0, center1 = 10.0, **kwargs):
        super(MetricsClass_BettingPL, self).__init__(name=name, **kwargs)

        self.bettingThreshold = bettingThreshold
        self.isFixedThreshold = isFixedThreshold
        self.center = center; self.center1 = center1

        self.PL = self.add_weight(name="PL", initializer="zeros", dtype = tf.float32)

        self.TP = self.add_weight(name="TP", initializer="zeros", dtype = tf.float32)
        self.TN = self.add_weight(name="TN", initializer="zeros", dtype = tf.float32)
        self.FP = self.add_weight(name="FP", initializer="zeros", dtype = tf.float32)
        self.FN = self.add_weight(name="FN", initializer="zeros", dtype = tf.float32)

        self.PerformanceIndices = [ self.GetPerformanceIndex, self.GetPerformanceIndex1, self.GetPerformanceIndex2, self.GetPerformanceIndex3 ]

    def reset_states(self):
    
        self.PL.assign(0.)        
        self.TP.assign(0.)
        self.TN.assign(0.)
        self.FP.assign(0.)
        self.FN.assign(0.)

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = {}
        base_config = super(MetricsClass_BettingPL, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def update_state(self, y_true, y_pred, sample_weight=None):

        # y_true = [odds, happen]
        # y_pred = sigmoid
       
        odds, happened = tf.split( y_true, num_or_size_splits = [ 1, 1 ], axis = -1 ) 
        probability, dummy = tf.split( y_pred, num_or_size_splits = [ 1, 1 ], axis = -1 )

        if self.isFixedThreshold : # default
            betting = tf.cast( tf.greater( probability, self.bettingThreshold ), tf.float32 )
        else :
            betting = tf.cast( tf.greater( probability, 1.0 / odds ), tf.float32 )

        profit = ( ( odds - 1.0 ) * ( happened ) + ( -1.0 ) * ( 1.0 - happened ) ) * betting
        profit = tf.reduce_sum( profit, axis = None )

        # tf.print('\n--------------', odds, happened, betting, profit) #============================

        self.PL.assign_add( profit )

        # No! tf.assert_equal( y_true.shape, y_pred.shape )

        TP = K.sum ( tf.cast(   tf.math.logical_and(    tf.greater(happened, 0.5),  tf.greater(betting, 0.5) ),  tf.float32 ) )
        TN = K.sum ( tf.cast(   tf.math.logical_and(    tf.logical_not(tf.greater(happened, 0.5)),     tf.logical_not(tf.greater(betting, 0.5))    ),  tf.float32 ) )
        FP = K.sum ( tf.cast(   tf.math.logical_and(    tf.logical_not( tf.greater(happened, 0.5) ),     tf.greater(betting, 0.5) ),  tf.float32 ) )
        FN = K.sum ( tf.cast(   tf.math.logical_and(    tf.greater(happened, 0.5),  tf.logical_not(tf.greater(betting, 0.5))    ),  tf.float32 ) )

        # No! tf.assert_equal( Total, TP + TN + FP + FN )

        self.TP.assign_add( TP )
        self.TN.assign_add( TN )
        self.FP.assign_add( FP )
        self.FN.assign_add( FN )

    def result(self):
        return  self.PL / (self.TP + self.TN + self.FP + self.FN) , self.TP, self.TN, self.FP, self.FN

    def PackOutputInDict(self, output ) :
        return { 'PL': output[0], 'TP': output[1], 'TN': output[2], 'FP': output[3], 'FN': output[4] }

    def Print (self, query_id, metrics ) :
        total = int( round ( metrics['TP'] + metrics['TN']  +metrics['FP']  + metrics['FN'] ) )

        couples = "Q{:>2} --- PL: {:.3f}, Total: {:>4}, TP: {:>4}, TN: {:>4}, FP: {:>4}, FN: {:>4}".format ( \
            query_id, metrics['PL'], total, int(round(metrics['TP'])), int(round(metrics['TN'])), int(round(metrics['FP'])), int(round(metrics['FN'])) )
        print (couples)

    epsilon = 1e-9

    def PrintModelSummary(self, viewpoint, query_id, query, seeking, modelSummary) :   

        PL = modelSummary['PL']

        TP = modelSummary['TP']
        TN = modelSummary['TN']
        FP = modelSummary['FP']
        FN = modelSummary['FN']
        Total = TP + TN + FP + FN

        Y_accuracy = int(round(TP*100/(TP+FP+epsilon)))
        Y_coverage = int(round(TP*100/(TP+FN+epsilon)))
        N_accuracy = int(round(TN*100/(TN+FN+epsilon)))
        N_coverage = int(round(TN*100/(TN+FP+epsilon)))
        Positive = int(round((TP+FN)*100/(Total+epsilon)))
        Negative = int(round((TN+FP)*100/(Total+epsilon)))
        Yes_rate = int(round((TP+FP)*100/(Total+epsilon)))
        No_rate = int(round((TN+FN)*100/(Total+epsilon)))

        PL_Y_100G = self.GetPerformanceIndex3(PL, Y_accuracy, Y_coverage, Positive, Yes_rate)
        PL_N_100G = self.GetPerformanceIndex3(PL, N_accuracy, N_coverage, Positive, No_rate)

        text = "Q{:>2} - [ {} ], PL_1 {:.3f}    Seeking {}   \
            Y - acc cov rel = {:>3} {:>3} {:>3}, PL_100 {:.2f}    \
            N - acc cov rel = {:>3} {:>3} {:>3}, PL_100 {:.2f} ".format ( \
            query_id, query, PL, seeking, \
            Y_accuracy, Y_coverage, Positive, PL_Y_100G, \
            N_accuracy, N_coverage, Negative, PL_N_100G )

        print ( text )

    def GetPerformanceIndex(self, PL, accuracy, coverage, relevance, purposeRate100):
        return accuracy * tf.keras.activations.sigmoid( 4.0 * ( ( coverage - self.center ) / self.center ) ) / ( relevance + epsilon )

    def GetPerformanceIndex1(self, PL, accuracy, coverage, relevance, purposeRate100):
        return accuracy * tf.keras.activations.sigmoid( 4.0 * ( ( coverage - self.center1 ) / self.center1 ) ) / ( relevance + epsilon )

    def GetPerformanceIndex2(self, PL, accuracy, coverage, relevance, purposeRate100):
        return PL

    def GetPerformanceIndex3(self, PL, accuracy, coverage, relevance, purposeRate100):

        bookieMargin = 0.20
        winningProbPerBetting = 1.0 * accuracy / 100
        fairOdds = 100.0 / ( relevance + epsilon )
        bookieOdds = ( fairOdds - 1.0 ) * ( 1.0 - bookieMargin ) + 1.0
        bettingCounts100 = purposeRate100

        return (  winningProbPerBetting * ( bookieOdds - 1.0 ) + ( 1.0 - winningProbPerBetting ) * (-1.0) ) * bettingCounts100

    def IsBetterModel(self, viewpoint, seeking, metrics_new, metrics ) :
        PL_new = metrics_new['PL']
        TP_new = metrics_new['TP']
        TN_new = metrics_new['TN']
        FP_new = metrics_new['FP']
        FN_new = metrics_new['FN']
        Total_new = TP_new + TN_new + FP_new + FN_new

        Y_accuracy_new = TP_new * 100 / (TP_new + FP_new + epsilon)
        Y_coverage_new = TP_new * 100 / (TP_new + FN_new + epsilon)
        N_accuracy_new = TN_new * 100 / (TN_new + FN_new + epsilon)
        N_coverage_new = TN_new * 100 / (TN_new + FP_new + epsilon)
        Positive_new = (TP_new + FN_new) * 100 / (Total_new + epsilon)
        Negative_new = (TN_new + FP_new) * 100 / (Total_new + epsilon)

        PL = metrics['PL']
        TP = metrics['TP']
        TN = metrics['TN']
        FP = metrics['FP']
        FN = metrics['FN']
        Total = TP + TN + FP + FN

        Y_accuracy = TP * 100 / (TP+FP+epsilon)
        Y_coverage = TP * 100 / (TP+FN+epsilon)
        N_accuracy = TN * 100 / (TN+FN+epsilon)
        N_coverage = TN * 100 / (TN+FP+epsilon)
        Positive = (TP+FN) * 100 / (Total+epsilon)
        Negative = (TN+FP) * 100 / (Total+epsilon)

        performanceIndex = self.PerformanceIndices[viewpoint]

        if seeking == 'Yes!' :
            YesRate100_new = 100.0 * (TP_new + FP_new) / ( Total_new + epsilon )
            performance_new = performanceIndex(PL_new, Y_accuracy_new, Y_coverage_new, Positive_new, YesRate100_new)
            performance_new = tf.cast( performance_new, tf.float32 )

            YesRate100 = 100.0 * (TP + FP) / ( Total + epsilon )
            performance = performanceIndex(PL, Y_accuracy, Y_coverage, Positive, YesRate100)
            performance = tf.cast( performance, tf.float32 )

            return ( performance * (1.0 + 0.001) ) < performance_new, performance, performance_new
 
        elif seeking == 'Noo!' :
            NoRate100_new = 100.0 * (TN_new + FN_new) / ( Total_new + epsilon)
            performance_new = performanceIndex(PL_new, N_accuracy_new, N_coverage_new, Negative_new, NoRate100_new)
            performance_new = tf.cast( performance_new, tf.float32 )

            NoRate100 = 100.0 * (TN + FN) / ( Total + epsilon)
            performance = performanceIndex(PL, N_accuracy, N_coverage, Negative, NoRate100)
            performance = tf.cast( performance, tf.float32 )

            return ( performance * (1.0 + 0.001) ) < performance_new, performance, performance_new

#========================================================================================================

class MinusBettingProfit(tf.keras.losses.Loss):
    def __init__(self, name='MBP'):
        super(MinusBettingProfit, self).__init__(name=name)

    def call(self, y_true, y_pred):
        # y_true = [odds, happen]
        # y_pred = sigmoid

        #tf.debugging.assert_all_finite( y_pred, '\n ---------------------- y_pred non in MBP' )

        odds, happened = tf.split( y_true, num_or_size_splits = [ 1, 1 ], axis = -1 )
        bettingProb, dummy = tf.split( y_pred, num_or_size_splits = [ 1, 1 ], axis = -1 )

        profit = - ( ( odds - 1.0 ) * ( happened ) + ( -1.0 ) * ( 1.0 - happened ) ) * bettingProb

        minusBettingProfit = tf.reduce_mean( profit, axis = None )

        return minusBettingProfit


#=============================================================================================================================================================================
       
r1 = 2.0
beta = -0.2
gamma = -0.8


class MetricsClass_BettingPLExt(tf.keras.metrics.Metric):
    nOutputs = 5
    defaultOutput = { 'PL' : -1e7, 'PO': 0., 'TP' : 0., 'FN' : 0., 'NE' : 0., 'TN' : 0.,  'FP' : 0., 'PLMin' : 0. } 

    # slope = 4.0 constant, as sigmoind (5) = 1 (nearly) is the earliest 1 intuitively.
    def __init__(self, query_id, name="PL", center = 20.0, center1 = 10.0, **kwargs):
        super(MetricsClass_BettingPLExt, self).__init__(name=name, **kwargs)

        self.query_id = query_id

        self.center = center; self.center1 = center1

        self.PL = self.add_weight(name="PL", initializer="zeros", dtype = tf.float32)

        self.PO = self.add_weight(name="PO", initializer="zeros", dtype = tf.float32)
        self.NE = self.add_weight(name="NE", initializer="zeros", dtype = tf.float32)

        self.TP = self.add_weight(name="TP", initializer="zeros", dtype = tf.float32)
        self.TN = self.add_weight(name="TN", initializer="zeros", dtype = tf.float32)
        self.FP = self.add_weight(name="FP", initializer="zeros", dtype = tf.float32)
        self.FN = self.add_weight(name="FN", initializer="zeros", dtype = tf.float32)

        self.PLMin = self.add_weight(name="PLMin", initializer="zeros", dtype = tf.float32)
        self.PLLast = self.add_weight(name="PLLast", initializer="zeros", dtype = tf.float32)

        # self.PerformanceIndices = [ self.GetPerformanceIndex, self.GetPerformanceIndex1, self.GetPerformanceIndex2, self.GetPerformanceIndex3 ]
        self.PerformanceIndices = [ self.GetPerformancePL] #, self.GetPerformancePLByRisk ]

    def reset_states(self):
    
        self.PL.assign(0.)

        self.PO.assign(0.)
        self.NE.assign(0.)

        self.TP.assign(0.)
        self.TN.assign(0.)
        self.FP.assign(0.)
        self.FN.assign(0.)

        self.PLMin.assign(0.)
        self.PLLast.assign(0.)

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = {}
        base_config = super(MetricsClass_BettingPLExt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def update_state(self, y_true, y_pred, sample_weight=None):

        #tf.print( y_true[0] )
        #tf.print( y_pred[0] )

        # y_true = [odds, notHappen]
        # y_pred = sigmoidNotHappened
       
        rows, odds, notHappened = tf.split( y_true, num_or_size_splits = [ 1, 1, 1 ], axis = -1 ) 
        rows = tf.squeeze(rows); odds = tf.squeeze(odds); notHappened = tf.squeeze(notHappened)
        y_pred = tf.squeeze(y_pred)
        no_betting_prob = y_pred

        happened = 1.0 - notHappened
        betting_prob = 1.0 - no_betting_prob

        betting_decision = Get_Betting_Decision( betting_prob, odds, self.query_id )
        profit_series = Get_Profit_Series( odds, happened, betting_decision )
        profit_series_acc = Get_Profit_Series_Accumulated( profit_series )
        profit = Get_Profit ( profit_series )

        self.PL.assign_add( profit )

        PO = K.sum ( tf.cast(   tf.greater(happened, 0.5),  tf.float32 ) )
        NE = K.sum ( tf.cast(   tf.logical_not( tf.greater(happened, 0.5) ),  tf.float32 ) )

        TP = K.sum ( tf.cast(   tf.math.logical_and(    tf.greater(happened, 0.5),  tf.greater(betting_decision, 0.5) ),  tf.float32 ) )
        TN = K.sum ( tf.cast(   tf.math.logical_and(    tf.logical_not(tf.greater(happened, 0.5)),     tf.logical_not(tf.greater(betting_decision, 0.5))    ),  tf.float32 ) )
        FP = K.sum ( tf.cast(   tf.math.logical_and(    tf.logical_not(tf.greater(happened, 0.5) ),     tf.greater(betting_decision, 0.5) ),  tf.float32 ) )
        FN = K.sum ( tf.cast(   tf.math.logical_and(    tf.greater(happened, 0.5),  tf.logical_not(tf.greater(betting_decision, 0.5))    ),  tf.float32 ) )

        # No! tf.assert_equal( Total, TP + TN + FP + FN )

        #tf.print( TP, TN, FP, FN )

        self.PO.assign_add( PO )
        self.NE.assign_add( NE )

        self.TP.assign_add( TP )
        self.TN.assign_add( TN )
        self.FP.assign_add( FP )
        self.FN.assign_add( FN )

        self.PLMin.assign( tf.math.minimum( self.PLMin, self.PLLast + tf.reduce_min( profit_series_acc ) ) )
        self.PLLast.assign_add( profit_series_acc[-1] )

        #if profit == 0 :
        #    tf.print( 'profit = 0', betting, TP, TN, FP, FN, self.TP, self.TN, self.FP, self.FN, profit, self.PL )
        #else :
        #    tf.print( 'profit != 0', betting, TP, TN, FP, FN, self.TP, self.TN, self.FP, self.FN, profit, self.PL )

    def result(self):
        #tf.print( self.PL, self.TP + self.TN + self.FP + self.FN )
        return  self.PL / (self.TP + self.TN + self.FP + self.FN) ,     self.PO, self.TP, self.FN,      self.NE, self.TN, self.FP,      self.PLMin

    def PackOutputInDict(self, output ) :
        return { 'PL' : output[0],      'PO': output[1], 'TP' : output[2], 'FN' : output[3],     'NE' : output[4], 'TN' : output[5],  'FP' : output[6], 'PLMin' : output[7] }

    def Print (self, query_id, queryString, metrics ) :
        total = int( round ( metrics['TP'] + metrics['TN']  + metrics['FP']  + metrics['FN'] ) )

        couples = "Q{:>2} {} --- PL: {:7.4f},  Total: {:>4},  PO = TP + FN : {:>4} = {:>4} + {:>4},  NE = TN + FP : {:>4} = {:>4} + {:>4}   PLMin: {:>5}".format ( \
            query_id, queryString, metrics['PL'], total,  int(round(metrics['PO'])), int(round(metrics['TP'])), int(round(metrics['FN'])),\
                int(round(metrics['NE'])), int(round(metrics['TN'])), int(round(metrics['FP'])), int(round(metrics['PLMin'])) )
        print (couples)

    epsilon = 1e-9

    def PrintModelSummary(self, viewpoint, query_id, query, seeking, modelSummary) :   

        PL = modelSummary['PL']

        TP = modelSummary['TP']
        TN = modelSummary['TN']
        FP = modelSummary['FP']
        FN = modelSummary['FN']
        PLMin = modelSummary['PLMin']

        Total = TP + TN + FP + FN

        Y_accuracy = int(round(TP*100/(TP+FP+epsilon)))     # what percents of YES is correct?
        Y_coverage = int(round(TP*100/(TP+FN+epsilon)))     # what percents of Positive is predicted Yes? 
        N_accuracy = int(round(TN*100/(TN+FN+epsilon)))     # what percents of NO is correct?
        N_coverage = int(round(TN*100/(TN+FP+epsilon)))     # what percents of Negative is predicted No?
        Positive = int(round((TP+FN)*100/(Total+epsilon)))  # what percents of Total is Positive? 
        Negative = int(round((TN+FP)*100/(Total+epsilon)))  # what percents of Total is Negative?
        Yes_rate = int(round((TP+FP)*100/(Total+epsilon)))  # what precents of Total is Yes? - relevance
        No_rate = int(round((TN+FN)*100/(Total+epsilon)))   # what percents of Total is No? - relevance
        CorrectPerWrong = (TP + TN ) / (FP + FN + epsilon)

        PL_Y_100G = self.GetPerformancePL(PL, Y_accuracy, Y_coverage, Positive, Yes_rate, PLMin)
        PL_N_100G = self.GetPerformancePL(PL, N_accuracy, N_coverage, Positive, No_rate, PLMin)

        text = "Q{:>2} - [ {} ], PL {:7.4f} Seek {}, PLMin {:>5}, Y - acc cov rel = {:>3} {:>3} {:>3},   N - acc cov rel = {:>3} {:>3} {:>3}, \
            TP FN TN FP = {:>3} {:>3} {:>3} {:>3}".format ( \
            query_id, query, PL, seeking, int(round(PLMin)), \
            Y_accuracy, Y_coverage, Positive, \
            N_accuracy, N_coverage, Negative, \
            int(TP), int(FN), int(TN), int(FP) )

        print ( text )

    def GetPerformanceIndex(self, PL, accuracy, coverage, relevance, purposeRate100, PLMin):
        return accuracy * tf.keras.activations.sigmoid( 4.0 * ( ( coverage - self.center ) / self.center ) ) / ( relevance + epsilon )

    def GetPerformanceIndex1(self, PL, accuracy, coverage, relevance, purposeRate100, PLMin):
        return accuracy * tf.keras.activations.sigmoid( 4.0 * ( ( coverage - self.center1 ) / self.center1 ) ) / ( relevance + epsilon )

    def GetPerformancePLByRisk(self, PL, accuracy, coverage, relevance, purposeRate100, PLMin):
        return PL / max ( 1.0, - PLMin * 1.2 )

    def GetPerformancePL(self, PL, accuracy, coverage, relevance, purposeRate100, PLMin):
        return PL

    def GetPerformanceIndex4(self, PL, accuracy, coverage, relevance, purposeRate100, PLMin):

        bookieMargin = 0.15
        winningProbPerBetting = 1.0 * accuracy / 100 # winning prob per bet.
        fairOdds = 100.0 / ( relevance + epsilon )  # 
        bookieOdds = ( fairOdds - 1.0 ) * ( 1.0 - bookieMargin ) + 1.0 # bookie odds
        bettingCounts100 = purposeRate100 #  how many times per 100 games we will bet?

        return (  winningProbPerBetting * ( bookieOdds - 1.0 ) + ( 1.0 - winningProbPerBetting ) * (-1.0) ) * bettingCounts100

    def IsBetterModel(self, viewpoint, seeking, metrics_new, metrics ) :
        PL_new = metrics_new['PL']
        TP_new = metrics_new['TP']
        TN_new = metrics_new['TN']
        FP_new = metrics_new['FP']
        FN_new = metrics_new['FN']
        PLMin_new = metrics_new['PLMin']
        Total_new = TP_new + TN_new + FP_new + FN_new

        Y_accuracy_new = TP_new * 100 / (TP_new + FP_new + epsilon)
        Y_coverage_new = TP_new * 100 / (TP_new + FN_new + epsilon)
        N_accuracy_new = TN_new * 100 / (TN_new + FN_new + epsilon)
        N_coverage_new = TN_new * 100 / (TN_new + FP_new + epsilon)
        Positive_new = (TP_new + FN_new) * 100 / (Total_new + epsilon)
        Negative_new = (TN_new + FP_new) * 100 / (Total_new + epsilon)

        PL = metrics['PL']
        TP = metrics['TP']
        TN = metrics['TN']
        FP = metrics['FP']
        FN = metrics['FN']
        PLMin = metrics['PLMin']
        Total = TP + TN + FP + FN

        Y_accuracy = TP * 100 / (TP+FP+epsilon)
        Y_coverage = TP * 100 / (TP+FN+epsilon)
        N_accuracy = TN * 100 / (TN+FN+epsilon)
        N_coverage = TN * 100 / (TN+FP+epsilon)
        Positive = (TP+FN) * 100 / (Total+epsilon)
        Negative = (TN+FP) * 100 / (Total+epsilon)

        performanceIndex = self.PerformanceIndices[viewpoint]

        if seeking == 'Yes!' :
            YesRate100_new = 100.0 * (TP_new + FP_new) / ( Total_new + epsilon )
            performance_new = performanceIndex(PL_new, Y_accuracy_new, Y_coverage_new, Positive_new, YesRate100_new, PLMin_new)
            performance_new = tf.cast( performance_new, tf.float32 )

            YesRate100 = 100.0 * (TP + FP) / ( Total + epsilon )
            performance = performanceIndex(PL, Y_accuracy, Y_coverage, Positive, YesRate100, PLMin)
            performance = tf.cast( performance, tf.float32 )

            return ( performance * (1.0 + 1e-8) ) < performance_new, performance, performance_new
 
        elif seeking == 'Noo!' :
            NoRate100_new = 100.0 * (TN_new + FN_new) / ( Total_new + epsilon)
            performance_new = performanceIndex(PL_new, N_accuracy_new, N_coverage_new, Negative_new, NoRate100_new, PLMin_new)
            performance_new = tf.cast( performance_new, tf.float32 )

            NoRate100 = 100.0 * (TN + FN) / ( Total + epsilon)
            performance = performanceIndex(PL, N_accuracy, N_coverage, Negative, NoRate100, PLMin)
            performance = tf.cast( performance, tf.float32 )

            return ( performance * (1.0 + 1e-8) ) < performance_new, performance, performance_new


# =================================


class MinusBettingExt(tf.keras.losses.Loss):
    def __init__(self, query_id, name='MBP'):
        super(MinusBettingExt, self).__init__(name=name)

        self.query_id = query_id

    def call(self, y_true, y_pred):

        rows, odds, notHappened = tf.split( y_true, num_or_size_splits = [ 1, 1, 1 ], axis = -1 )

        rows = tf.squeeze(rows); odds = tf.squeeze(odds); notHappened = tf.squeeze(notHappened)
        y_pred = tf.squeeze(y_pred)
        no_betting_prob = y_pred

        #tf.print( sigmoidNotHappened )

        happened = 1.0 - notHappened
        betting_prob = 1.0 - no_betting_prob

        """
        #happened, betting_prob, betting_decision = Get_Betting_Mix( y_true, y_pred )
        betting = tf.cast( tf.greater( betting_prob, 0.5 ), tf.float32 )
        penalty = tf.cast( tf.logical_and( tf.less(betting, 0.5), tf.greater(happened, 0.5)  ),  tf.float32 ) * 0.01 # no_betting while happened, 1 % penalty to add to losses of profit type.

        profit = ( ( odds - 1.0 ) * ( happened ) + ( -1.0 ) * ( 1.0 - happened ) ) * betting_prob
        loss1 = - tf.reduce_mean( profit, axis = None )

        loss2 = (   (odds - 1.0) * (1.0 - betting_prob)    -    (odds - 1.0) * betting_prob  ) * happened       +      betting_prob * ( 1.0 - happened )
        loss2 = tf.reduce_mean( loss2, axis = None ) # do NOT negate.

        loss3 = tf.keras.losses.binary_crossentropy( happened, betting_prob ) # reduced auto.
        loss3 = tf.reduce_mean( loss3, axis = None )

        profit_happened = ( ( odds - 1.0 ) * ( happened ) + ( -1.0 ) * ( 1.0 - happened ) )
        #profit_happened = ( tf.cast( tf.greater( profit_happened, 0), tf.float32 ) - 0.5 ) * 4.0 # -2 for loss, +2 for profit
        profit_prob = tf.cast( tf.greater( profit_happened, 0), tf.float32 ) # 0 for loss (-1), 1 for profit (odd-1)
        loss4 = tf.keras.losses.binary_crossentropy( profit_prob, betting_prob )
        loss4 = tf.reduce_mean( loss4, axis = None )
        """

        """
        betting_decision = Get_Betting_Decision( betting_prob, odds, self.query_id )
        profit_series = Get_Profit_Series( odds, happened, betting_decision )
        profit_series_acc = Get_Profit_Series_Accumulated( profit_series )
        profit = Get_Profit ( profit_series )
        """

        betting_prob_target = Get_Betting_Prob_Target( r1, beta, gamma, odds, happened )
        loss5 = tf.keras.losses.binary_crossentropy( betting_prob_target, betting_prob )

        #min = tf.reduce_min( betting_prob_target ); max = tf.reduce_max( betting_prob_target ); mean = tf.reduce_mean( betting_prob_target )
        #tf.print( betting_prob_target )

        augment = 1.0 / ( 1.01 - betting_prob_target + 1e-6 )
        loss5 = loss5 * augment

        loss5 = tf.reduce_mean( loss5, axis = None )

        #risk = tf.math.maximum( 1.0, - tf.reduce_min(profit_series_acc) * 1.2 )
        #loss5 = loss5 * risk

        return loss5 # loss3 #loss1 + loss2 + loss3



"""
import tensorflow as tf
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# evenly sampled time at 200ms intervals

r1 = 10.0
beta = -0.2
gamma = -1.0

r0 = beta * r1 / ( 1.0 - beta )
slope = beta / r0
alpha =  - gamma / ( slope * ( 1.0 - r0 ) + 1e-7 )

profit_on_bet = np.arange(-1.5, r1 + 0.5, 0.1)
betting_prob_target = tf.keras.activations.sigmoid( 3.5 * tf.keras.activations.relu( x = slope * ( profit_on_bet + r0 ), alpha = alpha ) )

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_title("prob_target per Profit_on_bet")
ax1.set_ylabel('prob_target')
ax1.set_xlabel('profit_on_bet')
ax1.set_yticks([0.2, 0.5, 0.8], minor=False)
ax1.set_xticks([-1.0, 0.0, 1.0], minor=True)
ax1.yaxis.grid(True, which='major')
ax1.xaxis.grid(True, which='minor')
ax1.grid(True)

ax1.plot( profit_on_bet, betting_prob_target )

plt.show()

"""