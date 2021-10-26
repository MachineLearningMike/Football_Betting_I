import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from M_BettingMath import *
from M_Config import *

      
class Metrics_Bet_Worth(tf.keras.metrics.Metric):
    nOutputs = 5
    defaultOutput = { 'PL' : -1e7, 'PO': 0., 'TP' : 0., 'FN' : 0., 'NE' : 0., 'TN' : 0.,  'FP' : 0., 'PLMin' : 0. } 

    # slope = 4.0 constant, as sigmoind (5) = 1 (nearly) is the earliest 1 intuitively.
    def __init__(self, query_id, name="PL", center = 20.0, center1 = 10.0, **kwargs):
        super().__init__(name=name, **kwargs)

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
        self.PerformanceIndices = self.GetPerformancePL

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
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def update_state(self, y_true, y_pred, sample_weight=None):
       
        rows, odds, happened = tf.split( y_true, num_or_size_splits = [ 1, 1, 1 ], axis = -1 ) 
        rows = tf.squeeze(rows); odds = tf.squeeze(odds); happened = tf.squeeze(happened)
        y_pred = tf.squeeze(y_pred)
        bet_worth = y_pred

        # bet_worth_target = Get_Bet_Worth_Target_V1( r1, beta, gamma, odds, happened )
        #bet_decision, profit, threshold = Get_Best_Profit ( bet_worth, odds, happened )
      
        #tf.print( tf.reduce_max(bet_worth) )
        #tf.print( tf.reduce_max(bet_decision) )

        #tf.print( 'head', odds[:6] ) #``````````````````````````````````````````````````````````````
        #tf.print( 'tail', odds[-6:] ) 

        bet_decision = Get_Bet_Decision( bet_worth, odds, self.query_id )
        profit_series = Get_Profit_Series( odds, happened, bet_decision )
        profit_series_acc = Get_Profit_Series_Accumulated( profit_series )
        profit = Get_Profit ( profit_series )

        self.PL.assign_add( profit )

        PO = K.sum ( tf.cast(   tf.greater(happened, 0.5),  tf.float32 ) )
        NE = K.sum ( tf.cast(   tf.logical_not( tf.greater(happened, 0.5) ),  tf.float32 ) )

        TP = K.sum ( tf.cast(   tf.math.logical_and(    tf.greater(happened, 0.5),  tf.greater(bet_decision, 0.5) ),  tf.float32 ) )
        TN = K.sum ( tf.cast(   tf.math.logical_and(    tf.logical_not(tf.greater(happened, 0.5)),     tf.logical_not(tf.greater(bet_decision, 0.5))    ),  tf.float32 ) )
        FP = K.sum ( tf.cast(   tf.math.logical_and(    tf.logical_not(tf.greater(happened, 0.5) ),     tf.greater(bet_decision, 0.5) ),  tf.float32 ) )
        FN = K.sum ( tf.cast(   tf.math.logical_and(    tf.greater(happened, 0.5),  tf.logical_not(tf.greater(bet_decision, 0.5))    ),  tf.float32 ) )

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
        #    tf.print( 'profit = 0', bet, TP, TN, FP, FN, self.TP, self.TN, self.FP, self.FN, profit, self.PL )
        #else :
        #    tf.print( 'profit != 0', bet, TP, TN, FP, FN, self.TP, self.TN, self.FP, self.FN, profit, self.PL )

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

    def PrintModelSummary(self, query_id, query, seeking, modelSummary) :   

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
        winningProbPerBet = 1.0 * accuracy / 100 # winning prob per bet.
        fairOdds = 100.0 / ( relevance + epsilon )  # 
        bookieOdds = ( fairOdds - 1.0 ) * ( 1.0 - bookieMargin ) + 1.0 # bookie odds
        betCounts100 = purposeRate100 #  how many times per 100 games we will bet?

        return (  winningProbPerBet * ( bookieOdds - 1.0 ) + ( 1.0 - winningProbPerBet ) * (-1.0) ) * betCounts100

    def IsBetterModel(self, seeking, metrics_new, metrics ) :
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

        performanceIndex = self.PerformanceIndices

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


# ===========================================================================================================================


class Metrics_Bet_Decision(tf.keras.metrics.Metric):
    nOutputs = 5
    defaultOutput = { 'PL' : -1e7, 'PO': 0., 'TP' : 0., 'FN' : 0., 'NE' : 0., 'TN' : 0.,  'FP' : 0., 'PLMin' : 0. } 

    # slope = 4.0 constant, as sigmoind (5) = 1 (nearly) is the earliest 1 intuitively.
    def __init__(self, query_id, name="PL", center = 20.0, center1 = 10.0, **kwargs):
        super().__init__(name=name, **kwargs)

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
        self.PerformanceIndices = self.GetPerformancePL

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
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def update_state(self, y_true, y_pred, sample_weight=None):
       
        rows, odds, happened = tf.split( y_true, num_or_size_splits = [ 1, 1, 1 ], axis = -1 ) 
        rows = tf.squeeze(rows); odds = tf.squeeze(odds); happened = tf.squeeze(happened)
        y_pred = tf.squeeze(y_pred)
        bet_worth = y_pred

        # bet_worth_target = Get_Bet_Worth_Target_V1( r1, beta, gamma, odds, happened )
        #bet_decision, profit, threshold = Get_Best_Profit ( bet_worth, odds, happened )
      
        #tf.print( tf.reduce_max(bet_worth) )
        #tf.print( tf.reduce_max(bet_decision) )

        #tf.print( 'head', odds[:6] ) #``````````````````````````````````````````````````````````````
        #tf.print( 'tail', odds[-6:] ) 

        bet_decision = Get_Bet_Decision( bet_worth, odds, self.query_id )
        profit_series = Get_Profit_Series( odds, happened, bet_decision )
        profit_series_acc = Get_Profit_Series_Accumulated( profit_series )
        profit = Get_Profit ( profit_series )

        self.PL.assign_add( profit )

        PO = K.sum ( tf.cast(   tf.greater(happened, 0.5),  tf.float32 ) )
        NE = K.sum ( tf.cast(   tf.logical_not( tf.greater(happened, 0.5) ),  tf.float32 ) )

        TP = K.sum ( tf.cast(   tf.math.logical_and(    tf.greater(happened, 0.5),  tf.greater(bet_decision, 0.5) ),  tf.float32 ) )
        TN = K.sum ( tf.cast(   tf.math.logical_and(    tf.logical_not(tf.greater(happened, 0.5)),     tf.logical_not(tf.greater(bet_decision, 0.5))    ),  tf.float32 ) )
        FP = K.sum ( tf.cast(   tf.math.logical_and(    tf.logical_not(tf.greater(happened, 0.5) ),     tf.greater(bet_decision, 0.5) ),  tf.float32 ) )
        FN = K.sum ( tf.cast(   tf.math.logical_and(    tf.greater(happened, 0.5),  tf.logical_not(tf.greater(bet_decision, 0.5))    ),  tf.float32 ) )

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
        #    tf.print( 'profit = 0', bet, TP, TN, FP, FN, self.TP, self.TN, self.FP, self.FN, profit, self.PL )
        #else :
        #    tf.print( 'profit != 0', bet, TP, TN, FP, FN, self.TP, self.TN, self.FP, self.FN, profit, self.PL )

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

    def PrintModelSummary(self, query_id, query, seeking, modelSummary) :   

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
        winningProbPerBet = 1.0 * accuracy / 100 # winning prob per bet.
        fairOdds = 100.0 / ( relevance + epsilon )  # 
        bookieOdds = ( fairOdds - 1.0 ) * ( 1.0 - bookieMargin ) + 1.0 # bookie odds
        betCounts100 = purposeRate100 #  how many times per 100 games we will bet?

        return (  winningProbPerBet * ( bookieOdds - 1.0 ) + ( 1.0 - winningProbPerBet ) * (-1.0) ) * betCounts100

    def IsBetterModel(self, seeking, metrics_new, metrics ) :
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

        performanceIndex = self.PerformanceIndices

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



