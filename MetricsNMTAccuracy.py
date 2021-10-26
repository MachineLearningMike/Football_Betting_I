import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from Config import *


class MetricsNMTAccuracy(tf.keras.metrics.Metric):

    def __init__(self, tokens, name="M7", **kwargs):
        super(MetricsNMTAccuracy, self).__init__(name=name, **kwargs)

        # tokens: [ [ token for Yes answer, token for No answer for query ] for query in queries ]

        tokens = tf.cast( tf.constant(tokens), tf.int32 ) # then: tokens.shape = ( len(queries), 2 )
        self.nQueries = tokens.shape[-2]

        self.positive_tokens, self.negative_tokens = tf.split( tokens, [1, 1], axis = -1 ) # then: [query, token for Yes answer], [query, token for No answer]

        self.TP_p = self.add_weight(name="TP_p", shape = (self.nQueries,), initializer="zeros", dtype = tf.float32)
        self.TN_p = self.add_weight(name="TN_p", shape = (self.nQueries,), initializer="zeros", dtype = tf.float32)
        self.FP_p = self.add_weight(name="FP_p", shape = (self.nQueries,), initializer="zeros", dtype = tf.float32)
        self.FN_p = self.add_weight(name="FN_p", shape = (self.nQueries,), initializer="zeros", dtype = tf.float32)

        self.TP_n = self.add_weight(name="TP_n", shape = (self.nQueries,), initializer="zeros", dtype = tf.float32)
        self.TN_n = self.add_weight(name="TN_n", shape = (self.nQueries,), initializer="zeros", dtype = tf.float32)
        self.FP_n = self.add_weight(name="FP_n", shape = (self.nQueries,), initializer="zeros", dtype = tf.float32)
        self.FN_n = self.add_weight(name="FN_n", shape = (self.nQueries,), initializer="zeros", dtype = tf.float32)

    def reset_states(self):
        self.TP_p.assign( np.zeros( shape = (self.nQueries,), dtype = np.float32 ) )
        self.TN_p.assign( np.zeros( shape = (self.nQueries,), dtype = np.float32 ) )
        self.FP_p.assign( np.zeros( shape = (self.nQueries,), dtype = np.float32 ) )
        self.FN_p.assign( np.zeros( shape = (self.nQueries,), dtype = np.float32 ) )

        self.TP_n.assign( np.zeros( shape = (self.nQueries,), dtype = np.float32 ) )
        self.TN_n.assign( np.zeros( shape = (self.nQueries,), dtype = np.float32 ) )
        self.FP_n.assign( np.zeros( shape = (self.nQueries,), dtype = np.float32 ) )
        self.FN_n.assign( np.zeros( shape = (self.nQueries,), dtype = np.float32 ) )

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = {}
        base_config = super(MetricsNMTAccuracy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast( y_true, tf.int32 )

        y_true, _ = tf.split( y_true, [ y_true.shape[-1] - 1, 1 ], axis = -1 ) # Remove the last element of each example, which is the End.
        y_pred, _ = tf.split( y_pred, [ y_pred.shape[-2] - 1, 1 ], axis = -2 ) # Accordingly, remove the last row of each example, which are logits for the End.

        # _, y_pred, _ = tf.split( y_pred, [1, -1, 1], axis = -1 ) # LINE A. Remove the first and the last row of each example, which are logits for Start and End.

        y_pred = tf.keras.activations.softmax( y_pred, axis = -1 )
        y_pred = tf.cast( tf.argmax( y_pred, axis = -1 ), tf.int32 )

        # y_pred += 1 # LINE B. Couples with LINE A. Put or remove the two lines together. ===================================
       
        positive_tokens = tf.cast( tf.repeat( tf.transpose(self.positive_tokens), y_true.shape[0], axis = -2 ), tf.int32 ) # align with y_true, expand for batch
        negative_tokens = tf.cast( tf.repeat( tf.transpose(self.negative_tokens), y_true.shape[0], axis = -2 ), tf.int32 ) # align with y_true, expand for batch
       
        TP_p = tf.logical_and( tf.equal( y_true, positive_tokens ), tf.equal( y_pred, positive_tokens ) )
        TN_p = tf.logical_and( tf.logical_not( tf.equal( y_true, positive_tokens ) ), tf.logical_not( tf.equal( y_pred, positive_tokens ) ) )
        FP_p = tf.logical_and( tf.logical_not( tf.equal( y_true, positive_tokens ) ), tf.equal( y_pred, positive_tokens ) )
        FN_p = tf.logical_and( tf.equal( y_true, positive_tokens ), tf.logical_not( tf.equal( y_pred, positive_tokens ) ) )

        TP_n = tf.logical_and( tf.equal( y_true, negative_tokens ), tf.equal( y_pred, negative_tokens ) )
        TN_n = tf.logical_and( tf.logical_not( tf.equal( y_true, negative_tokens ) ), tf.logical_not( tf.equal( y_pred, negative_tokens ) ) )
        FP_n = tf.logical_and( tf.logical_not( tf.equal( y_true, negative_tokens ) ), tf.equal( y_pred, negative_tokens ) )
        FN_n = tf.logical_and( tf.equal( y_true, negative_tokens ), tf.logical_not( tf.equal( y_pred, negative_tokens ) ) )

        TP_p = tf.keras.backend.sum( tf.cast(TP_p, tf.float32), axis = 0 )
        TN_p = tf.keras.backend.sum( tf.cast(TN_p, tf.float32), axis = 0 )
        FP_p = tf.keras.backend.sum( tf.cast(FP_p, tf.float32), axis = 0 )
        FN_p = tf.keras.backend.sum( tf.cast(FN_p, tf.float32), axis = 0 )

        TP_n = tf.keras.backend.sum( tf.cast(TP_n, tf.float32), axis = 0 )
        TN_n = tf.keras.backend.sum( tf.cast(TN_n, tf.float32), axis = 0 )
        FP_n = tf.keras.backend.sum( tf.cast(FP_n, tf.float32), axis = 0 )
        FN_n = tf.keras.backend.sum( tf.cast(FN_n, tf.float32), axis = 0 )

        self.TP_p.assign_add( TP_p )
        self.TN_p.assign_add( TN_p )
        self.FP_p.assign_add( FP_p )
        self.FN_p.assign_add( FN_p )

        self.TP_n.assign_add( TP_n )
        self.TN_n.assign_add( TN_n )
        self.FP_n.assign_add( FP_n )
        self.FN_n.assign_add( FN_n )

    def result(self):
        return  (self.TP_p, self.TN_p, self.FP_p, self.FN_p) , (self.TP_n, self.TN_n, self.FP_n, self.FN_n )

    def PackOutputInDict(self, output_p, output_n ) :
        return { 'TP': output_p[0], 'TN': output_p[1], 'FP': output_p[2], 'FN': output_p[3] }

    epsilon = 1e-9

    def Print (self, metrics ) :

        (TP_p, TN_p, FP_p, FN_p) , (TP_n, TN_n, FP_n, FN_n ) = metrics.numpy()

        nQueries = TP_p.shape[-1]

        for query_id in range( nQueries ) :

            total_p = int( TP_p[query_id] + TN_p[query_id] + FP_p[query_id] + FN_p[query_id] )
            total_n = int( TP_n[query_id] + TN_n[query_id] + FP_n[query_id] + FN_n[query_id] )

            Y_accuracy = int(round(TP_p[query_id]*100/(TP_p[query_id]+FP_p[query_id]+epsilon)))
            Y_coverage = int(round(TP_p[query_id]*100/(TP_p[query_id]+FN_p[query_id]+epsilon)))

            N_accuracy = int(round(TP_n[query_id]*100/(TP_n[query_id]+FP_n[query_id]+epsilon)))
            N_coverage = int(round(TP_n[query_id]*100/(TP_n[query_id]+FN_n[query_id]+epsilon)))

            Y_rate = int(round((TP_p[query_id]+FN_p[query_id])*100/(total_p+epsilon)))
            N_rate = int(round((TP_n[query_id]+FN_n[query_id])*100/(total_n+epsilon)))

            couples = "localQ{:>2} --- Y-acc Y-cov Y-rate: {:>3} {:>3} {:>3},     N-acc N-cov N-rate {:>3} {:>3} {:>3}".format ( \
                query_id, Y_accuracy, Y_coverage, Y_rate, N_accuracy, N_coverage, N_rate  )

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

        text = "Q{:>2} - [ {} ]    Seeking {}   \
            Y - acc cov rel, per = {:>3} {:>3} {:>3},   {:.2f}    \
            N - acc cov rel, per = {:>3} {:>3} {:>3},   {:.2f} ".format ( \
            query_id, query, seeking, \
            Y_accuracy, Y_coverage, Positive, self.PerformanceIndices[viewpoint](Y_accuracy, Y_coverage, Positive), \
            N_accuracy, N_coverage, Negative, self.PerformanceIndices[viewpoint](N_accuracy, N_coverage, Negative) )
        print ( text )

    def GetPerformanceIndex(self, accuracy, coverage, relevance ) :
        return accuracy * tf.keras.activations.sigmoid( 4.0 * ( ( coverage - self.center ) / self.center ) ) / ( relevance + epsilon )

    def GetPerformanceIndex1(self, accuracy, coverage, relevance ) :
        return accuracy * tf.keras.activations.sigmoid( 4.0 * ( ( coverage - self.center1 ) / self.center1 ) ) / ( relevance + epsilon )

    def GetPerformanceIndex2(self, accuracy, coverage, relevance ) :
        return accuracy * tf.keras.activations.sigmoid( 4.0 * ( ( coverage - self.center2 ) / self.center2 ) ) / ( relevance + epsilon )

    def IsBetterModel(self, viewpoint, seeking, metrics_new, metrics ) :
        TP_new = metrics_new['TP']
        TN_new = metrics_new['TN']
        FP_new = metrics_new['FP']
        FN_new = metrics_new['FN']
        Total_new = TP_new + TN_new + FP_new + FN_new

        Y_accuracy_new = int(round( TP_new * 100 / (TP_new + FP_new + epsilon) ))
        Y_coverage_new = int(round( TP_new * 100 / (TP_new + FN_new + epsilon) ))
        N_accuracy_new = int(round( TN_new * 100 / (TN_new + FN_new + epsilon) ))
        N_coverage_new = int(round( TN_new * 100 / (TN_new + FP_new + epsilon) ))
        Positive_new = int(round( (TP_new + FN_new) * 100 / (Total_new + epsilon) ))
        Negative_new = int(round( (TN_new + FP_new) * 100 / (Total_new + epsilon) ))

        TP = metrics['TP']
        TN = metrics['TN']
        FP = metrics['FP']
        FN = metrics['FN']
        Total = TP + TN + FP + FN

        Y_accuracy = int(round( TP * 100 / (TP+FP+epsilon) ))
        Y_coverage = int(round( TP * 100 / (TP+FN+epsilon) ))
        N_accuracy = int(round( TN * 100 / (TN+FN+epsilon) ))
        N_coverage = int(round( TN * 100 / (TN+FP+epsilon) ))
        Positive = int(round( (TP+FN) * 100 / (Total+epsilon) ))
        Negative = int(round( (TN+FP) * 100 / (Total+epsilon) ))

        performanceIndex = self.PerformanceIndices[viewpoint]

        if seeking == 'Yes!' :
            performance_new = performanceIndex( Y_accuracy_new, Y_coverage_new, Positive_new)
            performance = performanceIndex( Y_accuracy, Y_coverage, Positive )
            return performance + 0.0001 < performance_new, performance, performance_new
 
        elif seeking == 'Noo!' :
            performance_new = performanceIndex( N_accuracy_new, N_coverage_new, Negative_new)
            performance = performanceIndex( N_accuracy, N_coverage, Negative )
            return performance + 0.0001 < performance_new, performance, performance_new
 
