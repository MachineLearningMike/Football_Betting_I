import tensorflow as tf
import tensorflow.keras.backend as K

class ThomasQueries(tf.keras.metrics.Metric):

    def __init__(self, name="ThomasQueries", **kwargs):
        super(ThomasQueries, self).__init__(name=name, **kwargs)

        self.nMatches = self.add_weight(name="nMatches", initializer="zeros", dtype = tf.float32)

        self.home_over_0_goals_equality = self.add_weight(name="home_over_0_goals_equality", initializer="zeros", dtype = tf.float32)
        self.away_over_0_goals_equality = self.add_weight(name="away_over_0_goals_equality", initializer="zeros", dtype = tf.float32)
        self.both_over_0_goals_equality = self.add_weight(name="both_over_0_goals_equality", initializer="zeros", dtype = tf.float32)
        self.under_2_teams_over_0_goals_equality = self.add_weight(name="under_2_teams_over_0_goals_equality", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_2_goals_equality = self.add_weight(name="over_0_teams_over_2_goals_equality", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_under_3_goals_equality = self.add_weight(name="over_0_teams_under_3_goals_equality", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_1_goals_equality = self.add_weight(name="over_0_teams_over_1_goals_equality", initializer="zeros", dtype = tf.float32)
        self.home_win_equality = self.add_weight(name="home_win_equality", initializer="zeros", dtype = tf.float32)
        self.away_win_equality = self.add_weight(name="away_win_equality", initializer="zeros", dtype = tf.float32)
        self.draw_equality = self.add_weight(name="draw_equality", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_equality = self.add_weight(name="true_pred_matching_equality", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_0_equality = self.add_weight(name="true_pred_matching_1_0_equality", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_1_equality = self.add_weight(name="true_pred_matching_1_1_equality", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_2_1_equality = self.add_weight(name="true_pred_matching_2_1_equality", initializer="zeros", dtype = tf.float32)

        self.home_over_0_goals_precision = self.add_weight(name="home_over_0_goals_precision", initializer="zeros", dtype = tf.float32)
        self.away_over_0_goals_precision = self.add_weight(name="away_over_0_goals_precision", initializer="zeros", dtype = tf.float32)
        self.both_over_0_goals_precision = self.add_weight(name="both_over_0_goals_precision", initializer="zeros", dtype = tf.float32)
        self.under_2_teams_over_0_goals_precision = self.add_weight(name="under_2_teams_over_0_goals_precision", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_2_goals_precision = self.add_weight(name="over_0_teams_over_2_goals_precision", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_under_3_goals_precision = self.add_weight(name="over_0_teams_under_3_goals_precision", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_1_goals_precision = self.add_weight(name="over_0_teams_over_1_goals_precision", initializer="zeros", dtype = tf.float32)
        self.home_win_precision = self.add_weight(name="home_win_precision", initializer="zeros", dtype = tf.float32)
        self.away_win_precision = self.add_weight(name="away_win_precision", initializer="zeros", dtype = tf.float32)
        self.draw_precision = self.add_weight(name="draw_precision", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_precision = self.add_weight(name="true_pred_matching_precision", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_0_precision = self.add_weight(name="true_pred_matching_1_0_precision", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_1_precision = self.add_weight(name="true_pred_matching_1_1_precision", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_2_1_precision = self.add_weight(name="true_pred_matching_2_1_precision", initializer="zeros", dtype = tf.float32)

        self.home_over_0_goals_denominator = self.add_weight(name="home_over_0_goals_denominator", initializer="zeros", dtype = tf.float32)
        self.away_over_0_goals_denominator = self.add_weight(name="away_over_0_goals_denominator", initializer="zeros", dtype = tf.float32)
        self.both_over_0_goals_denominator = self.add_weight(name="both_over_0_goals_denominator", initializer="zeros", dtype = tf.float32)
        self.under_2_teams_over_0_goals_denominator = self.add_weight(name="under_2_teams_over_0_goals_denominator", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_2_goals_denominator = self.add_weight(name="over_0_teams_over_2_goals_denominator", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_under_3_goals_denominator = self.add_weight(name="over_0_teams_under_3_goals_denominator", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_1_goals_denominator = self.add_weight(name="over_0_teams_over_1_goals_denominator", initializer="zeros", dtype = tf.float32)
        self.home_win_denominator = self.add_weight(name="home_win_denominator", initializer="zeros", dtype = tf.float32)
        self.away_win_denominator = self.add_weight(name="away_win_denominator", initializer="zeros", dtype = tf.float32)
        self.draw_denominator = self.add_weight(name="draw_denominator", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_denominator = self.add_weight(name="true_pred_matching_denominator", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_0_denominator = self.add_weight(name="true_pred_matching_1_0_denominator", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_1_denominator = self.add_weight(name="true_pred_matching_1_1_denominator", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_2_1_denominator = self.add_weight(name="true_pred_matching_2_1_denominator", initializer="zeros", dtype = tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        
        #tf.config.experimental_run_functions_eagerly(True)
        #tf.print("++++++++", zip( y_pred.shape, y_true, y_pred ) )
        nMathes = K.sum( K.ones_like( y_true[0]) )
        #print(K.ones_like(y_true[0]), nMathes)
        self.nMatches.assign_add( nMathes )
        #tf.config.experimental_run_functions_eagerly(False)

        y_pred = tf.math.round( y_pred )

        true = tf.cast(y_true, tf.int32)
        pred = tf.cast(y_pred, tf.int32)
        # -----------------------------------------  Basic Zone ---------------------------------------------------

        # true.shape
        true_over_0_goals = K.greater( true, K.ones_like(true) * 0 )
        pred_over_0_goals = K.greater( pred, K.ones_like(pred) * 0 )
        true_over_1_goals = K.greater( true, K.ones_like(true) * 1 )
        pred_over_1_goals = K.greater( pred, K.ones_like(pred) * 1 )
        true_over_2_goals = K.greater( true, K.ones_like(true) * 2 )
        pred_over_2_goals = K.greater( pred, K.ones_like(pred) * 2 )
        true_under_3_goals = K.less( true, K.ones_like(true) * 3 )
        pred_under_3_goals = K.less( pred, K.ones_like(pred) * 3 )
        # true[0].shape
        true_both_over_0_goals = tf.logical_and( true_over_0_goals[0], true_over_0_goals[1] )
        pred_both_over_0_goals = tf.logical_and( pred_over_0_goals[0], pred_over_0_goals[1] )
        true_under_2_teams_over_0_goals = tf.logical_not( tf.logical_and( true_over_0_goals[0], true_over_0_goals[1] ) )
        pred_under_2_teams_over_0_goals = tf.logical_not( tf.logical_and( pred_over_0_goals[0], pred_over_0_goals[1] ) )
        true_over_0_teams_over_2_goals = tf.logical_or( true_over_2_goals[0], true_over_2_goals[1] )
        pred_over_0_teams_over_2_goals = tf.logical_or( pred_over_2_goals[0], pred_over_2_goals[1] )
        true_over_0_teams_under_3_goals = tf.math.logical_or( true_under_3_goals[0], true_under_3_goals[1] )
        pred_over_0_teams_under_3_goals = tf.math.logical_or( pred_under_3_goals[0], pred_under_3_goals[1] )
        true_over_0_teams_over_1_goals = tf.math.logical_or( true_over_1_goals[0], true_over_1_goals[1]  )
        pred_over_0_teams_over_1_goals = tf.math.logical_or( pred_over_1_goals[0], pred_over_1_goals[1]  )
        true_home_win = K.greater( true[0], true[1] )
        pred_home_win = K.greater( pred[0], pred[1] )
        true_away_win = K.less( true[0], true[1] )
        pred_away_win = K.less( pred[0], pred[1] )
        true_draw = K.equal( true[0], true[1] )
        pred_draw = K.equal( pred[0], pred[1] )
        true_pred_matching = tf.math.logical_and( K.equal( true[0], pred[0] ), K.equal( true[1], pred[1] ) )
        true_1_0 = tf.math.logical_and( K.equal( true[0], 1 ), K.equal( true[1], 0 ) )
        pred_1_0 = tf.math.logical_and( K.equal( pred[0], 1 ), K.equal( pred[1], 0 ) )
        true_1_1 = tf.math.logical_and( K.equal( true[0], 1 ), K.equal( true[1], 1 ) )
        pred_1_1 = tf.math.logical_and( K.equal( pred[0], 1 ), K.equal( pred[1], 1 ) )
        true_2_1 = tf.math.logical_and( K.equal( true[0], 2 ), K.equal( true[1], 1 ) )
        pred_2_1 = tf.math.logical_and( K.equal( pred[0], 2 ), K.equal( pred[1], 1 ) )

        # -----------------------------------------  Eqaulity Zone ---------------------------------------------------
 
        # shape = ()
        home_over_0_goals_sum_equality = K.sum(K.cast(K.equal(true_over_0_goals[0], pred_over_0_goals[0]), dtype=tf.float32))
        away_over_0_goals_sum_equality = K.sum(K.cast(K.equal(true_over_0_goals[1], pred_over_0_goals[1]), dtype=tf.float32))
        both_over_0_goals_sum_equality = K.sum(K.cast(K.equal(true_both_over_0_goals, pred_both_over_0_goals), dtype=tf.float32))
        under_2_teams_over_0_goals_sum_equality = K.sum(K.cast(K.equal(true_under_2_teams_over_0_goals, pred_under_2_teams_over_0_goals), dtype=tf.float32))
        over_0_teams_over_2_goals_sum_equality = K.sum(K.cast(K.equal(true_over_0_teams_over_2_goals, pred_over_0_teams_over_2_goals), dtype=tf.float32))
        over_0_teams_under_3_goals_sum_equality = K.sum(K.cast(K.equal(true_over_0_teams_under_3_goals, pred_over_0_teams_under_3_goals), dtype=tf.float32))
        over_0_teams_over_1_goals_sum_equality = K.sum(K.cast(K.equal(true_over_0_teams_over_1_goals, pred_over_0_teams_over_1_goals), dtype=tf.float32))
        home_win_sum_equality = K.sum(K.cast(K.equal(true_home_win, pred_home_win), dtype=tf.float32))
        away_win_sum_equality = K.sum(K.cast(K.equal(true_away_win, pred_away_win), dtype=tf.float32))
        draw_sum_equality = K.sum(K.cast(K.equal(true_draw, pred_draw), dtype=tf.float32))
        true_pred_matching_sum_equality = K.sum(K.cast(true_pred_matching, dtype=tf.float32))
        true_pred_matching_1_0_sum_equality = K.sum(K.cast(K.equal(true_1_0, pred_1_0), dtype=tf.float32))
        true_pred_matching_1_1_sum_equality = K.sum(K.cast(K.equal(true_1_1, pred_1_1), dtype=tf.float32))
        true_pred_matching_2_1_sum_equality = K.sum(K.cast(K.equal(true_2_1, pred_2_1), dtype=tf.float32))

        self.home_over_0_goals_equality.assign_add(home_over_0_goals_sum_equality)
        self.away_over_0_goals_equality.assign_add(away_over_0_goals_sum_equality)
        self.both_over_0_goals_equality.assign_add(both_over_0_goals_sum_equality)
        self.under_2_teams_over_0_goals_equality.assign_add( under_2_teams_over_0_goals_sum_equality)
        self.over_0_teams_over_2_goals_equality.assign_add(over_0_teams_over_2_goals_sum_equality)
        self.over_0_teams_under_3_goals_equality.assign_add(over_0_teams_under_3_goals_sum_equality)
        self.over_0_teams_over_1_goals_equality.assign_add(over_0_teams_over_1_goals_sum_equality)
        self.home_win_equality.assign_add( home_win_sum_equality)
        self.away_win_equality.assign_add( away_win_sum_equality)
        self.draw_equality.assign_add( draw_sum_equality)
        self.true_pred_matching_equality.assign_add( true_pred_matching_sum_equality)
        self.true_pred_matching_1_0_equality.assign_add( true_pred_matching_1_0_sum_equality)
        self.true_pred_matching_1_1_equality.assign_add( true_pred_matching_1_1_sum_equality)
        self.true_pred_matching_2_1_equality.assign_add( true_pred_matching_2_1_sum_equality)


        # -----------------------------------------  Precision Zone ---------------------------------------------------

        # shape = ()
        home_over_0_goals_sum_precision = K.sum(K.cast(tf.math.logical_and(true_over_0_goals[0], pred_over_0_goals[0]), dtype=tf.float32))
        away_over_0_goals_sum_precision = K.sum(K.cast(tf.math.logical_and(true_over_0_goals[1], pred_over_0_goals[1]), dtype=tf.float32))
        both_over_0_goals_sum_precision = K.sum( K.cast(tf.math.logical_and(true_both_over_0_goals, pred_both_over_0_goals), dtype=tf.float32))
        under_2_teams_over_0_goals_sum_precision = K.sum(K.cast(tf.math.logical_and(true_under_2_teams_over_0_goals, pred_under_2_teams_over_0_goals), dtype=tf.float32))
        over_0_teams_over_2_goals_sum_precision = K.sum(K.cast(tf.math.logical_and(true_over_0_teams_over_2_goals, pred_over_0_teams_over_2_goals), dtype=tf.float32))
        over_0_teams_under_3_goals_sum_precision = K.sum(K.cast(tf.math.logical_and(true_over_0_teams_under_3_goals, pred_over_0_teams_under_3_goals), dtype=tf.float32))
        over_0_teams_over_1_goals_sum_precision = K.sum(K.cast(tf.math.logical_and(true_over_0_teams_over_1_goals, pred_over_0_teams_over_1_goals), dtype=tf.float32))
        home_win_sum_precision = K.sum( K.cast(tf.math.logical_and(true_home_win, pred_home_win), dtype=tf.float32))
        away_win_sum_precision = K.sum( K.cast(tf.math.logical_and(true_away_win, pred_away_win), dtype=tf.float32))
        draw_sum_precision = K.sum( K.cast(tf.math.logical_and(true_draw, pred_draw), dtype=tf.float32))  # -----------------------------
        true_pred_matching_sum_precision = K.sum(K.cast(true_pred_matching, dtype=tf.float32))
        true_pred_matching_1_0_sum_precision = K.sum(K.cast(tf.math.logical_and(true_1_0, pred_1_0), dtype=tf.float32))
        true_pred_matching_1_1_sum_precision = K.sum(K.cast(tf.math.logical_and(true_1_1, pred_1_1), dtype=tf.float32))
        true_pred_matching_2_1_sum_precision = K.sum(K.cast(tf.math.logical_and(true_2_1, pred_2_1), dtype=tf.float32))

        self.home_over_0_goals_precision.assign_add(home_over_0_goals_sum_precision)
        self.away_over_0_goals_precision.assign_add(away_over_0_goals_sum_precision)
        self.both_over_0_goals_precision.assign_add(both_over_0_goals_sum_precision)
        self.under_2_teams_over_0_goals_precision.assign_add( under_2_teams_over_0_goals_sum_precision)
        self.over_0_teams_over_2_goals_precision.assign_add(over_0_teams_over_2_goals_sum_precision)
        self.over_0_teams_under_3_goals_precision.assign_add(over_0_teams_under_3_goals_sum_precision)
        self.over_0_teams_over_1_goals_precision.assign_add(over_0_teams_over_1_goals_sum_precision)
        self.home_win_precision.assign_add( home_win_sum_precision)
        self.away_win_precision.assign_add( away_win_sum_precision)
        self.draw_precision.assign_add( draw_sum_precision)
        self.true_pred_matching_precision.assign_add( true_pred_matching_sum_precision)
        self.true_pred_matching_1_0_precision.assign_add(true_pred_matching_1_0_sum_precision)
        self.true_pred_matching_1_1_precision.assign_add(true_pred_matching_1_1_sum_precision)
        self.true_pred_matching_2_1_precision.assign_add(true_pred_matching_2_1_sum_precision)


        # shape = ()
        home_over_0_goals_sum_denominator = K.sum(K.cast(pred_over_0_goals[0], dtype=tf.float32))
        away_over_0_goals_sum_denominator = K.sum(K.cast(pred_over_0_goals[1], dtype=tf.float32))
        both_over_0_goals_sum_denominator = K.sum( K.cast(pred_both_over_0_goals, dtype=tf.float32))
        under_2_teams_over_0_goals_sum_denominator = K.sum(K.cast(pred_under_2_teams_over_0_goals, dtype=tf.float32))
        over_0_teams_over_2_goals_sum_denominator = K.sum(K.cast(pred_over_0_teams_over_2_goals, dtype=tf.float32))
        over_0_teams_under_3_goals_sum_denominator = K.sum(K.cast(pred_over_0_teams_under_3_goals, dtype=tf.float32))
        over_0_teams_over_1_goals_sum_denominator = K.sum(K.cast(pred_over_0_teams_over_1_goals, dtype=tf.float32))
        home_win_sum_precision_denominator = K.sum(K.cast(pred_home_win, dtype=tf.float32))
        away_win_sum_denominator = K.sum( K.cast(pred_away_win, dtype=tf.float32))
        draw_sum_denominator = K.sum( K.cast(pred_draw, dtype=tf.float32))
        true_pred_matching_sum_denominator = K.sum(K.cast(true_pred_matching, dtype=tf.float32))
        true_pred_matching_1_0_sum_denominator = K.sum(K.cast(pred_1_0, dtype=tf.float32))
        true_pred_matching_1_1_sum_denominator = K.sum(K.cast(pred_1_1, dtype=tf.float32))
        true_pred_matching_2_1_sum_denominator = K.sum(K.cast(pred_2_1, dtype=tf.float32))

        self.home_over_0_goals_denominator.assign_add(home_over_0_goals_sum_denominator)
        self.away_over_0_goals_denominator.assign_add(away_over_0_goals_sum_denominator)
        self.both_over_0_goals_denominator.assign_add(both_over_0_goals_sum_denominator)
        self.under_2_teams_over_0_goals_denominator.assign_add(under_2_teams_over_0_goals_sum_denominator)
        self.over_0_teams_over_2_goals_denominator.assign_add(over_0_teams_over_2_goals_sum_denominator)
        self.over_0_teams_under_3_goals_denominator.assign_add(over_0_teams_under_3_goals_sum_denominator)
        self.over_0_teams_over_1_goals_denominator.assign_add(over_0_teams_over_1_goals_sum_denominator)
        self.home_win_denominator.assign_add(home_win_sum_precision_denominator)
        self.away_win_denominator.assign_add(away_win_sum_denominator)
        self.draw_denominator.assign_add(draw_sum_denominator)
        self.true_pred_matching_denominator.assign_add(true_pred_matching_sum_denominator)
        self.true_pred_matching_1_0_denominator.assign_add(true_pred_matching_1_0_sum_denominator)
        self.true_pred_matching_1_1_denominator.assign_add(true_pred_matching_1_1_sum_denominator)
        self.true_pred_matching_2_1_denominator.assign_add(true_pred_matching_2_1_sum_denominator)




    def result(self):

        epsilon = 1e-8

        return \
            tf.round( self.home_over_0_goals_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.home_over_0_goals_precision * 100 / (self.home_over_0_goals_denominator + epsilon) ), \
            tf.round( self.away_over_0_goals_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.away_over_0_goals_precision * 100 / (self.away_over_0_goals_denominator + epsilon) ), \
            tf.round( self.both_over_0_goals_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.both_over_0_goals_precision * 100 / (self.both_over_0_goals_denominator + epsilon) ), \
            tf.round(self.under_2_teams_over_0_goals_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.under_2_teams_over_0_goals_precision * 100 / (self.under_2_teams_over_0_goals_denominator + epsilon) ), \
            tf.round( self.over_0_teams_over_2_goals_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.over_0_teams_over_2_goals_precision * 100 / (self.over_0_teams_over_2_goals_denominator + epsilon) ), \
            tf.round( self.over_0_teams_under_3_goals_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.over_0_teams_under_3_goals_precision * 100 / (self.over_0_teams_under_3_goals_denominator + epsilon) ), \
            tf.round( self.over_0_teams_over_1_goals_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.over_0_teams_over_1_goals_precision * 100 / (self.over_0_teams_over_1_goals_denominator + epsilon) ), \
            tf.round( self.home_win_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.home_win_precision * 100 / (self.home_win_denominator + epsilon) ), \
            tf.round( self.away_win_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.away_win_precision * 100 / (self.away_win_denominator + epsilon) ), \
            tf.round( self.draw_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.draw_precision * 100 / (self.draw_denominator + epsilon) ), \
            tf.round( self.true_pred_matching_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.true_pred_matching_precision * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.true_pred_matching_1_0_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.true_pred_matching_1_0_precision * 100 / (self.true_pred_matching_1_0_denominator + epsilon) ), \
            tf.round( self.true_pred_matching_1_1_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.true_pred_matching_1_1_precision * 100 / (self.true_pred_matching_1_1_denominator + epsilon) ), \
            tf.round( self.true_pred_matching_2_1_equality * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.true_pred_matching_2_1_precision * 100 / (self.true_pred_matching_2_1_denominator + epsilon) ), \
