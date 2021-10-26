import tensorflow as tf
import tensorflow.keras.backend as K

class ThomasQueriesYesNo(tf.keras.metrics.Metric):

    def __init__(self, name="ThomasQueriesYesNo", **kwargs):
        super(ThomasQueriesYesNo, self).__init__(name=name, **kwargs)

        self.nMatches = self.add_weight(name="nMatches", initializer="zeros", dtype = tf.float32)

        self.home_over_0_goals_y = self.add_weight(name="home_over_0_goals_y", initializer="zeros", dtype = tf.float32)
        self.away_over_0_goals_y = self.add_weight(name="away_over_0_goals_y", initializer="zeros", dtype = tf.float32)
        self.both_over_0_goals_y = self.add_weight(name="both_over_0_goals_y", initializer="zeros", dtype = tf.float32)
        self.under_2_teams_over_0_goals_y = self.add_weight(name="under_2_teams_over_0_goals_y", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_2_goals_y = self.add_weight(name="over_0_teams_over_2_goals_y", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_under_3_goals_y = self.add_weight(name="over_0_teams_under_3_goals_y", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_1_goals_y = self.add_weight(name="over_0_teams_over_1_goals_y", initializer="zeros", dtype = tf.float32)
        self.home_win_y = self.add_weight(name="home_win_y", initializer="zeros", dtype = tf.float32)
        self.away_win_y = self.add_weight(name="away_win_y", initializer="zeros", dtype = tf.float32)
        self.draw_y = self.add_weight(name="draw_y", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_y = self.add_weight(name="true_pred_matching_y", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_0_y = self.add_weight(name="true_pred_matching_1_0_y", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_1_y = self.add_weight(name="true_pred_matching_1_1_y", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_2_1_y = self.add_weight(name="true_pred_matching_2_1_y", initializer="zeros", dtype = tf.float32)

        self.home_over_0_goals_y_count = self.add_weight(name="home_over_0_goals_y_count", initializer="zeros", dtype = tf.float32)
        self.away_over_0_goals_y_count = self.add_weight(name="away_over_0_goals_y_count", initializer="zeros", dtype = tf.float32)
        self.both_over_0_goals_y_count = self.add_weight(name="both_over_0_goals_y_count", initializer="zeros", dtype = tf.float32)
        self.under_2_teams_over_0_goals_y_count = self.add_weight(name="under_2_teams_over_0_goals_y_count", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_2_goals_y_count = self.add_weight(name="over_0_teams_over_2_goals_y_count", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_under_3_goals_y_count = self.add_weight(name="over_0_teams_under_3_goals_y_count", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_1_goals_y_count = self.add_weight(name="over_0_teams_over_1_goals_y_count", initializer="zeros", dtype = tf.float32)
        self.home_win_y_count = self.add_weight(name="home_win_y_count", initializer="zeros", dtype = tf.float32)
        self.away_win_y_count = self.add_weight(name="away_win_y_count", initializer="zeros", dtype = tf.float32)
        self.draw_y_count = self.add_weight(name="draw_y_count", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_y_count = self.add_weight(name="true_pred_matching_y_count", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_0_y_count = self.add_weight(name="true_pred_matching_1_0_y_count", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_1_y_count = self.add_weight(name="true_pred_matching_1_1_y_count", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_2_1_y_count = self.add_weight(name="true_pred_matching_2_1_y_count", initializer="zeros", dtype = tf.float32)


        self.home_over_0_goals_n = self.add_weight(name="home_over_0_goals_n", initializer="zeros", dtype = tf.float32)
        self.away_over_0_goals_n = self.add_weight(name="away_over_0_goals_n", initializer="zeros", dtype = tf.float32)
        self.both_over_0_goals_n = self.add_weight(name="both_over_0_goals_n", initializer="zeros", dtype = tf.float32)
        self.under_2_teams_over_0_goals_n = self.add_weight(name="under_2_teams_over_0_goals_n", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_2_goals_n = self.add_weight(name="over_0_teams_over_2_goals_n", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_under_3_goals_n = self.add_weight(name="over_0_teams_under_3_goals_n", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_1_goals_n = self.add_weight(name="over_0_teams_over_1_goals_n", initializer="zeros", dtype = tf.float32)
        self.home_win_n = self.add_weight(name="home_win_n", initializer="zeros", dtype = tf.float32)
        self.away_win_n = self.add_weight(name="away_win_n", initializer="zeros", dtype = tf.float32)
        self.draw_n = self.add_weight(name="draw_n", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_n = self.add_weight(name="true_pred_matching_n", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_0_n = self.add_weight(name="true_pred_matching_1_0_n", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_1_n = self.add_weight(name="true_pred_matching_1_1_n", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_2_1_n = self.add_weight(name="true_pred_matching_2_1_n", initializer="zeros", dtype = tf.float32)

        self.home_over_0_goals_n_count = self.add_weight(name="home_over_0_goals_n_count", initializer="zeros", dtype = tf.float32)
        self.away_over_0_goals_n_count = self.add_weight(name="away_over_0_goals_n_count", initializer="zeros", dtype = tf.float32)
        self.both_over_0_goals_n_count = self.add_weight(name="both_over_0_goals_n_count", initializer="zeros", dtype = tf.float32)
        self.under_2_teams_over_0_goals_n_count = self.add_weight(name="under_2_teams_over_0_goals_n_count", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_2_goals_n_count = self.add_weight(name="over_0_teams_over_2_goals_n_count", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_under_3_goals_n_count = self.add_weight(name="over_0_teams_under_3_goals_n_count", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_1_goals_n_count = self.add_weight(name="over_0_teams_over_1_goals_n_count", initializer="zeros", dtype = tf.float32)
        self.home_win_n_count = self.add_weight(name="home_win_n_count", initializer="zeros", dtype = tf.float32)
        self.away_win_n_count = self.add_weight(name="away_win_n_count", initializer="zeros", dtype = tf.float32)
        self.draw_n_count = self.add_weight(name="draw_n_count", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_n_count = self.add_weight(name="true_pred_matching_n_count", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_0_n_count = self.add_weight(name="true_pred_matching_1_0_n_count", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_1_1_n_count = self.add_weight(name="true_pred_matching_1_1_n_count", initializer="zeros", dtype = tf.float32)
        self.true_pred_matching_2_1_n_count = self.add_weight(name="true_pred_matching_2_1_n_count", initializer="zeros", dtype = tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        
        nMathes = K.sum( K.ones_like(y_true[0]) )
        self.nMatches.assign_add( nMathes )

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

        # ------------------------------------- Yes zone -------------------------------------------- 

        # shape = ()
        home_over_0_goals_sum_y = K.sum(K.cast(tf.math.logical_and(true_over_0_goals[0], pred_over_0_goals[0]), dtype=tf.float32))
        away_over_0_goals_sum_y = K.sum(K.cast(tf.math.logical_and(true_over_0_goals[1], pred_over_0_goals[1]), dtype=tf.float32))
        both_over_0_goals_sum_y = K.sum( K.cast(tf.math.logical_and(true_both_over_0_goals, pred_both_over_0_goals), dtype=tf.float32))
        under_2_teams_over_0_goals_sum_y = K.sum(K.cast(tf.math.logical_and(true_under_2_teams_over_0_goals, pred_under_2_teams_over_0_goals), dtype=tf.float32))
        over_0_teams_over_2_goals_sum_y = K.sum(K.cast(tf.math.logical_and(true_over_0_teams_over_2_goals, pred_over_0_teams_over_2_goals), dtype=tf.float32))
        over_0_teams_under_3_goals_sum_y = K.sum(K.cast(tf.math.logical_and(true_over_0_teams_under_3_goals, pred_over_0_teams_under_3_goals), dtype=tf.float32))
        over_0_teams_over_1_goals_sum_y = K.sum(K.cast(tf.math.logical_and(true_over_0_teams_over_1_goals, pred_over_0_teams_over_1_goals), dtype=tf.float32))
        home_win_sum_y = K.sum( K.cast(tf.math.logical_and(true_home_win, pred_home_win), dtype=tf.float32))
        away_win_sum_y = K.sum( K.cast(tf.math.logical_and(true_away_win, pred_away_win), dtype=tf.float32))
        draw_sum_y = K.sum( K.cast(tf.math.logical_and(true_draw, pred_draw), dtype=tf.float32))  # -----------------------------
        true_pred_matching_sum_y = K.sum(K.cast(true_pred_matching, dtype=tf.float32))
        true_pred_matching_1_0_sum_y = K.sum(K.cast(tf.math.logical_and(true_1_0, pred_1_0), dtype=tf.float32))
        true_pred_matching_1_1_sum_y = K.sum(K.cast(tf.math.logical_and(true_1_1, pred_1_1), dtype=tf.float32))
        true_pred_matching_2_1_sum_y = K.sum(K.cast(tf.math.logical_and(true_2_1, pred_2_1), dtype=tf.float32))

        self.home_over_0_goals_y.assign_add(home_over_0_goals_sum_y)
        self.away_over_0_goals_y.assign_add(away_over_0_goals_sum_y)
        self.both_over_0_goals_y.assign_add(both_over_0_goals_sum_y)
        self.under_2_teams_over_0_goals_y.assign_add( under_2_teams_over_0_goals_sum_y)
        self.over_0_teams_over_2_goals_y.assign_add(over_0_teams_over_2_goals_sum_y)
        self.over_0_teams_under_3_goals_y.assign_add(over_0_teams_under_3_goals_sum_y)
        self.over_0_teams_over_1_goals_y.assign_add(over_0_teams_over_1_goals_sum_y)
        self.home_win_y.assign_add( home_win_sum_y)
        self.away_win_y.assign_add( away_win_sum_y)
        self.draw_y.assign_add( draw_sum_y)
        self.true_pred_matching_y.assign_add( true_pred_matching_sum_y)
        self.true_pred_matching_1_0_y.assign_add(true_pred_matching_1_0_sum_y)
        self.true_pred_matching_1_1_y.assign_add(true_pred_matching_1_1_sum_y)
        self.true_pred_matching_2_1_y.assign_add(true_pred_matching_2_1_sum_y)

        # shape = ()
        home_over_0_goals_sum_y_count = K.sum(K.cast(pred_over_0_goals[0], dtype=tf.float32))
        away_over_0_goals_sum_y_count = K.sum(K.cast(pred_over_0_goals[1], dtype=tf.float32))
        both_over_0_goals_sum_y_count = K.sum( K.cast(pred_both_over_0_goals, dtype=tf.float32))
        under_2_teams_over_0_goals_sum_y_count = K.sum(K.cast(pred_under_2_teams_over_0_goals, dtype=tf.float32))
        over_0_teams_over_2_goals_sum_y_count = K.sum(K.cast(pred_over_0_teams_over_2_goals, dtype=tf.float32))
        over_0_teams_under_3_goals_sum_y_count = K.sum(K.cast(pred_over_0_teams_under_3_goals, dtype=tf.float32))
        over_0_teams_over_1_goals_sum_y_count = K.sum(K.cast(pred_over_0_teams_over_1_goals, dtype=tf.float32))
        home_win_sum_y_y_count = K.sum(K.cast(pred_home_win, dtype=tf.float32))
        away_win_sum_y_count = K.sum( K.cast(pred_away_win, dtype=tf.float32))
        draw_sum_y_count = K.sum( K.cast(pred_draw, dtype=tf.float32))
        true_pred_matching_sum_y_count = K.sum(K.cast(true_pred_matching, dtype=tf.float32))
        true_pred_matching_1_0_sum_y_count = K.sum(K.cast(pred_1_0, dtype=tf.float32))
        true_pred_matching_1_1_sum_y_count = K.sum(K.cast(pred_1_1, dtype=tf.float32))
        true_pred_matching_2_1_sum_y_count = K.sum(K.cast(pred_2_1, dtype=tf.float32))

        self.home_over_0_goals_y_count.assign_add(home_over_0_goals_sum_y_count)
        self.away_over_0_goals_y_count.assign_add(away_over_0_goals_sum_y_count)
        self.both_over_0_goals_y_count.assign_add(both_over_0_goals_sum_y_count)
        self.under_2_teams_over_0_goals_y_count.assign_add(under_2_teams_over_0_goals_sum_y_count)
        self.over_0_teams_over_2_goals_y_count.assign_add(over_0_teams_over_2_goals_sum_y_count)
        self.over_0_teams_under_3_goals_y_count.assign_add(over_0_teams_under_3_goals_sum_y_count)
        self.over_0_teams_over_1_goals_y_count.assign_add(over_0_teams_over_1_goals_sum_y_count)
        self.home_win_y_count.assign_add(home_win_sum_y_y_count)
        self.away_win_y_count.assign_add(away_win_sum_y_count)
        self.draw_y_count.assign_add(draw_sum_y_count)
        self.true_pred_matching_y_count.assign_add(true_pred_matching_sum_y_count)
        self.true_pred_matching_1_0_y_count.assign_add(true_pred_matching_1_0_sum_y_count)
        self.true_pred_matching_1_1_y_count.assign_add(true_pred_matching_1_1_sum_y_count)
        self.true_pred_matching_2_1_y_count.assign_add(true_pred_matching_2_1_sum_y_count)

        # ------------------------------------- No zone -------------------------------------------- 

        # shape = ()
        home_over_0_goals_sum_n = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_over_0_goals[0], pred_over_0_goals[0])), dtype=tf.float32))
        away_over_0_goals_sum_n = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_over_0_goals[1], pred_over_0_goals[1])), dtype=tf.float32))
        both_over_0_goals_sum_n = K.sum( K.cast(tf.math.logical_not(tf.math.logical_or(true_both_over_0_goals, pred_both_over_0_goals)), dtype=tf.float32))
        under_2_teams_over_0_goals_sum_n = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_under_2_teams_over_0_goals, pred_under_2_teams_over_0_goals)), dtype=tf.float32))
        over_0_teams_over_2_goals_sum_n = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_over_0_teams_over_2_goals, pred_over_0_teams_over_2_goals)), dtype=tf.float32))
        over_0_teams_under_3_goals_sum_n = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_over_0_teams_under_3_goals, pred_over_0_teams_under_3_goals)), dtype=tf.float32))
        over_0_teams_over_1_goals_sum_n = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_over_0_teams_over_1_goals, pred_over_0_teams_over_1_goals)), dtype=tf.float32))
        home_win_sum_n = K.sum( K.cast(tf.math.logical_not(tf.math.logical_or(true_home_win, pred_home_win)), dtype=tf.float32))
        away_win_sum_n = K.sum( K.cast(tf.math.logical_not(tf.math.logical_or(true_away_win, pred_away_win)), dtype=tf.float32))
        draw_sum_n = K.sum( K.cast(tf.math.logical_not(tf.math.logical_or(true_draw, pred_draw)), dtype=tf.float32))  # -----------------------------
        true_pred_matching_sum_n = K.sum(K.cast(true_pred_matching, dtype=tf.float32))
        true_pred_matching_1_0_sum_n = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_1_0, pred_1_0)), dtype=tf.float32))
        true_pred_matching_1_1_sum_n = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_1_1, pred_1_1)), dtype=tf.float32))
        true_pred_matching_2_1_sum_n = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_2_1, pred_2_1)), dtype=tf.float32))

        self.home_over_0_goals_n.assign_add(home_over_0_goals_sum_n)
        self.away_over_0_goals_n.assign_add(away_over_0_goals_sum_n)
        self.both_over_0_goals_n.assign_add(both_over_0_goals_sum_n)
        self.under_2_teams_over_0_goals_n.assign_add( under_2_teams_over_0_goals_sum_n)
        self.over_0_teams_over_2_goals_n.assign_add(over_0_teams_over_2_goals_sum_n)
        self.over_0_teams_under_3_goals_n.assign_add(over_0_teams_under_3_goals_sum_n)
        self.over_0_teams_over_1_goals_n.assign_add(over_0_teams_over_1_goals_sum_n)
        self.home_win_n.assign_add( home_win_sum_n)
        self.away_win_n.assign_add( away_win_sum_n)
        self.draw_n.assign_add( draw_sum_n)
        self.true_pred_matching_n.assign_add( true_pred_matching_sum_n)
        self.true_pred_matching_1_0_n.assign_add(true_pred_matching_1_0_sum_n)
        self.true_pred_matching_1_1_n.assign_add(true_pred_matching_1_1_sum_n)
        self.true_pred_matching_2_1_n.assign_add(true_pred_matching_2_1_sum_n)

        # shape = ()
        home_over_0_goals_sum_n_count = nMathes - K.sum(K.cast(pred_over_0_goals[0], dtype=tf.float32))
        away_over_0_goals_sum_n_count = nMathes - K.sum(K.cast(pred_over_0_goals[1], dtype=tf.float32))
        both_over_0_goals_sum_n_count = nMathes - K.sum( K.cast(pred_both_over_0_goals, dtype=tf.float32))
        under_2_teams_over_0_goals_sum_n_count = nMathes - K.sum(K.cast(pred_under_2_teams_over_0_goals, dtype=tf.float32))
        over_0_teams_over_2_goals_sum_n_count = nMathes - K.sum(K.cast(pred_over_0_teams_over_2_goals, dtype=tf.float32))
        over_0_teams_under_3_goals_sum_n_count = nMathes - K.sum(K.cast(pred_over_0_teams_under_3_goals, dtype=tf.float32))
        over_0_teams_over_1_goals_sum_n_count = nMathes - K.sum(K.cast(pred_over_0_teams_over_1_goals, dtype=tf.float32))
        home_win_sum_n_n_count = nMathes - K.sum(K.cast(pred_home_win, dtype=tf.float32))
        away_win_sum_n_count = nMathes - K.sum( K.cast(pred_away_win, dtype=tf.float32))
        draw_sum_n_count = nMathes - K.sum( K.cast(pred_draw, dtype=tf.float32))
        true_pred_matching_sum_n_count = nMathes - K.sum(K.cast(true_pred_matching, dtype=tf.float32))
        true_pred_matching_1_0_sum_n_count = nMathes - K.sum(K.cast(pred_1_0, dtype=tf.float32))
        true_pred_matching_1_1_sum_n_count = nMathes - K.sum(K.cast(pred_1_1, dtype=tf.float32))
        true_pred_matching_2_1_sum_n_count = nMathes - K.sum(K.cast(pred_2_1, dtype=tf.float32))

        self.home_over_0_goals_n_count.assign_add(home_over_0_goals_sum_n_count)
        self.away_over_0_goals_n_count.assign_add(away_over_0_goals_sum_n_count)
        self.both_over_0_goals_n_count.assign_add(both_over_0_goals_sum_n_count)
        self.under_2_teams_over_0_goals_n_count.assign_add(under_2_teams_over_0_goals_sum_n_count)
        self.over_0_teams_over_2_goals_n_count.assign_add(over_0_teams_over_2_goals_sum_n_count)
        self.over_0_teams_under_3_goals_n_count.assign_add(over_0_teams_under_3_goals_sum_n_count)
        self.over_0_teams_over_1_goals_n_count.assign_add(over_0_teams_over_1_goals_sum_n_count)
        self.home_win_n_count.assign_add(home_win_sum_n_n_count)
        self.away_win_n_count.assign_add(away_win_sum_n_count)
        self.draw_n_count.assign_add(draw_sum_n_count)
        self.true_pred_matching_n_count.assign_add(true_pred_matching_sum_n_count)
        self.true_pred_matching_1_0_n_count.assign_add(true_pred_matching_1_0_sum_n_count)
        self.true_pred_matching_1_1_n_count.assign_add(true_pred_matching_1_1_sum_n_count)
        self.true_pred_matching_2_1_n_count.assign_add(true_pred_matching_2_1_sum_n_count)


    def result(self):

        epsilon = 1e-8

        return \
            tf.round( self.home_over_0_goals_y * 100 / (self.home_over_0_goals_y_count + epsilon) ), \
            tf.round( self.home_over_0_goals_n * 100 / (self.home_over_0_goals_n_count + epsilon) ), \
            tf.round( self.away_over_0_goals_y * 100 / (self.away_over_0_goals_y_count + epsilon) ), \
            tf.round( self.away_over_0_goals_n * 100 / (self.away_over_0_goals_n_count + epsilon) ), \
            tf.round( self.both_over_0_goals_y * 100 / (self.both_over_0_goals_y_count + epsilon) ), \
            tf.round( self.both_over_0_goals_n * 100 / (self.both_over_0_goals_n_count + epsilon) ), \
            tf.round( self.under_2_teams_over_0_goals_y * 100 / (self.under_2_teams_over_0_goals_y_count + epsilon) ), \
            tf.round( self.under_2_teams_over_0_goals_n * 100 / (self.under_2_teams_over_0_goals_n_count + epsilon) ), \
            tf.round( self.over_0_teams_over_2_goals_y * 100 / (self.over_0_teams_over_2_goals_y_count + epsilon) ), \
            tf.round( self.over_0_teams_over_2_goals_n * 100 / (self.over_0_teams_over_2_goals_n_count + epsilon) ), \
            tf.round( self.over_0_teams_under_3_goals_y * 100 / (self.over_0_teams_under_3_goals_y_count + epsilon) ), \
            tf.round( self.over_0_teams_under_3_goals_n * 100 / (self.over_0_teams_under_3_goals_n_count + epsilon) ), \
            tf.round( self.over_0_teams_over_1_goals_y * 100 / (self.over_0_teams_over_1_goals_y_count + epsilon) ), \
            tf.round( self.over_0_teams_over_1_goals_n * 100 / (self.over_0_teams_over_1_goals_n_count + epsilon) ), \
            tf.round( self.home_win_y * 100 / (self.home_win_y_count + epsilon) ), \
            tf.round( self.home_win_n * 100 / (self.home_win_n_count + epsilon) ), \
            tf.round( self.away_win_y * 100 / (self.away_win_y_count + epsilon) ), \
            tf.round( self.away_win_n * 100 / (self.away_win_n_count + epsilon) ), \
            tf.round( self.draw_y * 100 / (self.draw_y_count + epsilon) ), \
            tf.round( self.draw_n * 100 / (self.draw_n_count + epsilon) ), \
            tf.round( self.true_pred_matching_y * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.true_pred_matching_n * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.true_pred_matching_1_0_y * 100 / (self.true_pred_matching_1_0_y_count + epsilon) ), \
            tf.round( self.true_pred_matching_1_0_n * 100 / (self.true_pred_matching_1_0_n_count + epsilon) ), \
            tf.round( self.true_pred_matching_1_1_y * 100 / (self.true_pred_matching_1_1_y_count + epsilon) ), \
            tf.round( self.true_pred_matching_1_1_n * 100 / (self.true_pred_matching_1_1_n_count + epsilon) ), \
            tf.round( self.true_pred_matching_2_1_y * 100 / (self.true_pred_matching_2_1_y_count + epsilon) ), \
            tf.round( self.true_pred_matching_2_1_n * 100 / (self.true_pred_matching_2_1_n_count + epsilon) ), \
