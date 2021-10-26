import tensorflow as tf
import tensorflow.keras.backend as K

class SoccerMetrics(tf.keras.metrics.Metric):

    def __init__(self, name="SoccerMetrics", **kwargs):
        super(SoccerMetrics, self).__init__(name=name, **kwargs)

        self.nMatches = self.add_weight(name="nMatches", initializer="zeros", dtype = tf.float32)
        self.nCalls = self.add_weight(name="nMatches", initializer="zeros", dtype = tf.float32)

        self.home_over_0_goals_TPTN = self.add_weight(name="home_over_0_goals_TPTN", initializer="zeros", dtype = tf.float32)
        self.away_over_0_goals_TPTN = self.add_weight(name="away_over_0_goals_TPTN", initializer="zeros", dtype = tf.float32)
        self.both_over_0_goals_TPTN = self.add_weight(name="both_over_0_goals_TPTN", initializer="zeros", dtype = tf.float32)
        self.under_2_teams_over_0_goals_TPTN = self.add_weight(name="under_2_teams_over_0_goals_TPTN", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_2_goals_TPTN = self.add_weight(name="over_0_teams_over_2_goals_TPTN", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_under_3_goals_TPTN = self.add_weight(name="over_0_teams_under_3_goals_TPTN", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_1_goals_TPTN = self.add_weight(name="over_0_teams_over_1_goals_TPTN", initializer="zeros", dtype = tf.float32)
        self.home_win_TPTN = self.add_weight(name="home_win_TPTN", initializer="zeros", dtype = tf.float32)
        self.away_win_TPTN = self.add_weight(name="away_win_TPTN", initializer="zeros", dtype = tf.float32)
        self.draw_TPTN = self.add_weight(name="draw_TPTN", initializer="zeros", dtype = tf.float32)
        self.pin_point_TPTN = self.add_weight(name="pin_point_TPTN", initializer="zeros", dtype = tf.float32)
        self.pin_point_1_0_TPTN = self.add_weight(name="pin_point_1_0_TPTN", initializer="zeros", dtype = tf.float32)
        self.pin_point_1_1_TPTN = self.add_weight(name="pin_point_1_1_TPTN", initializer="zeros", dtype = tf.float32)
        self.pin_point_2_1_TPTN = self.add_weight(name="pin_point_2_1_TPTN", initializer="zeros", dtype = tf.float32)

        # -------------------------------------- Yes-accuracy ----------------------------------------------

        self.home_over_0_goals_TP = self.add_weight(name="home_over_0_goals_TP", initializer="zeros", dtype = tf.float32)
        self.away_over_0_goals_TP = self.add_weight(name="away_over_0_goals_TP", initializer="zeros", dtype = tf.float32)
        self.both_over_0_goals_TP = self.add_weight(name="both_over_0_goals_TP", initializer="zeros", dtype = tf.float32)
        self.under_2_teams_over_0_goals_TP = self.add_weight(name="under_2_teams_over_0_goals_TP", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_2_goals_TP = self.add_weight(name="over_0_teams_over_2_goals_TP", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_under_3_goals_TP = self.add_weight(name="over_0_teams_under_3_goals_TP", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_1_goals_TP = self.add_weight(name="over_0_teams_over_1_goals_TP", initializer="zeros", dtype = tf.float32)
        self.home_win_TP = self.add_weight(name="home_win_TP", initializer="zeros", dtype = tf.float32)
        self.away_win_TP = self.add_weight(name="away_win_TP", initializer="zeros", dtype = tf.float32)
        self.draw_TP = self.add_weight(name="draw_TP", initializer="zeros", dtype = tf.float32)
        self.pin_point_TP = self.add_weight(name="pin_point_TP", initializer="zeros", dtype = tf.float32)
        self.pin_point_1_0_TP = self.add_weight(name="pin_point_1_0_TP", initializer="zeros", dtype = tf.float32)
        self.pin_point_1_1_TP = self.add_weight(name="pin_point_1_1_TP", initializer="zeros", dtype = tf.float32)
        self.pin_point_2_1_TP = self.add_weight(name="pin_point_2_1_TP", initializer="zeros", dtype = tf.float32)


        self.home_over_0_goals_pred = self.add_weight(name="home_over_0_goals_pred", initializer="zeros", dtype = tf.float32)
        self.away_over_0_goals_pred = self.add_weight(name="away_over_0_goals_pred", initializer="zeros", dtype = tf.float32)
        self.both_over_0_goals_pred = self.add_weight(name="both_over_0_goals_pred", initializer="zeros", dtype = tf.float32)
        self.under_2_teams_over_0_goals_pred = self.add_weight(name="under_2_teams_over_0_goals_pred", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_2_goals_pred = self.add_weight(name="over_0_teams_over_2_goals_pred", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_under_3_goals_pred = self.add_weight(name="over_0_teams_under_3_goals_pred", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_1_goals_pred = self.add_weight(name="over_0_teams_over_1_goals_pred", initializer="zeros", dtype = tf.float32)
        self.home_win_pred = self.add_weight(name="home_win_pred", initializer="zeros", dtype = tf.float32)
        self.away_win_pred = self.add_weight(name="away_win_pred", initializer="zeros", dtype = tf.float32)
        self.draw_pred = self.add_weight(name="draw_pred", initializer="zeros", dtype = tf.float32)
        self.pin_point_pred = self.add_weight(name="pin_point_pred", initializer="zeros", dtype = tf.float32)
        self.pin_point_1_0_pred = self.add_weight(name="pin_point_1_0_pred", initializer="zeros", dtype = tf.float32)
        self.pin_point_1_1_pred = self.add_weight(name="pin_point_1_1_pred", initializer="zeros", dtype = tf.float32)
        self.pin_point_2_1_pred = self.add_weight(name="pin_point_2_1_pred", initializer="zeros", dtype = tf.float32)

        self.home_over_0_goals_true = self.add_weight(name="home_over_0_goals_true", initializer="zeros", dtype = tf.float32)
        self.away_over_0_goals_true = self.add_weight(name="away_over_0_goals_true", initializer="zeros", dtype = tf.float32)
        self.both_over_0_goals_true = self.add_weight(name="both_over_0_goals_true", initializer="zeros", dtype = tf.float32)
        self.under_2_teams_over_0_goals_true = self.add_weight(name="under_2_teams_over_0_goals_true", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_2_goals_true = self.add_weight(name="over_0_teams_over_2_goals_true", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_under_3_goals_true = self.add_weight(name="over_0_teams_under_3_goals_true", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_1_goals_true = self.add_weight(name="over_0_teams_over_1_goals_true", initializer="zeros", dtype = tf.float32)
        self.home_win_true = self.add_weight(name="home_win_true", initializer="zeros", dtype = tf.float32)
        self.away_win_true = self.add_weight(name="away_win_true", initializer="zeros", dtype = tf.float32)
        self.draw_true = self.add_weight(name="draw_true", initializer="zeros", dtype = tf.float32)
        self.pin_point_true = self.add_weight(name="pin_point_true", initializer="zeros", dtype = tf.float32)
        self.pin_point_1_0_true = self.add_weight(name="pin_point_1_0_true", initializer="zeros", dtype = tf.float32)
        self.pin_point_1_1_true = self.add_weight(name="pin_point_1_1_true", initializer="zeros", dtype = tf.float32)
        self.pin_point_2_1_true = self.add_weight(name="pin_point_2_1_true", initializer="zeros", dtype = tf.float32)

        # -------------------------------------- No-accuracy ----------------------------------------------

        self.home_over_0_goals_TN = self.add_weight(name="home_over_0_goals_TN", initializer="zeros", dtype = tf.float32)
        self.away_over_0_goals_TN = self.add_weight(name="away_over_0_goals_TN", initializer="zeros", dtype = tf.float32)
        self.both_over_0_goals_TN = self.add_weight(name="both_over_0_goals_TN", initializer="zeros", dtype = tf.float32)
        self.under_2_teams_over_0_goals_TN = self.add_weight(name="under_2_teams_over_0_goals_TN", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_2_goals_TN = self.add_weight(name="over_0_teams_over_2_goals_TN", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_under_3_goals_TN = self.add_weight(name="over_0_teams_under_3_goals_TN", initializer="zeros", dtype = tf.float32)
        self.over_0_teams_over_1_goals_TN = self.add_weight(name="over_0_teams_over_1_goals_TN", initializer="zeros", dtype = tf.float32)
        self.home_win_TN = self.add_weight(name="home_win_TN", initializer="zeros", dtype = tf.float32)
        self.away_win_TN = self.add_weight(name="away_win_TN", initializer="zeros", dtype = tf.float32)
        self.draw_TN = self.add_weight(name="draw_TN", initializer="zeros", dtype = tf.float32)
        self.pin_point_TN = self.add_weight(name="pin_point_TN", initializer="zeros", dtype = tf.float32)
        self.pin_point_1_0_TN = self.add_weight(name="pin_point_1_0_TN", initializer="zeros", dtype = tf.float32)
        self.pin_point_1_1_TN = self.add_weight(name="pin_point_1_1_TN", initializer="zeros", dtype = tf.float32)
        self.pin_point_2_1_TN = self.add_weight(name="pin_point_2_1_TN", initializer="zeros", dtype = tf.float32)

    def reset_states(self):

        self.nMatches.assign(0.)
        self.nCalls.assign(0.)

        self.home_over_0_goals_TPTN.assign(0.)
        self.away_over_0_goals_TPTN.assign(0.)
        self.both_over_0_goals_TPTN.assign(0.)
        self.under_2_teams_over_0_goals_TPTN.assign(0.)
        self.over_0_teams_over_2_goals_TPTN.assign(0.)
        self.over_0_teams_under_3_goals_TPTN.assign(0.)
        self.over_0_teams_over_1_goals_TPTN.assign(0.)
        self.home_win_TPTN.assign(0.)
        self.away_win_TPTN.assign(0.)
        self.draw_TPTN.assign(0.)
        self.pin_point_TPTN.assign(0.)
        self.pin_point_1_0_TPTN.assign(0.)
        self.pin_point_1_1_TPTN.assign(0.)
        self.pin_point_2_1_TPTN.assign(0.)

        # -------------------------------------- Yes-accuracy ----------------------------------------------

        self.home_over_0_goals_TP.assign(0.)
        self.away_over_0_goals_TP.assign(0.)
        self.both_over_0_goals_TP.assign(0.)
        self.under_2_teams_over_0_goals_TP.assign(0.)
        self.over_0_teams_over_2_goals_TP.assign(0.)
        self.over_0_teams_under_3_goals_TP.assign(0.)
        self.over_0_teams_over_1_goals_TP.assign(0.)
        self.home_win_TP.assign(0.)
        self.away_win_TP.assign(0.)
        self.draw_TP.assign(0.)
        self.pin_point_TP.assign(0.)
        self.pin_point_1_0_TP.assign(0.)
        self.pin_point_1_1_TP.assign(0.)
        self.pin_point_2_1_TP.assign(0.)


        self.home_over_0_goals_pred.assign(0.)
        self.away_over_0_goals_pred.assign(0.)
        self.both_over_0_goals_pred.assign(0.)
        self.under_2_teams_over_0_goals_pred.assign(0.)
        self.over_0_teams_over_2_goals_pred.assign(0.)
        self.over_0_teams_under_3_goals_pred.assign(0.)
        self.over_0_teams_over_1_goals_pred.assign(0.)
        self.home_win_pred.assign(0.)
        self.away_win_pred.assign(0.)
        self.draw_pred.assign(0.)
        self.pin_point_pred.assign(0.)
        self.pin_point_1_0_pred.assign(0.)
        self.pin_point_1_1_pred.assign(0.)
        self.pin_point_2_1_pred.assign(0.)

        self.home_over_0_goals_true.assign(0.)
        self.away_over_0_goals_true.assign(0.)
        self.both_over_0_goals_true.assign(0.)
        self.under_2_teams_over_0_goals_true.assign(0.)
        self.over_0_teams_over_2_goals_true.assign(0.)
        self.over_0_teams_under_3_goals_true.assign(0.)
        self.over_0_teams_over_1_goals_true.assign(0.)
        self.home_win_true.assign(0.)
        self.away_win_true.assign(0.)
        self.draw_true.assign(0.)
        self.pin_point_true.assign(0.)
        self.pin_point_1_0_true.assign(0.)
        self.pin_point_1_1_true.assign(0.)
        self.pin_point_2_1_true.assign(0.)

        # -------------------------------------- No-accuracy ----------------------------------------------

        self.home_over_0_goals_TN.assign(0.)
        self.away_over_0_goals_TN.assign(0.)
        self.both_over_0_goals_TN.assign(0.)
        self.under_2_teams_over_0_goals_TN.assign(0.)
        self.over_0_teams_over_2_goals_TN.assign(0.)
        self.over_0_teams_under_3_goals_TN.assign(0.)
        self.over_0_teams_over_1_goals_TN.assign(0.)
        self.home_win_TN.assign(0.)
        self.away_win_TN.assign(0.)
        self.draw_TN.assign(0.)
        self.pin_point_TN.assign(0.)
        self.pin_point_1_0_TN.assign(0.)
        self.pin_point_1_1_TN.assign(0.)
        self.pin_point_2_1_TN.assign(0.)



    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        
        nMathes = K.sum( K.ones_like(y_true[0]) )
        self.nMatches.assign_add( nMathes )
        self.nCalls.assign_add( 1.0 )

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
        pin_point = tf.math.logical_and( K.equal( true[0], pred[0] ), K.equal( true[1], pred[1] ) )
        true_1_0 = tf.math.logical_and( K.equal( true[0], 1 ), K.equal( true[1], 0 ) )
        pred_1_0 = tf.math.logical_and( K.equal( pred[0], 1 ), K.equal( pred[1], 0 ) )
        true_1_1 = tf.math.logical_and( K.equal( true[0], 1 ), K.equal( true[1], 1 ) )
        pred_1_1 = tf.math.logical_and( K.equal( pred[0], 1 ), K.equal( pred[1], 1 ) )
        true_2_1 = tf.math.logical_and( K.equal( true[0], 2 ), K.equal( true[1], 1 ) )
        pred_2_1 = tf.math.logical_and( K.equal( pred[0], 2 ), K.equal( pred[1], 1 ) )

        # -----------------------------------------  Averge Zone ---------------------------------------------------
 
        # shape = ()
        home_over_0_goals_TPTN = K.sum(K.cast(K.equal(true_over_0_goals[0], pred_over_0_goals[0]), dtype=tf.float32))
        away_over_0_goals_TPTN = K.sum(K.cast(K.equal(true_over_0_goals[1], pred_over_0_goals[1]), dtype=tf.float32))
        both_over_0_goals_TPTN = K.sum(K.cast(K.equal(true_both_over_0_goals, pred_both_over_0_goals), dtype=tf.float32))
        under_2_teams_over_0_goals_TPTN = K.sum(K.cast(K.equal(true_under_2_teams_over_0_goals, pred_under_2_teams_over_0_goals), dtype=tf.float32))
        over_0_teams_over_2_goals_TPTN = K.sum(K.cast(K.equal(true_over_0_teams_over_2_goals, pred_over_0_teams_over_2_goals), dtype=tf.float32))
        over_0_teams_under_3_goals_TPTN = K.sum(K.cast(K.equal(true_over_0_teams_under_3_goals, pred_over_0_teams_under_3_goals), dtype=tf.float32))
        over_0_teams_over_1_goals_TPTN = K.sum(K.cast(K.equal(true_over_0_teams_over_1_goals, pred_over_0_teams_over_1_goals), dtype=tf.float32))
        home_win_TPTN = K.sum(K.cast(K.equal(true_home_win, pred_home_win), dtype=tf.float32))
        away_win_TPTN = K.sum(K.cast(K.equal(true_away_win, pred_away_win), dtype=tf.float32))
        draw_TPTN = K.sum(K.cast(K.equal(true_draw, pred_draw), dtype=tf.float32))
        pin_point_TPTN = K.sum(K.cast(pin_point, dtype=tf.float32))
        pin_point_1_0_TPTN = K.sum(K.cast(K.equal(true_1_0, pred_1_0), dtype=tf.float32))
        pin_point_1_1_TPTN = K.sum(K.cast(K.equal(true_1_1, pred_1_1), dtype=tf.float32))
        pin_point_2_1_TPTN = K.sum(K.cast(K.equal(true_2_1, pred_2_1), dtype=tf.float32))

        self.home_over_0_goals_TPTN.assign_add(home_over_0_goals_TPTN)
        self.away_over_0_goals_TPTN.assign_add(away_over_0_goals_TPTN)
        self.both_over_0_goals_TPTN.assign_add(both_over_0_goals_TPTN)
        self.under_2_teams_over_0_goals_TPTN.assign_add( under_2_teams_over_0_goals_TPTN)
        self.over_0_teams_over_2_goals_TPTN.assign_add(over_0_teams_over_2_goals_TPTN)
        self.over_0_teams_under_3_goals_TPTN.assign_add(over_0_teams_under_3_goals_TPTN)
        self.over_0_teams_over_1_goals_TPTN.assign_add(over_0_teams_over_1_goals_TPTN)
        self.home_win_TPTN.assign_add( home_win_TPTN)
        self.away_win_TPTN.assign_add( away_win_TPTN)
        self.draw_TPTN.assign_add( draw_TPTN)
        self.pin_point_TPTN.assign_add( pin_point_TPTN)
        self.pin_point_1_0_TPTN.assign_add( pin_point_1_0_TPTN)

        # -------------------------------------- TP Base ----------------------------------------------

        # shape = ()
        home_over_0_goals_TP = K.sum(K.cast(tf.math.logical_and(true_over_0_goals[0], pred_over_0_goals[0]), dtype=tf.float32))
        away_over_0_goals_TP = K.sum(K.cast(tf.math.logical_and(true_over_0_goals[1], pred_over_0_goals[1]), dtype=tf.float32))
        both_over_0_goals_TP = K.sum( K.cast(tf.math.logical_and(true_both_over_0_goals, pred_both_over_0_goals), dtype=tf.float32))
        under_2_teams_over_0_goals_TP = K.sum(K.cast(tf.math.logical_and(true_under_2_teams_over_0_goals, pred_under_2_teams_over_0_goals), dtype=tf.float32))
        over_0_teams_over_2_goals_TP = K.sum(K.cast(tf.math.logical_and(true_over_0_teams_over_2_goals, pred_over_0_teams_over_2_goals), dtype=tf.float32))
        over_0_teams_under_3_goals_TP = K.sum(K.cast(tf.math.logical_and(true_over_0_teams_under_3_goals, pred_over_0_teams_under_3_goals), dtype=tf.float32))
        over_0_teams_over_1_goals_TP = K.sum(K.cast(tf.math.logical_and(true_over_0_teams_over_1_goals, pred_over_0_teams_over_1_goals), dtype=tf.float32))
        home_win_TP = K.sum( K.cast(tf.math.logical_and(true_home_win, pred_home_win), dtype=tf.float32))
        away_win_TP = K.sum( K.cast(tf.math.logical_and(true_away_win, pred_away_win), dtype=tf.float32))
        draw_TP = K.sum( K.cast(tf.math.logical_and(true_draw, pred_draw), dtype=tf.float32))  # -----------------------------
        pin_point_TP = K.sum(K.cast(pin_point, dtype=tf.float32))
        pin_point_1_0_TP = K.sum(K.cast(tf.math.logical_and(true_1_0, pred_1_0), dtype=tf.float32))
        pin_point_1_1_TP = K.sum(K.cast(tf.math.logical_and(true_1_1, pred_1_1), dtype=tf.float32))
        pin_point_2_1_TP = K.sum(K.cast(tf.math.logical_and(true_2_1, pred_2_1), dtype=tf.float32))

        self.home_over_0_goals_TP.assign_add(home_over_0_goals_TP)
        self.away_over_0_goals_TP.assign_add(away_over_0_goals_TP)
        self.both_over_0_goals_TP.assign_add(both_over_0_goals_TP)
        self.under_2_teams_over_0_goals_TP.assign_add( under_2_teams_over_0_goals_TP)
        self.over_0_teams_over_2_goals_TP.assign_add(over_0_teams_over_2_goals_TP)
        self.over_0_teams_under_3_goals_TP.assign_add(over_0_teams_under_3_goals_TP)
        self.over_0_teams_over_1_goals_TP.assign_add(over_0_teams_over_1_goals_TP)
        self.home_win_TP.assign_add( home_win_TP)
        self.away_win_TP.assign_add( away_win_TP)
        self.draw_TP.assign_add( draw_TP)
        self.pin_point_TP.assign_add( pin_point_TP)
        self.pin_point_1_0_TP.assign_add(pin_point_1_0_TP)
        self.pin_point_1_1_TP.assign_add(pin_point_1_1_TP)
        self.pin_point_2_1_TP.assign_add(pin_point_2_1_TP)

     # -------------------------------------- TP over Predction ----------------------------------------------

        # shape = ()

        home_over_0_goals_pred = K.sum(K.cast(pred_over_0_goals[0], dtype=tf.float32))
        away_over_0_goals_pred = K.sum(K.cast(pred_over_0_goals[1], dtype=tf.float32))
        both_over_0_goals_pred = K.sum( K.cast(pred_both_over_0_goals, dtype=tf.float32))
        under_2_teams_over_0_goals_pred = K.sum(K.cast(pred_under_2_teams_over_0_goals, dtype=tf.float32))
        over_0_teams_over_2_goals_pred = K.sum(K.cast(pred_over_0_teams_over_2_goals, dtype=tf.float32))
        over_0_teams_under_3_goals_pred = K.sum(K.cast(pred_over_0_teams_under_3_goals, dtype=tf.float32))
        over_0_teams_over_1_goals_pred = K.sum(K.cast(pred_over_0_teams_over_1_goals, dtype=tf.float32))
        home_win_pred = K.sum(K.cast(pred_home_win, dtype=tf.float32))
        away_win_pred = K.sum( K.cast(pred_away_win, dtype=tf.float32))
        draw_pred = K.sum( K.cast(pred_draw, dtype=tf.float32))
        pin_point_pred = K.sum(K.cast(pin_point, dtype=tf.float32))
        pin_point_1_0_pred = K.sum(K.cast(pred_1_0, dtype=tf.float32))
        pin_point_1_1_pred = K.sum(K.cast(pred_1_1, dtype=tf.float32))
        pin_point_2_1_pred = K.sum(K.cast(pred_2_1, dtype=tf.float32))
        
        self.home_over_0_goals_pred.assign_add(home_over_0_goals_pred)
        self.away_over_0_goals_pred.assign_add(away_over_0_goals_pred)
        self.both_over_0_goals_pred.assign_add(both_over_0_goals_pred)
        self.under_2_teams_over_0_goals_pred.assign_add(under_2_teams_over_0_goals_pred)
        self.over_0_teams_over_2_goals_pred.assign_add(over_0_teams_over_2_goals_pred)
        self.over_0_teams_under_3_goals_pred.assign_add(over_0_teams_under_3_goals_pred)
        self.over_0_teams_over_1_goals_pred.assign_add(over_0_teams_over_1_goals_pred)
        self.home_win_pred.assign_add(home_win_pred)
        self.away_win_pred.assign_add(away_win_pred)
        self.draw_pred.assign_add(draw_pred)
        self.pin_point_pred.assign_add(pin_point_pred)
        self.pin_point_1_0_pred.assign_add(pin_point_1_0_pred)
        self.pin_point_1_1_pred.assign_add(pin_point_1_1_pred)
        self.pin_point_2_1_pred.assign_add(pin_point_2_1_pred)


        # -------------------------------------- TP over Truth ----------------------------------------------

        home_over_0_goals_true = K.sum(K.cast(true_over_0_goals[0], dtype=tf.float32))
        away_over_0_goals_true = K.sum(K.cast(true_over_0_goals[1], dtype=tf.float32))
        both_over_0_goals_true = K.sum( K.cast(true_both_over_0_goals, dtype=tf.float32))
        under_2_teams_over_0_goals_true = K.sum(K.cast(true_under_2_teams_over_0_goals, dtype=tf.float32))
        over_0_teams_over_2_goals_true = K.sum(K.cast(true_over_0_teams_over_2_goals, dtype=tf.float32))
        over_0_teams_under_3_goals_true = K.sum(K.cast(true_over_0_teams_under_3_goals, dtype=tf.float32))
        over_0_teams_over_1_goals_true = K.sum(K.cast(true_over_0_teams_over_1_goals, dtype=tf.float32))
        home_win_true = K.sum(K.cast(true_home_win, dtype=tf.float32))
        away_win_true = K.sum( K.cast(true_away_win, dtype=tf.float32))
        draw_true = K.sum( K.cast(true_draw, dtype=tf.float32))
        pin_point_true = K.sum(K.cast(pin_point, dtype=tf.float32))
        pin_point_1_0_true = K.sum(K.cast(true_1_0, dtype=tf.float32))
        pin_point_1_1_true = K.sum(K.cast(true_1_1, dtype=tf.float32))
        pin_point_2_1_true = K.sum(K.cast(true_2_1, dtype=tf.float32))

        self.home_over_0_goals_true.assign_add(home_over_0_goals_true)
        self.away_over_0_goals_true.assign_add(away_over_0_goals_true)
        self.both_over_0_goals_true.assign_add(both_over_0_goals_true)
        self.under_2_teams_over_0_goals_true.assign_add(under_2_teams_over_0_goals_true)
        self.over_0_teams_over_2_goals_true.assign_add(over_0_teams_over_2_goals_true)
        self.over_0_teams_under_3_goals_true.assign_add(over_0_teams_under_3_goals_true)
        self.over_0_teams_over_1_goals_true.assign_add(over_0_teams_over_1_goals_true)
        self.home_win_true.assign_add(home_win_true)
        self.away_win_true.assign_add(away_win_true)
        self.draw_true.assign_add(draw_true)
        self.pin_point_true.assign_add(pin_point_true)
        self.pin_point_1_0_true.assign_add(pin_point_1_0_true)
        self.pin_point_1_1_true.assign_add(pin_point_1_1_true)
        self.pin_point_2_1_true.assign_add(pin_point_2_1_true)


        # -------------------------------------- TN Base ----------------------------------------------

        # shape = ()
        home_over_0_goals_TN = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_over_0_goals[0], pred_over_0_goals[0])), dtype=tf.float32))
        away_over_0_goals_TN = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_over_0_goals[1], pred_over_0_goals[1])), dtype=tf.float32))
        both_over_0_goals_TN = K.sum( K.cast(tf.math.logical_not(tf.math.logical_or(true_both_over_0_goals, pred_both_over_0_goals)), dtype=tf.float32))
        under_2_teams_over_0_goals_TN = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_under_2_teams_over_0_goals, pred_under_2_teams_over_0_goals)), dtype=tf.float32))
        over_0_teams_over_2_goals_TN = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_over_0_teams_over_2_goals, pred_over_0_teams_over_2_goals)), dtype=tf.float32))
        over_0_teams_under_3_goals_TN = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_over_0_teams_under_3_goals, pred_over_0_teams_under_3_goals)), dtype=tf.float32))
        over_0_teams_over_1_goals_TN = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_over_0_teams_over_1_goals, pred_over_0_teams_over_1_goals)), dtype=tf.float32))
        home_win_TN = K.sum( K.cast(tf.math.logical_not(tf.math.logical_or(true_home_win, pred_home_win)), dtype=tf.float32))
        away_win_TN = K.sum( K.cast(tf.math.logical_not(tf.math.logical_or(true_away_win, pred_away_win)), dtype=tf.float32))
        draw_TN = K.sum( K.cast(tf.math.logical_not(tf.math.logical_or(true_draw, pred_draw)), dtype=tf.float32))  # -----------------------------
        pin_point_TN = K.sum(K.cast(pin_point, dtype=tf.float32))
        pin_point_1_0_TN = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_1_0, pred_1_0)), dtype=tf.float32))
        pin_point_1_1_TN = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_1_1, pred_1_1)), dtype=tf.float32))
        pin_point_2_1_TN = K.sum(K.cast(tf.math.logical_not(tf.math.logical_or(true_2_1, pred_2_1)), dtype=tf.float32))

        self.home_over_0_goals_TN.assign_add(home_over_0_goals_TN)
        self.away_over_0_goals_TN.assign_add(away_over_0_goals_TN)
        self.both_over_0_goals_TN.assign_add(both_over_0_goals_TN)
        self.under_2_teams_over_0_goals_TN.assign_add( under_2_teams_over_0_goals_TN)
        self.over_0_teams_over_2_goals_TN.assign_add(over_0_teams_over_2_goals_TN)
        self.over_0_teams_under_3_goals_TN.assign_add(over_0_teams_under_3_goals_TN)
        self.over_0_teams_over_1_goals_TN.assign_add(over_0_teams_over_1_goals_TN)
        self.home_win_TN.assign_add( home_win_TN)
        self.away_win_TN.assign_add( away_win_TN)
        self.draw_TN.assign_add( draw_TN)
        self.pin_point_TN.assign_add( pin_point_TN)
        self.pin_point_1_0_TN.assign_add(pin_point_1_0_TN)
        self.pin_point_1_1_TN.assign_add(pin_point_1_1_TN)
        self.pin_point_2_1_TN.assign_add(pin_point_2_1_TN)

        # -------------------------------------- No-accuracy over Predction ----------------------------------------------



        # -------------------------------------- No-accuracy over Truth ----------------------------------------------



    def result(self):

        epsilon = 1e-3

        return \
            \
            tf.round( self.home_over_0_goals_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.home_over_0_goals_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.home_over_0_goals_TP * 100 / (self.home_over_0_goals_pred + epsilon) ), \
            tf.round( self.home_over_0_goals_TP * 100 / (self.home_over_0_goals_true + epsilon) ), \
            tf.round( self.home_over_0_goals_TN * 100 / (self.nMatches - self.home_over_0_goals_pred + epsilon) ), \
            tf.round( self.home_over_0_goals_TN * 100 / (self.nMatches - self.home_over_0_goals_true + epsilon) ), \
            \
            tf.round( self.away_over_0_goals_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.away_over_0_goals_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.away_over_0_goals_TP * 100 / (self.away_over_0_goals_pred + epsilon) ), \
            tf.round( self.away_over_0_goals_TP * 100 / (self.away_over_0_goals_true + epsilon) ), \
            tf.round( self.away_over_0_goals_TN * 100 / (self.nMatches - self.away_over_0_goals_pred + epsilon) ), \
            tf.round( self.away_over_0_goals_TN * 100 / (self.nMatches - self.away_over_0_goals_true + epsilon) ), \
            \
            tf.round( self.both_over_0_goals_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.both_over_0_goals_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.both_over_0_goals_TP * 100 / (self.both_over_0_goals_pred + epsilon) ), \
            tf.round( self.both_over_0_goals_TP * 100 / (self.both_over_0_goals_true + epsilon) ), \
            tf.round( self.both_over_0_goals_TN * 100 / (self.nMatches - self.both_over_0_goals_pred + epsilon) ), \
            tf.round( self.both_over_0_goals_TN * 100 / (self.nMatches - self.both_over_0_goals_true + epsilon) ), \
            \
            tf.round(self.under_2_teams_over_0_goals_true * 100 / (self.nMatches + epsilon) ), \
            tf.round(self.under_2_teams_over_0_goals_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.under_2_teams_over_0_goals_TP * 100 / (self.under_2_teams_over_0_goals_pred + epsilon) ), \
            tf.round( self.under_2_teams_over_0_goals_TP * 100 / (self.under_2_teams_over_0_goals_true + epsilon) ), \
            tf.round( self.under_2_teams_over_0_goals_TN * 100 / (self.nMatches - self.under_2_teams_over_0_goals_pred + epsilon) ), \
            tf.round( self.under_2_teams_over_0_goals_TN * 100 / (self.nMatches - self.under_2_teams_over_0_goals_true + epsilon) ), \
            \
            tf.round( self.over_0_teams_over_2_goals_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.over_0_teams_over_2_goals_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.over_0_teams_over_2_goals_TP * 100 / (self.over_0_teams_over_2_goals_pred + epsilon) ), \
            tf.round( self.over_0_teams_over_2_goals_TP * 100 / (self.over_0_teams_over_2_goals_true + epsilon) ), \
            tf.round( self.over_0_teams_over_2_goals_TN * 100 / (self.nMatches - self.over_0_teams_over_2_goals_pred + epsilon) ), \
            tf.round( self.over_0_teams_over_2_goals_TN * 100 / (self.nMatches - self.over_0_teams_over_2_goals_true + epsilon) ), \
            \
            tf.round( self.over_0_teams_under_3_goals_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.over_0_teams_under_3_goals_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.over_0_teams_under_3_goals_TP * 100 / (self.over_0_teams_under_3_goals_pred + epsilon) ), \
            tf.round( self.over_0_teams_under_3_goals_TP * 100 / (self.over_0_teams_under_3_goals_true + epsilon) ), \
            tf.round( self.over_0_teams_under_3_goals_TN * 100 / (self.nMatches - self.over_0_teams_under_3_goals_pred + epsilon) ), \
            tf.round( self.over_0_teams_under_3_goals_TN * 100 / (self.nMatches - self.over_0_teams_under_3_goals_true + epsilon) ), \
            \
            tf.round( self.over_0_teams_over_1_goals_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.over_0_teams_over_1_goals_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.over_0_teams_over_1_goals_TP * 100 / (self.over_0_teams_over_1_goals_pred + epsilon) ), \
            tf.round( self.over_0_teams_over_1_goals_TP * 100 / (self.over_0_teams_over_1_goals_true + epsilon) ), \
            tf.round( self.over_0_teams_over_1_goals_TN * 100 / (self.nMatches - self.over_0_teams_over_1_goals_pred + epsilon) ), \
            tf.round( self.over_0_teams_over_1_goals_TN * 100 / (self.nMatches - self.over_0_teams_over_1_goals_true + epsilon) ), \
            \
            tf.round( self.home_win_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.home_win_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.home_win_TP * 100 / (self.home_win_pred + epsilon) ), \
            tf.round( self.home_win_TP * 100 / (self.home_win_true + epsilon) ), \
            tf.round( self.home_win_TN * 100 / (self.nMatches - self.home_win_pred + epsilon) ), \
            tf.round( self.home_win_TN * 100 / (self.nMatches - self.home_win_true + epsilon) ), \
            \
            tf.round( self.away_win_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.away_win_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.away_win_TP * 100 / (self.away_win_pred + epsilon) ), \
            tf.round( self.away_win_TP * 100 / (self.away_win_true + epsilon) ), \
            tf.round( self.away_win_TN * 100 / (self.nMatches - self.away_win_pred + epsilon) ), \
            tf.round( self.away_win_TN * 100 / (self.nMatches - self.away_win_true + epsilon) ), \
            \
            tf.round( self.draw_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.draw_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.draw_TP * 100 / (self.draw_pred + epsilon) ), \
            tf.round( self.draw_TP * 100 / (self.draw_true + epsilon) ), \
            tf.round( self.draw_TN * 100 / (self.nMatches - self.draw_pred + epsilon) ), \
            tf.round( self.draw_TN * 100 / (self.nMatches - self.draw_true + epsilon) ), \
            \
            tf.round( self.pin_point_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.pin_point_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.pin_point_TP * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.pin_point_TP * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.pin_point_TN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.pin_point_TN * 100 / (self.nMatches + epsilon) ), \
            \
            tf.round( self.pin_point_1_0_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.pin_point_1_0_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.pin_point_1_0_TP * 100 / (self.pin_point_1_0_pred + epsilon) ), \
            tf.round( self.pin_point_1_0_TP * 100 / (self.pin_point_1_0_true + epsilon) ), \
            tf.round( self.pin_point_1_0_TN * 100 / (self.nMatches - self.pin_point_1_0_pred + epsilon) ), \
            tf.round( self.pin_point_1_0_TN * 100 / (self.nMatches - self.pin_point_1_0_true + epsilon) ), \
            \
            tf.round( self.pin_point_1_1_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.pin_point_1_1_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.pin_point_1_1_TP * 100 / (self.pin_point_1_1_pred + epsilon) ), \
            tf.round( self.pin_point_1_1_TP * 100 / (self.pin_point_1_1_true + epsilon) ), \
            tf.round( self.pin_point_1_1_TN * 100 / (self.nMatches - self.pin_point_1_1_pred + epsilon) ), \
            tf.round( self.pin_point_1_1_TN * 100 / (self.nMatches - self.pin_point_1_1_true + epsilon) ), \
            \
            tf.round( self.pin_point_2_1_true * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.pin_point_2_1_TPTN * 100 / (self.nMatches + epsilon) ), \
            tf.round( self.pin_point_2_1_TP * 100 / (self.pin_point_2_1_pred + epsilon) ), \
            tf.round( self.pin_point_2_1_TP * 100 / (self.pin_point_2_1_true + epsilon) ), \
            tf.round( self.pin_point_2_1_TN * 100 / (self.nMatches - self.pin_point_2_1_pred + epsilon) ), \
            tf.round( self.pin_point_2_1_TN * 100 / (self.nMatches - self.pin_point_2_1_true + epsilon) ), \
            \
            #self.nMatches, self.nMatches, self.nCalls, self.nCalls, self.nCalls, self.nCalls, \


    def Answer(self, score):

        y_true = tf.transpose(y_true)
        y_pred = tf.transpose(y_pred)
        
        nMathes = K.sum( K.ones_like(y_true[0]) )
        self.nMatches.assign_add( nMathes )
        self.nCalls.assign_add( 1.0 )

        y_pred = tf.math.round( y_pred )

        true = tf.cast(y_true, tf.int32)
        pred = tf.cast(y_pred, tf.int32)

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
        pin_point = tf.math.logical_and( K.equal( true[0], pred[0] ), K.equal( true[1], pred[1] ) )
        true_1_0 = tf.math.logical_and( K.equal( true[0], 1 ), K.equal( true[1], 0 ) )
        pred_1_0 = tf.math.logical_and( K.equal( pred[0], 1 ), K.equal( pred[1], 0 ) )
        true_1_1 = tf.math.logical_and( K.equal( true[0], 1 ), K.equal( true[1], 1 ) )
        pred_1_1 = tf.math.logical_and( K.equal( pred[0], 1 ), K.equal( pred[1], 1 ) )
        true_2_1 = tf.math.logical_and( K.equal( true[0], 2 ), K.equal( true[1], 1 ) )
        pred_2_1 = tf.math.logical_and( K.equal( pred[0], 2 ), K.equal( pred[1], 1 ) )
