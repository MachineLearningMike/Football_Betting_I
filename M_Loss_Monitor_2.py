import numpy as np
import tensorflow as tf

from M_Config import *

class Loss_Monitor_2() :

    def __init__( self, subject, epochs, mSLD, depth, generosity ): 

        # mSol : minimum Speed Of Loss Drop

        """
        super_config['.loss_patterns'] = [ \
            { 'xSOLD' : 1.0, 'xEpochs' : 1.00 }, \
            { 'xSOLD' : 0.6, 'xEpochs' : 0.60 }, \
            { 'xSOLD' : 0.2, 'xEpochs' : 0.20 }, \
        ]
        """
        self.subject = subject
        self.epochs = epochs

        self.mSLD = mSLD

        self.patterns = super_config['.loss_patterns']
        self.depth = int( min( depth, len( self.patterns ) ) )
        self.generosity = generosity

        self.call_counter = 0
        self.nan_counter = 0
        self.counter_wanderding = 0
        self.warning_counters = [0] * self.depth
        self.prev_loss = 1e9 # ------------------------- cannot be float('inf').
        self.min_loss = 1e9

        self.len_xSLDs = 10 # --------------------= 10?
        self.recent_xSLDs = [ 999. ] * self.len_xSLDs

        self.mLoss = float('inf')


    def on_each_loss( self, curr_loss ):
        self.call_counter += 1

        stop = False
        speedup = False

        reports = []

        if tf.math.is_nan( curr_loss ) :
            self.nan_counter += 1
            report = { 'category' : 'loss is Nan', 'parameter' : self.nan_counter }
            reports.append( report )
        else :
            self.nan_counter = 0


            print("---------- {} --------".format( self.subject ) )

            #--------------------- Detect frequent lower SLD, include quiting.
            loss_drop = self.prev_loss - curr_loss
            SLD = loss_drop / abs( curr_loss + 1e-7 ) # Standard Loss Drop.

            stop = False
            for depth in range ( self.depth ) :
                if stop is not True : stop = self.Countdown( depth, SLD )
                else : break

            if stop is True : 
                report = { 'category' : 'SLD_lower_many_times', 'parameter' : 0 }
                reports.append( report )

            """
            #--------------------- Detect wandering
            
            if curr_loss < self.mLoss : #------------- This is the goal, actually.
                self.mLoss = curr_loss
                report = { 'category' : 'best_valley_found', 'parameter' : 0 }
                reports.append( report )
                self.counter_wanderding = 0

            else :
                self.counter_wanderding += 1
                wandering_threshold = int(  0.06 * self.epochs * self.generosity ) + 1
                if self.counter_wanderding > wandering_threshold :
                    print(" - !!!!!!!! Wandering in a loss valley. Restoring to the previous best valley.".format() )
                    report = { 'category' : 'wandering_for_long', 'parameter' : 0 }
                    reports.append( report )
                    self.counter_wanderding = 0
                else :
                    print(" - {} more wandering steps, before taken to the previous best loss valley.".format( wandering_threshold - self.counter_wanderding ) )

            """
            #--------------------- Detect wrong speeds

            if curr_loss < self.min_loss : self.min_loss = curr_loss

            if self.call_counter > 1 and curr_loss < self.min_loss : # Leave it to optimizer if curr_loss >= self.min_loss ( or the best-yet loss )
                xSLD = SLD / self.mSLD
                self.recent_xSLDs.pop( 0 )
                self.recent_xSLDs.append( xSLD )
                array_xSLDs = np.array( self.recent_xSLDs )
                mean_xSLDs = np.mean( array_xSLDs )
                std_xSLDs = np.std( array_xSLDs )

                score = xSLD # mean_xSLDs # / ( 1. + std_xSLDs )
                #score = SLD - self.mSLD
                
                speed_x_at_score_0 = 2.5 ; score_for_speed_x_1 = 20.0

                def get_speed_x( score ) :   
                    beta = 1.0 / speed_x_at_score_0; alpha = ( 1.0 - beta ) / score_for_speed_x_1
                    speed_x = 1.0 / ( alpha * score + beta )
                    return speed_x

                speed_x = 1.0

                if score > 1e-7 : speed_x = get_speed_x( score )
                elif score < -1e-7 : speed_x = get_speed_x( - score )
                else : speed_x = speed_x_at_score_0

                print(" - SLD: {:0.7f}, xSLD: {:.7f}, mean_xSLDs: {:.7f}, speed_x:{:.4f}".format( SLD, xSLD, mean_xSLDs, speed_x ) )
                report = { 'category' : 'speed_could_be_better', 'parameter' : speed_x }
                reports.append( report )

        self.prev_loss = curr_loss

        return reports


    def Countdown( self, depth, SLD ) :

        stop_signal = False
    
        threshold_SLD = self.patterns[depth]['xSLD'] * self.mSLD * self.generosity
        threshold_count = int( self.patterns[depth]['xEpochs'] * self.epochs * self.generosity )

        if SLD < threshold_SLD :
            self.warning_counters[ depth ] += 1

            if self.warning_counters[depth] >= threshold_count :
                print( " - SLD was less than {:.6f} at depth {} over {} times. Quiting.".format( threshold_SLD, depth, threshold_count - 1 ) )
                stop_signal = True
            else :
                print( " - SLD was less than {:.6f} at depth {}. Warning Countdown: {}".format( threshold_SLD, depth, threshold_count - self.warning_counters[depth] ) )
                stop_signal = False
        else :
            pass
            # self.warning_counters[ depth ] = max( self.warning_counters[ depth ] - 1, 0 )

        return stop_signal