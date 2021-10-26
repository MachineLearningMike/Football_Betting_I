import numpy as np
import tensorflow as tf

from M_Config import *

class Loss_Monitor() :

    def __init__( self, lossMonitorParams ):
        
        self.patience = lossMonitorParams['.patience']
        self.min_loss_improvement_per_loss = lossMonitorParams['min_loss_improvement_per_loss']
        self.countdown_params = lossMonitorParams['countdown_params']
        self.message_heading = lossMonitorParams['message_heading']

        self.call_counter = 0
        self.warning_counters = [0] * len( self.countdown_params )
        self.prev_avg_delta_loss_per_loss = 0.0
        self.curr_avg_delta_loss_per_loss = 0.0
        self.recent_losses = [ float('inf') ] * (self.patience + 1)

        self.recent_times = []
        self.prev_margin_to_speedup = 1.


    def on_each_loss( self, loss ):

        self.call_counter += 1

        stop = False
        speedup = False

        if tf.math.is_nan( loss ) :
            pass
        else :
            self.recent_losses = self.recent_losses[1:] + [ loss ]

            if self.call_counter >= 2 :
                negative_starting = max( - self.call_counter, - len(self.recent_losses) )
                nLosses_involved = abs ( negative_starting + 1 )

                sum = 0.0
                for loss in self.recent_losses[ negative_starting : - 1 ] : sum += loss
                prev_avg_loss = sum / nLosses_involved

                sum = 0.0
                for loss in self.recent_losses[ negative_starting + 1 : ] : sum += loss
                curr_avg_loss = sum / nLosses_involved

                avg_delta_loss = prev_avg_loss - curr_avg_loss # hopefully be >= 0.
                self.curr_avg_delta_loss_per_loss = avg_delta_loss / abs( curr_avg_loss + 1e-7 )
                
                times = round(self.curr_avg_delta_loss_per_loss/self.min_loss_improvement_per_loss)

                if len( self.recent_times ) < self.patience + 1 : self.recent_times.append( times )
                else: self.recent_times = self.recent_times[1:] + [ times ]
                array_times = np.array( self.recent_times )
                #weights = range( len(array_times) )
                avg_times = np.average( array_times ) #, weights = weights )
                std_times = np.std( array_times )
                margin_to_speedup = avg_times / ( 1. + std_times )
                if margin_to_speedup > 0. and margin_to_speedup / self.prev_margin_to_speedup > 0.9 : speedup = True
                else : speedup = False

                print("{} ===== avg_delta_loss_per_loss: {:.7f}. {:.0f} times: {:.7}. speed-up: {}".\
                    format( self.message_heading, self.curr_avg_delta_loss_per_loss, round( times ), self.min_loss_improvement_per_loss, speedup ) )

                for n in range ( len( self.countdown_params ) ) :
                    cd = self.countdown_params[ n ]
                    if stop is not True : stop = self.Countdown( n, cd['loss_x'], cd['patience_x'] )

                self.prev_avg_delta_loss_per_loss = self.curr_avg_delta_loss_per_loss

        return stop, speedup


    def Countdown(self, counterId, loss_factor, patience_factor ) :
        stop_signal = False

        if self.curr_avg_delta_loss_per_loss < self.min_loss_improvement_per_loss * loss_factor :
            self.warning_counters[counterId] += 1

            acting_patience = int( self.patience * patience_factor )
            if acting_patience <= 0 : raise Exception("patience_factor {} is low.".format(patience_factor) )

            if self.warning_counters[counterId] >= acting_patience :
                print( 'avg_delta_loss_per_loss is not enough with loss_factor {} to continue. Quiting.'.format( loss_factor ) )
                stop_signal = True
            else :
                print( 'avg_delta_loss_per_loss is not enough with loss_factor {}. Warning countdown {:.0f}'\
                    .format( loss_factor, int( self.patience * patience_factor ) - self.warning_counters[counterId] ) )
        else:
            self.warning_counters[counterId] -= 1
            self.warning_counters[counterId] = max( self.warning_counters[counterId], 0 )

        return stop_signal