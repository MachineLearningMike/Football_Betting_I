import tensorflow as tf

from M_Config import *
from M_Loss_Monitor_2 import *

import copy

class Early_Stop_2(tf.keras.callbacks.Callback):

    def __init__(self, subject, loss_to_monitor, epochs, mSLD, depth, generosity  ):

        super().__init__()

        self.subject = subject
        self.loss_to_monitor = loss_to_monitor

        self.best_valley = None

        self.loss_monitor = Loss_Monitor_2( subject, epochs, mSLD, depth, generosity )

        self.initial_lr = None # self.model is None now.
        
        self.over_learning_threshold = 0.1        
        self.over_learning_counter = 0
        self.prev_over_learning_rate = float('inf')


    def on_epoch_end(self, epoch, logs={}):

        if self.initial_lr is None : 
            self.initial_lr =  tf.keras.backend.get_value( self.model.optimizer.lr )

        loss = logs.get( self.loss_to_monitor )

        if loss is None: raise Exception( "Early stopping requires {} available!".format( self.loss_to_monitor ) )
      
        self.model.stop_training = False

        loss = logs.get( 'loss' )
        val_loss = logs.get( 'val_loss', None )

        #----------------------------------------------------------------------

        if val_loss is not None :
            over_learning_rate = abs ( val_loss - loss / ( loss + 1e-9 ) )

            """
            if self.prev_over_learning_rate < self.over_learning_threshold and self.over_learning_threshold <= over_learning_rate : #------------------------ Over-learnng rate.
                print( "Over-learning rate {:.4f} is beyond {:.2f}. Quiting".format( over_learning_rate, 0.1 ) )
                self.model.stop_training = True # Over-learning
            if over_learning_rate > self.over_learning_threshold :
                self.over_learning_counter += 1
                if self.over_learning_counter > 99999 :
                    self.model.stop_training = True
            """
            self.prev_over_learning_rate = over_learning_rate

        #-----------------------------------------------------------------------

        reports = self.loss_monitor.on_each_loss( val_loss )

        for report in reports :

            if self.model.stop_training is True : break

            if report['category'] == 'best_valley_found' : # report in None if loss fed to loss monitor is NaN.

                self.best_valley = { 'weights' : self.model.get_weights(), 'lRate' : self.model.optimizer.lr  }

            elif report['category'] == 'wandering_for_long' : # Go back to the best_valley with a reduced learning rate.

                if self.best_valley is not None :
                    self.model.set_weights( self.best_valley['weights'] )
                    self.ChangeLearningRate( 0.3 ) #--------------- 0.5 ? # The reason we wondered in a wrong loss valley was that learning rate was too large.

            elif report['category'] == 'speed_could_be_better' :

                self.ChangeLearningRate( report['parameter'] )

            elif report['category'] == 'SLD_lower_many_times' :

                self.model.stop_training = True #------------------------------ Quit

            elif report['category'] == 'normal_situation' : pass

            elif  report['category'] == 'loss is Nan' and report['parameter'] > 5 :

                self.model.stop_training = True #------------------------------ Quit

        return


    def is_monotonicDecay(self, lossList):

        isMonoDecay = True

        if len( lossList ) > 1:
            prevLoss = lossList[0]
            for loss in lossList[1:] :
                if loss > prevLoss :
                    isMonoDecay = False
                    break
                else: 
                    prevLoss = loss
        
        return isMonoDecay


    def ChangeLearningRate( self, multiplier ) :

        curr_lr = self.model.optimizer.lr
        new_lr = curr_lr * multiplier
        new_lr = min( new_lr, self.initial_lr * 100 ) # not greater than 100 times the initial rate.
        new_lr = max( new_lr, self.initial_lr / 100 ) # not less than a 100th of the initial rate.
        self.model.optimizer.lr = new_lr

        return curr_lr, new_lr
