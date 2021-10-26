import tensorflow as tf

from M_Config import *
from M_Loss_Monitor import *


class EarlyStop_0(tf.keras.callbacks.Callback):

    def __init__(self, coordinatorParams, lossMonitorParams ):

        super().__init__()

        self.monitor = coordinatorParams['monitor']
        self.patience = coordinatorParams['.patience']
        self.verbose = coordinatorParams['verbose']

        self.loss_monitor = Loss_Monitor( lossMonitorParams )

        self.initial_lr = None # self.model is None at this time.


    def on_epoch_end(self, epoch, logs={}):

        if self.initial_lr is None : self.initial_lr =  tf.keras.backend.get_value( self.model.optimizer.lr )

        loss = logs.get(self.monitor)
        if loss is None: warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        self.model.stop_training, speedup = self.loss_monitor.on_each_loss( loss )

        if self.model.stop_training is False : # Be cautious.
            if speedup is True : self.ChangeLearningRate( 0.2 / self.patience )
            else : self.ChangeLearningRate( -0.8 / self.patience )

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


    def ChangeLearningRate( self, changeRate ) :

        curr_lr = tf.keras.backend.get_value( self.model.optimizer.lr )
        new_lr = curr_lr * ( 1.0 + changeRate  )
        new_lr = min( new_lr, self.initial_lr * 100 )
        new_lr = max( new_lr, self.initial_lr / 100 )
        tf.keras.backend.set_value( self.model.optimizer.lr, new_lr )

        return curr_lr, new_lr

    def ChangeLearningRateWithAplpha( self, alpha ) :

        curr_lr = tf.keras.backend.get_value( self.model.optimizer.lr)
        new_lr = curr_lr * alpha
        new_lr = min( new_lr, self.initial_lr * 100 )
        new_lr = max( new_lr, self.initial_lr / 100 )
        tf.keras.backend.set_value( self.model.optimizer.lr, new_lr )

        return curr_lr, new_lr