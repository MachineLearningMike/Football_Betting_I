import numpy as np
import tensorflow as tf

from M_Config import *

from M_EarlyStop_0 import *
from M_Early_Stop_2 import *
from M_Loss_Monitor import *

from memory_profiler import profile

class EarlyStopMaxLoss( tf.keras.callbacks.Callback ):

    def __init__(self, monitor, patience, leftop_epoloss_list, **kwargs):
        super(EarlyStopMaxLoss, self).__init__(**kwargs)
        self.patience = patience
        self.leftop_epoloss_list = leftop_epoloss_list
        self.monitor = monitor

        self.minumum = float('inf')
        self.previous = float('inf')
        self.counter = 0
        self.countre2 = 0

    def on_epoch_end(self, epoch, logs=None):

        current = logs.get(self.monitor)

        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return

        if current < self.minumum : 
            self.minumum = current
            self.counter = 0
            self.countre2 = 0
        else :
            self.counter += 1


        if self.counter > self.patience :
            print( "????? No improvement for {} successive epochs. Auditing...".format( self.patience ), end = '' )

            epoch_all_beyond = True
            
            for leftop_epoloss in self.leftop_epoloss_list :
                min_epochs = leftop_epoloss[0]
                max_loss = leftop_epoloss[1]
                
                if epoch < min_epochs : epoch_all_beyond = epoch_all_beyond and False
                
                if ( epoch > min_epochs ) & ( current > max_loss ) :
                    print( "????? " + " Leftop ( epoch, {} ) = ( {}, {:.5f} ) is beyond ( {}, {:.5f} ). Quiting...".\
                        format( self.monitor, epoch, current, min_epochs, max_loss ) )
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    break
            
            if self.model.stop_training is False :
                if epoch_all_beyond is True :
                    self.countre2 += 1
                    if self.countre2 > self.patience :
                        print( "????? Idle for over {} times. Quiting... ".format( self.patience ) )
                        self.stopped_epoch = epoch
                        self.model.stop_training = True

            if self.model.stop_training is False :
                print( "!!!!! No reason to quit. Keep going.")
        else :
            self.model.stop_training = False

        return

class Model_Fitter() :

    def __init__( self, x_data, y_data ):

        super().__init__()

        self.x_data = x_data
        self.y_data = y_data

        self.model = None

    def Create_Fit_Params( self, train_params, custom ) :

        fitParams = train_params['.fitParams']

        custom_fit_params = {}
        custom_fit_params['.epochs'] = custom.get( '.epochs', fitParams['.epochs'] )
        custom_fit_params['EarlyStop'] = custom.get( 'EarlyStop', fitParams['EarlyStop'] )
        custom_fit_params['stop_monitor'] = custom.get( 'stop_monitor', fitParams['stop_monitor'] )

        custom_fit_params['subject'] = custom.get( 'subject', 'Unknown subject' )
        custom_fit_params['.mSLD'] = custom.get( '.mSLD', fitParams['.mSLD'] )
        custom_fit_params['depth'] = custom.get( 'depth', fitParams['depth'] )
        custom_fit_params['.generosity'] = custom.get( '.generosity', fitParams['.generosity'] )

        return custom_fit_params


    def Fit_Model_Once( self, model, train_params, custom_fit_params, x_data, y_data, patience = 10, leftop_epoloss_list = None ) :

        modelParams = train_params['modelParams']; fitParams = train_params['.fitParams']
        trainResources = train_params['trainResources']; dataUse = train_params['.dataUse']

        output_names = [ model.get_config()['output_layers'][n][0] for n in range( len( model.get_config()['output_layers'] ) ) ]

        x_train_data = { name : x_data['train'] for name in output_names }
        y_train_data = { name : y_data['train'] for name in output_names }

        valid_data = ( { name : x_data['valid'] for name in output_names }, { name : y_data['valid'] for name in output_names } )

        callbacks = []

        #-----------------------------------------------------------------
        checkpoint_monitor = fitParams['check_monitor']
        if valid_data is None : checkpoint_monitor = 'loss'

        check_filePath = super_config['CheckPoints']
        save_best_only = tf.keras.callbacks.ModelCheckpoint(\
            filepath = check_filePath, monitor = checkpoint_monitor, save_best_only = True, save_weights_only = True, verbose = 2 \
        )
        callbacks.append( save_best_only )

        #------------------------------------------------------------------

        if custom_fit_params['EarlyStop'] == 'Yes' : 
            if valid_data is None : earlyStopParams['monitor'] = 'loss' #========================== Contaminating relayed params ?
            #early_stop_callback = EarlyStop_0( earlyStopParams, lossMonitorParams )
            early_stop_generocity = Early_Stop_2( 'Loss Review --- '  + custom_fit_params['subject'], custom_fit_params['stop_monitor'], custom_fit_params['.epochs'], \
                custom_fit_params['.mSLD'], custom_fit_params['depth'], custom_fit_params['.generosity']  )
            callbacks.append( early_stop_generocity )
        #-------------------------------------------------------------------

        if leftop_epoloss_list is not None :
            early_stop_val_loss = EarlyStopMaxLoss( monitor = 'val_loss', patience = patience, leftop_epoloss_list = leftop_epoloss_list ) # Let min_epochs be an argument to this funciton.
            callbacks.append( early_stop_val_loss )
        #-------------------------------------------------------------------

        model.reset_metrics()
    
        model.fit( x_train_data, y_train_data, validation_data = valid_data, callbacks = callbacks, \
            shuffle = dataUse['.ShuFit'], validation_freq = 1, batch_size = dataUse['.BSizeFit'], \
            epochs = custom_fit_params['.epochs'], verbose = 2 \
        ) # verbose 0 makes the system stoned.

        model.load_weights( filepath = check_filePath ) # because of cp_save_best_only

        del train_params, custom_fit_params, x_data, y_data

        return model


    def Get_Array_Numbers_Weights( self, model ) : # all, trainable, non-trainable

        ret = np.array( [ \
            np.sum( [ np.prod(v.shape) for v in model.get_weights() ] ), \
            np.sum( [ np.prod(v.shape) for v in model.trainable_weights ] ), \
            np.sum( [ np.prod(v.shape) for v in model.non_trainable_weights ] ) \
        ] )

        del model

        return ret

    def Copy_Model( self, model, with_compilation = False ) :

        clone = tf.keras.models.clone_model( model )

        clone.set_weights( model.get_weights() )

        # for s_layer, t_layer in zip( model.layers, clone.layers ) : t_layer.set_weights( s_layer.get_weights() )

        # clone.build( ...  )

        if with_compilation and train_params is not None :
            pass

            #clone.compile( optimizer = model.optimizer, loss = model.compiled_loss, metrics = model.compiled_metrics )
            #del clone.optimizer, clone.compiled_loss, clone.compiled_metrics
            #clone.optimizer, clone.compiled_loss, clone.compiled_metrics = model.optimizer, model.compiled_loss, model.compiled_metrics

        del model

        return clone


    def Is_Consistent_Weight_Numbers( self, model, core_model, closing_model ) :
        
        w_model = self.Get_Array_Numbers_Weights( model )
        w_core_morel = self.Get_Array_Numbers_Weights( core_model )
        w_closing_model = self.Get_Array_Numbers_Weights( closing_model )
        
        ret = np.all( w_model == w_core_morel + w_closing_model )

        del model, core_model, closing_model, w_model, w_core_morel, w_closing_model

        return ret


    def Assess_Loss( self, model, loss_type ) :

        train_loss, valid_loss, loss = None, None, None

        if loss_type in [ 'loss_train', 'average' ] :
            train_loss = self.Get_Model_Loss( model, 'train', self.x_data, self.y_data )
            loss = train_loss
        if loss_type in [ 'loss_valid', 'average' ] :
            valid_loss = self.Get_Model_Loss( model, 'valid', self.x_data, self.y_data )
            loss = valid_loss
        if loss_type in [ 'loss_test' ] :
            test_loss = self.Get_Model_Loss( model, 'test', self.x_data, self.y_data )
            loss = test_loss
        if loss_type == 'average' : 
            loss = ( train_loss + valid_loss ) / 2

        if loss is None :
            raise Exception( "Help!" )

        return loss


    def Get_Model_Loss( self, model, segment, x_data, y_data ) :

        losses = model.evaluate( x = x_data[segment], y = y_data[segment], verbose = 0 )

        loss = losses
        if isinstance( losses, list ) : loss = losses[0]

        del model, segment, x_data, y_data

        return loss

