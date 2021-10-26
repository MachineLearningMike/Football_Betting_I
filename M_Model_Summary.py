import numpy as np
import tensorflow as tf

from M_Config import *

from M_EarlyStop_0 import *
from M_Early_Stop_2 import *
from M_Loss_Monitor import *

from M_Model_Fitter import *
from M_Model_Slots import *

from memory_profiler import profile

class Model_Summary( Model_Fitter ) :

    def __init__( self, x_data, y_data, loss_to_compare ):

        self.loss_to_compare = loss_to_compare

        super().__init__( x_data, y_data )

        self.model = None
        self.summary = {}
        self.loss = float('inf')


    def Build_Model( self, params, model_slots, model_name, version_string ) :

        model_id = model_slots.GetLatestModelNumber( model_name, version_string )

        if model_id < 0 : # No model exists.

            model_min = None; loss_min = float('inf')

            params_vulnerable = params.copy()

            print("****************************************** Simpleton: Searching for better initial parameters.")

            verbosity = tf.compat.v1.logging.get_verbosity()
            tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )

            for _ in range( params['pick_trials'] ) :
                model, loss = self.Get_Test_Model( params_vulnerable )
                if loss < loss_min :
                    model_min = model; loss_min = loss
                    print( "-------------------- min_loss : {:.5f}".format( loss_min ) )

            tf.compat.v1.logging.set_verbosity( verbosity )


            print("******************************************* Simpleton: Improving initial parameters.")

            del self.model, self.loss
            self.model, self.loss = model_min, loss_min

            model = self.Fit_Model( model_min, self.x_data, self.y_data, params )
            loss = self.Assess_Loss( model_min, self.loss_to_compare )

            if loss < self.loss :
                del self.model, self.loss
                self.model, self.loss = model, loss
            else :
                del model, loss

            self.summary.update( { 'loss_train' : self.Assess_Loss( self.model, 'loss_train' ) } )
            self.summary.update( { 'loss_valid' : self.Assess_Loss( self.model, 'loss_valid' ) } )
            self.summary.update( { 'loss_test' : self.Assess_Loss( self.model, 'loss_test' ) } )

            print("****************************************** Simpleton: loss_train {:.4f}, loss_valid {:.4f}, loss_test {:.4f}"\
                .format( self.summary['loss_train'], self.summary['loss_valid'], self.summary['loss_test'] ) )

            model_id = self.Save_Model( model_slots, model_name, version_string )

        return model_id


    def Save_Model( self, model_slots, model_name, version_string ) :

        model_number = model_slots.SaveModel( model_name, version_string, self.model, self.summary, forced_number = -1 )

        return model_number


    def Load_Model( self, model_slots, model_name, version_string, number = -1 ) :

        self.model, self.summary = model_slots.LoadModel( model_name, version_string, number = number, custom_objects = {}, defaultSummary = {}, summary_only = False, withWeights = True )

        return self.model is not None


    def Apply_Model_To( self, summaries ) :

        preds = self.model( summaries )

        return preds


    def Get_Test_Model( self, params ) :

        model = self.Create_Model( params )
        model = self.Compile_Model( model, params )

        params['epochs'] = 1
        model = self.Fit_Model( model, self.x_data, self.y_data, params )

        loss = self.Assess_Loss( model, self.loss_to_compare )

        del params

        return model, loss


    def Create_Model( self, params ) :

        inputs = self.inputs

        input_fields = []
        rows = tf.keras.layers.Flatten( ) ( self.inputs['main'] )
        input_fields.append( rows )

        base_tensor = None
        if len( input_fields ) > 1 :
            base_tensor = tf.keras.layers.concatenate( input_fields , axis = -1 )
        elif len( input_fields ) == 1 : 
            base_tensor = input_fields[0]

        model = None

        if base_tensor is not None :

            for idx in range( len( params[ 'dense_layers' ] ) ) :
                dim_x, activation = params[ 'dense_layers' ][ idx ]
                dimension = dim_x * self.width_x_data

                base_tensor = self.Add_Dense_Layer( base_tensor, dimension, activation )
            
            base_tensor = self.Add_Dense_Layer( base_tensor, dimension = 1, activation = params['closing_act'], name = 'Output_00' )

            model = tf.keras.models.Model( inputs = inputs, outputs = [ base_tensor ], name = "model" )

            del base_tensor

        del inputs

        return model


    def Add_Dense_Layer( self, dense_tensor, dimension, activation, name = None ) : # PENDING - place BatchNormalization properly, jsut before activiation.

        #if dropoutRate > 0 : dense_tensor = tf.keras.layers.Dropout( dropoutRate ) ( dense_tensor ) #=========================== Dropout               
        #dense_tensor = tf.keras.layers.BatchNormalization()( dense_tensor ) # ======================== Batch Normalization                  
        
        dense_tensor = tf.keras.layers.Dense( units = dimension, activation = activation, name = name, \
        #kernel_initializer = initializers.Zeros(), bias_initializer = initializers.Zeros(), \
        #kernel_regularizer = regularizer, bias_regularizer = regularizer, \
        #kernel_constraint = tf.keras.constraints.MaxNorm(2) DO NOT give too much constraints.\
        ) ( dense_tensor )

        return dense_tensor


    def Compile_Model( self, model, params ) :

        optimizer = params.get( '.optimizer', tf.optimizers.Adadelta() )

        losses = { 'Output_00' : params.get( 'loss', tf.keras.losses.BinaryCrossentropy() ) }
        metrics = { 'Output_00' : params.get( 'metrics', tf.keras.metrics.BinaryCrossentropy() ) }

        model.compile( optimizer = optimizer, loss = losses, metrics = metrics )

        return model


    def Fit_Model( self, model, x_data, y_data, params ) :

        x_train_data = { 'Output_00' : x_data['train'] }
        y_train_data = { 'Output_00' : y_data['train'] }

        valid_data = ( { 'Output_00' : x_data['valid'] }, { 'Output_00' : y_data['valid'] } )

        callbacks = []

        checkpoint_monitor = params['check_monitor']
        if valid_data is None : checkpoint_monitor = 'loss'

        check_filePath = params['checkpoints']
        save_best_only = tf.keras.callbacks.ModelCheckpoint(\
            filepath = check_filePath, monitor = checkpoint_monitor, save_best_only = True, save_weights_only = True, verbose = 2 \
        )
        callbacks.append( save_best_only )

        model.reset_metrics()
        
        model.fit( x_train_data, y_train_data, validation_data = valid_data, callbacks = callbacks, \
            shuffle = True, validation_freq = 1, \
            epochs = params['epochs'], verbose = 2 \
        ) # verbose 0 makes system stoned.

        model.load_weights( filepath = check_filePath ) # because of cp_save_best_only

        del params, x_data, y_data

        return model

"""
params = {}

# a list of ( dim_x, dropout, activation )

params['pick_trials'] = 20

x = 5
params[ 'dense_layers' ] = [ ( x, 'linear' ), ( x, 'relu' ), ( x, 'relu' ), ( x, 'relu' ), ( x, 'relu' ) ]
params['closing_act'] = 'linear'
params[ 'optimizer'] = tf.keras.optimizers.Adam
params['loss'] = tf.keras.losses.mean_absolute_error
params[ 'metrics'] = tf.keras.metrics.mean_absolute_error
params['check_monitor'] = 'val_loss'
params['checkpoints'] = 'CheckPoint_Simpleton'
params['epochs'] = 100

data_exp = [ [ x + y for x in range( 5 ) ] for y in range( 100 ) ]
data_exp = np.array( data_exp, dtype = np.float32 )
data_obs = data_exp[:, 0] + data_exp[:, 1] + data_exp[:, 1] * data_exp[:, 2] + data_exp[:, 3] / ( data_exp[:, 4] + 1 )

x_data = {}
x_data['train'] = data_exp[:50, :]; x_data['valid'] = data_exp[50:80, :]; x_data['test'] = data_exp[80:, :]
y_data = {}
y_data['train'] = data_obs[:50]; y_data['valid'] = data_obs[50:80]; y_data['test'] = data_obs[80:]

singletone = Model_Simpleton( x_data, y_data )

model = singletone.Get_Model( params )

pass
"""