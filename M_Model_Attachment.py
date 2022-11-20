import tensorflow as tf
import numpy as np

from M_Config import *
from M_EarlyStop_0 import *

from Task_Params_Feeder import *

from M_Model_Fitter import *

class Attachment( Model_Fitter ) :

    def __init__( self, branch_id, att_id, baseline_loss, dims_core, config, core_model, closing_model, x_data, y_data ) :

        super().__init__( x_data, y_data )

        self.branch_id = branch_id
        self.att_id = att_id
        self.baseline_loss = baseline_loss

        self.dims_core = dims_core
        
        self.config = config

        # Non-None core_model is used for a special purpose, although self.core_model is normally assumed to be created and trained by this object.
        self.core_model = core_model 

        self.closing_model = self.Copy_Model( closing_model ) # It's a contract not to change caller's closing_model.

        self.dim_socket = closing_model.inputs[0].shape[-1]
        assert self.Is_End_With_Socket( self.dims_core )

        self.inputs = tf.keras.layers.Input( shape = ( self.dim_socket, ), dtype = tf.float32 )

        self.model = None
        self.train_params = None

        self.core_model_name = self.Create_Core_Name( self.att_id, self.branch_id )
        self.closing_model_name = self.closing_model.name
        self.model_name = 'Attachment_' + str( self.att_id ).zfill( 2 )

        return

    def Is_End_With_Socket( self, dims_core ) :
        if len( dims_core ) > 0 and dims_core[-1] != self.dim_socket :
            raise Exception( "dims_core[-1] {} mismatches '.D.X.Soc' {}".format( dims_core[-1], self.config['.D.X.Soc'] ) )
        else : return True

    def nWeights( self, model ) :  return np.sum([np.prod(v.shape) for v in model.trainable_variables])

    def Train( self, train_params ) :

        self.train_params = train_params

        # Sure Consistent( self.model, self.core_model, self.closing_model ), after this call.
        self.model, self.core_model = self.Create_Models( self.train_params )

        assert self.Is_Consistent_Weight_Numbers( self.model, self.core_model, self.closing_model )

        print( "\nD. ======= Att-{}, Train: Starting......... loss = {:.5f}, nWeights = {:,}\n".format( self.att_id, self.baseline_loss, self.nWeights( self.model )  ) )

        self.model = self.Compile_Model( self.model, self.train_params )

        self.model, self.core_model, self.closing_model = self.Fit_Model( self.model, self.train_params )

        #self.core_model, self.closing_model = self.

        assert self.Is_Consistent_Weight_Numbers( self.model, self.core_model, self.closing_model )

        print( "\nD. ======= Att-{}, Train: Done.\n".format( self.att_id  ) )

        return self.Assess_Loss( self.model, self.config['loss_unified'] )


    def Get_Count_Params( self ) :
        return self.model.count_params()


    def Divide_Tensor( self, tensor, axis, dim_outputs ) :

        dim_total = 0; indices = [0]
        for dim in dim_outputs : 
            dim_total += dim; indices.append( dim_total )

        assert tensor.shape[ axis ] == dim_total
        tensors = [ tensor[:, indices[n] : indices[n+1] ] for n in range( len(indices) - 1 ) ]

        return tensors


    def Create_Models( self, train_params ) :

        core_model = self.Create_Core_Model( self.inputs, self.dims_core, self.x_data, train_params )
    
        model = self.Compose_Core_Closing_Models( core_model, self.closing_model, self.config )

        assert self.Is_Consistent_Weight_Numbers( model, core_model, self.closing_model )

        # Sure Consistent ( model, core_model, self.closing_model )
        return model, core_model


    def Compose_Core_Closing_Models( self, core_model, closing_model, config ) :

        x = core_model( self.inputs ) # core_model inputs a total_X, amd outputs a tensor of dim_socket size.
        x = closing_model( x ) # closing_model inputs a tensor of dim_socket size, and outputs tensors of self.dim_closing size.

        model = self.Create_Model_From_Tensor( inputs = self.inputs, tensor = x, branch_id = self.branch_id, att_id = self.att_id, config = config )

        assert self.Is_Consistent_Weight_Numbers( model, core_model, closing_model )

        return model


    def Create_Model_From_Tensor( self, inputs, tensor, branch_id, att_id, config ) :

        # All in order to assign correct loss functions. See Compile_Model.
        if config['.Out.Wise'] is True : # Just one base_model.
            base_models = [ tf.keras.models.Model( inputs = inputs, outputs = tensor, name = self.Create_Base_Name( att_id, branch_id ) ) ] # It's self.branch_id, which is actually its output id.
        elif config['.Out.Wise'] is False : # Multiple base_models.
            outputs = self.Divide_Tensor( tensor = tensor, axis = -1, dim_outputs = config['Dim.Outs'] )
            nOutputs = len( config['Dim.Outs'] )
            base_models = [ tf.keras.models.Model( inputs = inputs, outputs = outputs[outputId], name = self.Create_Base_Name( att_id, outputId ) ) for outputId in range( nOutputs ) ]

        model = tf.keras.models.Model( inputs = inputs, outputs = [ base_model( inputs ) for base_model in base_models ], name = self.model_name )

        return model


    def Divide_Tensor( self, tensor, axis, dim_outputs ) :

        dim_total = 0; indices = [0]
        for dim in dim_outputs : 
            dim_total += dim; indices.append( dim_total )

        assert tensor.shape[ axis ] == dim_total
        tensors = [ tensor[:, indices[n] : indices[n+1] ] for n in range( len(indices) - 1 ) ]

        return tensors


    def Idenize( self, model, x_data, train_params ):

        model.compile( optimizer = tf.keras.optimizers.Adadelta( 0.9 ), loss = tf.losses.mean_absolute_error, metrics = None )

        x_data_new = { 'train' : x_data['train'], 'valid' : x_data['valid'] }
        y_data_new = { 'train' : x_data['train'], 'valid' : x_data['valid'] } # so, x_train_data == y_train_data.

        print( "\nD. ======= Att-{}, Idenize: Starting.........\n".format( self.att_id  ) )

        custom = { 'subject' : "Fitting attachment.core(x_data) to x_data", '.epochs' : self.config['AttIdenizeEpo']  }
        custom_fit_params = self.Create_Fit_Params( train_params, custom )

        patience = 20
        leftop_epoloss_list = [ ( 25, float('inf') ) ]
        model = self.Fit_Model_Once( model, train_params, custom_fit_params, x_data_new, y_data_new, patience, leftop_epoloss_list )

        print( "\nD. ======= Att-{}, Idenize: Done.\n".format( self.att_id  ) )

        return model


    def Create_Base_Name( self, att_id, outputId ) :
        return 'OUT_' + str( att_id ).zfill( 2 ) + '_' + str( outputId ).zfill( 2 )


    def Create_Core_Name( self, att_id, branch_id ) :
        return 'AttCore' +  str( att_id ).zfill( 2 ) + '_' + str( branch_id ).zfill( 2 )


    def Pick_Random_Activation( self ) :
        rand = np.random.randint( low = 0, high = len( self.config['Att.Act'] ), dtype = int )
        activation = self.config['Att.Act'][ rand ]
        print( "----------- Activation picked = {}".format( activation ) )

        return activation


    def Create_Core_Model( self, inputs, dims_core, x_data, train_params ) :

        def Add_Dense_Layer( dense_tensor, dim, kernel_init, bias_init, act, dropoutRate ) : # PENDING - place BatchNormalization properly, jsut before activiation.

            if dropoutRate > 0 : dense_tensor = tf.keras.layers.Dropout( dropoutRate ) ( dense_tensor ) #=========================== Dropout               
            #dense_tensor = tf.keras.layers.BatchNormalization()( dense_tensor ) # ======================== Batch Normalization                  
           
            dense_tensor = tf.keras.layers.Dense( units = dim, activation = act, \
                kernel_initializer = kernel_init, bias_initializer = bias_init ) ( dense_tensor )

            return dense_tensor
        #-----------------------------------------------------------------------------------------

        modelParams = train_params['modelParams']
        base_tensor = inputs

        activation = None

        if self.config['.Att.Ver'] == 0 : # Debugging purpose.
            kernel_init = tf.keras.initializers.Identity()
            bias_init = tf.keras.initializers.Zeros()
            base_tensor = tf.keras.layers.Dense( units = dims_core[-1], activation='linear', \
                kernel_initializer = kernel_init, bias_initializer = bias_init ) ( base_tensor )
        else :
            loop = 0
            dropoutRate = modelParams['.dropIn']

            for dim in  dims_core :
         
                if loop > 0 : dropoutRate = dropoutRate * 2.5                        
                kernel_init, bias_init = None, None
                act = self.Pick_Random_Activation()
                activation = act

                if self.config['.Att.Ver'] in [ 3, 4 ] :
                    #{ '0' : debuggin. ('Unit core weights', 'Fitting on identical data'). '1' : (0,0), '2': (0,1), '3': (1, 0), '4' : (1, 1) }
                    kernel_init = tf.keras.initializers.Identity()
                    bias_init = tf.keras.initializers.Zeros()
         
                base_tensor = Add_Dense_Layer( base_tensor, dim, kernel_init, bias_init, act, dropoutRate )
                loop += 1

        model = tf.keras.models.Model( inputs = inputs, outputs = base_tensor, name = self.core_model_name )

        if self.config['.Att.Ver'] in [ 2, 4 ] :

            model = self.Idenize( model, x_data, train_params )
                        
        return model


    def Get_Inverse_Activation( self, activation ) :
        def identical( x ) : return x
        def logit( x ) : return tf.math.log( x / ( 1.0 - x ) )
        def in_tanh( x ) : return 0.5 * tf.math.log( ( 1.0 + x ) / ( 1.0 - x ) )
        inverse = identical
        if activation == 'sigmoid' : inverse = logit
        elif activation == 'tanh' : inverse = in_tanh
        return inverse


    def Compile_Model( self, model, train_params ) :

        modelParams = train_params['modelParams']; trainResources = train_params['trainResources']
        feeder = Params_Feeder()
        optimizer = feeder.translator_optimizer( modelParams['.optmer'] )
        del feeder

        losses = None

        if self.config['.Out.Wise'] is True :
            losses = { self.Create_Base_Name( self.att_id, self.branch_id ) : trainResources['.loss']( self.branch_id ) }

        elif self.config['.Out.Wise'] is False :
            nOutputs = len( self.config['Dim.Outs'] )
            losses = { self.Create_Base_Name( self.att_id, outputId ) : trainResources['.loss']( outputId ) for outputId in range( nOutputs ) }

        # Or, simply - losses = trainResources['.loss']( 0 )
        metrics = None #{ None for query_id in self.task.queries }

        model.compile( optimizer = optimizer, loss = losses, metrics = metrics )

        return model

    def Fit_Model( self, model, train_params ) :

        print( "\nD. ======= Att-{}, Fit_Model: Starting.........\n".format( self.att_id  ) )

        custom = { 'subject' : "Fitting an attachment", '.epochs' : self.config['AttachEpo'] } # 3 epochs only, it proves that attachments tend to fit at the first epoch and no more improvement.

        if self.config['.Att.Ver'] == 0 : 
            weights_before = self.core_model.get_weights()
            custom['.epochs'] = 1 # if 0 is impossible.

        custom_fit_params = self.Create_Fit_Params( train_params, custom )

        patience = 15
        leftop_epoloss_list = [ ( 15, self.baseline_loss ) ]
        model = self.Fit_Model_Once( model, train_params, custom_fit_params, self.x_data, self.y_data, patience, leftop_epoloss_list )

        if self.config['.Att.Ver'] == 0 :
            self.core_model.set_weights( weights_before )

        core_model, closing_model = self.Decompose_Model( model )

        print( "\nD. ======= Att-{}, Fit_Model: Done.\n".format( self.att_id  ) )

        # Sure Consistent( model, core_model, closing_model )
        return model, core_model, closing_model


    def Decompose_Model( self, model ) :
        #--------------------- Updating core_model and closing_model, in accordance with model update.
        core_model = model.get_layer(model.output_names[0]).get_layer( self.core_model_name )
        closing_model = model.get_layer(model.output_names[0]).get_layer( self.closing_model_name )
        
        return core_model, closing_model


    def Produce_Output( self, segment ) :

        output = self.model( self.x_data[ segment ], training = False )

        output = output if isinstance( output, list ) else [ output ]

        return output


    def Produce_Core_Output( self, segment ) :
        
        core_output = self.core_model( self.x_data[ segment ], training = False )

        core_output = output if isinstance( core_output, list ) else [ core_output ]

        return core_output


    

