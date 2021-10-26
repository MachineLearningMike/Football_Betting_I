import numpy as np
import tensorflow as tf

from M_Config import *

from M_EarlyStop_0 import *
from M_Loss_Monitor import *
from M_Loss_Monitor_2 import *

from M_Model_Attachment import *

from M_Model_Fitter import *
from Task_Params_Feeder import *

from memory_profiler import profile

class Model_Branch( Model_Fitter ):

    def __init__( self, branch_id, inputs, base_X, dim_closing, act_closing, config, x_data, y_data ):

        super().__init__( x_data, y_data )

        self.branch_id = branch_id
        self.inputs = inputs
        self.base_X = base_X # base_X is coming from inputs.
        self.dim_closing = dim_closing
        self.act_closing = act_closing

        self.dim_socket = base_X.shape[-1] * config['.D.X.Soc']
        assert self.Is_End_With_Socket( config['dims.core'] ) == True
        assert self.Is_End_With_Socket( config['dims.attach'] ) == True

        self.nBranches = len( config['Dim.Outs'] ) if config['.Out.Wise'] else 1

        self.config = config.copy() # We don't want possible changes propageted to the calling module.

        self.train_params = None

        self.model = None
        self.core_model = None
        self.closing_model = None

        self.emb_homes = None ; self.emb_aways = None

        self.history = None

        self.model_name = 'Branch_' + str( self.branch_id ).zfill( 2 )
        self.core_model_name = 'core_model'
        self.closing_model_name = 'closing_model'

    def Is_End_With_Socket( self, dims ) :
        if isinstance( dims[0], list ) :
            for sub_dims in dims :
                if self.Is_End_With_Socket( sub_dims ) is True : continue
                else : return False
            return True
        else :
            if len( dims ) > 0 and dims[-1] != self.dim_socket :
                raise Exception( "dims[-1] {} mismatches '.D.X.Soc' {}".format( dims[-1], self.config['.D.X.Soc'] ) )
            else: return True

    def Get_Count_Params( self ) :
        return self.model.count_params()

    def Get_MinLoss( self, history, val_weight ) :
        assert 0. <= val_weight and val_weight <= 1.

        loss = np.array( history.get( 'loss', None ) )
        val_loss = np.array( history.get( 'val_loss', None ) )

        eloss = None
        if len( loss.shape ) > 0 :
            if len( loss.shape ) == len( val_loss.shape ) :
                eloss = val_weight * val_loss + ( 1. - val_weight ) * loss
            elif len( val_loss.shape ) == 0 :
                eloss = loss
            else : raise Exception('Help :)')
        else : raise Exception('Help :)')

        min = np.min( eloss )
        if min is None : min = float('inf')

        return min

    def nWeights( self, model ) :  return np.sum([np.prod(v.shape) for v in model.trainable_variables])

    def Train( self, train_params ) :
        
        print( "======================================================================================================")

        print( "\nA. ======= B-{}, Train: Starting.........\n".format( self.branch_id ) )

        self.train_params = train_params # This is setting self.train_params.

        model_min, core_model_min, closing_model_min = None, None, None ; loss_min = float('inf')

        verbosity = tf.compat.v1.logging.get_verbosity()
        tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )

        #-------------------------------------- Searching for a baseline model ------------------------------

        print( "\nA. ======= B-{}, Train: Searching for a baseline model.........\n".format( self.branch_id ) )

        for loop in range ( self.config['pick_trials'] ) :
           
            # Sure Consistent( model, core_model, closing_model )
            model, core_model, closing_model, loss = self.Pickup_Candidate( train_params, self.config, loop )

            if loss < loss_min :
                del model_min, core_model_min, closing_model_min
                model_min, core_model_min, closing_model_min, loss_min = model, core_model, closing_model, loss
                print( "--------------------------------------------------- min_loss : {:.5f}\n".format( loss_min ) )
            else :
                del model, core_model, closing_model, loss

        # Sure Consistent( model, core_model, closing_model )
        model, core_model, closing_model, loss = model_min, core_model_min, closing_model_min, loss_min

        print( "\nA. ======= B-{}, Train: Found a baseline model. loss = {:.5f}, nWeights = {:,}\n".format( self.branch_id, loss, self.nWeights( model ) ) )

        #-----------------------------------------------------------------------------------------------------------

        print( "\nA. ======= B-{}, Train: Fitting the baseline model.........\n".format( self.branch_id ) )
        model_org = self.Deep_Copy_Model( model, train_params, self.config, copy_optimizer = True )
        loss_org = loss = self.Assess_Loss( model_org, self.config['loss_unified'] )

        # Before and after calling, we are sure Consistent( model, core_model, closing_model )
        model, core_model, closing_model, model_status = self.Fit_Baseline( model, core_model, closing_model, self.train_params, self.config ) # includes adding and training attachments.

        loss = self.Assess_Loss( model, self.config['loss_unified'] )
        print( "\nA. ======= B-{}, Train: Fitted the baseline model. loss = {:.5f}, nWeights = {:,}\n".format( self.branch_id, loss, self.nWeights( model ) ) )

        print( "\nA. ======= B-{}, Train: Conclusion: ".format( self.branch_id ), end = '' )
        if loss < loss_org :
            print( "!!!!! Loss improved from the baseline: model_status = {}, Loss = {:.5f} ---> {:.5f} nWeights = {:,}\n".format( model_status, loss_org, loss, self.nWeights( model ) ) )
            self.model, self.core_model, self.closing_model, self.loss = model, core_model, closing_model, loss
        else :
            print( "????? Loss NOT improved from the baseline: model_status = {}. Loss = {:.5f} ---> {:.5f} nWeights = {:,}\n".format( model_status, loss_org, loss, self.nWeights( model ) ) )
            print( "\nA. ======= B-{}, Train: Back to the baseline model. Loss = {:.5f}, nWeights = {:,}\n".format( self.branch_id, loss_org, self.nWeights( model_org ) ) )
            self.model = model_org
            self.core_model, self.closing_model = self.Decompose_Model( self.model ) # , although we know they are core_model_min, and closing_model_min, respectively.

        # Sure Consistent( self.model, self.core_model, self.closing_model )
        return self.model


    def Pickup_Candidate( self, train_params, config, loop ) :

        model, core_model, closing_model = self.Create_Models( train_params, config )
        model = self.Compile_Model( model, train_params, config )

        train_params_copy = self.train_params.copy()

        print( "\nB. ======= B-{}, Pickup_Candidate: loop = {}".format( self.branch_id, loop ) )
        custom = { 'subject' : "B. ======= B-{}, Fitting baseline candidate.........".format( self.branch_id ), '.epochs' : 1 } # 3, instead of 1, to train optimizer.
        custom_fit_params = self.Create_Fit_Params( train_params_copy, custom )
        assert self.Is_Consistent_Weight_Numbers( model, core_model, closing_model )

        model = self.Fit_Model_Once( model, train_params_copy,  custom_fit_params, self.x_data, self.y_data )

        loss = self.Assess_Loss( model, self.config['loss_unified'] )
        core_model, closing_model = self.Decompose_Model( model )

        # Sure Cosistent( model, core_model, closing_model )
        return model, core_model, closing_model, loss


    def Create_Models( self, train_params, config ) :

        core_model = self.Create_Core_Model( self.inputs, self.base_X, train_params['modelParams'], config )
        closing_model = self.Create_Closing_Model( self.dim_socket, self.dim_closing, train_params['modelParams'], config )

        model = self.Compose_Core_Closing_Models( core_model, closing_model, config )

        return model, core_model, closing_model


    def Compose_Core_Closing_Models( self, core_model, closing_model, config ) :

        x = core_model( self.inputs ) # core_model inputs a base_X, amd outputs a tensor of dim_socket size.
        x = closing_model( x ) # closing_model inputs a tensor of dim_socket size, and outputs tensors of self.dim_closing size.

        model = self.Create_Model_From_Tensor( inputs = self.inputs, tensor = x, branch_id = self.branch_id, config = config )

        assert self.Is_Consistent_Weight_Numbers( model, core_model, closing_model )

        return model


    def Create_Model_From_Tensor( self, inputs, tensor, branch_id, config ) :

        # All in order to assign correct loss functions. See Compile_Model.
        if config['.Out.Wise'] is True : # Just one base_model.
            base_models = [ tf.keras.models.Model( inputs = inputs, outputs = tensor, name = self.Create_Base_Name( branch_id ) ) ] # It's self.branch_id, which is actually its output id.
        elif config['.Out.Wise'] is False : # Multiple base_models.
            outputs = self.Divide_Tensor( tensor = tensor, axis = -1, dim_outputs = config['Dim.Outs'] )
            nOutputs = len( config['Dim.Outs'] )
            base_models = [ tf.keras.models.Model( inputs = inputs, outputs = outputs[outputId], name = self.Create_Base_Name( outputId ) ) for outputId in range( nOutputs ) ]

        model = tf.keras.models.Model( inputs = inputs, outputs = [ base_model( inputs ) for base_model in base_models ], name = self.model_name )

        return model


    def Create_Base_Name(self, outputId ) :
        return 'OUT_' + str( outputId ).zfill( 2 )  # use log( N_QUERIES )


    def Create_Core_Model( self, inputs, base_X, modelParams, config ) :

        def Add_Dense_Layer( dense_tensor, dimension, activation, dropoutRate, modelParams ) : # PENDING - place BatchNormalization properly, jsut before activiation.

            if dropoutRate > 0 : dense_tensor = tf.keras.layers.Dropout( dropoutRate ) ( dense_tensor ) #=========================== Dropout               
            
            if self.config['UseBN'] :
                dense_tensor = tf.keras.layers.BatchNormalization()( dense_tensor ) # ======================== Batch Normalization                  
            
            regularizer = modelParams['.regur'] # Consider not to regularize weights, as BatchNormalization is there, and as we need free weights.

            dense_tensor = tf.keras.layers.Dense( units = dimension, activation = activation, 
            #kernel_initializer = initializers.Zeros(), bias_initializer = initializers.Zeros(), \
            #kernel_regularizer = regularizer, bias_regularizer = regularizer, \
            #kernel_constraint = tf.keras.constraints.MaxNorm(2) DO NOT give too much constraints.\
            ) ( dense_tensor )
            return dense_tensor

        inputs = inputs
        base_tensor = base_X # base_X comes from inputs.

        actions = modelParams['.action']

        loop = 0
        for dim in  config['dims.core'] :
            dropoutRate = modelParams['.dropIn']
            if loop > 0 : dropoutRate = dropoutRate * 2.5
            base_tensor = Add_Dense_Layer( base_tensor, dim, actions[ loop ], dropoutRate, modelParams )
            loop += 1

        model = tf.keras.models.Model( inputs = inputs, outputs = base_tensor, name = self.core_model_name )

        del inputs, base_X

        return model


    def Create_Closing_Model( self, dim_socket, dim_closing, modelParams, config ) :
        
        input = tf.keras.layers.Input( shape = ( dim_socket,), dtype = tf.float32 )

        be_careful_with_softmax = config['.Out.Wise'] is False and self.act_closing == 'softmax'

        activation = self.act_closing
        if be_careful_with_softmax : activation = 'linear'

        # Consider adding Dropout layer.
        tensor = input
        tensor = tf.keras.layers.BatchNormalization()( input ) # ======================== Batch Normalization                  

        tensor = tf.keras.layers.Dense( units = dim_closing, activation = activation ) ( tensor )

        if be_careful_with_softmax : 
            outputs = self.Divide_Tensor( tensor = tensor, axis = -1, dim_outputs = config['Dim.Outs'] )
            outputs = [ tf.keras.layers.Softmax( axis = -1 )( output ) for output in outputs ]
            tensor = tf.concat( outputs, axis = -1 )

        model = tf.keras.models.Model( inputs = input, outputs = tensor, name = self.closing_model_name )

        return model
        

    def Divide_Tensor( self, tensor, axis, dim_outputs ) :

        dim_total = 0; indices = [0]
        for dim in dim_outputs : 
            dim_total += dim; indices.append( dim_total )

        assert tensor.shape[ axis ] == dim_total
        tensors = [ tensor[:, indices[n] : indices[n+1] ] for n in range( len(indices) - 1 ) ]

        return tensors


    def Decompose_Model( self, model ) :
        #--------------------- Updating core_model and closing_model, in accordance with model update.
        core_model = model.get_layer(model.output_names[0]).get_layer( self.core_model_name )
        closing_model = model.get_layer(model.output_names[0]).get_layer( self.closing_model_name )
        
        return core_model, closing_model



    def Compile_Model( self, model, train_params = None, config = None, branch_id = -1 ) : # Do NOT remove the default value of train_params. Outside modules don't know train_params.

        # This generalize the utility of this method, so it can be used from other modules.
        if train_params is None : train_params = self.train_params
        if branch_id < 0 : branch_id = self.branch_id
        if config is None : config = self.config

        modelParams = train_params['modelParams']; trainResources = train_params['trainResources']
        feeder = Params_Feeder()
        optimizer = feeder.translator_optimizer( modelParams['.optmer'] )
        del feeder

        losses = None
        if config['.Out.Wise'] is True :
            losses = { self.Create_Base_Name( branch_id ) : trainResources['.loss']( branch_id ) } # loss
        elif config['.Out.Wise'] is False :
            nOutputs = len( config['Dim.Outs'] )
            losses = { self.Create_Base_Name( outputId ) : trainResources['.loss']( outputId ) for outputId in range( nOutputs ) }
        # Or, simply - losses = trainResources['.loss']( 0 )
        metrics = None #{ None for query_id in self.task.queries }

        model.compile( optimizer = optimizer, loss = losses, metrics = metrics )

        return model

    def Deep_Copy_Model( self, model, train_params, config, copy_optimizer = False ) :

        clone = self.Copy_Model( model ) # copy over the structrue.
        clone = self.Compile_Model( clone, train_params, config )

        if copy_optimizer is True :

            symbolic_weights = getattr( model.optimizer, 'weights' )
            optimizer_weights_source = tf.keras.backend.batch_get_value( symbolic_weights )

            trainables_temp = model.trainable_variables.copy() # Copy it to avoid contaminating the model.
            zero_grads = [ tf.zeros_like( w ) for w in trainables_temp ]
            clone.optimizer.apply_gradients( zip( zero_grads, trainables_temp ) ) # The optimizer has now suitable weights structure ready.

            #if len( weight_values ) > 0 : 
            try :
                clone.optimizer.set_weights( optimizer_weights_source )
            except :
                print( "Deep_Copy_Model: close.optimizer.set_weights( weight_values) raised an exception." )

        return clone


    def Fit_Baseline( self, model, core_model, closing_model, train_params, config, max_val_loss = None ) :
        # Caller is responsible for Consistent( model, core_model, closing_model )

        model_status = 'baseline'

        #------------------------------------ Main_Fitting -------------------------------------------

        main_success = False

        print( "\nB. ======= B-{}, Fit_Baseline: Main_Fitting......... \n".format( self.branch_id ) )

        model_org = self.Deep_Copy_Model( model, train_params, config, copy_optimizer = True )
        loss_org = self.Assess_Loss( model_org, self.config['loss_unified'] )

        custom = { 'subject' : "B. ======= B-{}, Fit_Baseline: Main_Fitting loop.".format( self.branch_id ), '.epochs' : config['BrMainEpo'] }
        custom_fit_params = self.Create_Fit_Params( train_params, custom )
        fluctuation_space = 0.0 # ------------- Application-specific.

        patience = 10; leftop_epoloss_list = [ ( 15, loss_org ) ]
        model = self.Fit_Model_Once( model, train_params, custom_fit_params, self.x_data, self.y_data, patience, leftop_epoloss_list ) # model normally changes.
        print( "\nB. ======= B-{}, Fit_Baseline: Concluding Main_Fitting......... ".format( self.branch_id ) )

        # model has changed by the preceding call to Fit_Model_Once, whether loss was improved or not.
        core_model, closing_model = self.Decompose_Model( model )

        assert self.Is_Consistent_Weight_Numbers( model, core_model, closing_model )
        loss = self.Assess_Loss( model, self.config['loss_unified'] )

        if loss_org <= loss :
            print( "B. ======= B-{}, Fit_Baseline: ????? Model Main_Fitted is NOT better. Loss = {:.5f} ---> {:.5f}, nWeights = {:,}".format( self.branch_id, loss_org, loss, self.nWeights( model ) ) )
            model = model_org; loss = self.Assess_Loss( model, self.config['loss_unified'] )
            model_status = model_status
            print( "B. ======= B-{}, Fit_Baseline: Back to '{}': Loss = {:.5f}, nWeights = {:,}".format( self.branch_id, model_status, loss, self.nWeights( model ) ) )
        else :
            print( "B. ======= B-{}, Fit_Baseline: !!!!!! Model Main_Fitted IS better. Loss = {:.5f} ---> {:.5f}, nWeights = {:,}".format( self.branch_id, loss_org, loss, self.nWeights( model ) ) )
            model = model; loss = loss
            model_status = model_status + '-mained'
            print( "B. ======= B-{}, Fit_Baseline: Forward to '{}': Loss = {:.5f}, nWeights = {:,}".format( self.branch_id, model_status, loss, self.nWeights( model ) ) )

            main_success = True

        # state : consistent (model, core_model, closing_model), train_params, config, model_status

        #------------------------------------ Attaching -------------------------------------------
        att_success = False

        if config['.Attach'] is True :

            print( "\nB. ======= B-{}, Fit_Baseline: Attatching......... \n".format( self.branch_id ) )

            model_org = self.Deep_Copy_Model( model, train_params, config, copy_optimizer = False )
            core_model_org = self.Copy_Model( core_model )

            loss_org = self.Assess_Loss( model_org, self.config['loss_unified'] )

            # Before and after calling, consistent( model, core_model, closing_model )
            model, core_model, closing_model, attached = self.Attach_Submodels( train_params, model, core_model, closing_model, config )

            print( "\nB. ======= B-{}, Fit_Baseline: Concluding Attatching.........".format( self.branch_id ) )

            if model is not None and attached is True :

                assert self.Is_Consistent_Weight_Numbers( model, core_model, closing_model )
                loss = self.Assess_Loss( model, self.config['loss_unified'] )

                if loss_org <= loss : # Give up the new ones, and take the originals back.
                    print( "B. ======= B-{}, Fit_Baseline: ????? Model with attachments is NOT better. Loss = {:.5f} ---> {:.5f}, nWeights = {:,}".format( self.branch_id, loss_org, loss, self.nWeights( model ) ) )
                    model, core_model = model_org, core_model_org; loss = self.Assess_Loss( model, self.config['loss_unified'] )
                    model_status = model_status
                    print( "B. ======= B-{}, Fit_Baseline: Back to '{}': Loss = {:.5f}, nWeights = {:,}".format( self.branch_id, model_status, loss, self.nWeights( model ) ) )
                else :
                    print( "B. ======= B-{}, Fit_Baseline: !!!!! Model with attachments IS better. Loss = {:.5f} ---> {:.5f}, nWeights = {:,}".format( self.branch_id, loss_org, loss, self.nWeights( model ) ) )
                    model, core_model = model, core_model; loss = loss
                    model_status = model_status + '-attached'
                    print( "B. ======= B-{}, Fit_Baseline: Forward to '{}': Loss = {:.5f}, nWeights = {:,}".format( self.branch_id, model_status, loss, self.nWeights( model ) ) )

                    att_success = True # ------------------------------------------------
            else :
                print( "B. ======= B-{}, Fit_Baseline: ????? No sub-models attached.".format( self.branch_id ) )
                model, core_model = model_org, core_model_org; loss = self.Assess_Loss( model, self.config['loss_unified'] )
                model_status = model_status
                print( "B. ======= B-{}, Fit_Baseline: Remaining at '{}': Loss = {:.5f}, nWeights = {:,}".format( self.branch_id, model_status, loss, self.nWeights( model ) ) )

            # state : model, core_model, closing_model, train_params, config, model_status

        #------------------------------------ Final_Fitting -------------------------------------------

        final_success = False

        if att_success is True and config['BrFinal'] is True :

            print( "\nB. ======= B-{}, Fit_Baseline: Final_Fitting......... \n".format( self.branch_id ) )

            model_org = self.Deep_Copy_Model( model, train_params, config, copy_optimizer = False )
            loss_org = self.Assess_Loss( model_org, self.config['loss_unified'] )
    
            custom = { 'subject' : "B. ======= B-{}, Fit_Baseline: Final_Fitting loop.".format( self.branch_id ), '.epochs' : config['BrFinalEpo'] }
            custom_fit_params = self.Create_Fit_Params( train_params, custom )

            patience = 5
            leftop_epoloss_list = [ ( 10, loss_org ) ]
            model = self.Fit_Model_Once( model, train_params, custom_fit_params, self.x_data, self.y_data, patience, leftop_epoloss_list )

            # model has changed by the preceding call to Fit_Model_Once, whether loss was improved or not.
            core_model, closing_model = self.Decompose_Model( model )

            print( "\nB. ======= B-{}, Fit_Baseline: Concluding Final_Fitting......... ".format( self.branch_id ) )

            assert self.Is_Consistent_Weight_Numbers( model, core_model, closing_model )
            loss = self.Assess_Loss( model, self.config['loss_unified'] )

            if loss_org <= loss :
                print( "B. ======= B-{}, Fit_Baseline: ????? Model Final_Fitted is NOT better. Loss = {:.5f} ---> {:.5f}, nWeights = {:,}".format( self.branch_id, loss_org, loss, self.nWeights( model ) ) )
                model = model_org; loss = self.Assess_Loss( model, self.config['loss_unified'] )
                model_status = model_status
                print( "B. ======= B-{}, Fit_Baseline: Back to '{}': Loss = {:.5f}, nWeights = {:,}".format( self.branch_id, model_status, loss, self.nWeights( model ) ) )
            else :
                print( "B. ======= B-{}, Fit_Baseline: !!!!!! Model Final_Fitted IS better. Loss = {:.5f} ---> {:.5f}, nWeights = {:,}".format( self.branch_id, loss_org, loss, self.nWeights( model ) ) )
                model = model; loss = loss
                model_status = model_status + '-finaled'
                print( "B. ======= B-{}, Fit_Baseline: Forward to '{}': Loss = {:.5f}, nWeights = {:,}".format( self.branch_id, model_status, loss, self.nWeights( model ) ) )

                final_success = True
        
        elif att_success is not True and config['BrFinal'] is True :
            print( "\nB. ======= B-{}, Fit_Baseline: Final_Fitting skipped.".format( self.branch_id ) )

        # Sure Consistent( model, core_model, closing_model )
        return model, core_model, closing_model, model_status


    def Attach_Submodels( self, train_params, model, core_model, closing_model, config ) :
        # Caller is responsible for Consistent( model, core_model, closing_model )

        print( "\nC. ======= B-{}, Attach_Submodels: Starting......... ".format( self.branch_id ) )

        loss_org = self.Assess_Loss( model, self.config['loss_unified'] )
        nWeights_org = np.sum([np.prod(v.shape) for v in model.trainable_variables])

        prev_att = self # PENDING. Vulnerable extreamly. It requires self.core_model == core_model.
        model, loss_after, attached = None, float('inf'), False
        #modelParams = train_params['modelParams']; fitParams = train_params['.fitParams']; trainResources = train_params['trainResources']; dataUse = train_params['.dataUse']

        lossMonitorParams = {}
        lossMonitorParams['.patience'] = 3 if super_config['.Test'] else 5 # Finding attachment is fast.
        lossMonitorParams['min_loss_improvement_per_loss'] = 0.03
        lossMonitorParams['countdown_params'] = [ { 'loss_x' : 1000., 'patience_x' : 1. } ] if super_config['.Test'] else [ { 'loss_x' : 1., 'patience_x' : 1. } ]
        lossMonitorParams['message_heading'] = "Adding attachments"
        loss_monitor = Loss_Monitor( lossMonitorParams )

        attachments = []
        min_loss = loss_org
        config_copy = config.copy()
        dims_left = config['dims.attach'].copy()

        if len( dims_left ) > 0 : # Prepare for the next loop.
            # Sure Consistent( prev_att.model, prev_att.core_model, prev_att.closing_model ), after this call.
            prev_att = Attachment( self.branch_id, -1, None, [], None, core_model, closing_model, self.x_data, self.y_data ) # No change to core_model and closing_model.
            prev_core_output = { segment : prev_att.Produce_Core_Output( segment ) for segment in [ seg for seg, _ in self.y_data.items() ] }

        loop = -1
        while len( dims_left ) > loop + 1 :
            loop += 1
            print( "\nC. ======= B-{}, Attach_Submodels: Looping......... {}".format( self.branch_id, loop ) )

            dims = dims_left[ loop ] # [ dims_left.pop() ] # Open to change policy.
            att_id = len( attachments )
            
            # prev_closing_model is NOT changed by this call by the contract bound with Attachment. So, sure Consistent( prev_att )
            attachment = Attachment( self.branch_id, att_id, min_loss, dims, config_copy, None, prev_att.closing_model, prev_core_output, self.y_data )

            # Calls to Attachment keep Consistent( attachment ).
            loss = attachment.Train( train_params ) # PENDING. optimizer should get faster as attachments grow.

            if min_loss > loss :
                attachments.append( [ attachment, loss ] )

                print( "\nC. ======= B-{}, Attach_Submodels: Loop = {} !!!!! success. {}-th attachment added, loss = {:.5f} ---> {:.5f}".format( self.branch_id, loop, len( attachments ), min_loss, loss ) )

                if len( dims_left ) > 0 : # Prepare for the next loop.
                    prev_att = attachment
                    prev_core_output = { segment : prev_att.Produce_Core_Output( segment ) for segment in [ seg for seg, _ in self.y_data.items() ] }

                min_loss = loss

            else :
                print( "\nC. ======= B-{}, Attach_Submodels: Loop = {} ????? failure. loss = {:.5f} ---> {:.5f}".format( self.branch_id, loop, min_loss, loss ) )

                if len( dims_left ) > 0 : # Prepare for the next loop.
                    prev_att = prev_att
                    prev_core_output = prev_core_output

            stop, speedup = loss_monitor.on_each_loss( loss )           
            if stop is True : break

        # closing_model.trainable = True #=== Recover / Unfrozen === Ugly hidden relation with model. No way to extract closing_model back from model. Relation proved.

        attached = len( attachments ) > 0

        if attached :

            print( "\nC. ======= B-{}, Attach_Submodels: Chaining {} attachments......... Expected loss = {:.5f}".format( self.branch_id, len( attachments ), min_loss ) )
            
            x = core_model( self.inputs ) # core_model == self.core_model may NOT hold.
            for [att, loss] in attachments :  
                x = att.core_model( x )

            core_model = tf.keras.models.Model( inputs = self.inputs, outputs = x, name = self.core_model_name )
            [ last_att, last_loss ] = attachments[ -1 ]
            closing_model = last_att.closing_model
            model = self.Compose_Core_Closing_Models( core_model, closing_model, config )

            model = self.Deep_Copy_Model( model, train_params, config, copy_optimizer = False )
            loss_after = self.Assess_Loss( model, self.config['loss_unified'] ) 
            print( "\nC. ======= B-{}, Attach_Submodels: Chained {} attachments. Loss = {:.5f}".format( self.branch_id, len( attachments ), loss_after ) )

        else :
            print( "\nC. ======= B-{}, Attach_Submodels: No attachments created.".format( self.branch_id ) )

        # Sure Consistent( model, core_model, closing_model )
        return model, core_model, closing_model, attached # is this true loss of current model after it's created and compled again?


    def Produce_Core_Output( self, segment, input = None, aux_input = None ) : # The argument signature i

        core_output = self.core_model( self.x_data[ segment ], training = False )

        core_output = output if isinstance( core_output, list ) else [ core_output ]

        return core_output


