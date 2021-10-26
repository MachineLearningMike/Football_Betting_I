import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers


from EarlyStop import *
from EarlyStop_Patient_AvgDeltaLossPerLoss import *

from M_Model_Base import *

class Model_Teams( Model_Base ) :

    def __init__( self, queries, dataset, modelSlots, superModelObjects ) :

        print( 'Constructing a Model_Teams.')

        super().__init__( queries, dataset, modelSlots, superModelObjects )

        self.model_name = 'Model_Teams'

        if superModelObjects != None and len(superModelObjects) > 0 :
            for superModelObject in superModelObjects :
                superModelObject.Register_SubModel( self )

        self.output_name = 'teams_output'
        self.total_inputs = [ self.dataset.inp_leagues, self.dataset.inp_homes, self.dataset.inp_aways, self.dataset.inp_situation ]


    def Create_Model( self, round, modelParams, trainResources ) :

        logits = self.Create_Prob_Logits( modelParams['model_complexity'], modelParams )
        
        prob = trainResources['pred_filters'] ( trainResources['pred_filterCore'], prob_logits = logits, name = self.output_name ) # trainResources['pred_filterCore'] = sigmoid. shape = (None, len(queries))

        self.model = tf.keras.Model ( \
            inputs = self.total_inputs, \
            outputs = prob, \
            name = self.model_name \
        )

        self.best_model_propageted = False


    def Create_Prob_Logits( self, complexity, modelParams ) : 

        tensor = None

        if complexity > 0 :

            league_dim = modelParams['league_dim']
            team_dim = modelParams['team_dim']

            hidden_dim_factor = modelParams['hidden_dim_factor']
            activation = modelParams['.action']

            regularizer = modelParams['.regur']
            regularRate = modelParams['regularRate']

            if regularizer == 'None': regularizerObj = None
            elif regularizer == 'l1' : regularizerObj = tf.keras.regularizers.l1(l = regularRate)
            elif regularizer == 'l2' : regularizerObj = tf.keras.regularizers.l2(l = regularRate)
        
            leagueEmbedding = tf.keras.layers.Embedding( self.dataset.nLeagues, league_dim, input_length=1, name = 'leagueEmbedding' ) (self.dataset.inp_leagues) # regularizers, constraint
            homeEmbedding = tf.keras.layers.Embedding( self.dataset.nTeams, team_dim, input_length=1, name = 'homeEmbedding' ) (self.dataset.inp_homes)
            awayEmbedding = tf.keras.layers.Embedding( self.dataset.nTeams, team_dim, input_length=1, name = 'awayEmbedding' ) (self.dataset.inp_aways)           
            situation = tf.keras.layers.Flatten( name = 'flat_situation' ) ( self.dataset.inp_situation )
            #odds = tf.keras.layers.Flatten( name = 'flat_odds' ) ( self.dataset.inp_odds )

            initial_tensor = tf.keras.layers.concatenate( [ leagueEmbedding, homeEmbedding, awayEmbedding, situation ], axis = -1, name = 'initial_tensor' )
           
            def EnhanceTensor( tensor, complexity ) :
                assert complexity > 0

                tensor = tf.keras.layers.BatchNormalization( ) (tensor) # ======================== Batch Normalization

                for n in range( complexity ) :

                    dropoutRate = modelParams['.dropIn']
                    if n > 0 : dropoutRate = dropoutRate * 2.5

                    if dropoutRate > 0 :
                        tensor = tf.keras.layers.Dropout( dropoutRate ) (tensor) #=========================== Dropout

                    hidden_dim = int( initial_tensor.shape[-1] * 100 * self.complexity_factors[complexity][n] / 100 )
                    #tensor = tf.keras.layers.BatchNormalization()(tensor) # ======================== Batch Normalization
                    tensor = tf.keras.layers.Dense( \
                        hidden_dim, \
                        activation = activation, \
                        use_bias = True, \
                        #kernel_initializer = initializers.Zeros(), \
                        #bias_initializer = initializers.Zeros(), \
                        kernel_regularizer = regularizerObj, \
                        bias_regularizer = regularizerObj, \
                        kernel_constraint = tf.keras.constraints.MaxNorm(2) \
                        ) ( tensor )

                    tensor = tf.keras.layers.BatchNormalization()(tensor) # ======================== Batch Normalization
                
                return tensor

            tensor = EnhanceTensor( initial_tensor, complexity )
 
            prob_logits = tf.keras.layers.Dense( len(self.queries), activation = 'linear', use_bias = True, kernel_regularizer = regularizerObj, bias_regularizer = regularizerObj) ( tensor )

        return prob_logits



    def Compile_Model( self, round, modelParams, trainResources, dataUse ) :

        optimizerObj = trainResources['optimizers'][ modelParams['optimizerObjects_pointer'] ]

        self.model.compile( \
            optimizer = optimizerObj, \
            loss = trainResources['.loss'](), \
            # metrics = metrics  \
        )  # You can defer setting metrics.

        # NO! self.best_weights_loaded = False


    def Fit_Model( self, round, fitParams, trainResources, dataUse ) :

        if self.superModels != None and len(self.superModels) > 0 :
            best_output = self.superModels[0].best_output['train']
            output_from_super = best_output
            x_train_data = self.Create_X_Data( 'train', output_from_super = output_from_super )
        else :
            x_train_data = self.Create_X_Data( 'train' )

        y_train_data = self.Pack_Y_Data( trainResources, labels = self.dataset.train_labels )

        callbacks = []

        #self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        cp_save_best_only = tf.keras.callbacks.ModelCheckpoint(\
            filepath = self.checkpoint_path, \
            monitor = fitParams['monitor'], \
            save_best_only = True, \
            save_weights_only = True, \
            verbose = 0 \
        )
        callbacks.append( cp_save_best_only )

        # restore_best_weights = True causes len(weights), NoneType has no len()
        # baseline argument done not work.

        if fitParams['EarlyStop'] == 'Yes' :
            ##cp_earlyStopping = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience = fitParams['.patience'] )
            ##callbacks.append( cp_earlyStopping )
            
            cp_earlyStop_Patient_AvgDeltaLossPerLoss = EarlyStop_Patient_AvgDeltaLossPerLoss ( \
                monitor = fitParams['monitor'], \
                patience = fitParams['.patience'], \
                min_loss_improvement_per_loss = fitParams['min_loss_improvement_per_loss'], \
                verbose=0, \
            )
            callbacks.append( cp_earlyStop_Patient_AvgDeltaLossPerLoss )

      
        self.model.reset_metrics()

        history = self.model.fit( \
            x_train_data, y_train_data, \
            validation_split = dataUse['valSplit'], \
            shuffle = dataUse['.ShuFit'], \
            validation_freq = 1, \
            batch_size = dataUse['.BSizeFit'],  \
            callbacks = callbacks, \
            epochs = fitParams['.epochs'], \
            verbose = 2 \
        ) # verbose 0 makes system stoned. 

        self.model.load_weights( filepath = self.checkpoint_path ) # because of cp_save_best_only
        self.best_model_propageted = False

        return



    def Create_X_Data( self, segment, output_from_super = None ) :

        # output_from_super is a tensor output from super model. It's None, as this is the topmose model.

        x_data = None

        if output_from_super == None : # Do NOT accept output_from_super. It is the topmost model, having no super model.
            leagues, homes, aways, situation, _, _, _, _ = self.dataset.GetDataSegment( segment )
            x_data = self.Pack_X_Data( leagues = leagues, homes = homes, aways = aways, situation = situation )

        return x_data


    def Pack_X_Data( self, **cols ) :
         return { 'league': cols['leagues'], 'home': cols['homes'], 'away': cols['aways'], 'situation': cols['situation'] }


    def Pack_Y_Data( self, trainResources, **cols ) :

        hGoal = cols['labels'][:,0]
        aGoal = cols['labels'][:,1]

        tensor = trainResources['true_filters']( hGoal, aGoal, name = self.output_name )

        return tensor


    def Unpack_Y_Data( self, y_data, query_id ) :
        return y_data.get(self.Create_Bet_Worth_Name(query_id), None)


    def Get_Modelled_Y_Data( self, segment ) :
        x_data = self.Create_X_Data( segment )
        y_data = self.model( x_data )
        
        return y_data


    def Get_Custom_Objects ( self ) :
        
        return { } # keys must be a class name, not a layer name.


    def Load_Best_Model( self, round, trainResources ) :

        custom_objects = self.Get_Custom_Objects()
        model, summary = self.model_slots.LoadModel( self.model_name, custom_objects = custom_objects,  defaultSummary = {}, summary_only = False )

        if model is not None :

            self.model = model
            self.best_model_propageted = False

            self.Propagate_Best_Model( round, trainResources )

        # We are not sure if a model has been loaded. We simply tried to load a best model, which may not exist.

        return


    def Propagate_Best_Model( self, round, trainResources ) :

        found_a_best_model = False

        is_best_model = self.performance.Is_Best_Model( 'test', self.model_name )

        if  is_best_model and not self.best_model_propageted :

            print( '------------------------------------ This is a better model.' )

            found_a_best_model = True

            for segment in self.dataset.standard_segments: 
                profit = self.performance.Synchnonize_With_Best_Model( segment, True, self.model_name, input_from_super = None )
    
            self.best_model_propageted = True # Sure?

            summary = { 'drivenBy' : 'Teams' }
            self.SaveModelSuperward( summary )
            self.SaveModelSubward( summary )

        else :
            pass

        return ( found_a_best_model and self.best_model_propageted )


    def Produce_Output( self, segment, input ) :

        prob_teams = None; home_emb = None; away_emb = None

        if self.model != None :
            trainable = self.model.trainable
            
            self.model.trainable = False
            # 'input' must be None, as this it the topmost model.
            x_data = self.Create_X_Data( segment, output_from_super = input )
            prob_teams = self.model( x_data, training = False )

            league_emb = [ self.model.get_layer('leagueEmbedding').get_weights() for query_id in self.queries ]            
            home_emb = [ self.model.get_layer('homeEmbedding').get_weights() for query_id in self.queries ]
            away_emb = [ self.model.get_layer('awayEmbedding').get_weights() for query_id in self.queries ]

            self.model.trainable = trainable

        return prob_teams, league_emb, home_emb, away_emb


    def GetPredictionByQuery( self, models, x_data ) :

        predictions = []

        for queryPlace in range(len(self.queries)) :
            query_id = self.queries[queryPlace]
            
            Y_model = models[query_id][0]
            N_model = models[query_id][1]

            if Y_model == None or N_model == None : continue
                
            Y_model_pred = Y_model( x_data, training = False ) # BatchNormalization requires model(x, training = False), than Y_model.predict( x_data )
            Y_model_pred = tf.squeeze(Y_model_pred, axis = -1) # then, [x_data]
            Y_model_pred = Y_model_pred.numpy()

            predictions.append( Y_model_pred )

        return predictions

