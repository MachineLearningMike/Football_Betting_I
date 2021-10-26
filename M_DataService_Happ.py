import tensorflow as tf
import numpy as np

from M_Config import *
from M_DataService import *

class DataService_Happ( DataService ) :

    def __init__( self, dataset = None ) :
        super().__init__()
        self.dataset = dataset

        return

    def Change_Dataset( self, dataset ) :
        self.dataset = dataset

        return


    def Get_Inputs( self, config ) :

        inputs = {}

        if config['In.Leag'] is True : inputs.update( { 'leagues': self.dataset.inp_leagues } )
        if config['In.Home'] is True : inputs.update( { 'homes' : self.dataset.inp_homes } )
        if config['In.Away'] is True : inputs.update( { 'aways' : self.dataset.inp_aways } )
        if config['In.Situ'] is True : inputs.update( { 'situation' : self.dataset.inp_situation } )
        if config['In.Odds'] is True : inputs.update( { 'odds' : self.dataset.inp_odds } )
        if config['In.RowId'] is True : inputs.update( { 'rows' : self.dataset.inp_rows } )
        if config['In.X'] is True : pass #================================================ No separate input layer needed.

        if super_config['Contest'] :
            if config['.In.lCoun'] is True : inputs.update( { 'lCountries': self.dataset.inp_lCountries } )
            if config['.In.hCoun'] is True : inputs.update( { 'hCountries': self.dataset.inp_hCountries } )
            if config['.In.aCoun'] is True : inputs.update( { 'aCountries': self.dataset.inp_aCountries } )

        elif not super_config['Contest'] :
            if config['In.PGoals'] is True : inputs.update( { 'p_goals' : self.dataset.inp_p_goals } )
            if config['In.PHapp'] is True : inputs.update( { 'p_happened' : self.dataset.inp_p_happened } )
            if config['In.Goals'] is True : inputs.update( { 'goals' : self.dataset.inp_goals } )

        return inputs


    def Get_Base_X( self, inputs, config, train_params, branch_id = -1 ):

        modelParams = train_params['modelParams']

        homeEmbedder = None; awayEmbedder = None

        input_fields = []

        if config['In.Leag'] is True :
            league_dim = modelParams['.d.leag']
            leagueEmbedding = tf.keras.layers.Embedding( self.dataset.nLeagues, league_dim, input_length=1 ) ( inputs['leagues'] ) # regularizers, constraint
            input_fields.append( leagueEmbedding )

        homeEmbedder = None
        if config['In.Home'] is True :
            team_dim = modelParams['.d.team']
            homeEmbedder = tf.keras.layers.Embedding( self.dataset.nTeams, team_dim, input_length = 1 )
            homeEmbedding = homeEmbedder ( inputs['homes'] )
            input_fields.append( homeEmbedding )

        awayEmbedder = None
        if config['In.Away'] is True :
            team_dim = modelParams['.d.team']
            awayEmbedder = tf.keras.layers.Embedding( self.dataset.nTeams, team_dim, input_length = 1 )
            awayEmbedding = awayEmbedder ( inputs['aways'] )
            input_fields.append( awayEmbedding )

        if config['In.Situ'] is True :
            situation = tf.keras.layers.Flatten( ) ( inputs['situation'] )
            input_fields.append( situation )

        if config['In.Odds'] is True :
            odds = tf.keras.layers.Flatten( ) ( inputs['odds'] )
            if branch_id >= 0 :
                idx = [ [query_id] for query_id in super_config.GroupId_To_Queries( branch_id ) ]
                odds = tf.transpose( tf.gather_nd( tf.transpose( odds ), idx ) )
            input_fields.append( odds )

        if config['In.RowId'] is True :
            rows = tf.keras.layers.Flatten( ) ( inputs['rows'] )
            input_fields.append( rows )

        if homeEmbedder is not None and config['In.X'] is True :
            awayAsHomeEmbedding = homeEmbedder.__call__( inputs['aways'], training = False )
            input_fields.append( awayAsHomeEmbedding )

        if awayEmbedder is not None and config['In.X'] is True :
            homeAsAwayEmbedding = awayEmbedder.__call__( inputs['homes'], training = False )
            input_fields.append( homeAsAwayEmbedding )

        if super_config['Contest'] :

            country_dim = modelParams['.d.coun']

            if config['.In.lCoun'] is True :
                CountryEmbedder = tf.keras.layers.Embedding( self.dataset.nCountries, country_dim, input_length = 1 )
                CountryEmbedding = CountryEmbedder ( inputs['lCountries'] )
                input_fields.append( CountryEmbedding )

            if config['.In.hCoun'] is True :
                CountryEmbedder = tf.keras.layers.Embedding( self.dataset.nCountries, country_dim, input_length = 1 )
                CountryEmbedding = CountryEmbedder ( inputs['hCountries'] )
                input_fields.append( CountryEmbedding )

            if config['.In.aCoun'] is True :
                CountryEmbedder = tf.keras.layers.Embedding( self.dataset.nCountries, country_dim, input_length = 1 )
                CountryEmbedding = CountryEmbedder ( inputs['aCountries'] )
                input_fields.append( CountryEmbedding )

        elif not super_config['Contest'] :

            if config['In.PGoals'] is True :
                p_goals = tf.keras.layers.Flatten( ) ( inputs['p_goals'] )
                input_fields.append( p_goals )

            if config['In.PHapp'] is True :
                p_happened = tf.keras.layers.Flatten( ) ( inputs['p_happened'] )
                input_fields.append( p_happened )

            if config['In.Goals'] is True :
                goals = tf.keras.layers.Flatten( ) ( inputs['goals'] )
                input_fields.append( goals )

        base_X = None
        if len( input_fields ) > 1 : base_X = tf.keras.layers.concatenate( input_fields , axis = -1 )
        elif len( input_fields ) > 0 : base_X = input_fields[ 0 ]
        else : raise Exception("Input cannot be of zero length.")

        del inputs, config, train_params

        return base_X, homeEmbedder, awayEmbedder



    def Prepare_Data( self, config, train_params ) :

        x_data = { segment: self.Create_X_Data( segment, config = config, train_params = train_params ) for segment in self.dataset.standard_segments }
        y_data = { segment: self.Create_Y_Data( segment, config = config, train_params = train_params ) for segment in self.dataset.standard_segments }

        return x_data, y_data

    def Create_X_Data( self, segment, config, train_params, aux_input = None ) :

        x_data = None
        #superModels = self.task.Get_superModels()
        #if superModels != None and len( superModels ) > 0 :  x_data = superModels[0].Produce_Output( segment, False,  input = None, aux_input = None )

        if x_data is None :
            if segment in self.dataset.standard_segments :

                leagues = self.dataset.Data( 'leagues', segment )
                homes = self.dataset.Data( 'homes', segment )
                aways = self.dataset.Data( 'aways', segment )
                situation = self.dataset.Data( 'situation', segment )
                rows = self.dataset.Data( 'rows', segment )
                odds = self.dataset.Data( 'odds', segment )

                if super_config['Contest'] :
                    lCountries = self.dataset.Data( 'lCountries', segment )
                    hCountries = self.dataset.Data( 'hCountries', segment )
                    aCountries = self.dataset.Data( 'aCountries', segment )

                elif not super_config['Contest'] :
                    goals = self.dataset.Data( 'goals', segment )
                    p_goals = self.dataset.Data( 'p_goals', segment )
                    p_happened = self.dataset.Data( 'p_happened', segment )

            elif aux_input is not None :
                leagues, homes, aways, situation, rows, odds, goals, p_goals, p_happened, _ = aux_input

            else :
                raise Exception('Non-standard segment without auxilary input.')

            x_data = {}

            if config['In.Leag'] is True : x_data.update( { 'leagues' : leagues } )
            if config['In.Home'] is True : x_data.update( { 'homes' : homes } )
            if config['In.Away'] is True : x_data.update( { 'aways' : aways } )
            if config['In.Situ'] is True : x_data.update( { 'situation' :situation } )
            if config['In.Odds'] is True : x_data.update( { 'odds' : odds } )
            if config['In.RowId'] is True : x_data.update( { 'rows' : rows } )

            if super_config['Contest'] :
                if config['.In.lCoun'] is True : x_data.update( { 'lCountries' : lCountries } )
                if config['.In.hCoun'] is True : x_data.update( { 'hCountries' : hCountries } )
                if config['.In.aCoun'] is True : x_data.update( { 'aCountries' : aCountries } )                

            elif not super_config['Contest'] :
                # It's important to minimize the size of x_data, for the limited memory of GPU.
                if config['In.PGoals'] is True : x_data.update( { 'p_goals' : p_goals } )
                if config['In.PHapp'] is True : x_data.update( { 'p_happened' : p_happened } )
                if config['In.Goals'] is True : x_data.update( { 'goals' : goals } )

        return x_data


    def Create_Y_Data( self, segment, config, train_params ) :

        rows = self.dataset.Data( 'rows', segment )

        if super_config['Contest'] :
            # This is an ugly patch. If we don't accept this patch, then all the loss classes written for Thomas task, 
            # which uniformly accept rows, goals, odds, and happened, can not be used for the Contest task, which actually has no goals.
            # So, this 'goals' is just a placeholder created for re-use of Thomas task's loss classes in the Contest task.
            goals = np.stack( [ rows, rows ], axis = -1 )

        elif not super_config['Contest'] :
            goals = self.dataset.Data( 'goals', segment )
        
        odds = self.dataset.Data( 'odds', segment )
        happened_true = self.dataset.Data( 'happened', segment )

        rows = rows.reshape( [ -1, 1 ] )

        tensor = tf.concat( [ rows, goals, odds, happened_true ], axis = -1 ) # The order of placement is coupled with the Loss function, where they are split back.

        return tensor


    def Get_Labels( self, games_plus ) :
        # This code can have different versions.

        odds = self.dataset.Data( 'odds', games_plus ) # GQ
        happened_true = self.dataset.Data( 'happened', games_plus ) # GQ

        labels_dict = { 'odds' : odds, 'happened' : happened_true }

        return labels_dict