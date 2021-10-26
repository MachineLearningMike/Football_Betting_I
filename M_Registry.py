import numpy as np

import pandas as pd
import os

from M_Excel_Driver import *

from M_Model_Simpleton import *

from M_Database import *

class Registry() :
    
    def __init__( self, modelBase, queries, target_segment, train, valid, debugging = '' ) :

        self.modelBase = modelBase
        self.queries = queries
        self.target_segment = target_segment
        self.train = train
        self.valid = valid
        self.debugging = debugging

        self.score_renewed = False

        self.account_tv_path_base = os.path.join( super_config['Summary'], super_config['job_prefix'] + "_Accounts" + "_M_" + str( super_config['.M.Day'] ).zfill(4) + '_' + self.train + '.' + self.valid )
        self.account_tvt_path_base = self.account_tv_path_base + '.' + self.target_segment

        self.scores = {}
        self.debugging = debugging

        self.outputs_dict_mid = {}; self.stakes_dict_mid = {}; self.profits_dict_mid = {}

        self.account_folder_outputs = self.account_tvt_path_base + ".outputs"
        if not os.path.exists( self.account_folder_outputs ) : os.mkdir( self.account_folder_outputs )

        self.account_folder_stakes = self.account_tvt_path_base + ".stakes"
        if not os.path.exists( self.account_folder_stakes ) : os.mkdir( self.account_folder_stakes )

        self.account_folder_profits = self.account_tvt_path_base + ".profits"
        if not os.path.exists( self.account_folder_profits ) : os.mkdir( self.account_folder_profits )

        self.account_folder_scores = self.account_tvt_path_base +  ".views"
        if not os.path.exists( self.account_folder_scores ) : os.mkdir( self.account_folder_scores )
        self.filePath_scores = os.path.join( self.account_folder_scores, 'Total' + '.view' )

        load = LoadBinaryData( self.filePath_scores )
        if load is not None : self.scores = load

        """
        print('Loading stakes...', end = '')
        load = LoadBinaryData( self.account_path_stakes )
        print('not found' if load is None else 'done')
        if load is not None : self.stakes_dict_mid = load # self.scores might not up-to-date with self.stakes_dict_mid...

        print('Loading profits...', end = '')
        load = LoadBinaryData( self.account_path_profits )
        print('not found' if load is None else 'done')
        if load is not None : self.profits_dict_mid = load # self.scores might not up-to-date with self.stakes_dict_mid...

        print('Loading views...', end = '')
        load = LoadBinaryData( self.account_path_scores )
        print('not found' if load is None else 'done')
        if load is not None : self.scores = load # self.scores might not up-to-date with self.stakes_dict_mid...
        """
        return # self.scores is a dictionary of nGames-long lists of nQueries-long lists of scores, with the key of str(model_id)


    def Get_nModels( self ) : return len( self.stakes_dict_mid )

    def Register_Models( self, models_id ) :

        state_changed = False

        stakes, profits = None, None

        for model_id in models_id :

            print( "Load_Model_Outcomes : model_id = {}...".format( model_id ) )
            model_id_registered = \
                ( self.outputs_dict_mid is not None and self.outputs_dict_mid.get( str( model_id ), None ) is not None ) \
                and ( self.stakes_dict_mid is not None and self.stakes_dict_mid.get( str( model_id ), None ) is not None ) \
                and ( self.prifits_dict_mid is not None and self.profits_dict_mid.get( str( model_id ), None ) is not None )

            if model_id_registered :
                pass
            else :           
                outputs, stakes, profits = self.Load_Model_Outcomes( model_id, debugging = self.debugging ) # Duplicated models are detected and not registered.

                if outputs is not None and stakes is not None and profits is not None :
                    self.outputs_dict_mid.update( { str( model_id ) : outputs } ) # A segment-keyed dictionary of numpy arrays. ( nCheckGames, nQueries )
                    self.stakes_dict_mid.update( { str( model_id ) : stakes } ) # A segment-keyed dictionary of numpy arrays. ( nCheckGames, nQueries )
                    self.profits_dict_mid.update( { str( model_id ) : profits } ) # A segment-keyed dictionary of numpy arrays. ( nCheckGames, nQueries )

                    state_changed = True
                    self.score_renewed = False

                del outputs, stakes, profits

        #if state_changed :
        #    SaveBinaryData( ( self.stakes_dict_mid, self.profits_dict_mid ), self.account_path_stakes ) # No need to ensure self.scores is up-to-date with self.stakes...

        return


    def Load_Model_Outcomes( self, model_id, loaded = False, debugging = '' ) :

        def max_error( tensor1, tensor2 ) :
            #max = np.nanmax( np.abs( np.nan_to_num( arr1 ) - np.nan_to_num( arr2 ) ) )
            max = tf.reduce_max( tf.abs( tensor1 - tensor2 ) ).numpy()
            return max

        outputs, stakes, profits = None, None, None

        # Try from file, first.
        fileNameBase = 'Model_' + str( model_id ).zfill(6)
        filePath_output = os.path.join( self.account_folder_outputs, fileNameBase + '.output' )
        filePath_stake = os.path.join( self.account_folder_stakes, fileNameBase + '.stake' )
        filePath_profit = os.path.join( self.account_folder_profits, fileNameBase + '.profit' )

        outputs = LoadBinaryData( filePath_output )
        stakes = LoadBinaryData( filePath_stake )
        profits = LoadBinaryData( filePath_profit )

        if outputs is None or stakes is None or profits is None :

            loaded, summary, number_to_load = self.modelBase.LoadModel( number = model_id, summary_only = False, withWeights = True)

            if loaded :
                segments = [ self.train, self.valid, self.target_segment ] 
                ######### decision to include target_segment, whatever it is. Even 'predict', which has no 'happened', returning None profit.

                outputs, stakes, profits = self.modelBase.Get_Outcomes_On_Current_Model( segments, debugging = debugging )
            
                if stakes is not None :

                    duplicate = False
                    for model_key, stakes_existing in self.stakes_dict_mid.items() : # key = model_id, value = stakes for the model.
                        if max_error( stakes_existing[ self.train ], stakes[ self.train ] ) < 1e-5 :
                            duplicate = True ; break
                    
                    if duplicate is True : # stake_in_GQ is a unique ditribution of stake, yet.
                        stakes = None; profits = None
            
            if outputs is not None and stakes is not None and profits is not None :
                SaveBinaryData( outputs, filePath_output )
                SaveBinaryData( stakes, filePath_stake )
                SaveBinaryData( profits, filePath_profit )
            else :
                debug = 3
        
                
        return outputs, stakes, profits # None signals that loading model failed.


    def Register_Scores( self, models_id, schemes ) :

        for model_id in models_id :
            for scheme in schemes :

                print( "Expand_Existing_Scores : model_id = {}, scheme = {}".format( model_id, scheme ) )

                scores_scheme = self.scores.get( scheme, None )

                if scores_scheme is None : 
                    scores_scheme = {}
                    self.scores.update( { scheme : scores_scheme } )

                mkey = str( model_id )
                if scores_scheme.get( mkey, None ) is None : # Try find from storage. If not found, build and save it.
                    outputs = self.outputs_dict_mid[ mkey ]
                    stakes = self.stakes_dict_mid[ mkey ]
                    profits = self.profits_dict_mid[ mkey ]

                    scores_model_scheme, _, _ = self.modelBase.gauge.Select_Predictable_Games( self.modelBase, outputs, stakes, profits, target_segment = self.target_segment, model_id = model_id, scheme = scheme, cutRate = super_config['.CutRate'] )
                    
                    scores_scheme.update( { mkey : np.array( scores_model_scheme ) } ) # Create an array of shape ( game, query )
                    del scores_model_scheme

                    self.scores.update( { scheme : scores_scheme } )

                    SaveBinaryData( self.scores, self.filePath_scores )

                    del scheme

        return self.scores
   

    def Get_Views( self, scores, models_id_to_have ) :

        scores_new = []; schemes = []

        models_id = None
        for scheme, scores_dict_M_GQ in scores.items() :

            schemes.append( scheme )

            scores_GMQ, models_id_M = self.Transform_Dict_M_GQ_To_Array_GMQ( scores_dict_M_GQ, models_id_to_have = models_id_to_have )
            if models_id is None : models_id = models_id_M # Take the first coming.
            if models_id is not None : assert np.array_equal( models_id, models_id_M )

            scores_new.append( scores_GMQ )
       
        scores = np.array( scores_new, dtype = np.float32 ) # then in ( S, G, M, Q )

        # Sort them in scheme.
        schemes = np.array( schemes )
        ind = np.argsort( schemes, axis = 0 )
        schemes = schemes[ ind ]
        scores = scores[ ind ]

        scores = np.moveaxis( scores, 1, 0 ) # then in ( G, S, M, Q )
        scores_GMQS = np.moveaxis( scores, 1, -1 ) # then in ( G, M, Q, S )

        return scores_GMQS, models_id, schemes, self.target_segment



    def Summarize_Views( self, views ) :

        scores_GMQS, models_id, schemes, target_segment = views
        assert scores_GMQS.shape[ 1 ] == len( models_id )
        assert scores_GMQS.shape[ 3 ] == len( schemes )

        self.summary_folder = self.account_tvt_path_base + ".summaries"
        if not os.path.exists( self.summary_folder ) : 
            os.mkdir( self.summary_folder )

        summaries_list_list = []

        nGames = scores_GMQS.shape[ 0 ]

        for queryPlace in range( scores_GMQS.shape[2] ) :

            query_id = self.queries[ queryPlace ]

            do_summarize = False

            print( 'Loading existing summaries for query {}...'.format( query_id ), end = '' )
            path_summaries = os.path.join( self.summary_folder, 'Query_' + str(query_id).zfill(2) + ".summaries" )
            load = LoadBinaryData( path_summaries )
            print( 'not found' if load is None else 'done' )

            if load is None :
                do_summarize = True 
            else :
                ( summaries_list, views_summaries_frames ) = load

                sample_frame = None
                for title, frame in views_summaries_frames.items() :
                    sample_frame = frame; break

                # Target:
                # If a model or scheme is added to scores after last saving summaries_list/views_summaries_frames, then do_summarize.

                if do_summarize == False :  # summaries_list of shape ( nGames, nSchemes )
                    if  len( summaries_list ) != scores_GMQS.shape[ 0 ] or \
                        len( summaries_list[0] ) != len( schemes ) or \
                        len( sample_frame.index ) != len( schemes ) or \
                        len( sample_frame.columns ) != len( models_id ) + 2 :   # 2 for 'Item' and 'average'

                        # summaries_list NOT covers the same number of queries as scores do.
                        # summaries_list NOT covers teh same number of schemes as scores do
                        # views_summaries_frames NOT covers the same number of schemes as scores do.
                        # views_summaries_frames NOT overs the same number of models as scores do.

                        do_summarize = True

                if do_summarize == False :
                    for model_id in models_id :
                        if self.Create_Column_Name( model_id ) not in sample_frame.columns :
                            do_summarize = True ; break

                if do_summarize == False :
                    for scheme in schemes : 
                        if scheme not in sample_frame.index :
                            do_summarize = True ; break

                if do_summarize == False :
                    # the list of schemes in views_summaries_frames is the same as that of scores.
                    for scheme_id in range( len( schemes ) ) :
                        if sample_frame.loc[ scheme_id, 'Item' ] != schemes[ scheme_id ] :
                            do_summarize = True ; break
                if do_summarize == False :
                    # the list of models_id in views_summaries_frames is the same as that of scores.
                    for model_idx in range( len( models_id ) ) :
                        if int( sample_frame.columns[ model_idx + 1 ][2:]) !=  models_id[ model_idx ] :
                            do_summarize = True ; break
            
            if do_summarize :

                print( 'Summarizing and saving on query {}...'.format( query_id) )

                summaries_list = []
                views_summaries_frames = {}

                nFrames = 0
                for game_id in range( nGames ) :

                    scores_GMQS[ game_id, :, queryPlace, : ]

                    scores_MS = scores_GMQS[ game_id, :, queryPlace, : ]
                    summary_S = np.nanmean( scores_MS, axis = 0 ) # each row is averaged, to form a nSchemes-long array.          
                    summaries_list.append( summary_S )

                    if nFrames < super_config['nFramesToShow'] :

                        vs_dict = {}
                        vs_dict.update( { 'Item' : schemes } )

                        for model_idx in range( scores_GMQS.shape[ 1 ] ) :
                            model_id = models_id[ model_idx ]
                            vs_dict.update( { self.Create_Column_Name( model_id ) : scores_MS[ model_idx, : ] } ) # : for schemes
                        
                        vs_dict.update( { 'average' : summary_S } )

                        vs_frame = pd.DataFrame( vs_dict )
                        views_summaries_frames.update( { "Game_" + str( game_id ) : vs_frame } )

                    nFrames += 1
                
                SaveBinaryData( ( summaries_list, views_summaries_frames ), path_summaries ) # summaries_GS

            summaries_list_list.append( summaries_list ) # summaries_QGS

        summaries_QGS = np.array( summaries_list_list, dtype = np.float32 )

        return summaries_QGS, models_id, schemes, target_segment # ( query, game, scheme )

    def Create_Column_Name( self, model_id ) :
        return 'V_' + str( model_id ).zfill( 6 )

    def Extend_Views( self, views, extension_dict ) : 

        scores_GMQS, models_id, schemes, target_segment = views
        assert scores_GMQS.shape[ 1 ] == len( models_id )
        assert scores_GMQS.shape[ 3 ] == len( schemes )

        scores_MGQS = np.moveaxis( scores_GMQS, 1, 0 ); del scores_GMQS
        scores_MSGQ = np.moveaxis( scores_MGQS, 3, 1 )

        scores_MSGQ_new = []
        for ext_name, ext in extension_dict.items() : # ext in ( G, Q )
            ext_SGQ = np.expand_dims( ext, axis = 0 ) # The new axis is Scheme axis, S.

            for model_idx in range( scores_MSGQ.shape[0] ) :
                scores_SGQ = scores_MSGQ[ model_idx ]
                scores_SGQ = np.concatenate( [ scores_SGQ, ext_SGQ ], axis = 0 )

            scores_MSGQ_new.append( scores_SGQ )

            schemes = np.append( schemes, ext_name )
        
        scores_MSGQ = np.array( scores_MSGQ_new ) # model order is the same as they was. schemes is just appeneded with ext_names, in order.

        scores_MGQS = np.moveaxis( scores_MSGQ, 1, 3 ); del scores_MSGQ
        scores_GMQS = np.moveaxis( scores_MGQS, 1, 0 ); del scores_MGQS

        scores_GMQS_plus = scores_GMQS
        schemes_plus = schemes

        return scores_GMQS_plus, schemes_plus, target_segment



    def Extend_Summaries( self, summaries, extension_dict ) : 

        summaries_QGS, models_id, schemes, target_segment = summaries
        assert summaries_QGS.shape[ 2 ] == len( schemes )

        folder_summaries_plus = self.account_tvt_path_base + ".summaries_plus"
        if not os.path.exists( folder_summaries_plus ) : 
            os.mkdir( folder_summaries_plus )

        summaries_list_list_plus = []
        nGames = summaries_QGS.shape[ 1 ]
        excel_driver = Excel_Driver()

        for queryPlace in range( summaries_QGS.shape[ 0 ] ) :

            query_id = self.queries[ queryPlace ]

            do_extend = False

            path_summaries = os.path.join( self.summary_folder, 'Query_' + str(query_id).zfill(2) + ".summaries" )
            load = LoadBinaryData( path_summaries )
            assert load is not None
            summaries_list, views_summaries_frames = load

            print( 'Loading existing summaries_plus for query {}...'.format( query_id ), end = '' )
            path_summaries_plus = os.path.join( folder_summaries_plus, 'Query_' + str(query_id).zfill(2) + ".summaries_plus" )
            load = LoadBinaryData( path_summaries_plus )
            print( 'not found' if load is None else 'done' )

            if load is None :
                do_extend = True 
            else :
                ( summaries_list_plus, views_summaries_frames_plus ) = load

                sample_frame_plus = None
                for title, frame in views_summaries_frames_plus.items() :
                    sample_frame_plus = frame; break
            
                if do_extend == False :  #  summaries_list of shape ( nGames, nSchemes ), summaries_array of shape ( nQueries, nGames, nSchemes )
                    if  len( summaries_list_plus ) != summaries_QGS.shape[ 1 ] or \
                        len( summaries_list_plus[0] ) != summaries_QGS.shape[2] + len( extension_dict ) or \
                        len( sample_frame_plus.index ) != summaries_QGS.shape[2] + len( extension_dict ) or \
                        len( sample_frame_plus.columns ) != len( models_id ) + 2 or \
                        len( views_summaries_frames_plus ) != len( views_summaries_frames ) :                        

                        do_extend = True

                if do_extend == False :
                    for model_id in models_id :
                        if self.Create_Column_Name( model_id ) not in sample_frame_plus.columns :
                            do_extend = True ; break

                if do_extend == False :
                    for scheme in schemes : 
                        if scheme not in sample_frame_plus.index :
                            do_extend = True ; break

           
            if do_extend :

                erows = self.modelBase.task.dataset.Data( 'erows', target_segment )
                nGames = len( summaries_list )
                summaries_list_plus = []
                views_summaries_frames_plus = {}

                print( 'Extending and saving on query {}...'.format( query_id) )

                summaries_list_plus = summaries_list

                nFrames = 0

                for game_id in range( nGames ) :

                    frameIndexM_plus = views_summaries_frames.get( "Game_" + str( game_id ), None ) # summaries have simple keys.

                    for ext_name, extension in extension_dict.items() :
                        ext = extension[ game_id, queryPlace ] # ext is a scalar or an np.array of shape ( nLabels,).
                        summaries_list_plus[ game_id ] = np.append( summaries_list_plus[ game_id ], [ ext ], axis = 0 )
                        np.append( schemes, ext_name )

                        if nFrames < super_config['nFramesToShow'] :
                            row = [ 'LABEL.' + ext_name ] + [ ext for _ in range( frameIndexM_plus.shape[1] - 1 ) ]
                            frameIndexM_plus.loc[ frameIndexM_plus.shape[0] ] = row 
                            title = "Game {}, Segment '{}', Excel_Row {}".format( game_id, target_segment, erows[ game_id ] )
                            views_summaries_frames_plus.update( { title : frameIndexM_plus } )

                    nFrames += 1

                SaveBinaryData( ( summaries_list_plus, views_summaries_frames_plus ), path_summaries_plus )

                print( "Reformating summaries_plus on query {}...".format( query_id ), end = '' )
                filePath = os.path.join( folder_summaries_plus, "Query_" + str(query_id).zfill(2) + ".summaries_plus" + ".xlsx" )
                sheetName = 'Summaries_plus'
                excel_driver.Write_DataFrames( views_summaries_frames_plus, filePath, sheetName, nFrames = 20 )
                print( 'done.' )

            summaries_list_list_plus.append( summaries_list_plus )

        summaries_QGS_plus = np.array( summaries_list_list_plus, dtype = np.float32 )
        schemes_plus = schemes

        return summaries_QGS_plus, models_id, schemes_plus, target_segment


    def Get_Outcomes( self, segment, models_id_to_have = [] ) :

        outputs_GMQ, stakes_GMQ, profits_GMQ, models_id \
            = self.Get_Outcomes_In_Array( self.outputs_dict_mid, self.stakes_dict_mid, self.profits_dict_mid, segment, models_id_to_have = [] )

        return outputs_GMQ, stakes_GMQ, profits_GMQ, models_id


    def Get_Outcomes_In_Array( self, outputs, stakes, profits, segment, models_id_to_have = [] ) :

        outputs_GMQ, models_id_outcomes = self.Transform_Dict_M_Segment_GQ_To_Array_GMQ( outputs, segment, models_id_to_have )
        stakes_GMQ, models_id_stakes = self.Transform_Dict_M_Segment_GQ_To_Array_GMQ( stakes, segment, models_id_to_have )
        profits_GMQ, models_id_profits = self.Transform_Dict_M_Segment_GQ_To_Array_GMQ( profits, segment, models_id_to_have )

        assert np.array_equal( models_id_outcomes, models_id_stakes )
        assert np.array_equal( models_id_stakes, models_id_profits )

        return outputs_GMQ, stakes_GMQ, profits_GMQ, models_id_outcomes


    def Transform_Dict_M_Segment_GQ_To_Array_GMQ( self, things_dict_m_dict_seg_GQ, segment, models_id_to_have = [] ) :

        stakes_GMQ, profits_GMQ = [], []

        things_dict_M_GQ = { }
        for model_key, dict_seg_GQ in things_dict_m_dict_seg_GQ.items() :
            things_dict_M_GQ.update( { model_key : dict_seg_GQ[ segment ]  } )
        
        things_MGQ, models_id = self.Transform_Dict_M_GQ_To_Array_GMQ( things_dict_M_GQ, models_id_to_have = models_id_to_have )

        return things_MGQ, models_id

  
    def Transform_Dict_M_GQ_To_Array_GMQ( self, things_dict_M_GQ, models_id_to_have ) :

        things = []; models_id = []

        for mkey, things_model in things_dict_M_GQ.items() :
            model_id = int( mkey )
            if len( models_id_to_have ) <= 0 or model_id in models_id_to_have :
                models_id.append( int( model_id ) ) # A nModels-long list of models_id.
                things.append( things_model ) # A nModels-long list of ( nGames, nQueries )

        if len( models_id_to_have ) > 0 : assert len( models_id_to_have ) == len( models_id )

        models_id = np.array( models_id, dtype = np.int64 ) # shape : ( models_id_to_have, )
        things = tf.stack( things, axis = 0 ).numpy() # shape : ( models_id_to_have, nGames, nQueries )

        # Sort them in model_id.
        idx = np.argsort( models_id, axis = 0 )
        models_id = models_id[ idx ]
        things = things[ idx  ] # Then in ( M, G, Q )

        things_GMQ = np.moveaxis( things, 1, 0 )

        del things

        return things_GMQ, models_id


    #----------------------------------------------------



    def Get_Score_List_Descending( self, scheme ) :

        score_list = []

        if self.scores.get( scheme, None ) is not None :

            scores_dic = self.scores.get( scheme )

            for model_id, score_array in scores_dic.items() :
                score_list.append( score_array ) # score_array of shape ( nCheckGames, nQueries )
            
            score_list = np.array( score_list ) # ( M, G, Q ). Rounds each elem to an integer? No, it does 'int'. float('-inf')s in an np.array become -2147483648.
            score_list = np.concatenate( score_list, axis = 0 ) # ( N, Q ). Reduces the dimension of model_id.
            score_list = np.concatenate( score_list, axis = 0 ) # ( N, ). Reduces the dimension of games_id.
            score_list = sorted( list( score_list ), reverse = True ) # DO NOT use set(). DO NOT lose distribution info.

        return np.array( score_list )



    def Expand_Single_Scheme_Scores( self, models_id_to_have, scores_dict_M_GQ ) :

        models_id = []; scores = []; stakes = []

        for ( (key_score, _scores), (key_stake, _stakes) ) in zip( scores_dict_M_GQ.items(), self.stakes_dict_mid.items() ) :
            assert key_score == key_stake
            model_id = int( key_stake )
            if len( models_id_to_have ) <= 0 or model_id in models_id_to_have : 
                models_id.append( model_id ) # A nModels-long list of models_id.
                scores.append( _scores ) # A nModels-long list of ( nCheckGames, nQueries )
                stakes.append( _stakes[ self.target_segment ].numpy() ) # A nModels-long list of ( nCheckGames, nQueries )

        if len( models_id_to_have ) > 0 : assert len( models_id_to_have ) == len( models_id )

        models_id_M = np.array( models_id, dtype = np.int64 ) # shape : ( models_id_to_have, )
        scores_MGQ = np.array( scores, dtype = np.float32 ) # shape : ( models_id_to_have, nCheckGames, nQueries )
        stakes_MGQ = np.array( stakes, dtype = np.float32 ) # shape : ( models_id_to_have, nCheckGames, nQueries )

        # Sort them in model_id.
        idx = np.argsort( models_id_M, axis = 0 )
        models_id_M = models_id_M[ idx ]
        scores_MGQ = scores_MGQ[ idx  ]
        stakes_MGQ = stakes_MGQ[ idx  ]

        score_sorting_idx_QNI = None # Q: query_ids, N: ranking, I: 0 = index_to_model_id, 1 = game_id

        if len( models_id ) > 0 :
            score_sorting_idx_QNI = \
            np.array( \
                [ \
                    np.dstack( \
                        np.unravel_index( \
                            np.argsort( \
                                scores_MGQ[ :, :, queryPlace ].ravel() \
                                # So, this is sorting in model and game, for a query. \
                            ), \
                            ( scores_MGQ.shape[0], scores_MGQ.shape[1] ) \
                        ) \
                    )[0] \
                    for query_id in range( scores_MGQ.shape[2] ) \
                ],
                dtype = np.int64 \
            )
            # score_sorting_idx_QNI - Q: query_ids, N: ranking, I: 0 = index_to_model_id, 1 = game_id.
            # scores_MGQ[ score_sorting_idx_QNI[ q, n, 0 ], score_sorting_idx_QNI[ q, n, 0 ], q ] = scores_MGQ[ index_to_model_id, game_id ] = 32.3, for instance, is the n-th smallest score in query_id of q.
            # score_sorting_idx_QNI.shape = ( nQueries, models_id_to_have * nCheckGames, len( ( models_id_to_have, nCheckGames ) ) = 2 ). argsort[ query_id, nth ] = ( index_to_model_id, game_id )

        del models_id, scores, stakes

        return scores_MGQ, stakes_MGQ, score_sorting_idx_QNI, models_id_M

    #------------------------------------------------------------------------

    def Create_Outcomes_Views_Model_Wholesale( self, outcomes, views, labels_dict, queries ) :

        outputs_GMQ, stakes_GMQ, profits_GMQ, models_id_outcomes = outcomes
        scores_GMQS, models_id_views, schemes = views

        labels_id = []; labels_IGQ = [] # I for index.
        for label_id, label_GQ in labels_dict.items() :
            labels_id.append( label_id )
            labels_IGQ.append( label_GQ )

        lebels_id = np.array( labels_id )
        labels_IGQ = np.array( labels_IGQ )
        labels_GQI = np.moveaxis( labels_IGQ, 0, 2 )

        assert np.array_equal( models_id_outcomes, models_id_views )
        assert outputs_GMQ.shape[0] == scores_GMQS.shape[0] # Games
        assert outputs_GMQ.shape[0] == labels_GQI.shape[0] # Games
        assert outputs_GMQ.shape[1] == scores_GMQS.shape[1] # Models
        assert outputs_GMQ.shape[1] == len( models_id_outcomes ) # Redundant
        assert outputs_GMQ.shape[2] == scores_GMQS.shape[2] # Queries
        assert outputs_GMQ.shape[2] == labels_GQI.shape[1] # Queries
        assert scores_GMQS.shape[3] == len( schemes ) # Redundant

        outputs_GQM = np.moveaxis( outputs_GMQ, 1, 2 )
        scores_GQMS = np.moveaxis( scores_GMQS, 1, 2 )
        labels_GQI = np.moveaxis( labels_GQI, 0, 0 )

        models_id = [] # We need to devise a distinct name for model?


        return models_id

    

    def Create_Summaries_Model_Wholesale( self, summaries_plus, queries ) :

        summaries_QGS_plus, models_id, schemes_plus, target_segment = summaries_plus

        models_id = []

        for queryPlace in range( len( self.queries ) ) :
            query_id = queries[ queryPlace ]

            models_id.append( self.Create_Summaries_Model( summaries_QGS_plus[ queryPlace ], query_id ) )

        return models_id


    def Create_Outcomes_Views_Model_Wholesale( self, outcomes, views, labels, queries ) :
        
        models_id = []

        models_id = self.account_over_models.Create_Outcomes_Views_Model_Wholesale( outcomes, views, labels, queries )

        return models_id


    def Create_Summaries_Model( self, summaries_GS_plus, query_id ) : # query_id is not required, but used for a progress message.

        print( "***** ModelBase: Creating a summaries model {}".format( query_id ) )

        params = {}
        params['pick_trials'] = 50

        x = 7
        params[ 'dense_layers' ] = [ ( x, 'linear' ) ] + [ ( x, 'tanh' ) ] * 5 + [ ( x, 'tanh' ) ] * 5
        params['closing_act'] = 'sigmoid' # 'linear' worked well, for query0 at least.
        params[ 'optimizer'] = tf.keras.optimizers.Adam
        params['loss'] = Loss_Minus_Profit_On_Happened_Prob_Summary() # tf.keras.losses.mean_absolute_error
        params[ 'metrics'] = None #tf.keras.metrics.mean_absolute_error
        params['check_monitor'] = 'val_loss'
        params['checkpoints'] = 'CheckPoint_Simpleton'
        params['epochs'] = 1000

        model_id = None

        # This will avoid an empty array. summaries_plue may be empty if no event is recorded. The dummy query for correct_score_group is an example of such.
        if len( summaries_GS_plus.shape ) > 1 :

            data_explanatory = summaries_GS_plus[ :, : -2 ]
            data_observation = summaries_GS_plus[ :, -2 : ]

            nExamples = summaries_GS_plus.shape[0]
            nTrain = int( nExamples * 50 / 100 )
            nValid = int( nExamples * 30 / 100 )
            assert nTrain > 0 and nValid > 0 and nTrain + nValid <= nExamples

            x_data = {}
            x_data['train'] = data_explanatory[ : nTrain, :]; x_data['valid'] = data_explanatory[ nTrain : nTrain + nValid, : ]; x_data['test'] = data_explanatory[ nTrain + nValid :, :]
            y_data = {}
            y_data['train'] = data_observation[ : nTrain ]; y_data['valid'] = data_observation[ nTrain : nTrain + nValid ]; y_data['test'] = data_observation[ nTrain + nValid : ]

            simpleton = Model_Simpleton( x_data, y_data, loss_to_compare = 'loss_valid' )

            model_name, version_string = self.Get_Analyze_Model_Tags( query_id )

            model_id = simpleton.Build_Model( params, self.modelBase.task.model_slots, model_name, version_string )

        return model_id


    def Apply_Models_Wholesale( self, models_id, queries, games, summaries ) :

        summaries_QGS, _models_id, schemes, target_segment = summaries

        simpleton = Model_Simpleton( x_data = None, y_data = None, loss_to_compare = None )
        nGames = self.modelBase.task.dataset.Data( 'homes', games ).shape[0]

        preds = []

        for queryPlace in range( len( queries ) ) :
            query_id = queries[ queryPlace ]

            model_id = models_id[ queryPlace ]
            model_name, version_string = self.Get_Analyze_Model_Tags( query_id )

            loaded = simpleton.Load_Model( self.modelBase.task.model_slots, model_name, version_string, model_id )

            if loaded :
                games_id = range( nGames )
                summaries_q = summaries_QGS[ queryPlace ]
                preds.append( tf.squeeze( simpleton.Apply_Model_To( summaries_q ) ) )
            else :
                raise Exception( 'Help!' )

        #preds = np.squeeze( np.array( preds, dtype = np.float32 ) )
        preds = np.array( preds, dtype = np.float32 )

        # code for quick test ====================================================================================

        happ = self.modelBase.task.dataset.Data( 'happened', 'test' )
        happ_true = np.ndarray.astype( np.transpose( happ ), dtype = np.int32 )

        odds = self.modelBase.task.dataset.Data( 'odds', 'test' )
        odds = np.transpose( odds )

        # happ_pred = np.ndarray.astype( preds > 0.0, dtype = np.int32 ) # Use it when a model outputs oh-1.
        happ_pred = np.ndarray.astype( preds > 0.5, dtype = np.int32 ) # Use it when a model outputs probability.

        frames = {}
        for queryPlace in range( happ_true.shape[0] ) :

            query_id = queries[ queryPlace ]

            f1, precision, recall = self.modelBase.gauge.Get_F1_Precision_Recall_For_1_Of_0_And_1( happ_true[ queryPlace ], happ_pred[ queryPlace ] ) # Read this function!!!
            stake = ( odds[ queryPlace ] * happ_pred[ queryPlace ] )
            back_profit = ( odds[ queryPlace ] * happ_true[ queryPlace ] - 1 ) * stake
            cum_profit = np.zeros_like( back_profit )
            cum_profit[0] = back_profit[0]
            for idx in range( 1, back_profit.shape[0] ) :
                cum_profit[ idx ] = cum_profit[ idx - 1 ] + back_profit[ idx ]

            dict = {}
            dict.update( { 'Item' : [ 'Happ_True', 'Happ_Pred', 'Odds', 'Stake', 'Profit', 'cum.p' ] } )
            dict.update( { str( game_id ) : [ happ_true[ queryPlace, game_id ], happ_pred[ queryPlace, game_id ], \
                odds[ queryPlace, game_id ], stake[ game_id ], back_profit[ game_id ], cum_profit[ game_id ] ] for game_id in range( happ_true.shape[ 1 ] ) } )
            frame = pd.DataFrame( dict )

            title = "Query_ {}, f1 = {:.3f}, precision = {:.3f}, recall = {:.3f}, avg.stake/game = {:.3f}, avg.profit/game = {:.3f}, max.stake = {:.3f}, avg.stake/stake = {:.3f}, avg.profit/stake = {:.3f},  stake {} times / total {} chances".\
                format( str( query_id ).zfill(2), f1[1], precision[1], recall[1], np.average( stake ), np.average( back_profit ), np.max(stake), np.average( stake[ np.where( stake > 0.0 )[0] ] ), np.average( back_profit[ np.where( back_profit != 0.0 )[0] ] ), np.sum( np.ndarray.astype(stake > 0.0, np.int32 ) ), stake.shape[0]  )
            frames[ title ] = frame

        excel_driver = Excel_Driver()
        filePath = self.account_tv_path_base = os.path.join( super_config['Summary'], "Test Report" + "_M" + str( super_config['.M.Day'] ).zfill(4) + ".xlsx" )
        excel_driver.Write_DataFrames( frames, filePath, 'Compare' )

        return preds


    def Get_Analyze_Model_Tags( self, query_id ) :
        model_name = "Happened_Q" + str( query_id ).zfill(2)
        version_string = 'M' + str( super_config['.M.Day'] ).zfill(4)

        return model_name, version_string