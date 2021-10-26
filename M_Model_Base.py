import numpy as np
import tensorflow as tf
import pandas as pd
import os
import copy

from M_BettingMath import *
from M_Losses import *

from sklearn.cluster import SpectralClustering

from M_Config import *
from Task_Params_Feeder import *


from memory_profiler import profile

"""
from M_EarlyStop_0 import *
from M_Loss_Monitor import *
"""

from M_Visulal_Reporter import *

import xlsxwriter

from dateutil import parser
from M_Excel_Driver import *

from io import StringIO  # Python 3
import sys

from M_Model_Branch import *
from M_Model_Fitter import *
from M_Gauge import *

from M_Registry import *

class Model_Base( Model_Fitter ) :

    def __init__( self, task ) :

        super().__init__( None, None )

        self.task = task
        self.dataservice = task.dataservice
        self.config = task.config # Required to have the same, consistent interfece with Branch and Attachment.

        self.train_params = None # to be copied and shared to branches ( and then their attachments )

        self.branches = None
        self.branchTool = None

        self.model = None
        self.model_name = 'Branched'

        self.inputs = None
        self.output = None
        self.stakes = None; self.profits = None

        self.account_over_models = None

        #====================================
        self.gauge = None
        self.excel_driver = Excel_Driver()
        #====================================


    def Get_Inputs( self, config ) :
        return self.dataservice.Get_Inputs( config )
    def Get_Base_X( self, inputs, config, train_params, branch_id = -1 ) :
        return self.dataservice.Get_Base_X( inputs, config, train_params, branch_id )
    def Prepare_Data( self, config, train_params ) :
        return self.dataservice.Prepare_Data( config, train_params )
    def Create_X_Data( self, segment, config, train_params, aux_input = None ) :
        return self.dataservice.Create_X_Data( segment, config, train_params, aux_input )
    def Create_Y_Data( self, segment, config, train_params ) :
        return self.dataservice.Create_Y_Data( segment, config, train_params )
    def Get_Labels( self, games_plus ) :
        return self.dataservice.Get_Labels( games_plus )


    def Prepare_Session( self, margin, purpose ) :

        self.inputs = self.Get_Inputs( self.config )
        self.output = {}
        self.Initialize_Gauge( margin )

        return

    def Settle_With_Model_Store( self, model, outputId ) :

        model_chosen = None

        #-------------- A new model branch is trained. So, ...
        if self.config['Branch'] in [ 'better' ] : # Pursue better models.
            best_model, model_number = self.Load_Best_Branch( outputId )
            if best_model is not None :
                best_model = self.branchTool.Compile_Model( best_model, train_params = self.train_params, branch_id = outputId ) 

                best_score = self.Assess_Loss( best_model, self.config['loss_unified'] )
                score = self.Assess_Loss( model, self.config['loss_unified'] )

                if best_score < score : 
                    model_chosen = best_model #; del model
                else : 
                    model_chosen = model ; del best_model
            else :
                model_chosen = model
        else :
            model_chosen = model

        loss = self.Assess_Loss( model, self.config['loss_unified'] )

        self.Save_Branch( model, outputId, loss ) # save it even though it's not better.

        del model

        return model_chosen

    def Train( self, modelParams, fitParams, trainResources, dataUse ):

        stakes = {}; profits = {}

        self.train_params = { 'modelParams' : modelParams.copy(), '.fitParams' : fitParams.copy(), 'trainResources' : trainResources, '.dataUse' : dataUse.copy() }
        self.x_data, self.y_data = self.Prepare_Data( self.config, self.train_params )

        del self.branches, self.branchTool
        self.branches, self.branchTool = self.Create_Branches( self.inputs, self.config, self.train_params, self.x_data, self.y_data )

        branch_models = []

        for outputId in range( len( self.branches ) ) : #============================ Try distribute Trains.

            branch = self.branches[ outputId ]

            if isinstance( branch, Model_Branch ) :
                train_params = self.train_params
                model = branch.Train( self.train_params.copy() ) # eg: optimizers change their state, and each branch should enjoy the same train_param.
                model_chosen = self.Settle_With_Model_Store( model, outputId )
                branch_models.append( model_chosen )

            elif isinstance( branch, tf.keras.models.Model ) :
                model = branch
                branch_models.append( model )

        del self.model
        self.model = self.Combine_Branch_Models( branch_models )
        self.model = self.Compile_Model( self.model, self.train_params )

        self.Invalidate_Output_Subward( segment = None )

        print( "============= Main : Trainables = {:,d}".format( np.sum([np.prod(v.shape) for v in self.model.trainable_variables]) ) )
        
        #------------------------- Get Raw stakes, profits

        segments = super_config['segs.outcomes']
        outputs, stakes, profits = self.Get_Outcomes_On_Current_Model( segments, debugging = super_config['debugging_1'] )


        del self.stakes, self.profits
        self.stakes = stakes; self.profits = profits # Keep is for later use, in gauge.get_statistics. It takes long to compute them.
        del stakes, profits

        return self.stakes, self.profits

    #-----------------------------------------------------------------------------------------------

    def Select_Best_Viewers( self, queries, viewers, train, valid, test ) :

        max_scores = [ float('-inf') for query in queries ]
        best_viewers = [ None for query in queries ]
        best_outcomes = [ None for query in queries ]

        def Score( totol_profit, std, plRt, MDrop, f1, pre, rec ) :
            return (totol_profit / MDrop_q).numpy().item()

        for viewer in viewers :
            print( "Examining viewer {}...".format( viewer), end = '' )

            loaded, summary, model_number = self.LoadModel( number = viewer )
            assert loaded is True
            _, model_stat_dframe = self.gauge.Get_Model_Statistics( self )

            nQueries = len( queries )

            segment = valid # train? or train + valid?

            profit_q = model_stat_dframe.loc[ 'profit', segment ]; assert profit_q.shape[-1] == nQueries
            profit_cum_q = model_stat_dframe.loc[ 'cum', segment ]; assert profit_cum_q.shape[-1] == nQueries
            total_profit_q = profit_cum_q[-1]; assert total_profit_q.shape[-1] == nQueries
            safety_q = model_stat_dframe.loc[ 'safe', segment ]; assert len( safety_q ) == nQueries
            std_q = model_stat_dframe.loc[ 'std', segment ]; assert len( std_q ) == nQueries
            plRt_q = model_stat_dframe.loc[ 'plRt', segment ]; assert len( plRt_q ) == nQueries
            MDrop_q = model_stat_dframe.loc[ 'MDrop', segment ]; assert len( MDrop_q ) == nQueries
            f1_q = model_stat_dframe.loc[ 'f1', segment ]; assert len( f1_q ) == nQueries
            f1p_q = model_stat_dframe.loc[ 'f1p', segment ]; assert len( f1p_q ) == nQueries
            f1r_q = model_stat_dframe.loc[ 'f1r', segment ]; assert len( f1r_q ) == nQueries

            for queryPlace in range( len( queries ) ) :
                query = queries[ queryPlace ]
                outputId = super_q_group.Query_To_GroupId( query )

                score = Score( total_profit_q[ queryPlace], std_q[ queryPlace], plRt_q[ queryPlace], MDrop_q[queryPlace], f1_q[queryPlace], f1p_q[queryPlace], f1r_q[queryPlace] )
                if max_scores[ queryPlace ] < score :
                    print( "score = {:.4f} on query {}, ".format( score, query ), end = '' )
                    max_scores[ queryPlace ] = score
                    best_viewers[ queryPlace ] = viewer
                    best_outcomes[ queryPlace ] = self.Get_Outcomes_On_Current_Model( super_config['segs.outcomes'], debugging = super_config['debugging_1'] )
            print( "done." )
        
        return best_viewers, best_outcomes


    def Create_Views( self, queries, games, viewers, train, valid ) :
        views = None

        schemes = super_config['schemes']

        modelBase = self
        self.account_over_models = Registry( modelBase, queries, games, train, valid, debugging = super_config['debugging_1'] )
        self.account_over_models.Register_Models( viewers )

        scores = self.account_over_models.Register_Scores( viewers, schemes )

        views = self.account_over_models.Get_Views( scores, models_id_to_have = viewers )

        return views


    def Create_Outcomes( self, queries, games, viewers, train, valid ) :
        stakes = None

        modelBase = self
        self.account_over_models = Registry( modelBase, queries, games, train, valid, debugging = super_config['debugging_1'] )
        self.account_over_models.Register_Models( viewers )

        outcomes = self.account_over_models.Get_Outcomes( segment = games )

        return outcomes


    def Create_Summaries( self, queries, games, views ) :

        summaries = self.account_over_models.Summarize_Views( views )

        return summaries


    def Label_Summaries( self, summaries, labels_dict ) :

        summaries_plus = self.account_over_models.Extend_Summaries( summaries, labels_dict )

        return summaries_plus



    def Create_Summaries_Model_Wholesale( self, summaries_plus, queries ) :

        models_id = self.account_over_models.Create_Summaries_Model_Wholesale( summaries_plus, queries )

        return models_id


    def Apply_Models_Wholesale( self, models_id, queries, games, summaries ) :

        preds = self.account_over_models.Apply_Models_Wholesale( models_id, queries, games, summaries )

        return preds


    #------------------------------------------------------------------------------------------------


    def Get_Outcomes_On_Current_Model( self, segments, debugging = '' ):

        outputs = {}; stakes = {}; profits = {}

        for segment in segments :

            odds = self.task.dataset.Data( 'odds', segment )
            happened_true = self.task.dataset.Data( 'happened', segment )
            output = self.Sync_Output_Subward( segment, update_cache = True, subward_sync = False, debugging = debugging )
            
            debugging_arg = None
            if debugging ==  'follow_pred_happened' : debugging_arg = self.task.dataset.Data( 'p_happened', segment )
            elif debugging == 'follow_true_happened' : debugging_arg = happened_true

            stakes[ segment ], profits[ segment ] = self.Get_Outcomes( output, odds, happened_true = happened_true, debugging = debugging, debugging_arg = debugging_arg )

            outputs[ segment ] = tf.concat( output, axis = -1 )

        return outputs, stakes, profits


    def Save_Selection_Stat( self, stats_total ) :

        save_folder = super_config['Summary']
        filePath = os.path.join( save_folder, "Stats_Selection" + ".xlsx" )
        backup = os.path.join( save_folder, "Backup Stats_Selection" + ".xlsx" )

        session_timestamp_min = session_time.strftime("%m%d-%H%M")

        self.excel_driver.Ensure_File_Sheet_Exists( filePath, sheetName = session_timestamp_min)

        names = [ 'f1', 'pre', 'cov', 'mean', 'std' ]
        names = [ name + suffix for name in names for suffix in ['.h', '.a', '.mh', '.ma'] ]

        dictionaries = []
        for stats in stats_total : # across games.
            stat_home, stat_away, stat_med_home, stat_med_away = stats # for a sigle game
            if stat_home is None or stat_away is None or stat_med_home is None or stat_med_away is None : continue

            dict = {}
            arrays = [ arr[i] for i in range(5) for arr in [ stat_home, stat_away, stat_med_home, stat_med_away ] ]
            accuracy = stat_home[6] # hit.

            for query_id in self.task.queries :
                qstr = str( query_id ).zfill(2)
                for name, array in zip( names, arrays ) :
                    name = name + '.' + qstr
                    dict[ name ] = array[ query_id ]
                dict[ 'hit' + '.' + qstr ] = accuracy[ query_id ]

            dictionaries.append( dict )

        self.excel_driver.Save_Data_Dictionaries( filePath, sheetName = session_timestamp_min, dictionaries = dictionaries, overwrite = False, horizontal = True )

        self.excel_driver.Backup( filePath, backup )

        return


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


    def Create_Branches( self, inputs, config, train_params, x_data, y_data ) :

        branches = None

        def Create_QWise_Branch( outputId ) :
            base_X, homeEmbedder, awayEmbedder = self.Get_Base_X( inputs, config, train_params, outputId ) # Each branch has now its own, exclusive embeddings.
            config_copy = copy.deepcopy( config )
            config_copy['dims.core'] = [ base_X.shape[-1] * dim_x for dim_x in config_copy['DimX.core'] ]
            config_copy['dims.attach'] = [ [ base_X.shape[-1] * dim_x for dim_x in list_for_att ] for list_for_att in config_copy['DimX.attach'] ]
            branch = Model_Branch( outputId, inputs, base_X, config_copy['Dim.Outs'][ outputId ], config_copy['ActOuts'][ outputId ], \
                config_copy, x_data, y_data )
            return branch


        def Create_Monolithic_Branch() :
            base_X, homeEmbedder, awayEmbedder = self.Get_Base_X( inputs, config, train_params, outputId = -1 ) # Each branch has now its own, exclusive embeddings.
            config_copy = copy.deepcopy( config )
            config_copy['dims.core'] = [ base_X.shape[-1] * dim_x for dim_x in config_copy['DimX.core'] ]
            config_copy['dims.attach'] = [ [ base_X.shape[-1] * dim_x for dim_x in list_for_att ] for list_for_att in config_copy['DimX.attach'] ]

            dim_total_outputs = 0
            for dim in self.config['Dim.Outs'] : dim_total_outputs += dim
            branch = Model_Branch( 0, inputs, base_X, dim_total_outputs, config_copy['ActOuts'][ 0 ], \
                config_copy, x_data, y_data )
            return branch

        branchTool = Create_QWise_Branch( 0 )

        # if or not : self.config['.Attach'] is True :
        if self.config['.Out.Wise'] is True :

            branches = []

            for outputId in range ( len( self.config['Dim.Outs'] ) ) :

                if self.config['Branch'] in [ 'past', 'best' ] :
                    best_model, model_number = self.Load_Best_Branch( outputId )
                    if best_model is not None :
                        best_model = branchTool.Compile_Model( best_model, train_params = self.train_params, branch_id = outputId ) 
                        branches.append( best_model )
                    else :
                        branch = Create_QWise_Branch( outputId )
                        branches.append( branch )

                elif self.config['Branch'] in [ 'better' ] :
                    branch = Create_QWise_Branch( outputId )
                    branches.append( branch )

                elif self.config['Branch'] in [ 'new' ] :
                    model, model_number = self.Load_Branch_Current_Model( outputId )
                    if model is not None :
                        model = branchTool.Compile_Model( model, train_params = self.train_params, branch_id = outputId ) 
                        branches.append( model )
                    else :
                        branch = Create_QWise_Branch( outputId )
                        branches.append( branch )

        else :
            branch = Create_Monolithic_Branch()
            branches = [ branch ]

        del x_data, y_data

        return branches, branchTool


    def Combine_Branch_Models( self, branch_models ) :

        model = None

        if self.config['.Out.Wise'] is True :

            assert len( branch_models ) == len( self.config['Dim.Outs'] )
            outputs = [ b_model( self.inputs ) for b_model in branch_models ]

            base_models = [ tf.keras.models.Model( inputs = self.inputs, outputs = outputs[outputId], name = self.Create_Base_Name( outputId ) ) for outputId in range( len( outputs ) ) ]

            model = tf.keras.models.Model( inputs = self.inputs, outputs = [ base_model( self.inputs ) for base_model in base_models ], name = self.model_name )

        elif self.config['.Out.Wise'] is False :

            assert len( branch_models ) == 1
            model = branch_models[0]

            #outputs = [ branch_models[0]( self.inputs ) ]
            #base_models = [ tf.keras.models.Model( inputs = self.inputs, outputs = outputs[outputId], name = self.Create_Base_Name( outputId ) ) for outputId in range( len( outputs ) ) ]

        # Inline-Checking - DONE ------------------------------------------------------- Keep this code.
        """
        def max_error( tensor1, tensor2 ) :
            #max = np.nanmax( np.abs( np.nan_to_num( arr1 ) - np.nan_to_num( arr2 ) ) )
            max = tf.reduce_max( tf.abs( tensor1 - tensor2 ) ).numpy()
            return max

        for segment in self.task.dataset.standard_segments :
            model_output = model( self.x_data[segment], training = False )
            model_output = model_output if isinstance( model_output, list ) else [ model_output ]

            for branch_id in range( len( branch_models ) ) :
                branch_model = branch_models[ branch_id ]
                branch_output = branch_model( self.x_data[segment], training = False )
                assert max_error( model_output[ branch_id ], branch_output ) < 1e-5
        """

        while len( branch_models ) > 0 :
            del branch_models[ 0 ]
        del branch_models

        return model
            

    def Divide_Tensor( self, tensor, axis, dim_outputs ) :

        dim_total = 0; indices = [0]
        for dim in dim_outputs : 
            dim_total += dim; indices.append( dim_total )

        assert tensor.shape[ axis ] == dim_total
        tensors = [ tensor[:, indices[n] : indices[n+1] ] for n in range( len(indices) - 1 ) ]

        return tensors


    def Compile_Model( self, model, train_params ) : # Why compile? Losses were applied in branches. We must comple it again because it a combination of branch models.

        modelParams = train_params['modelParams']; trainResources = train_params['trainResources']
        feeder = Params_Feeder()
        optimizer = feeder.translator_optimizer( modelParams['.optmer'] )
        del feeder

        losses = { self.Create_Base_Name( outputId ) : trainResources['.loss']( outputId ) for outputId in range ( len( self.config['Dim.Outs'] ) ) }
        # Or, simply - losses = trainResources['.loss']( 0 )
        metrics = None #{ None for query_id in self.task.queries }

        model.compile( optimizer = optimizer, loss = losses, metrics = metrics )

        return model


    def Fit_Model( self, model, train_params ) :

        custom = { 'subject' : "Fitting combined base model", '.epochs' : self.config['BaseEpo'] }
        custom_fit_params = self.Create_Fit_Params( train_params, custom )

        model = self.Fit_Model_Once( model, train_params, custom_fit_params, self.x_data, self.y_data )

        return model


    def Create_Base_Name(self, outputId ) :
        return 'OUT_' + str( outputId ).zfill( 2 )  # use log( N_QUERIES )


    def Create_Output_Key( self, segment ) :
        output_key = segment

        return output_key

    def Produce_Output( self, segment, update_cache = True, input = None, aux_input = None, debugging = False ) : # The argument signature i

        # PENDING : Let Model_Chained be the only object responsible for cache management. Remove output cache from this method.

        output = None

        output_key = self.Create_Output_Key( segment )

        if self.model != None :
            #trainable = self.model.trainable

            if self.output.get( output_key, None ) is not None :
                output = self.output[ output_key ]

            else :

                #self.model.trainable = False
                # 'input' must be None, as this it the topmost model.
                x_data = input
                if x_data is None : x_data = self.Create_X_Data( segment, self.config, self.train_params, aux_input = aux_input )

                output = self.model( x_data, training = False )
                output = output if isinstance( output, list ) else [ output ]

            if update_cache : # modelBase is a best model loaded from HDD.
                self.output.update( { output_key : output } )

        del input

        return output


    def Get_Outcomes( self, modelOutput, odds, happened_true = None, debugging = '' ) :
        """ To be overwritten."""

        stake = None; profit = None

        return stakes, profits


    def Sync_Output_Subward( self, segment, update_cache = True, input_from_super = None, aux_input = None, subward_sync = False, debugging = False ) :

        # This will be heavily overwritten and enriched by descendent classes, including Model_Chained.

        output = None
        
        if self.output_validity.get( segment, False ) is True : # Save time greatly.
            output = self.output[ segment ]
        else :
            output = self.Produce_Output( self, segment, update_cache, input = None, aux_input = None, debugging = debugging )
        
        return output


    # Open ================================================ Reporting ========================================================

    def Initialize_Gauge( self, margin, segment = 'test' ) :
        margins_by_segment = self.task.dataset.Get_Propositional_Margins( margin, segment )
        self.gauge = Gauge( margins_by_segment, self.task.queries, profit_weight = 1.0, safety_weight = 1.0 )

        return self.gauge


    def Report_Visual( self, segments, queryDic, show_foretest = True, good_queries = None, title = '', subtitle = '', save = False, save_number = -1, browse = False ) :

        visual_reporter = Visual_Reporter()

        _, model_stat_dframe = self.gauge.Get_Model_Statistics( self )

        for segment in segments :

            if self.stakes.get( segment, None ) is not None :

                string_out = StringIO(); sys.stdout = string_out
                self.Report_Text( segment, queryDic, model_stat_dframe )
                report_string = string_out.getvalue(); sys.stdout = sys.__stdout__
                print( report_string )

                save_folder = None
                if save : save_folder = super_config['Reports']
                visual_reporter.Draw_Report( segment, model_stat_dframe, queryDic, self.task.queries, title, subtitle, save_folder, save_number = save_number, browse = browse  )

        return


    def Report_Text( self, segment, queryDic, model_stat_dframe ) :

        #profit_cum, total_profit, safety, score = analysis_query_wise

        profit_cum = model_stat_dframe.loc[ 'cum', segment ]
        total_profit = profit_cum[ -1, : ]
        profit = model_stat_dframe.loc[ 'profit', segment ]
        safety = model_stat_dframe.loc[ 'safe', segment ]
        safety_c = model_stat_dframe.loc[ '*safe', segment ]
        f1 = model_stat_dframe.loc[ 'f1', segment ]
        n_profits = model_stat_dframe.loc[ 'plRt', segment ]

        print ( '+' * 30 + " Total Profit on '{}', {}".format(segment, self.model_name) )

        sum_total = 0.0; p_sum_total = 0.0; n_sum_total = 0.0

        for queryPlace in range( len( self.task.queries ) ) :
            query_id = self.task.queries[ queryPlace ]

            total = total_profit[ queryPlace ].numpy()
            sum_total += total
            if total >= 0.0 : p_sum_total += total
            else : n_sum_total -= total 
            print( "query_id: {:>2}, name: {:>}, total_profit: {:4.2f}, safety: {:4.2}, f1: {:4.2}".format( query_id, queryDic[query_id], total_profit[queryPlace].numpy(), safety[queryPlace], f1[queryPlace] ) )

        n_games = profit_cum.shape[0]
        print( '-' * 30 + " Sum_Total: {:5.2f} = {:5.2f} - {:5.2f}. NP_rate: {:5.2f}".format(sum_total, p_sum_total, n_sum_total, n_sum_total/(p_sum_total + 1e-9) ))
        print( '-' * 30 + " Sum_Total/N: {:5.4f} = {:5.4f} - {:5.4f}.".format(sum_total/n_games , p_sum_total/n_games, n_sum_total/n_games ))

        print ("Collective safety: {:4.2}".format( safety_c ) )

        del profit_cum, total_profit, profit, safety, safety_c, f1, n_profits

        return


    def Report_Model( self, trainState, stakes = None, profits = None ) :

        report_super_config = Report_super_config()
        report_datause =  self.task.Report_DataUse()
        report_config = self.task.Report_Config()

        param_arranged = trainState['.pick']
        report_param = self.task.Report_Param( param_arranged )

        report_fit = 'FitHistory -  '

        if stakes is not None and stakes.get('test', None) is not None :
            stake = stakes['test']
            report_fit += ", [Stake - max: {:5.2f}, mean: {:5.2f}, total: {:5.2f}]".format( tf.reduce_max( stake ), tf.reduce_mean( stake ), tf.reduce_sum( stake ) )

        if profits is not None and profits.get('test', None) is not None :
            profit = profits['test']
            profit_cum, total_profit, safety, score_inclusive = self.gauge.Analyze_Profit_Collectively( profit, 'test' )
            report_fit += ", PL/G: {:5.3f}, s'ty: {:5.3f}".format( total_profit/profit.shape[0], safety[0] )

        report = "; ".join( [ report_super_config, report_config, report_param, report_fit ] ) # report_datause, 

        return report

    # ================================================ Excel ========================================================



    def Excelize_Model_Data( self, filePath, backup, trainState, string_dict ) :

        #session_timestamp_min = parser.parse( super_config['sess_time'] ).strftime("%y%m%d-%H%M")

        session_timestamp_min = session_time.strftime("%m%d-%H%M")

        self.excel_driver.Ensure_File_Sheet_Exists( filePath, sheetName = session_timestamp_min)

        data_dictionaries, string_dict_updated = self.Get_Data_Dict( trainState, string_dict )

        total_dict = { 'Model' : self.model_save_number }
        for dict_name, dict_body in data_dictionaries.items() : # No use of dict_name however wellformed it is. We know them and the order they are arranged in in data_dictionaries.
            total_dict.update( dict_body )

        self.excel_driver.Save_Data_Dict( filePath, sheetName = session_timestamp_min, dictionary = total_dict, overwrite = False, horizontal = True )

        self.excel_driver.Backup( filePath, backup )

        return string_dict_updated # string_dict is managed by History object at a higher level.


    def Get_Data_Dict( self, trainState, string_dict ) :

        def GetOrCreateEntry( dict, key ) :
            max_number = -1; number_assigned = -1
            for string_dict, number_dict in dict.items() :
                if string_dict == key : 
                    number_assigned = number_dict
                    break
                elif max_number < number_dict : 
                    max_number = number_dict
            if number_assigned < 0 : # 'string_dict' dictionary has no value key. Add a new entry.
                number_assigned = max_number + 1
                strings.update( { key : number_assigned } ) # Historic 'string_dict' dictionary is taking a new entry. 

            return dict, number_assigned

        def Serialize_Dict( source, target, string_dict ) :

            if isinstance( source, dict ) :
                for key, value in source.items() :

                    if key[0] != '.' : continue

                    if isinstance( value, dict ) :
                        Serialize_Dict( value, target, string_dict )
                    elif isinstance( value, list ) or isinstance( value, tuple ) :
                        temp_dict = {}
                        for index in range( len( value ) ):
                            temp_dict[ key + '.' + str( index) ] = value[index]
                        Serialize_Dict( temp_dict, target, string_dict )

                    elif isinstance( value, int ) : # bool falls on this branch.
                        target[ key[ 1: ] ] = int(value) # To embrace boolean values.
                    elif isinstance( value, float ) or isinstance( value, np.float32 ) :
                        target[ key[ 1: ] ] = value
                    elif isinstance( value, str ) :
                        string_dict, number_assigned = GetOrCreateEntry( string_dict, value )
                        target[ key[ 1: ] ] = number_assigned

                    else : pass #raise Exception( 'Help' )
            else : raise Exception( 'Help' )

            return target, string_dict


        
        dictionaries = {}
        strings = string_dict # so history is updated on and renewed on each session.

        dict_source = trainState
        dict_trainState, strings = Serialize_Dict( dict_source, {}, strings )
        dictionaries['trainState'] = dict_trainState

        model_stat_dict, _ = self.gauge.Get_Model_Statistics( self )
        dict_model, strings = Serialize_Dict( model_stat_dict, {}, strings )
        dictionaries['model_summary'] = dict_model
        
        string_dict_updated = strings

        return dictionaries, string_dict_updated
