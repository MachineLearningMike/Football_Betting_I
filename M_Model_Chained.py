import tensorflow as tf
import numpy as np

from M_Config import *
from M_Model_Base import *

class Model_Chained( Model_Base ) :

    def __init__( self, task ) :

        super().__init__( task )

        self.model_save_number = -1

        self.model_version_string = self.task.config['.model.ver']
        self.branch_version_string = self.task.config['.branch.ver']

        return

    def GetModelNumbers( self ) :
        numbers_super = self.GetModelNumbersSuperward( count_self = False ) # None or Not None
        numbers_local = self.task.model_slots.GetExistingModelNumbers_Duo( self.model_name, self.model_version_string )
        numbers_sub = self.GetModelNumbersSubward( count_self = False ) # None or Not None

        numbers = numbers_local
        if numbers_super is not None : numbers = [ num for num in numbers if num in numbers_super ]
        if numbers_sub is not None : numbers = [ num for num in numbers if num in numbers_sub ]

        numbers = sorted( numbers )

        return numbers


    def GetNextSaveNumber( self ) :

        _, _, number = self.task.model_slots.GetNextSavePathAndNumber( self.model_name, forced_number = -1 )

        return number


    def LoadModel( self, number = -1, summary_only = False, withWeights = True) :

        number_to_load = -1

        numbers = self.GetModelNumbers()

        if number < 0 :
            if len( numbers ) > 0 : number_to_load = numbers[-1] # load the latest integral model.
        else :
            if len( numbers ) > 0 and number in numbers : number_to_load = number
        
        loaded = False; summary = None

        if number_to_load >= 0 :
            loaded_superward = self.LoadModelSuperward( load_self = False, forced_number = number_to_load, summary_only = summary_only, withWeights = withWeights )
            loaded_subward =self.LoadModelSubward( load_self = False, forced_number = number_to_load, summary_only = summary_only, withWeights = withWeights ) # False to avoid redundant saving.       

            model, summary = self.task.model_slots.LoadModel( self.model_name, self.model_version_string, number = number_to_load, \
                custom_objects = self.Get_Custom_Objects(), summary_only = summary_only, withWeights = withWeights )

            loaded = loaded_superward and loaded_subward

            if loaded is True :
                self.model = model
                self.Invalidate_Output_Subward( segment = None )

        return loaded, summary, number_to_load


    def GetNextAvailableSaveNumber( self ) :
        max_number = -1
        max_number = self.GetMaxModelNumberSuperward( max_number, count_self = True )
        max_number = self.GetMaxModelNumberSubward( max_number, count_self = False )
        return max_number + 1        


    def SaveModel( self, summary, forced_number = -1 ) :

        if forced_number < 0 : forced_number = self.GetNextAvailableSaveNumber()

        saved_super, number_super = self.SaveModelSuperward( summary, save_self = True, forced_number = forced_number )
        saved_sub, number_sub = self.SaveModelSubward( summary, save_self = False, forced_number = forced_number ) # False to avoid redundant saving.

        save_number = -1

        if super_config['save_model'] or True :

            if number_super == number_sub : save_number =  number_super
            elif saved_super is True and saved_sub is False : save_number =  number_super
            elif saved_super is False and saved_sub is True : save_number =  number_sub

            self.model_save_number = save_number # ================================= Keep save number, until next time.

        return save_number


    def GetModelNumbersSuperward( self, count_self = True ) :

        numbers = None; numbers_local = None

        superModels = self.task.Get_superModels()
        if superModels != None and len( superModels ) > 0 :
            for modelBase in superModels :
                numbers = modelBase.GetModelNumbersSuperward()

        if count_self :
            numbers_local = self.task.model_slots.GetExistingModelNumbers_Duo( self.model_name, self.model_version_string )


        common_numbers = None
        if numbers is not None :
            if numbers_local is not None : common_numbers = [ num for num in numbers if num in numbers_local ]
            else : common_numbers = numbers
        else :
            if numbers_local is not None : common_numbers = numbers_local

        return common_numbers


    def GetModelNumbersSubward( self, count_self = True ) :

        numbers = None; numbers_local = None

        if count_self :
            numbers_local = self.task.model_slots.GetExistingModelNumbers_Duo( self.model_name, self.model_version_string )

        subModels = self.task.Get_subModels()
        if subModels != None and len( subModels ) > 0 :
            for modelBase in subModels :
                numbers = modelBase.GetModelNumbersSubward()

        common_numbers = None
        if numbers is not None :
            if numbers_local is not None : common_numbers = [ num for num in numbers if num in numbers_local ]
            else : common_numbers = numbers
        else :
            if numbers_local is not None : common_numbers = numbers_local

        return common_numbers


    def GetMaxModelNumberSuperward( self, max, count_self = True ) :

        superModels = self.task.Get_superModels()
        if superModels != None and len( superModels ) > 0 :
            for modelBase in superModels :
                candidate = modelBase.GetMaxModelNumberSuperward( max )
                if max < candidate : max = candidate

        if count_self :
            numbers_local = self.task.model_slots.GetExistingModelNumbers_Duo( self.model_name, self.model_version_string )
            if len(numbers_local) > 0 and max < numbers_local[-1] : max = numbers_local[-1]

        return max


    def GetMaxModelNumberSubward( self, max, count_self = True ) :

        if count_self :
            numbers_local = self.task.model_slots.GetExistingModelNumbers_Duo( self.model_name, self.model_version_string )
            if len(numbers_local) > 0 and max < numbers_local[-1] : max = numbers_local[-1]

        subModels = self.task.Get_subModels()
        if subModels != None and len( subModels ) > 0 :
            for modelBase in subModels :
                candidate = modelBase.GetMaxModelNumberSubward( max )
                if max < candidate : max = candidate

        return max


    def SaveModelSuperward( self, summary, save_self = True, forced_number = -1 ) :

        min = float('inf'); max = float('-inf')
        something_saved = False

        superModels = self.task.Get_superModels()
        if superModels != None and len( superModels ) > 0 :
            for modelBase in superModels :
                number = modelBase.SaveModelSuperward( summary, forced_number = forced_number ) # Do NOT propagate save_self = False
                if min > number : min = number
                if max < number : max = number
                something_saved = True

        if save_self is True and self.model is not None :
            number = self.task.model_slots.SaveModel( self.model_name, self.model_version_string, model = self.model, summary = summary, forced_number = forced_number )
            if min > number : min = number
            if max < number : max = number
            something_saved = True

        if min == max : return something_saved, min
        else : return something_saved, -1            


    def SaveModelSubward( self, summary, save_self = True, forced_number = - 1 ) :

        min = float('inf'); max = float('-inf')
        something_saved = False

        if save_self is True and self.model is not None :
            number = self.task.model_slots.SaveModel( self.model_name, self.model_version_string, model = self.model, summary = summary, forced_number = forced_number )
            if min > number : min = number
            if max < number : max = number
            something_saved = True

        subModels = self.task.Get_subModels()
        if subModels != None and len( subModels ) > 0 :
            for modelBase in subModels :
                number = modelBase.SaveModelSubward( summary, forced_number = forced_number ) # Do NOT propagate save_self = False
                if min > number : min = number
                if max < number : max = number
                something_saved = True
        
        if min == max : return something_saved, min
        else : return something_saved, -1   


    def LoadModelSuperward( self, load_self = True, forced_number = -1, summary_only = False, withWeights = True ) :

        loaded = True

        superModels = self.task.Get_superModels()
        if superModels != None and len( superModels ) > 0 :
            for modelBase in superModels :
                ret = modelBase.LoadModelSuperward( forced_number = forced_number, summary_only = summary_only, withWeights = withWeights ) # Do NOT propagate save_self = False
                if ret is False : loaded = False

        if load_self is True :
            self.model, _ = self.task.model_slots.LoadModel( self.model_name, self.model_version_string, number = forced_number, \
                custom_objects = self.Get_Custom_Objects(), summary_only = summary_only, withWeights = withWeights )
            if self.model is None : loaded = False

        return loaded


    def LoadModelSubward( self, load_self = True, forced_number = -1, summary_only = False, withWeights = True ) :

        loaded = True

        if load_self is True :
            self.model, _ = self.task.model_slots.LoadModel( self.model_name, self.model_version_string, number = forced_number, \
                custom_objects = self.Get_Custom_Objects(), summary_only = summary_only, withWeights = withWeights )
            if self.model is None : loaded = False

        subModels = self.task.Get_subModels()
        if subModels != None and len( subModels ) > 0 :
            for modelBase in subModels :
                ret = modelBase.LoadModelSubward( forced_number = forced_number, summary_only = summary_only, withWeights = withWeights ) # Do NOT propagate save_self = False
                if ret is False : loaded = False

        return loaded



    def Load_Best_Model( self, round, trainResources, number = -1, force_best_save = False ) :

        custom_objects = self.Get_Custom_Objects()
        # Load the latest model.
        # model, summary = self.model_slots.LoadModel( self.model_name, self.model_version_string, number = number, custom_objects = custom_objects,  defaultSummary = {}, summary_only = False )
        loaded, summary, number = self.LoadModel( number = number, summary_only = False, withWeights = True )

        stakes = {}; profits = {}

        if loaded :

            for segment in self.task.dataset.standard_segments :

                odds = self.task.dataset.Data( 'odds', segment )
                happened_true = self.task.Data( 'happened', segment )
                output = self.Sync_Output_Subward( segment, update_cache = True, input_from_super = None, subward_sync = True )
                stakes[ segment ], profits[ segment ] = self.Get_Outcomes( output, odds, happened_true = happened_true )

            if force_best_save :
                trainState = summary.get( 'trainState', None )
                fit_history = summary.get( 'fit_history', None )
                found_and_propageted, save_number, good_queries = self.Save_If_Best_Model( round, trainState, trainResources, save = force_best_save, forced_number = number )
                summary.update( { 'save_number': save_number, 'good_queries' : good_queries } )

        return loaded, summary, number, stakes, profits


    def Save_If_Best_Model( self, round, trainState, trainResources, save = True, forced_number = -1 ) :
        
        found_a_best_model = False; save_number = -1; good_queries = None 

        if self.model is None :
            pass
        else :
        
            model_file_name = ''

            is_best, good_queries = self.gauge.Is_Good_Model( self, trainState['.fitParams'], trainResources )

            found_a_best_model = tf.reduce_any( is_best )

            found_a_best_model = True #================================== PENDING - Parameterize this action.

            if  found_a_best_model or super_config['.Test'] :
                print( '------------------------------------ This is a good model.' )

                # self.Sync_Output_Subward( segment, update_cache= True, subward_sync = True ) ?

                if save is True :
                    summary = {
                        'trainState' : trainState, \
                        'good_queries' : good_queries.numpy(), \
                    }
                    save_number = self.SaveModel( summary, forced_number = forced_number )

            else :
                pass

            # Tensorflow Error? This block doesn't work for Config['.Out.Wise'] = False
            """
            loss = self.Get_MinLoss( val_weight = 1. )
            if self.best_loss is None or self.best_loss > loss :
                self.best_loss = loss
                self.best_model = tf.keras.models.clone_model( self.model )
                self.best_model.set_weights( self.model.get_weights() )
                #for sLayer, tLayer in zip( self.model.layers, self.best_model.layers ) :
                #    w = sLayer.get_weights(); tLayer.set_weights( w )
            """

        return found_a_best_model, save_number, good_queries


    def Invalidate_Output_Subward( self, segment = None ) :

        if segment is None :
            for key, value in self.output.items() :
                if value is not None : del value
            self.output = {}
        else :
            output_key = self.Create_Output_Key( segment )
            if self.output.get( output_key, None ) is not None : self.output.pop( output_key )
    
        subModels = self.task.Get_subModels()
        if subModels is not None :
            for model in subModels :
                model.Invalidate_Output_Subward( segment )

        return


    def Sync_Output_Subward( self, segment, update_cache = True, input_from_super = None, aux_input = None, subward_sync = False, debugging = False ) :

        output = None
        re_produced = False

        output_key = self.Create_Output_Key( segment )

        if self.model != None :
            if self.output.get( output_key, None ) is not None :
                output = self.output[ output_key ]
            else :
                output = self.Produce_Output( segment, update_cache = update_cache, input = input_from_super, aux_input = aux_input, debugging = debugging )
                self.output[ output_key ] = output
                re_produced = True

            if subward_sync is True :

                if output is not None and re_produced is True :
                    subModels = self.task.Get_subModels()
                    if subModels is not None :
                        for model in subModels :
                            model.Sync_Output_Subward( segment, update_cache = update_cache, input_from_super = output, aux_input = aux_input, subward_sync = subward_sync, debugging = debugging ) # let the sub inherit aux_input.

        return output


    def Get_Output_With_No_Subward_Sync( self, segment ) :
        
        return self.Sync_Output_Subward( segment, update_cache = True, input_from_super = None, aux_input = None, subward_sync = False )


    def Redefine_Models_Summary( self ) :

        model_numbers = self.GetModelNumbers()

        for number in model_numbers :

            loaded, summary, number, stakes, profits = self.Load_Best_Model( round = -1, fitParams = self.task.fitParams, trainResources = self.task.trainResources, number = number, force_best_save = True )

            if loaded :
                subtitle = modelBase.Model_Report( param_arranged = summary.get( 'param_arranged', None ), fit_history = summary.get( 'fit_history', None ), stakes = stakes, profits = profits )
                self.Report_Visual( self.task.dataset.standard_segments, self.task.queryDic, show_foretest = True, good_queries = summary.get('good_queries', self.task.queries ), title = 'Refining Load', subtitle = subtitle, save = (good_queries.shape[0] > 0 or super_config['.Test']), save_number = number ) #, browse = True )
        return
    
    def Summarize_Models( self ) :

        model_numbers = self.GetModelNumbers()

        stakes = {} ; profits = {} ; counts = 0

        for number in model_numbers :
            loaded, summary, number, stakes_local, profits_local = self.Load_Best_Model( round = -1, fitParams = self.task.fitParams, trainResources = self.task.trainResources, number = number, force_best_save = False ) # Loads all chained models.

            for segment in self.task.dataset.standard_segments :

                if stakes .get( segment, None ) is None : stakes [ segment ] = stakes_local[ segment ]
                else : stakes [ segment ] += stakes_local[ segment ]

                if profits.get( segment, None ) is None : profits[ segment ] = profits_local[ segment ]
                else : profits[ segment ] += profits_local[ segment ]
            
            counts += 1

        if counts > 0 :
            for segment in self.task.dataset.standard_segments :
                stakes[ segment ] /= counts
                profits[ segment ] /= counts

        #for segment in self.task.dataset.standard_segments :
        #    _, _, _, _, _, odds, _, _, _, happened_true = self.task.dataset.GetDataSegment( segment )
        #    #odds = tf.convert_to_tensor( odds )
        #    #happened_true = tf.convert_to_tensor( happened_true )
        #    stake = stake_cum[ segment ]
        #    profit = Get_Pofit_On_Hap( odds, happened_true, stake_cum[ segment ] )

        subtitle = modelBase.Model_Report( stakes = stakes, profits = profits )
        self.Report_Visual( self.task.dataset.standard_segments, self.task.queryDic, title = 'Models Summarized', subtitle = subtitle, show_foretest = False, browse = True )

        return


    def Save_Branch( self, model, branchId, val_loss ):
        
        if super_config['save_model'] :

            forced_number = self.GetNextAvailableSaveNumber()
            self.task.model_slots.Save_Branch( self.model_name, self.branch_version_string, model, branchId, val_loss, forced_number )

        return

    def Load_Branch_Current_Model( self, branchId, custom_objects = {} ) :

        forced_number = self.GetNextAvailableSaveNumber()
        return self.task.model_slots.Load_Best_Branch( self.model_name, self.branch_version_string, branchId, forced_number, custom_objects )


    def Load_Best_Branch( self, branchId, custom_objects = {}  ):

        return self.task.model_slots.Load_Best_Branch( self.model_name, self.branch_version_string, branchId, custom_objects = custom_objects )



