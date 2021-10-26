import tensorflow as tf
import numpy as np

import os
import sys

from M_Metrics import *
from M_Losses import *
from M_Football import *

from M_Dataset import *
from M_Model_Slots import *
from M_Database import *
from M_DataChecker import *

from Task_History import *

from Task_Params_Feeder import *

from M_Config import *

class Task_Base( Params_Feeder ) :


    def Report_DataUse( self ) : return "DataUse - {}".format( self.dataUse )


    def __init__( self, taskName, dataservice, superTasks = None, is_train_chain = False, checkMargin = 100 ) : 

        super().__init__()

        self.modelsPerQuery = 2
        self.queryDic = super_config['QDic']
        self.queries = super_q_group.Queries()


        self.taskName = taskName
        self.dataservice = dataservice

        self.history = History( self.taskName )
                
        self.dataUse = {} #===================== dataUse. Anchor to this dictionary object.

        self.modelBase = None
        self.superTasks = superTasks
        self.subTasks = None
        self.is_train_chain = is_train_chain
        self.checkMargin = checkMargin

        self.model_slots = ModelSlots( self.taskName, self.modelsPerQuery, self.queryDic, self.queries )


        #self.trainState = self.Collect_TrainState() # Very beginning of trainState.

        #self.Initialze_TrainState() # TrainState may or may not include dataset.

        #================================== TrainState Managment

        self.Initialize_Fit_Context() # PENDING - Why here? while Prepare_Train() is out there seperately.


        self.pathToFolder = self.model_slots.pathToFolder

        try:
            if not os.path.exists( self.pathToFolder ) or not os.path.isdir( self.pathToFolder ) :
                access_rights = 0o755
                os.mkdir( self.pathToFolder, access_rights )
        except OSError :
            raise Exception( "Couldn't create a folder: ", self.pathToFolder )
        else:
            print ("Successfully created the directory %s" % self.pathToFolder)

        self.TrainStateFilePath = os.path.join( super_config['Summary'], super_config['job_prefix'] + "_TrainState_" + self.taskName + ".txt")


    #================================================================

    def Initialize_Fit_Context(self) :

        self.fitParams = {}
        self.fitParams['has_train_focus_token'] = False
        self.fitParams['has_train_focus_token'] = False
        self.fitParams['has_had_no_focus'] = True
        self.fitParams['roundCounter'] = 0

        self.trainResources = {}


    def Prepare_Session( self, trainState, purpose ) :

        self.Implement_Volatile_TrainState( trainState )

        #self.Initialize_Fit_Context()
           
        self.dataUse = self.Rebuild_DataUse( trainState['.dataUse'], trainState['.super_config']['.Stage'] )

        dictionaries = self.history.Load_Dictionaries( path_from_caller = None, dict_names = ['leagues', 'teams', 'countries'] )
        
        name_variations = self.history.Load_Dictionaries( path_from_caller = "Manually_Built Dictionary.xlsx", dict_names = ['team_name_variations'], list_mode = True )
        name_variations_total = {}
        for dict_name, dict_body in name_variations.items() : name_variations_total.update( dict_body )

        dictionaries['name_variations'] = name_variations_total

        self.data, dataUse_updated, dictionaries_updated = PrepareData( dataUse = self.dataUse, dictionaries = dictionaries, from_save = super_config['db_from_save'] ) # ========== from_save == True for test.

        self.dataUse.update( dataUse_updated )

        #if dictionaries != dictionaries_updated :
        self.Save_Dictionaries( dictionaries_updated )

        self.dataset = Dataset( self.dataUse, self.data, self.queries )
        self.dataservice.Change_Dataset( self.dataset )

        dataChecker = DataChecker()
        # assert self.dataset.Checked_OK( dataChecker, dictionaries_updated ) is True

        self.modelBase.Prepare_Session( self.checkMargin, purpose )

        return


    def Save_Dictionaries( self, dictionaries_org ) :

        dictionaries = dictionaries_org.copy()

        name_variations = dictionaries.get('name_variations', None)
        if name_variations is not None : dictionaries.pop('name_variations') # name_variations must not changed by code.
        clusters_dict = dictionaries.get( 'clusters', None )
        if clusters_dict is not None : dictionaries.pop( 'clusters' )

        self.history.Save_Dictionaries( path_from_caller = None, dictionaries = dictionaries, overwrite = True )

        if clusters_dict is not None :
            self.history.Save_Dictionaries( path_from_caller = None, dictionaries = { 'clusters' : clusters_dict }, overwrite = True, list_mode = True )


    def Rebuild_DataUse( self, dataUse, trainStage ) : # trainStage is part of trainState, and related to super+_config['.stage']

        if super_config['Contest'] :

            dataUse['DB.Id'] = 12 # ============================================== 4,7
            dataUse['.BSize'] = None
            dataUse['.m.Attends'] = 10 #=================
            dataUse['.BSizeFit'] = super_config['.BSize']
            dataUse['.ShuFit'] = False # used in model.fit()
            dataUse['m.cluster.size'] = 0.05 # The size of minimum-sized cluster of teams, to count, measured in portion ( < 1.0 ) of total number of teams.
            
            dataUse['.relDepth'] = super_config['.relDepth']

            nCheck = super_config['.NCheck'] #======================== 

            # 0 < anchor and anchor < nRows
            dataUse['anchorToLast'] = nCheck  * check_day #nCheck # anchor = nRows - dataUse['anchorToLast']

            # 0 <= origin and origin < anchor
            dataUse['orgToLast'] = -1 # if dataUse['orgToLast'] < 0 : origin = 0 else : origin = nRows - dataUse['orgToLast']. data rows before 'origin' are not used at all.

            # anchor <= check_fr and check_fr < check_to and check_to <= nRows
            dataUse['offCheck'] = 0 # check_fr = anchor + dataUse['offCheck']. Absolute.
            dataUse['nCheck'] = nCheck # check_to = check_fr + dataUse['nCheck']. Absolute.

            # origin <= test_fr and test_fr < test_to and test_to <= anchor
            dataUse['offTest'] = 0 # test_to = anchor - dataUse['offTest']. Absolute.
            dataUse['nTest'] = nCheck # test_fr = test_to- dataUse['nTest']. Absolute.

            # origin == train_fr and train_fr < train_to and train_to == valid_fr and valid_fr <= valid_to and valid_to <= anchor
            dataUse['.nTrain'] = 70000 # This and the next tell just the propositional rate of each.
            dataUse['.nValid'] = 30000 #1000 # This and the above tell just the propositional rate of each.
            dataUse['offValid'] = 0 # The size is subject to the propositional rate defined above.
            dataUse['.TVRate'] = dataUse['.nTrain'] /  dataUse['.nValid']


        elif not super_config['Contest'] :

            dataUse['DB.Id'] = 14 # 11, 13 ============================================== 4,7

            dataUse['M.Day'] = super_config['.M.Day']
            dataUse['C.Day'] = super_config['.C.Day']

            dataUse['.BSize'] = None
            dataUse['.m.Attends'] = 10 #=================
            dataUse['.BSizeFit'] = super_config['.BSize']
            dataUse['.ShuFit'] = False # used in model.fit()
            dataUse['m.cluster.size'] = 0.01 # The size of minimum-sized cluster of teams to have in db, to count, measured in portion ( < 1.0 ) of total number of teams.

            dataUse['.relDepth'] = super_config['.relDepth']
            
            # origin == train_fr and train_fr < train_to and train_to == valid_fr and valid_fr <= valid_to and valid_to <= anchor
            dataUse['.nTrain'] = 75000 # This and the next tell just the propositional rate of each.
            dataUse['.nValid'] = 25000 #1000 # This and the above tell just the propositional rate of each.
            
        return dataUse



    def Get_Loss_Class( self, loss_string ) : # Placeholder.
        # Overwritten by descendents.
        return None

    def Get_Round_String( self ) :

        str = ''
        if self.superTasks != None and len( self.superTasks ) > 0 :
            str += '('
            for superTask in self.superTasks :
                str = str + superTask.Get_Round_String() + ', '
            str += ')'
        str = str + ' - ' + self.taskName + " Round {} ".format( self.fitParams['roundCounter'] )

        return str


    def Register_SubTask( self, subTask ) :

        if subTask != None :
            if self.subTasks == None : self.subTasks = []
            self.subTasks.append( subTask )

    def Get_superModels( self ) :
        if self.superTasks is not None :
            return [ task.modelBase for task in self.superTasks ]
        else :
            return None

    def Get_subModels( self ) :
        if self.subTasks is not None :
            return [ task.modelBase for task in self.subTasks + [] ]
        else :
            return None

    def Take_Focus( self ) :
        self.fitParams['has_train_focus_token'] = True


    def Create_TrainResources( self ) : return None # Placeholder to be overwritten by descendants.


    def Implement_Volatile_TrainState( self, trainState ) :

        self.trainResources = self.Create_TrainResources( trainState['.config'] )


    #================================== TrainStateer =====================

    def SaveTrainState( self, trainState ) :

        try :
            SaveJsonData( trainState, self.TrainStateFilePath )
        except :
            raise Exception( 'Failed to write to ', self.TrainStateFilePath )


    def LoadTrainState( self, update_config = False ) :

        trainState = None
        is_first_session = True

        try :
            trainState = LoadJsonData( self.TrainStateFilePath )
            is_first_session = False
        except :
            trainState = None
            is_first_session = True

        if trainState is not None and update_config is True :
            trainState['.super_config'].update( super_config )
            trainState['.config'].update( self.config )

        return trainState, is_first_session


    def Is_Valid_TrainState( self ) :

        # PENDING - Ask the relevant obejct, for validity of pick_pointer, ...
        return True


    def Collect_TrainState( self ) :

        trainState = { \
            'version' : super_config['.Ver'], \
            '.super_config' : super_config, \
            '.dataUse' : self.dataUse, \
            '.config' : self.config, \
            '.fitParams' : self.fitParams, \
            'params_unarranged' : self.params_unarranged, \
            'pick_pointer' : self.pick_pointer, \
            '.pick' : self.current_pick, \
            'valid' : self.Is_Valid_TrainState()
        }

        return trainState



    def ClearTrainState( self ) :

        DeleteFile( self.TrainStateFilePath )       
        
        return



    def Initialze_TrainState( self ) : # A placeholder, to be overwritten by descentants.

        # Assume an existing trainState.

        self.GetInformed( 'Initialize_TrainState' ) # BEFORE initializing TrainState.

        self.__iter__() # This is the core of Initialize_TrainState()

        self.trainState = self.Get_To_Next_TrainState() # Next() is required after _iter_().

        return self.trainState


    def Restore_TrainState( self, trainState ) :

        self.GetInformed( 'Restore_TrainState' )

        global super_config

        if trainState['version'] > super_config['.Ver'] :
            raise Exception( "Error. The code version {} is less than the train state version {}.".format( super_config['.Ver'], trainState['version'] ) )

        super_config['.Ver'] = trainState['version']
        super_config.update( trainState['.super_config'] )
        self.dataUse.update( trainState['.dataUse'] )
        self.config.update( trainState['.config'] )
        self.fitParams.update( trainState['.fitParams'] )

        # PENDING: The following values might conflict with each other. Use a consistency checker method.
        self.__iter__()
        self.params_unarranged = trainState['params_unarranged']
        self.pick_pointer = trainState['pick_pointer']
        self.current_pick = trainState['.pick']
        # where to put trainState['valid'] ?


        self.Implement_Volatile_TrainState( trainState )
        self.trainState = trainState

        return self.trainState


    def Get_To_Next_TrainState( self, trainState = None, context = None ) :

        # Assume an existing trainState, which are collected by self.Collect_TrainState()

        valid = True
        try : self.__next__()
        except : valid = False

        trainState = self.Collect_TrainState()
        trainState['valid'] = valid

        if valid : self.Implement_Volatile_TrainState( trainState ) # Implement, because it's changed.
        self.trainState = trainState

        self.SaveTrainState( self.trainState ) # Save, valid or not. Save, because it's changed.

        self.GetInformed( 'Get_To_Next_TrainState' )

        return self.trainState

  

    def Initialize_TrainState_With_Params( self, d_day_delta = 1 ) :

        # Assume an existing trainState.

        if super_config['session'] == 'train' : data_day = self.trainState['.super_config']['.M.Day'] + d_day_delta
        elif super_config['session'] == 'test' : data_day = self.trainState['.super_config']['.C.Day'] + d_day_delta

        trainState = self.Initialze_TrainState()

        if super_config['session'] == 'train' : trainState['.super_config']['.M.Day'] = data_day
        elif super_config['session'] == 'test' : trainState['.super_config']['.C.Day'] = data_day

        self.trainState = self.Restore_TrainState( trainState )

        self.SaveTrainState( self.trainState ) # Special. 'trainState' is newly created, than existing, and needs to save.


    #======================== Chain_Trainter =================================


    def Find_Train_Focus_Task( self, find_superwards ) :

        focus_task = None

        tasks_to_lookup = None
        if find_superwards : tasks_to_lookup = self.superTasks
        else : tasks_to_lookup = self.subTasks

        if tasks_to_lookup is not None and len( tasks_to_lookup ) > 0 :
            for task in tasks_to_lookup :
                if task.fitParams['has_train_focus_token'] : focus_task = task
                else : focus_task = task.Find_Train_Focus_Token( find_superwards )
                if focus_task is not None: break

        return focus_task


    def Find_Focus_Task_All_Around( self ) :

        focus_task = None
        if self.fitParams['has_train_focus_token'] : focus_task = self
        
        if focus_task is None :
            focus_task = self.Find_Train_Focus_Token( find_superwards = True )
            if focus_task is None :
                focus_task = self.Find_Train_Focus_Token( find_superwards = False )

        return focus_task


    def GetModelNumbers( self, nViewers = -1 ) :

        numbers = self.modelBase.GetModelNumbers()
        numbers = sorted( numbers )

        return numbers[ - min( len( numbers ), nViewers ) : ] if nViewers >= 0 else numbers


    def Train( self, input_train_focus_token, nTargetModels = -1 ) :

        """
        State: It's not clear that at least one task has a train focus.
        ToDo: If self has_focus, then really train. If not, then let a focus task get a chance to train.
        """

        if input_train_focus_token is True :
            """
            CASE.4 
            A train focus is relayed from the caller, which is previous in this search tree and that called this train.
            Take the relayedon token and and 'take the train steps' with the token. A train step knows what to do.
            """
            self.fitParams['has_train_focus_token'] = True

        else :
            """
            Case.5 
            The previosu task had not focus token. Follow the following rules.
            """
            train_focus_task = self.Find_Focus_Task_All_Around()

            if train_focus_task is None : 
                """
                CASE.1
                No token relayed from the caller. No task has a focus. The token may be lost by code bug, or intentionally. 
                Why not create and give a focus token to self. 'Take train steps' with the newly created token. A train step knows what to do.
                """ 
                self.fitParams['has_train_focus_token'] = True # Because there must exist a focus token, and there is no reason to yield it to other task.

            elif id( train_focus_task ) != id( self ) : 
                """
                CASE.2
                No token relayed from the caller. At least one task has a focus, and it's not self.
                'Take train steps' without token. A train step knows what to do.
                """
                self.fitParams['has_train_focus_token'] = False

            else : 
                """
                CASE.3
                No token relayed form the caller. At least one task, including self, has a focus. Self is the first finding in this search tree, having the chance first.
                Make it clear that self has a focus. 'Take train steps' with its focus token. A train step knows what to do.
                PENDING: Two or more focuses actually means a bug. The last finding of focus task in this search tree, is actually the corret focus task.
                """               
                self.fitParams['has_train_focus_token'] = True


        if self.fitParams['has_train_focus_token'] : self.fitParams['has_had_no_focus'] = False

        context = { 'is_first_loop' : True, 'nModels_to_find' : 0, 'nModels_found' : 0 }

        """
        State: At least one task has a focus, including self. Self either has a focus or not.
        TODO : Play train steps as if self has a train focus. A train stop itself will take all the responsibiliy.
        """
        """
        TODO of a train step: This TODO is deduced from the above algorith.
        If self enters the step with a train focus, then self takes a 'real train step', and call 'train' on subward tasks to 'train' themselves with the token.
        Make sure the token is not lost during the subward callings, and return the token to the caller.

        If NOT, then there must be at least one subward task having a token. 
        Self skips the 'real train step', and tries to call 'train' on subward tasks without a token, until a token task finally gets a 'train' call, 'train's itsef, and returns its token.
        Go on to call 'train' on remaining subward tasks to 'train' them with the newly found token, and return the token to the caller. There MUST be a token to return to the caller.
        """

        trainState = self.trainState

        nExistingModels = len( self.modelBase.GetModelNumbers() )

        while trainState['valid'] is True and nExistingModels < nTargetModels :

            focus_token_before = self.fitParams['has_train_focus_token']

            context = self.Run_TrainState_Subward( trainState, context )
            
            focus_token_after = self.fitParams['has_train_focus_token']
            assert focus_token_before <= focus_token_after
           
            trainState = self.Get_To_Next_TrainState( trainState, context ) # task.trainState is updated.

            nExistingModels = len( self.modelBase.GetModelNumbers() )

            # ======== Special. Exit and re-enter the program so tensorflow's memory leak will be recovered from.
            import gc
            gc.collect()
            del gc.garbage[:]


        has_train_focus_token = self.fitParams['has_train_focus_token']
        self.fitParams['has_train_focus_token'] = False # When a task return train 

        return has_train_focus_token


    def Run_TrainState_Subward( self, trainState, context = None ) : # Make sure 'trainState' == self.trainState. Do NOT assign or Restore_TrainState inside the function.

        """
        TODO of a train step: This TODO is deduced from the caller's algorith.
        If self enters the step with a train focus, then self takes a 'real train step', and call 'train' on subward tasks to 'train' themselves with the token.
        Make sure the token is not lost during the subward callings, and return the token to the caller.

        If NOT, then there must be at least one subward task having a token. 
        Self skips the 'real train step', and tries to call 'train' on subward tasks without a token, until a token task finally gets a 'train' call, 'train's itsef, and returns its token.
        Go on to call 'train' on remaining subward tasks to 'train' them with the newly found token, and return the token to the caller. There MUST be a token to return to the caller.
        """

        tf.keras.backend.clear_session() #----------------------- Clear session.

        if context is None : context = { 'is_first_loop' : True, 'nModels_to_find' : 1, 'nModels_found' : 0 }

        is_first_loop = context['is_first_loop']
        nModels_to_find = context['nModels_to_find']
        nModels_found = context['nModels_found']

        save_number = -1 ; history = None; stakes = {}; profits = {}

        modelParams = self.Translate_Param( trainState['.pick'] )
        fitParams = trainState['.fitParams']
        trainResources = self.trainResources # PENDING : Get it directly from trainState, to ensure consistency.
        dataUse = trainState['.dataUse']

        if self.fitParams['has_train_focus_token'] : # Self engered with a token. Take real train steps : Set_Train_Params, Train.

            print( "\n====================== {}".format( self.Get_Round_String() ) )
            print( "nbestloss = {:.7f}".format( self.fitParams['bestLoss'] ) )
            # print( "Params - {}".format( self.modelBase.Report_Model( trainState ) ) )

            totalLoss_min = self.fitParams['bestLoss']

            self.dataset.Initialize() #---------------- Find proper place to do this.

            stakes, profits = self.modelBase.Train( modelParams, fitParams, trainResources, dataUse ) # end-to-end, from Prepare_Data to Fit.
            #self.modelBase.Invalidate_Output_Subward( segment = None )

            found_a_best_model, save_number, good_queries = self.modelBase.Save_If_Best_Model( round, trainState, trainResources )
            if found_a_best_model : 
                nModels_found += 1
 
        else : # Self entered withoud a token. Nothing to do, regarding the core algorithm.

            found_a_best_model = self.modelBase.Load_Best_Model( round = -1, trainResources = self.trainResources )
            #modelBase.Compile_Model( round, compilePoint, trainResources, dataUse )

        #-------------------------------------------------------

        if self.fitParams['has_train_focus_token'] :
            save = found_a_best_model or super_config['.Test']
            #if super_config['.Test'] : save_number = round
            title = '======== Detail ========'
            subtitle = self.modelBase.Report_Model( trainState, stakes = stakes, profits = profits )
            self.modelBase.Report_Visual( self.dataset.standard_segments, self.queryDic, good_queries = good_queries, title = title, subtitle = subtitle, save = save, save_number = save_number )

            #--------------------------------------------------
            save_folder = super_config['Summary']
            filepath = os.path.join( save_folder, 'Model_Stats_' + self.taskName + ".xlsx" )
            backup = os.path.join( save_folder, 'Backup Model_Stats_' + self.taskName + ".xlsx" )
            self.config['strings'].update( self.modelBase.Excelize_Model_Data( filepath, backup, self.trainState, self.config['strings'] ) )

        #-------------------------------------------------------

        if self.subTasks is not None and len( self.subTasks ) > 0 :

            token_to_relay_to_subs = self.fitParams['has_train_focus_token']

            for subTask in self.subTasks :
                if subTask.is_train_chain :
                    self.fitParams['has_train_focus_token'] = False # Whether token_to_relay_to_subs is True or False.
                    token_returned_from_sub = subTask.Train( token_to_relay_to_subs ) # A recursive call.
                    assert token_to_relay_to_subs <= token_returned_from_sub # ============================== So, True->False development is impossible.
                    self.fitParams['has_train_focus_token'] = token_returned_from_sub # Keep the True/False token returned from subward tasks.
                    token_to_relay_to_subs = self.fitParams['has_train_focus_token'] # Keep relaying on the True/False taoken.

        #-------------------------------------------------------

        if self.fitParams['has_train_focus_token'] : fitParams['roundCounter'] += 1

        del history, stakes, profits

        return context


    #====================== Connect to history ====================

    # Although self.config is defined in descendants, I put it here to avoid replicated copies in descendannts.
    def GetInformed( self, event ) :

        if event ==  'Get_To_Next_TrainState' :        
            self.history.Save_Dictionaries( path_from_caller = None, dictionaries = { 'strings' : self.config['strings'] }, overwrite = True )

        elif event == 'Restore_TrainState' :
            self.config['strings'].update( self.history.Load_Dictionaries( path_from_caller = None, dict_names = [ 'strings' ] )[ 'strings' ] )

        elif event == 'Initialize_TrainState' :
            self.config['strings'].update( self.history.Load_Dictionaries( path_from_caller = None, dict_names =[ 'strings' ] )[ 'strings' ] )

        return
