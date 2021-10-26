
import tensorflow as tf
from M_Config import *

from Task_Base import *
from Task_GLogits import *
from Task_Happs import *
from Task_Goals import *
from Task_Contest import *

from M_Model_Slots import *


def Run_Session( task, nModels_to_find = -1, update_task_param = False, nRounds = -1 ) :

    if super_config['session'] == 'train' :

        trainState, is_first_session = task.LoadTrainState( update_config = False ) # 'update_config' will update/overwrite persistent trainState with pre-set trainState.
        if trainState is not None and is_first_session is False : trainState = task.Restore_TrainState( trainState )
        else : trainState = task.Initialze_TrainState()

        task.Prepare_Session( trainState, purpose = 'train' ) #===================== Overall initializaion for Train(). Don't put it in Train().

        task.Train( input_train_focus_token = True, nTargetModels = nRounds ) # True means we have a train focus token and relay it to 'task'. If False, they will create one.
    
    elif super_config['session'] == 'test' :

        trainState = task.Initialze_TrainState() # Initialize deliberately to hard-coded parameters and to stay independent of saved trainstate.

        task.Prepare_Session( trainState, purpose = 'test' ) #===================== Overall initializaion for Train(). Don't put it in Train().

        task.Create_Views( progressive = True )

    elif super_config['session'] == 'average' :

        trainState = task.Initialze_TrainState() # Initialize deliberately to hard-coded parameters and to stay independent of saved trainstate.

        task.Prepare_Session( trainState, purpose = 'average' ) #===================== Overall initializaion for Train(). Don't put it in Train().

        task.Get_Average_Stakes( progressive = True )

    elif super_config['session'] == 'analyze' :

        trainState = task.Initialze_TrainState() # Initialize deliberately to hard-coded parameters and to stay independent of saved trainstate.

        task.Prepare_Session( trainState, purpose = 'analyze' ) #===================== Overall initializaion for Train(). Don't put it in Train().

        task.Analyze_Stakes( progressive = True )

        

    return


def Disable_GPU() :
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


if super_config['GPU'] is False : Disable_GPU()

#==========================================================================

if super_config['session'] == 'train' :
    rounds_per_day = 35
elif super_config['session'] == 'test' :
    rounds_per_day = 30 # 30 days.
elif super_config['session'] in [ 'average', 'analyze' ] :
    rounds_per_day = 1



while True :

    if not super_config['Contest'] :
        #tPrimary = Task_GLogits( superTasks = None, is_train_chain = True, checkMargin = super_config['Margin'] )
        tPrimary = Task_Happs( superTasks = None, is_train_chain = True, checkMargin = super_config['Margin'] ) # dataUse takes its default value. No dataset is built.
        #tPrimary = Task_Goals( superTasks = None, is_train_chain = True, checkMargin = super_config['Margin'] )
    else :
        tPrimary = Task_Contest( superTasks = None, is_train_chain = True, checkMargin = super_config['Margin'] ) # dataUse takes its default value. No dataset is built.

    Run_Session(  tPrimary, nModels_to_find = -1, update_task_param = False, nRounds = rounds_per_day ) # Load/Restore or Initialize trainState, Rebuild dataset, before Train().

    tPrimary.Initialize_TrainState_With_Params( d_day_delta = 1 ) # Initialize trainState with super_config['.M.Day'] + delta, restore and save it.

    del tPrimary

    if super_config['session'] in [ 'average', 'analyze' ] : break


