import tensorflow as tf
import numpy as np

from M_Config import *

from M_Metrics import *
from M_Losses import *
from M_Football import *

from Task_Base import *
from M_Model_Contest import *


class Task_Contest( Task_Base ) :

    config = {}

    config['.M.Score'] = super_config['.M.Score']

    #========================================================================================================
    config['Dim.Outs'] = super_q_group.Dimensions_GroupWise()
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ActOuts must be chosen in relation to loss function.
    config['ActOuts'] = ['sigmoid'] * super_q_group.nGroups()  # 'softmax', 'sigmoid', 'relu', 'tanh', 'linear' (logits). Let the layer before softmax produce activations freely.

    config['.loss'] = 'minus.profit.on.happened.prob' #'distance.to.1.on.happened.prob' # 'minus.profit.on.happened.prob', 'distance.to.1.on.happened.prob'

    config['.Attach'] = False #------------------------------------ 
    config['.Out.Wise'] = True
    config['.D.X.Soc'] = 3 if super_config['.Test'] else 15
    config['.Att.Len'] = 3 if super_config['.Test'] else 15 # split in between branch base and its attachments.
    config['Dim.Xs'] = [ config['.D.X.Soc'] ] * config['.Att.Len']
    config['Att.Act'] = [ 'relu', 'sigmoid', 'tanh', 'swish', 'softplus', 'selu' ]
    config['.Att.Ver'] = 4 # { '0' : debuggin. ('Unit core weights', 'Fitting on identical data'). '1' : (0,0), '2': (0,1), '3': (1, 0), '4' : (1, 1) }

    config['BaseEpo'] = 5 if super_config['.Test'] else 555
    config['BrMainEpo'] = 5 if super_config['.Test'] else 555
    config['BrFinalEpo'] = 5 if super_config['.Test'] else 555
    config['AttachEpo'] = 5 if super_config['.Test'] else 333
    config['AttIdenizeEpo'] = 5 if super_config['.Test'] else 55

    config['Uniform_Att'] = True
    if config['Uniform_Att'] is True :
        for factor in config['Dim.Xs'] :
            if factor != config['.D.X.Soc'] :
                raise Exception( 'Uneven factor found in uniform attachments.')

    config['In.Leag'] = True
    config['In.Home'] = True
    config['In.Away'] = True
    config['In.Situ'] = True
    config['In.Odds'] = True
    config['In.RowId'] = False
    config['In.X'] = True
    config['.In.lCoun'] = True
    config['.In.hCoun'] = True
    config['.In.aCoun'] = True

    config['.BSize'] = super_config['.BSize']

    config['strings'] = { 'None' : 0, } 

    #=========================================================================================================

    config['.BSize'] = super_config['.BSize']


    config['min_oh_1'] = 0.2
    config['thres_bWorth'] = 0.5
    config['full_oh_1'] = 2.5
    config['full_bWorth'] = 0.95
    config['max_stake'] = 2.0


    def Report_Config( self, config = None ) :
        if config is None : config = self.config
        return "Config - {}".format( config )


    def __init__( self, superTasks = None, is_train_chain = False, checkMargin = 100 ) :

        taskName = 'Task_Contest'

        super().__init__( taskName, superTasks, is_train_chain = is_train_chain, checkMargin = checkMargin )
       
        self.modelBase = Model_Contest( self )

        # This block should not precede the above block.
        if superTasks != None and len(superTasks) > 0 :
            for superTask in superTasks :
                superTask.Register_SubTask( self )


    def Initialize_Fit_Context(self) :

        super().Initialize_Fit_Context()

        #================== Iterator properties ====================

        self.params_unarranged = { \
            '.dim_Xs': { 'v_list': self.config['Dim.Xs'],  }, \
            '.d.coun' : { 'v_list': [ super_config['.d.coun'] ], }, \
            '.d.leag' : { 'v_list': [ super_config['.d.leag'] ], }, \
            '.d.team' : { 'v_list': [ super_config['.d.team'] ], }, \
            '.dropIn': { 'v_list': [ 0.1, 0.20, 0.25 ], }, \
            '.action' : { 'v_list' : [ 'relu', 'sigmoid', 'tanh' ], }, \
            '.regur' : { 'v_list' : [ [ [ 'None' ], [ 0.0001 ] ],  ], 'nested': True, 'translator' : '.regur' }, \
            '.optmer' : { 'v_list' : [ [ [ 'Adamax' ], [ 5e-3, 20e-3, 50e-3 ] ] ], 'nested': True, 'translator' : '.optmer'}, \
        }

        #================= Fit Params ================================

        self.fitParams['check_monitor'] = 'val_loss' # Try 'val_loss'. Switch to '.loss' if validation_data is None.
        self.fitParams['stop_monitor'] = 'val_loss' if super_config['.Test'] else 'val_loss'
        self.fitParams['.epochs'] = 1 if super_config['.Test'] else self.config['BrMainEpo']
        self.fitParams['bestLoss'] = float("inf")

        self.fitParams['.mSLD'] = 0.0005 # Standard Loss Drop. loss drop / abs( loss )
        self.fitParams['EarlyStop'] = 'Yes' if super_config['.Test'] else 'Yes'
        
        self.fitParams['depth'] = 3
        self.fitParams['.generosity'] = 1.0 if super_config['.Test'] else 0.2 #================================== generosity

        self.fitParams['modelsPerQ'] = self.modelsPerQuery
        self.fitParams['roundCounter'] = 0


        odds = 2.0; happened_true = 1.0; happened_pred = 1.0
        standard_profit_per_bet = Get_Pofit_On_Hap( odds, happened_true, Get_Stake_On_Hap( odds, happened_pred ) ) # 1.0 with current Stake_Strategy.
        standard_profit_per_bet = standard_profit_per_bet.numpy()
        standard_profit_per_game = standard_profit_per_bet * super_q_group.nGroups() # 1.0 * 5 = 5.0

        self.fitParams['train'+'slope'] = {}
        params = self.fitParams['train'+'slope']
        params['examination'] = 'collective_slope'
        params['slope_start_pct'] = 0.0
        params['slope_end_pct'] = 90.0
        params['slope_limit_dn'] = standard_profit_per_game * ( 1.0 - 5.0 )
        params['slope_limit_up'] = standard_profit_per_game * ( 1.0 - 5.5 )
        
        #================= Resources ================================

        self.trainResources.update( self.Create_TrainResources( self.config ) )

        return [ self.queryDic, self.queries, self.dataUse, self.params_unarranged, self.pick_pointer, self.fitParams ]


    def Create_TrainResources( self, config ) :

        resources = {}
        resources['.loss'] = self.Get_Loss_Class( config['.loss'] )
        resources['metrices'] = tf.keras.metrics.Accuracy

        return resources


    def GetTrainResources(self) :
        return self.trainResources


    def Get_Loss_Class( self, loss_string ) :

        loss_class = None
    
        if loss_string == 'mse.on.goals' :
            loss_class = Loss_MSE_On_Goals
        elif loss_string == 'mse.on.happened.prob' :
            loss_class = Loss_MSE_On_Happened_Prob
        elif loss_string == 'sparce.cat.x.entropy.on.glogits' :
            loss_class = Loss_Sparce_Cat_X_Entropy_On_GLogits
        elif loss_string == 'sparce.cat.x.entropy.on.gsoft' :
            loss_class = Loss_Sparce_Cat_X_Entropy_On_GSoft
        elif loss_string == 'max.profit.on.happened.prob' :
            loss_class = Loss_Max_Profit_On_Happened_Prob
        elif loss_string == 'mse.on.bet.worth.v1' :
            loss_class = Loss_MSE_On_Bet_Worth_V1
        elif loss_string == 'mse.on.bet.worth.v2' :
            loss_class = Loss_MSE_On_Bet_Worth_V2
        elif loss_string == 'max.profit.on.free.stake' :
            loss_class = Loss_Profit_On_Free_Stake

        elif loss_string == 'distance.to.1.on.happened.prob' :
            loss_class = Loss_Distance_To_1_On_Happened_Prob
        elif loss_string == 'minus.profit.on.happened.prob' :
            loss_class = Loss_Minus_Profit_On_Happened_Prob

        return loss_class



"""
t = Task_Goals( )
it = iter( t )
for param_arranged in it :
    param_bin = t.Translate_Param( param_arranged )
"""