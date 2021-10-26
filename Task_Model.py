import tensorflow as tf
import numpy as np

from M_Config import *

from M_Metrics import *
from M_Losses import *
from M_Football import *

from Task_Base import *
from M_Model_Model import *
from M_Model_Slots import *
from M_DataService_Model import *


class Task_Model( Task_Base ) :

    def Report_Config( self, config = None ) :
        if config is None : config = self.config
        text = ''
        for key, value in config.items() :
            if key.startswith('.') : text += ( key[1:] + ': ' + str( value ) + ', ' )
        for key, value in self.fitParams.items() :
            if key.startswith('.') : text += ( key[1:] + ': ' + str( value ) + ', ' )

        return text


    def __init__( self, base_outcomes, superTasks = None, is_train_chain = False, checkMargin = 100 ) :

        self.base_outputs, self.base_stakes, self.base_profits = {}, {}, {}
        for segment in super_config['segs.outcomes'] :
            self.base_outputs[segment] = tf.concat( [ base_outcomes[ queryPlace ][ 0 ][ segment ] for queryPlace in range( len( base_outcomes ) ) ], axis = -1 )
            self.base_stakes[segment] = tf.concat( [ base_outcomes[ queryPlace ][ 1 ][ segment ] for queryPlace in range( len( base_outcomes ) ) ], axis = -1 )
            self.base_profits[segment] = tf.concat( [ base_outcomes[ queryPlace ][ 2 ][ segment ] for queryPlace in range( len( base_outcomes ) ) ], axis = -1 )


        self.config = {}

        self.config['.M.Score'] = super_config['.M.Score']

        #========================================================================================================
        self.config['Dim.Outs'] = super_q_group.Dimensions_GroupWise()

        #!!!!!!!!!!!!!!!! Keep it 'sigmoid' for loss version 2 and over !!!!!!!!!!!!!! minus.profit.
        self.config['ActOuts'] = ['sigmoid'] * super_q_group.nGroups()  # 'softmax', 'sigmoid', 'relu', 'tanh', 'linear' (logits). Let the layer before softmax produce activations freely.

        self.config['.loss'] = 'minus.profit.on.happened.prob' # 'minus.profit.on.happened.prob'

        self.config['.Attach'] = True if super_config['.Test'] else False #---------------------------------------------------------------
       
        self.config['loss_unified'] = 'loss_valid' # 'average' # 'loss_train'

        self.config['.Out.Wise'] = True #--------------------------------------------------------------

        self.config['Branch'] = 'new' # 'new' 'past', 'better', 'new' #-----------------------------------

        self.config['.D.X.Soc'] = 5 # 10 if super_config['.Test'] else 10 # x74 Excessive inclusion, along with excessive config['D.X.Soc'] would lead to momory overflow.
        self.config['DimX.core'] = [ self.config['.D.X.Soc'] ] * ( 12 if super_config['.Test'] else 15 ) # Uneven, possibly.
        self.actions = [ \
            [ 'tanh', 'tanh', 'relu', 'tanh', 'tanh', 'tanh', 'relu', 'tanh', 'tanh', 'relu', 'tanh', 'tanh', 'tanh', 'tanh', 'relu' ], \
            [ 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'relu', 'tanh', 'tanh', 'tanh', 'relu', 'tanh' ], \
            [ 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid' ], \
            [ 'sigmoid', 'relu', 'sigmoid', 'relu', 'sigmoid', 'relu', 'sigmoid', 'relu', 'sigmoid', 'relu', 'sigmoid', 'relu', 'sigmoid', 'relu', 'sigmoid' ], \
            [ 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu' ], \
        ]
        for idx in range( len( self.actions ) ):
            assert len( self.actions[ idx ] ) >= len( self.config['DimX.core'] )

        # self.config['DimX.attach'] = [ [ self.config['.D.X.Soc'] ] * ( att_id + 1 ) for att_id in range( 5 if super_config['.Test'] else 10 )  ]

        nAtt_attempts = 5 if super_config['.Test'] else 10
        nAtt_layers = 1 # 2 or more didn't give better results, though lengths of attachemnts can be any longer.
        self.config['DimX.attach'] = [ [ self.config['.D.X.Soc'] ] * ( nAtt_layers ) for att_id in range( nAtt_attempts )  ]
        self.config['Att.Act'] = [ 'relu', 'sigmoid', 'tanh', 'swish', 'softplus', 'selu' ]
        self.config['.Att.Ver'] = 4 # { '0' : debuggin. ('Unit core weights', 'Fitting on identical data'). '1' : (0,0), '2': (0,1), '3': (1, 0), '4' : (1, 1) }

        self.config['UseBN'] = True
        self.config['pick_trials'] = 10 # Larger number of picking rounds might lead to little improvement of loss, as well as time waste.
        self.config['BrMainEpo'] = 100 if super_config['.Test'] else 111 # Large to grow embedding. EarlyStopped.
        self.config['BrFinal'] = True
        self.config['BrFinalEpo'] = 50 if super_config['.Test'] else 33 # Large to grow embedding.EarlyStopped.

        self.config['AttachEpo'] = 100 if super_config['.Test'] else 55 # Small for many attempts. EarlyStopped.
        self.config['AttIdenizeEpo'] = 100 if super_config['.Test'] else 33 # This loop is fast, each improving loss. EarlyStopped.

        for factor_list in self.config['DimX.attach'] :
            for factor in factor_list :
                if factor != self.config['.D.X.Soc'] :
                    raise Exception( 'Uneven factor found in uniform attachments.')

        #------------------------------------------------- Excessive inclusion, along with excessive confit['D.X.Soc'] would lead to Overflow Of Memory

        param_str = ''

        # Protocol : Do NOT put '_' in the version string.
        self.config['.branch.ver'] = super_config['job_prefix'] + 'M' + str( super_config['.M.Day'] ).zfill(4) + '' + param_str
        self.config['.model.ver'] = super_config['job_prefix'] + 'M' + str( super_config['.M.Day'] ).zfill(4) + '' + param_str

        self.config['.BSize'] = super_config['.BSize']

        self.config['strings'] = { 'None' : 0, } 

        self.config['.BSize'] = super_config['.BSize']

        #=========================================================================================================

        taskName = 'Task_Model'
        dataservice = DataService_Model()

        super().__init__( taskName, dataservice, superTasks, is_train_chain = is_train_chain, checkMargin = checkMargin )
       
        self.modelBase = Model_Model( self )

        # This block should not precede the above block.
        if superTasks != None and len(superTasks) > 0 :
            for superTask in superTasks :
                superTask.Register_SubTask( self )


    def Initialize_Fit_Context(self) :

        super().Initialize_Fit_Context()

        #================== Iterator properties ====================

        self.params_unarranged = { \
            '.action' : { 'v_list' : self.actions, }, \
            '.regur' : { 'v_list' : [ [ [ 'None' ], [ 0.0001 ] ],  ], 'nested': True, 'translator' : '.regur' }, \
            '.optmer' : { 'v_list' : [ ( 'Adadelta', 0.001 ), ( 'Adadelta', 0.01 ), ( 'Adadelta', 0.1 )  ] }, \
        } # NOTE!!!: learning rates are now temporarilly ignored, for test purpose.

        #================= Fit Params ================================

        self.fitParams['check_monitor'] = 'val_loss' # Try 'val_loss'. Switch to '.loss' if validation_data is None.
        self.fitParams['stop_monitor'] = 'val_loss' if super_config['.Test'] else 'val_loss'
        self.fitParams['.epochs'] = 1 if super_config['.Test'] else self.config['BrMainEpo']
        self.fitParams['bestLoss'] = float("inf")

        self.fitParams['.mSLD'] = 0.0005 # Standard Loss Drop. loss drop / abs( loss ), or loss drop per loss.
        self.fitParams['EarlyStop'] = 'No' if super_config['.Test'] else 'No' # Do not use it until validated again.
        
        self.fitParams['depth'] = 3
        self.fitParams['.generosity'] = 1.0 if super_config['.Test'] else 0.35 #================================== generosity

        self.fitParams['modelsPerQ'] = self.modelsPerQuery
        self.fitParams['roundCounter'] = 0

        odds = 2.0; happened_true = 1.0; happened_pred = 1.0
        standard_profit_per_bet = BettingMath.Get_Pofit_On_Hap( odds, happened_true, BettingMath.Get_Stake_On_Hap( odds, happened_pred ) ) # 1.0 with current Stake_Strategy.
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
                                                         