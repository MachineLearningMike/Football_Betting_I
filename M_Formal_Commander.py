
from Task_Happs import *
from Task_Model import *
from M_Model_Composite import *
from M_Config import super_q_group

# This class comes from the formal specifications "M_Formarl_Specifications.txt", 
# which captures the statistical, overall logic of thw whole pipeline.
# It intentionally excludes the details of lower-level modules.
# The mid details are captures by super_config, while the lower details by the config of task object.
# If you include something more, it's a sign that you're going beyond the designed realm of this class.

class Formal_Commander() :

    def __init__( self, nViewers, mode ) :
        
        self.nViewers = nViewers
        self.mode = mode
        self.final_viewer = None

        # Stage
        self.phase = None
        self.task = None
        self.modelBase = None

        return


    def Now( self ) :

        queries = super_q_group.Queries()
        train = 'train'
        valid = 'valid'
        test = 'test'
        nViewers = self.nViewers
        predict = 'predict'

        self.Go( queries, train, valid, test, nViewers, predict )


    def Go( self, queries, train, valid, test, nViewers, predict_games) :

        if self.mode == 1 :
            viewers, final_viewer, models_id, performance = self.Build_Process( queries, train, valid, test, nViewers )
        elif self.mode == 2 :
            viewers, final_viewer, models_id, performance = self.Build_Process_v2( queries, train, valid, test, nViewers )

        stakes = self.Produce_Stakes( queries, predict_games, viewers, final_viewer, models, train, valid )

        return stakes


    def Set_Stage( self, phase, base_outcomes = None ) :

        self.phase = phase 
        
        if phase == 1 :
            self.task = Task_Happs( superTasks = None, is_train_chain = True, checkMargin = super_config['Margin'] )
            self.modelBase = self.task.modelBase

        elif phase == 2 :
            self.task = Task_Model( base_outcomes, superTasks = None, is_train_chain = True, checkMargin = super_config['Margin'] )
            self.modelBase = self.task.modelBase

        return

    def Build_Process( self, queries, train, valid, test, nViewers ) :

        viewers = self.Create_Viewers( nViewers )

        examples_plus = valid

        models = self.Create_Models( queries, examples_plus, viewers, train, valid ) # views, summaries, summaries_plus, summaries->labels model.

        games = test
        eligible = self.Calculate_Eligible( queries, games, viewers, models, train, valid )

        games_plus = test
        SPB = self.Get_SPB(queries, games_plus, final_viewer, eligible )

        performance = self.fun_performance( SPB )

        return performance


    def Build_Process_v2( self, queries, train, valid, test, nViewers ) :

        examples_plus = valid

        model, outputs = self.Create_Models_v2( queries, examples_plus, train, valid, test, nViewers )

        games = test
        eligible = self.Calculate_Eligible( queries, games, viewers, models, train, valid )

        games_plus = test
        SPB = self.Get_SPB(queries, games_plus, final_viewer, eligible )

        performance = self.fun_performance( SPB )

        return performance



    def Create_Viewers( self, nViewers ) :

        trainState, is_first_session = self.task.LoadTrainState( update_config = False ) # 'update_config' will update/overwrite persistent trainState with pre-set trainState.
        if trainState is not None and is_first_session is False : trainState = self.task.Restore_TrainState( trainState )
        else : trainState = self.task.Initialze_TrainState()

        self.task.Prepare_Session( trainState, purpose = 'train' ) #===================== Overall initializaion for Train(). Don't put it in Train().

        self.task.Train( input_train_focus_token = True, nTargetModels = nViewers ) # True means we have a train focus token and relay it to 'task'. If False, they will create one.

        viewers = self.task.GetModelNumbers( nViewers )

        return viewers


    def Summarize_Viewers( self, viewers ) :

        #composite_model = Model_Composite( viewers )

        #return composite_model
        return None


    def Create_Models(self, queries, examples_plus, viewers, train, valid ) : 

        games = examples_plus
        views = self.modelBase.Create_Views( queries, games, viewers, train, valid )
        summaries = self.modelBase.Create_Summaries( queries, games, views )

        games_plus = examples_plus
        labels = self.modelBase.Get_Labels( games_plus )
        summaries_plus = self.modelBase.Label_Summaries( summaries, labels )

        models_id = []
        models_id = self.modelBase.Create_Summaries_Model_Wholesale( summaries_plus, queries )

        return models_id


    def Create_Models_v2(self, queries, examples_plus, train, valid, test, nViewers ) :

        self.Set_Stage( phase = 1 ) #---------------------------- Phase 1 set -------------------

        viewers = self.Create_Viewers( nViewers = nViewers )

        best_viewers, best_outcomes = self.modelBase.Select_Best_Viewers( queries, viewers, train, valid, test )

        self.Set_Stage( phase = 2, base_outcomes = best_outcomes ) #---------------------------- Phase 2 set -------------------

        viewers = self.Create_Viewers( nViewers = nViewers )

        best_viewers, best_outcomes = self.modelBase.Select_Best_Viewers( queries, viewers, train, valid, test )

        return best_viewer, best_outcomes 


    def Calculate_Eligible( self, queries, games, viewers, models_id, train, valid ) :

        views = self.modelBase.Create_Views( queries, games, viewers, train, valid )

        summaries = self.modelBase.Create_Summaries( queries, games, views )

        preds = self.modelBase.Apply_Models_Wholesale( models_id, queries, games, summaries )

        eligible = fun_eligible( preds )

        return eligible


    def Produce_Stakes( self, queries, predict_games, viewers, final_viewer, models, train, valid ) :

        eligible = Calculate_Eligible( queries, predict_games, viewers, models, train, valid )

        stakes = Stake.Viewer( queries, predict_predict, final_viewer, eligible )

        return stakes


commander = Formal_Commander( nViewers = 34, mode = 2 )
commander.Now()