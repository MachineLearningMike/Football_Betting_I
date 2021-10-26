import numpy as np
import pandas as pd
import tensorflow as tf
import math
import os
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

from M_Database import *

from M_BettingMath import *


class Gauge() :

    def __init__( self, margins, queries, profit_weight, safety_weight, **kwargs ) :
        super().__init__( **kwargs )

        self.margins = margins

        self.queries = queries

        assert profit_weight > 0
        self.profit_weight = profit_weight

        assert safety_weight > 0
        self.safety_weight = safety_weight

        self.gauge_path_base = os.path.join( super_config['Summary'], super_config['job_prefix'] + "_Accounts" + "_M" + str( super_config['.M.Day'] ).zfill(4) )

        self.cache_stats = None
        self.cache_stats_model = None

        self.schemes_dict = self.Populate_Schemes_Dict()

        #=========================

    def Get_F1_Precision_Recall_For_1_Of_0_And_1( self, true, pred, average = None ) :
        """
        How this function works:

          0.  Unique classes of label that appear in true and pred seem to be sorted and represented by their sort indices : 0, 1, ...
            Below, we assume there are only two unique labels - 0 and 1, giving nLabelClasses = 2, which covers all two-label trues and preds.

        1.  true.shape == pred.shape == ( d0, d1 ). d1 > 1
            - All f1, precision, and recall are of shape ( d1, )
            - They focus on 1s in true and pred.
                precision[i], for instance, is density( { j : true[j, i] == pred[j, i] == 1 } ) / density( { k : pred[k,i] == 1 } )

        2. true.shape == pred.shape == ( d0, )
            - All f1, precision, and recall are of shape ( nLabelClasses = 2 )
            - They focus on 0s first, and then 1s, of true and pred.
                precision[0] is density( { j : true[j] == pred[j] == 0 } ) / density( { k : pred[k] == 0 } )
                precision[1] is density( { j : true[j] == pred[j] == 1 } ) / density( { k : pred[k] == 1 } )
        """
        
        assert true.shape == pred.shape

        f1 = f1_score( true, pred, average = average )
        scores = precision_recall_fscore_support( true, pred, average = average )
        precision = scores[0]
        recall = scores[1]

        if len(true.shape) > 1 and true.shape[1] == 1 :
            if len( f1.shape ) > 0 : f1 = f1[ f1.shape[0] - 1 : f1.shape[0] ] # to get in array format.
            else : pass # Keep it as is. average == 'micro'
            if len( scores[0].shape ) > 0 : precision = scores[0][ scores[0].shape[0] - 1 : scores[0].shape[0] ] # to get in array format.
            else : pass
            if len( scores[1].shape ) > 0 : recall = scores[1][ scores[1].shape[0] - 1: scores[1].shape[0] ] # to get in array format.
            else : pass

        return f1, precision, recall


    def Is_Good_Model( self, modelBase, fitParams, trainResources ):
        
        slope_param = fitParams['train'+'slope']
        self.Examine_Slope( 'train', modelBase, slope_param )
        slope_min_train = slope_param['slope_min_found']
        slope_max_train = slope_param['slope_max_found']
        is_good_profit_train = slope_param['is_good_profit']
        good_queries_train = slope_param['good_queries']

        
        is_good = is_good_profit_train
        #is_good = tf.logical_and( is_good_profit_train, is_good_profit_test )
        good_queries = tf.squeeze( tf.where( is_good ), axis = -1 )

        return is_good, good_queries


    def Examine_Slope( self, segment, modelBase, slope_params ) :

        examination = slope_params['examination']
        slope_limit_up = slope_params['slope_limit_up']
        slope_limit_dn = slope_params['slope_limit_dn']
        start_pct = slope_params['slope_start_pct']
        end_pct = slope_params['slope_end_pct']

        def Slope_Of_Standard_Profit_Accum( profit_cum, start_pct, end_pct ) :
            top = profit_cum[-1]
            base_length = profit_cum.shape[0]
            assert base_length >= 2
            start = min( max( math.floor( 1.0 * base_length * start_pct / 100.0 ), 0 ), base_length - 2 )
            end = min( max( math.floor( 1.0 * base_length * end_pct / 100.0 ), start + 1 ), base_length - 1 ) # Exclude the last (top) point.
            indices_on_interval = tf.stack( [ tf.Variable( range( end - start ) ) ] * profit_cum.shape[-1], axis = -1 )
            std_slope_on_interval = tf.cast( ( top - profit_cum )[ start : end ], tf.float32 ) / tf.cast( base_length - start - 1 - indices_on_interval, tf.float32 )
            min_slope = tf.reduce_min( std_slope_on_interval, axis = 0 )
            max_slope = tf.reduce_max( std_slope_on_interval, axis = 0 )
            return min_slope, max_slope
            
        odds = modelBase.task.dataset.Data( 'odds', segment )
        happened_true = modelBase.task.dataset.Data( 'happened', segment )
        #output = modelBase.Get_Output_With_No_Subward_Sync( segment )
        #stake, profit = modelBase.Get_Outcomes( output, odds, happened_true = happened_true )
        outputs, stakes, profits = modelBase.Get_Outcomes_On_Current_Model( [ segment ] )
        output, stake, profit = outputs[ segment ], stakes[ segment ], profits[ segment ]

        if profit is not None :

            if examination == 'querywise_slope_of_standard_profit' :
                profit_cum, _, _, _ = self.Analyze_Profit( profit, segment ) # This line identifies this section.
                min_slope, max_slope = Slope_Of_Standard_Profit_Accum( profit_cum, start_pct, end_pct )

            elif examination == 'collective_slope' :
                profit_cum, _, _, _ = self.Analyze_Profit_Collectively( profit, segment ) # This line identifies this section.
                min_slope, max_slope = Slope_Of_Standard_Profit_Accum( profit_cum, start_pct, end_pct )

            else : raise Exception("Invalid 'examination'")

        else : raise Exception("Slope not found.")

        slope_params['slope_min_found'] = min_slope.numpy()
        slope_params['slope_max_found'] = max_slope.numpy()
        is_good_profit = tf.logical_and( tf.greater( min_slope, slope_limit_dn ), tf.greater( max_slope, slope_limit_up ) )
        slope_params['is_good_profit'] = is_good_profit.numpy()
        good_queries = tf.squeeze( tf.where( is_good_profit ), axis = -1 )
        slope_params['good_queries'] = good_queries.numpy()
        print( "Slope of Profit_Accum ------ {} [{}, {}] : ".format( segment, start_pct, end_pct ) +  \
            ', '.join('({:.4f})/{:.4f} - ({:.4f})/{:.4f}: {}'.format( min, slope_limit_dn, max, slope_limit_up, good ) for min, max, good in zip(min_slope, max_slope, is_good_profit ) ) )

        del segment, modelBase, slope_params

        return


    def Analyze_Profit( self, profit, segment ) : # profit = [ (None, 1) ] * len( queries ), or [ None, nQueries ]

        profit_cum = self.Get_Profit_Accum( profit ) # profit is changed. profit_cum.shape == (None, nQueries)
        total_profit = self.Get_Total_Profit( profit_cum ) # profit_cum is changed, if its shape was (None, nQueries). total_profit.shape = (nQueries,)
 
        profitability = self.Get_Profitability( profit_cum, segment ) # profit_cume.shape is not changed. profitability.shape == (nQueries,)
        safety = self.Get_Safety( profit_cum, segment ) # profit_cum is not changed. safety.shape == (nQueries,)
 
        #score = ( total_profit * self.profit_weight + safety * self.safety_weight ) / ( self.profit_weight + self.safety_weight + 1e-7 )
        score = profitability * safety
        score_inclusive = tf.where( safety <= 0.0, tf.where( profitability >= 0.0, -1e8, -1e9 ), score )   #================================= -1e8, -1e9

        del profit, segment

        return profit_cum, total_profit, safety, score_inclusive


    def Analyze_Profit_Collectively( self, profit, segment, good_queries = None ) : # profit = [ (None, 1) ] * len( queries )
       
        if good_queries is None : good_queries = self.queries

        if good_queries is not None and len( good_queries ) > 0 :
            profit = tf.concat( profit, axis = - 1 )

            profit = [ profit[ :, queryPlace ] for queryPlace in range( len( good_queries ) ) ]
            
            profit = tf.reduce_sum( tf.stack( profit, axis = -1 ), axis = -1 )
            profit = tf.expand_dims( profit, axis = -1 )
            profit = [ profit ]

            return self.Analyze_Profit( profit, segment )
        else :
            return None  # not None


    def Get_Profit_Accum( self, profit ) : # profit is a list of TensorShape[None, 1], or [None, nQueries]
        """
        profit = tf.concat( profit, axis = -1 ) # then profit.shape = (None, nQueries)
        a = tf.expand_dims( tf.ones_like( profit[:, 0] ), -1 )
        b = tf.transpose(a)
        lowerTriangle = tf.cast( tf.linalg.band_part(  a * b, -1, 0), tf.float32 ) # PENDING - momory overflow happens here.
        #profit_cum = tf.squeeze( tf.matmul( lowerTriangle, profit ) )
        profit_cum = tf.matmul( lowerTriangle, profit ) # then profit_cum.shape = (None, nQueries)
        """
        profit = tf.concat( profit, axis = 0 ) # then profit.shape = (None, nQueries)
        profit_cum = tf.cumsum( profit, axis = 0 )

        if profit_cum.shape[0] == 0 : # ugly patch? not elegant.
            profit_cum = tf.expand_dims( tf.Variable( tf.zeros( shape = ( profit_cum.shape[-1], ), dtype = tf.float32 ) ), axis = 0 )
        
        return profit_cum


    def Get_Total_Profit( self, profit_cum ) :
        # No more models following.
        profit_cum = tf.concat( profit_cum, axis = -1 )
        total_profit = tf.squeeze( profit_cum[-1, :] ) # then total_profit.shape = (nQueris,)

        return total_profit
      

    def Get_Profitability( self, profit_cum, segment ) :
        profitability = profit_cum[ -1 ] / self.margins[ segment ]

        return profitability


    def Get_Safety( self, profit_cum, segment ) :
        safety = ( self.margins[ segment ] + tf.reduce_min( profit_cum, axis = 0) ) / self.margins[ segment ]

        return safety


    def Get_Model_Statistics( self, modelBase ) :
        

        def ExcludeQueries( value_by_query, queries_to_exclude ):
            return [ value_by_query[n] for n in range( len( value_by_query ) ) if n not in queries_to_exclude ]


        def Get_Statistics( export_to_account, segment, fore_not_back, dummyQueries ) :

            #stake = modelBase.stakes[ segment ]
            #profit = modelBase.profits[ segment ]
            outputs, stakes, profits = modelBase.Get_Outcomes_On_Current_Model( [ segment ] )
            output = outputs[ segment ]
            stake = stakes[ segment ]
            profit = profits[ segment ]

            if export_to_account : 
                if fore_not_back is False : 
                    pre = '.' + segment[:2]; pre_hide = pre[1:]
                else : 
                    pre = '.' + segment[:2] + 'f' ; pre_hide = pre[1:]
            else : 
                if fore_not_back is False : 
                    pre = segment[:2] ; pre_hide = pre
                else : 
                    pre = segment[:2] + 'f' ; pre_hide = pre

            cum, total, safety, score = self.Analyze_Profit( profit, segment )

            happened_true_onehot = np.ndarray.astype( modelBase.task.dataset.Data( 'happened', segment ), np.int32 )
            happened_pred_onehot = tf.cast( tf.greater( stake, 0.0 ), tf.int32 )
            
            # modelOut : [ tensor shape of [ None, len(QGroup) ] for GRroup in Groups ]
            # steke, profit : tensor, shape of [ None, queries ]
            # happened_true_onehot, happened_pred_onehot : np.array, shape of ( None, queries ), all 1s and 0s.

            dict = {}; _list = []; indices = []

            #--------------------------- profit by query -------------------------------

            name = 'mean'
            value = [ value.item() for value in list( tf.reduce_mean( profit, axis = 0 ).numpy() ) ]
            value = ExcludeQueries( value, dummyQueries ); mean = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = 'std'
            value = [ value.item() for value in list( np.std( profit.numpy(), axis = 0 ) ) ]
            value = ExcludeQueries( value, dummyQueries ); std = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = 'msR'
            value = [ ( mean[n] / ( std[n] + 1e-9 ) ) for n in range( len( mean ) ) ]
            value = ExcludeQueries( value, dummyQueries ); msR = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )
            
            name = 'safe'
            value = [ value.item() for value in list( safety.numpy() ) ]
            value = ExcludeQueries( value, dummyQueries ); safe = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = 'MDrop'
            value = [ value.item() for value in list ( BettingMath.Get_MaxDrop( cum, axis = 0 ).numpy() ) ]
            value = ExcludeQueries( value, dummyQueries ); MDrop = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = 'msRD'
            value = [ msR[n] / ( MDrop[n] + 1e-7 ) for n in range( len( msR ) ) ]
            value = ExcludeQueries( value, dummyQueries ); msRD = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = 'msRDR'
            value = [ ( tf.nn.relu( msR[n] ) / ( tf.nn.relu( MDrop[n] ) + 1e-7 ) ).numpy().item() for n in range( len( msR ) ) ]
            value = ExcludeQueries( value, dummyQueries ); msRDR = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = 'plRt'
            value = [ value.item() for value in list ( ( tf.reduce_sum( tf.cast( tf.greater( profit, 0.0 ), tf.float32 ), axis = 0 ) / profit.shape[0] ).numpy() ) ]
            value = ExcludeQueries( value, dummyQueries ); plRt = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            _f1, _pre, _rec = self.Get_F1_Precision_Recall_For_1_Of_0_And_1( happened_true_onehot, happened_pred_onehot, average = None ) # Read this function!!!

            name = 'f1'
            value = list( _f1 ); f1 = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = 'f1p'
            value = list( _pre ); f1p = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )
            
            name = 'f1r'
            value = list( _rec ); f1r = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = 'profit'
            value = tf.concat( profit, axis = -1 ) if isinstance( profit, list) else profit; profit = value
            dict[ pre_hide + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = 'cum'
            value = tf.concat( cum, axis = -1 ) if isinstance( cum, list) else cum; cum = value
            dict[ pre_hide + '.' + name ] = value; _list.append( value ); indices.append( name )


            #----------------------------- profit collective --------------------------

            profit_c = tf.reduce_sum( profit, axis = -1 )
            cum_c, total_c, safety_c, score_c = self.Analyze_Profit_Collectively( profit, segment ) # profit, Not profit_c

            name = '*mean'
            value = tf.reduce_mean( profit_c, axis = 0 ).numpy().item(); mean = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )
            
            name = '*std'
            value = np.std( profit_c.numpy(), axis = 0 ).item(); std = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = '*msR'
            value = mean / ( std + 1e-9 ); msR = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = '*safe' 
            value = safety_c.numpy().item(); safe = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = '*MDrop'
            value = tf.reduce_max( BettingMath.Get_MaxDrop( cum_c, axis = 0 ) ).numpy().item(); MDrop = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            #msRateSafe = msRate * safe
            #dict[ pre + '.*msRS' ] = msRateSafe

            #msRateSafeRelu = ( tf.nn.relu( msRate ) * tf.nn.relu( safe ) ).numpy().item()
            #dict[ pre + '.*msRSR' ] = msRateSafeRelu

            name = '*msRD'
            value = msR / ( MDrop + 1e-7 ); msRD = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = '*msRDR'
            value = ( tf.nn.relu( msRD ) / ( tf.nn.relu( MDrop ) + 1e-7 ) ).numpy().item()
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = '*plRt'
            value = ( tf.reduce_sum( tf.cast( tf.greater( profit_c, 0.0 ), tf.float32 ), axis = 0 ) / profit_c.shape[0] ).numpy().item(); plRt = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            _f1, _pre, _rec = self.Get_F1_Precision_Recall_For_1_Of_0_And_1( happened_true_onehot, happened_pred_onehot, average = 'micro' ) # Read this function!!!

            name = '*f1'
            value = _f1; f1 = value # 'macro', 'weighted'
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )
           
            name = '*f1p'
            value = _pre; f1p = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )
            
            name = '*f1r'
            value = _rec; f1r = value
            dict[ pre + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = '*profit'
            value = tf.concat( profit_c, axis = -1 ) if isinstance( profit_c, list) else profit_c; profit = value
            dict[ pre_hide + '.' + name ] = value; _list.append( value ); indices.append( name )

            name = '*cum'          
            value = tf.concat( cum_c, axis = -1 ) if isinstance( cum_c, list) else cum_c; cum = value
            dict[ pre_hide + '.' + name ] = value; _list.append( value ); indices.append( name )

            #----------------------------- embeddings --------------------------

            return dict, _list, indices

        #=================================================================================================================================================

        dict_total = {}; data = {}; indices = None

        #--------------------------------------------------- Back Profit ----------------------------------------
        for segment in modelBase.task.dataset.standard_segments :

            dict, _list, indices = Get_Statistics( export_to_account = True, segment = segment, fore_not_back = False, dummyQueries = [] ) #modelBase.task.dummyQueries )
            dict_total.update( dict )
            data.update( { segment : _list } )
            
        dataframe = pd.DataFrame( data, index = indices )

        return dict_total, dataframe

    #============================================================================================================================
    #============================================================================================================================

    def Select_Predictable_Games( self, modelBase, outputs, stakes, profits, target_segment, model_id, scheme, cutRate = None ) :
        """
        This module actually wraps the main body to loading and saving the intermediatary statistics data, which is generated and used repeatedly by the main body.
        Why cache?
        The intermediatary stats depends on train, valid, target_segment, stakes, and profits, of which train and valid segments are implicit.
        As stakes and profits are generated by model, stakes and profits can be reduced to a simple model_id.
        The point is that stats is independent of scheme. It's a waste of time if we produce stats data again for each scheme.
        This is where this wrapper comes into play. It caches the stats data between schemes.
        There are multile schemes, and having stats independent of schemes reduces coding complexity greatly.
        Experiments showed that caching stats saves time greatly greatly. From 140 seconds to 0.5 per call, when the cache is avaliable.
        """

        # Feel free to modity dataset, by eliminating games from 'segment' segment. It will be re-initialized later.

        dataset = modelBase.task.dataset
        nGames = dataset.Data( 'homes', target_segment ).shape[0]

        #------------------------------------- Cache stats.

        # This block generates a many small cache file, one for each model_id. Let it load the files scilently.

        folder = self.gauge_path_base + '_train.valid.' + target_segment + ".view_base"
        if not os.path.exists( folder ) : os.mkdir( folder )
        stats_path = os.path.join( folder, 'Viewer_'+ str(model_id).zfill(6) + ".view_base"  )

        load = LoadBinaryData( stats_path ) # Load again for each model_id, which changes call by call.
        if load is not None : 
            self.cache_stats_model = load # self.scores might not up-to-date with self.stakes_dict_mid...     
        else : 
            self.cache_stats_model = [ None for _ in range ( nGames ) ]
            print( "Model {} is to newly register its statistics. Wait...".format( model_id ) )
       
        ret = self.Select_Predictable_Games_With_Stats( modelBase, outputs, stakes, profits, target_segment, scheme, cutRate = None )

        if not os.path.exists( stats_path ) :
            SaveBinaryData( self.cache_stats_model, stats_path )

        return ret

        # This block generates a single huge cache file, which adds 12 MB for each model_id.
        """
        stats_path = self.gauge_path_base + 'train.valid.' + target_segment + ".stats"

        if self.cache_stats is None :
            print('Loading stats...', end = '')
            load = LoadBinaryData( stats_path )
            print('not found' if load is None else 'done')
            if load is not None : self.cache_stats = load # self.scores might not up-to-date with self.stakes_dict_mid...
            else : self.cache_stats = {}

        self.cache_stats_model = self.cache_stats.get( str( model_id ), None )
        if self.cache_stats_model is None :
            self.cache_stats_model = [ None for _ in range ( nGames ) ]
        
        ret = self.Select_Predictable_Games_With_Stats( modelBase, stakes, profits, target_segment, scheme, cutRate = None )

        if self.cache_stats.get( str( model_id ), None ) is None :
            self.cache_stats.update( { str( model_id ) : self.cache_stats_model } )
            SaveBinaryData( self.cache_stats, stats_path )

        return ret
        """


    #-------------------------------------------------------------------------------------------

    def Populate_Schemes_Dict( self ) :

        # Each funciton 
        # takes odds, pHapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a
        # returns a nQueries-long array, which are the base score of a target game in each query_id.
        # The base score is a function of scheme, which is given by the key in the dictionary.

        return { \
            'happ' :       ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['happ'] ),
            'output' :       ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['output'] ),
            'stake' :       ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['stake'] ),
            'profit' :      ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['profit'] ),
            #------------------------------------------------------------------------
            'odds' :        ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : odds ),
            'pHapp' :       ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : phapp ),
            '_m.pre' :      ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['pre_0'] ),
            'm.pre' :       ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['pre_1'] ),
            '_m.acc' :      ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['acc'] ),
            'm.acc' :       ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['acc'] ),
            '_m.f1' :       ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['f1_0'] ),
            'm.f1' :        ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['f1_1'] ),
            #-----------------------------------------------------------------------
            '_m.std*f1' :   ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['std'] * stat_m['f1_0'] ),
            'm.std*f1' :    ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['std'] * stat_m['f1_1'] ),
            '_m.pre*pre' :  ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m_h['pre_0'] * stat_m_a['pre_0'] ),
            'm.pre*pre' :   ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m_h['pre_1'] * stat_m_a['pre_1'] ),
            '_m.acc*acc' :  ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m_h['acc'] * stat_m_a['acc'] ),
            'm.acc*acc' :   ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m_h['acc'] * stat_m_a['acc'] ),
            '_acc*acc' :    ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['acc'] * stat_a['acc'] ),
            'acc*acc' :     ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['acc'] * stat_a['acc'] ),
            '_m.f1*f1' :    ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m_h['f1_0'] * stat_m_a['f1_0'] ),
            'm.f1*f1' :     ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m_h['f1_1'] * stat_m_a['f1_1'] ),               
            #-----------------------------------------------------------------------
            '_m.std*std*f1*f1' :    ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['std'] * stat_a['std'] * stat_m_h['f1_0'] * stat_m_a['f1_0'] ),
            'm.std*std*f1*f1' :     ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['std'] * stat_a['std'] * stat_m_h['f1_1'] * stat_m_a['f1_1'] ),
            '_m.std*f1*f1' :        ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['std'] * stat_m_h['f1_0'] * stat_m_a['f1_0'] ),
            'm.std*f1*f1' :         ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m['std'] * stat_m_h['f1_1'] * stat_m_a['f1_1'] ),
            '_m.std*std*f1' :       ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m_a['std'] * stat_m_h['std'] * stat_m['f1_0'] ),
            'm.std*std*f1' :        ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_m_a['std'] * stat_m_h['std'] * stat_m['f1_1'] ),
            '_std*std' :            ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_a['std'] * stat_h['std'] ),
            'std*std' :             ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_a['std'] * stat_h['std'] ),
            '_f1*f1' :  ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['f1_0'] * stat_a['f1_0'] ),
            'f1*f1' :   ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['f1_1'] * stat_a['f1_1'] ),
            #------------------------------------------------------------------------
            '_std*std*f1' :     ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['std'] * stat_a['std'] * stat_m['f1_0'] ),
            'std*std*f1' :      ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['std'] * stat_a['std'] * stat_m['f1_1'] ),
            '_std*std*pre' :    ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['std'] * stat_a['std'] * stat_m['pre_0'] ),
            'std*std*pre' :     ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['std'] * stat_a['std'] * stat_m['pre_1'] ),
            '_std*std*f1*f1' :  ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['std'] * stat_a['std'] * stat_m_h['f1_0'] * stat_m_a['f1_0'] ),
            'std*std*f1*f1' :   ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['std'] * stat_a['std'] * stat_m_h['f1_1'] * stat_m_a['f1_1'] ),
            '_std*std*pre*pre' :( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['std'] * stat_a['std'] * stat_m_h['pre_0'] * stat_m_a['pre_0'] ),
            'std*std*pre*pre' : ( lambda odds, phapp, stat_h, stat_a, stat_m, stat_m_h, stat_m_a : stat_h['std'] * stat_a['std'] * stat_m_h['pre_1'] * stat_m_a['pre_1'] )
        }


    def Select_Predictable_Games_With_Stats( self, modelBase, outputs, stakes, profits, target_segment, scheme, cutRate = None ) :

        dataset = modelBase.task.dataset

        happ_tar = dataset.Data( 'happened', target_segment )
        output_tar = outputs[ target_segment ]
        stake_tar = stakes[ target_segment ]
        profit_tar = profits[ target_segment ]
        row_tar = dataset.Data( '', target_segment )

        homes_t = dataset.Data( 'homes', 'train' )
        aways_t = dataset.Data( 'aways', 'train' )
        happened_t = dataset.Data( 'happened', 'train' )
        stake_t = stakes['train'].numpy()
        profit_t = profits['train'].numpy()
        stake_bin_t = ( stake_t > 0 ).astype( stake_t.dtype )

        homes_v = dataset.Data( 'homes', 'valid' )
        aways_v = dataset.Data( 'aways', 'valid' )
        happened_v = dataset.Data( 'happened', 'valid' )
        stake_v = stakes['valid'].numpy()
        profit_v = profits['valid'].numpy()
        stake_bin_v = ( stake_v > 0 ).astype( stake_v.dtype )

        homes_tv = np.concatenate( [ homes_t, homes_v ], axis = 0 )
        aways_tv = np.concatenate( [ aways_t, aways_v ], axis = 0 )
        happened_tv = np.concatenate( [ happened_t, happened_v ], axis = 0 )
        stake_tv = np.concatenate( [ stake_t, stake_v ], axis = 0 )
        profit_tv = np.concatenate( [ profit_t, profit_v ], axis = 0 )
        stake_bin_tv = np.concatenate( [ stake_bin_t, stake_bin_v ], axis = 0 )

        odds_target = dataset.Data( 'odds', target_segment )
        pHapp_target = dataset.Data( 'p_happened', target_segment )


        def get_statistics( happened, stake_bin, profit, happ_game, output_game, stake_game, profit_game, rows ) :
            """
            Get a statistics from historic 'happened', 'stake_bin', and profit.

            input:
                happened : 
                stake_bin :
                profit :
                rows : the ids of concerned rows of happened, stake_bin, and profit.

            return:
                A collection of arrays of shape ( nQueries, ), plus stat['nGames'] which is a scalar.
                From a glabal point of view, this comes from all train, valid, target_segment, and viewer, but scheme.
            """

            stat = None

            nRows = len( rows )

            if nRows > 0 :

                happened = happened[ rows ]
                stake_bin = stake_bin[ rows ]
                profit = profit[ rows ]

                # Now, happened, stake_bin, and profit are of shape ( nRows, nQueries ).

                happened = np.round( happened ).astype( np.int64 ) # for safety purpose.
                stake_bin = np.round( stake_bin ).astype( np.int64 ) # for safety purpose.

                nNumbers_for_half_confidence = 5
                # The larger nNumbers_for_half_confidence, the slower growth of confidence over nNumbers.
                def confidence( nNumbers ) : return 1. - 1. / np.power( 2., nNumbers / nNumbers_for_half_confidence ) 

                stat = {}

                stat['nGames'] = happened.shape[0]

                #--------------------------------------------------------------------------------------------------
                # Scores are distributed over [ 0, inf ), with 0 being the worst score.
                # This rule is requres for average scores to have meaning.
                 #--------------------------------------------------------------------------------------------------

                # All of shape ( nRows, nQueries ) are going to be reduced to shape ( nQueries, )
               
                stat['happ'] = happ_game
                stat['output'] = output_game
                stat['stake'] = stake_game
                stat['profit'] = profit_game

                conf = confidence( nRows )
                stat['acc'] = np.sum( np.array( happened == stake_bin, np.float32 ), axis = 0 ) / nRows * conf
                stat['mean'] = np.mean( profit, axis = 0 ) * conf
                stat['std'] = np.std( profit, axis = 0 ) * conf

                conf_0_p = confidence( np.sum( np.array( stake_bin == 0, np.int32 ), axis = 0 ) ) # query-wise
                conf_0_r = confidence( np.sum( np.array( happened == 0, np.int32 ), axis = 0 ) ) # query-wise
                f1, pre, rec = self.Get_F1_Precision_Recall_For_1_Of_0_And_1( 1 - happened, 1 - stake_bin, average = None ) # Read this function!!!
                stat['f1_0'] = f1 * conf_0_p * conf_0_r
                stat['pre_0'] = pre * conf_0_p # what portion of stake_bin's 0s are a true 0, for each Q (label).
                stat['rec_0'] = rec * conf_0_r # what portion of true 0s are hit by a stake_bin's 0, for each Q (label).

                conf_1_p = confidence( np.sum( np.array( stake_bin == 1, np.int32 ), axis = 0 ) ) # query-wise
                conf_1_r = confidence( np.sum( np.array( happened == 1, np.int32 ), axis = 0 ) ) # query-wise
                f1, pre, rec = self.Get_F1_Precision_Recall_For_1_Of_0_And_1( happened, stake_bin, average = None ) # Read this function!!!
                stat['f1_1'] = f1 * conf_1_p * conf_1_r
                stat['pre_1'] = pre * conf_1_p # what portion of stake_bin's 1s are a true 1, for each Q (label).
                stat['rec_1'] = rec * conf_1_r # what portion of true 1s are hit by a stake_bin's 1, for each Q (label).

            return stat 

        """
        def get_statistics_2( happened, stake_bin, profit, rows ) :
            happened = happened[ rows ]
            profit = profit[ rows ]

            return get_statistics( happened, stake_bin, profit )
        """

        def get_stat( homes, aways, happened, stake_bin, profit, happ_game, output_game, stake_game, profit_game, home, away ) :

            if super_config['bidir'] :
                rows_home = np.concatenate( [ np.where( homes == home )[0], np.where( aways == home )[0] ], axis = 0 )
                rows_away = np.concatenate( [ np.where( homes == away )[0], np.where( aways == away )[0] ], axis = 0 )
            else :
                rows_home = np.where( homes == home )[0]
                rows_away = np.where( aways == away )[0]

            stat_h = get_statistics( happened, stake_bin, profit, happ_game, output_game, stake_game, profit_game, rows_home )
            stat_a = get_statistics( happened, stake_bin, profit, happ_game, output_game, stake_game, profit_game, rows_away )

            return stat_h, stat_a


        def get_med_stat( homes, aways, happened, stake_bin, profit, happ_game, output_game, stake_game, profit_game, home, away ) :

            aways_of_home = aways[ np.where( homes == home )[0] ]
            homes_of_away = homes[ np.where( aways == away )[0] ]
            mediators_HA = set( [ team for team in aways_of_home if team in homes_of_away ] )

            rows_from_home_HA = np.array( [ row for row in np.where( homes == home )[0] if aways[row] in mediators_HA ], dtype = np.int64 )
            rows_from_away_HA = np.array( [ row for row in np.where( aways == away )[0] if homes[row] in mediators_HA ], dtype = np.int64 )

            homes_of_home = homes[ np.where( aways == home )[0] ]
            aways_of_away = aways[ np.where( homes == away )[0] ]
            mediators_AH = set( [ team for team in homes_of_home if team in aways_of_away ] )

            rows_from_home_AH = np.array( [ row for row in np.where( aways == home )[0] if homes[row] in mediators_AH ], dtype = np.int64 )
            rows_from_away_AH = np.array( [ row for row in np.where( homes == away )[0] if aways[row] in mediators_AH ], dtype = np.int64 )

            if super_config['bidir'] :
                rows_from_home = np.concatenate( [ rows_from_home_HA, rows_from_home_AH ], axis = 0 )
                rows_from_away = np.concatenate( [ rows_from_away_HA, rows_from_away_AH ], axis = 0 )
                rows = np.concatenate( [ rows_from_home, rows_from_away ], axis = 0 )
            else :
                rows_from_home = rows_from_home_HA
                rows_from_away = rows_from_away_HA
                rows = np.concatenate( [ rows_from_home_HA, rows_from_away_HA ], axis = 0 )

            stat_med_home = get_statistics( happened, stake_bin, profit, happ_game, output_game, stake_game, profit_game, rows_from_home )
            stat_med_away = get_statistics( happened, stake_bin, profit, happ_game, output_game, stake_game, profit_game, rows_from_away )
            stat_med = get_statistics( happened, stake_bin, profit, happ_game, output_game, stake_game, profit_game, rows )

            return stat_med_home, stat_med_away, stat_med

        #==================================================================================== Score 1 ==============================================================

        def get_eligibility_score( game_id, home, away, scheme ) : # of a target game in each query.

            stats = None
            stat_h, stat_a, stat_m_h, stat_m_a, stat_m = None, None, None, None, None

            if self.cache_stats_model is not None : stats = self.cache_stats_model[ game_id ]

            if stats is not None : 
                stat_h, stat_a, stat_m_h, stat_m_a, stat_m = stats
            else :
                happ_game = happ_tar[ game_id ]
                output_game = output_tar[ game_id ]
                stake_game = stake_tar[ game_id ]
                profit_game = profit_tar[ game_id ]

                if super_config['valid_only'] :
                    stat_h, stat_a = get_stat( homes_v, aways_v, happened_v, stake_bin_v, profit_v, happ_game, output_game, stake_game, profit_game, home, away )
                    stat_m_h, stat_m_a, stat_m = get_med_stat( homes_v, aways_v, happened_v, stake_bin_v, profit_v, happ_game, output_game, stake_game, profit_game, home, away )
                else :
                    stat_h, stat_a = get_stat( homes_tv, aways_tv, happened_tv, stake_bin_tv, profit_tv, happ_game, output_game, stake_game, profit_game, home, away )
                    stat_m_h, stat_m_a, stat_m = get_med_stat( homes_tv, aways_tv, happened_tv, stake_bin_tv, profit_tv, happ_game, output_game, stake_game, profit_game, home, away )
                
                self.cache_stats_model[ game_id ] = stat_h, stat_a, stat_m_h, stat_m_a, stat_m
            #---------------------------------------------------------------------------------------------------------------------------------------------------------

            #if stat_h is None or stat_a is None or stat_m_h is None or stat_m_a is None :
            #    scores = np.array( [ super_config['no_score'] for _ in modelBase.task.queries ] )

            #else :
            #    scores = self.schemes_dict[ scheme ]( odds_target[ game_id ], pHapp_target[ game_id ], stat_h, stat_a, stat_m, stat_m_h, stat_m_a )

            try :
                scores = self.schemes_dict[ scheme ]( odds_target[ game_id ], pHapp_target[ game_id ], stat_h, stat_a, stat_m, stat_m_h, stat_m_a )
            except :
                # self.schemes_dict may raise an exception if one of its functions has to work on a None arguement.
                scores = np.array( [ super_config['no_score'] for _ in modelBase.task.queries ] )

            return scores # of a target game in each query.


        #====================================================================== Main body ==================================================

        games_to_take = []
        target_games = [ ( home, away ) for home, away in zip( dataset.Data( 'homes', target_segment ), dataset.Data( 'aways', target_segment ) ) ]

        if cutRate is not None :
            nGamesCutDown = int( len( target_games ) * cutRate ) #------------------------------- Rate
        else :
            nGamesCutDown = int( len( target_games ) * super_config['.CutRate'] ) #------------------------------- Rate

        indices_by_query = []; stats_total = []

        scores_by_game = [] # shape of [ nChechGames, nQueries, 2 ], game, query, f1

        for game_id, target_game in zip( range( len( target_games ) ), target_games ) :

            scores = get_eligibility_score( game_id, target_game[0], target_game[1], scheme )

            scores_by_game.append( list( scores ) ) # Grows to be [ score ] x nCheckGames            

        def select_games_of_higher_score( scores_by_game ) : # scores_by_game : scores of a game at each Q.

            scores_by_game = np.array( scores_by_game, dtype = np.float32 ) # array of shape : ( nCheckGames, nQueries ) - The eligibility of each predict game on each query. eg: ( 50, 13 )
            indices = np.argsort( scores_by_game, axis = 0 ) # array of shape : ( nCheckGames, nQueries ) - rowIds of games, sorted in the eligibility score, for each query. eg: ( 50, 13 )
            games_selected = indices[ nGamesCutDown : ] # rowIds giving smaller eligibility socre comes first. Get rid of the early-coming indices, giving rowIds selected on each query. eg: ( 5, 13 )
            # predict this way: ss = np.take_along_axis( scores_by_game, indices, axis = 0). ss should give scored_by_game sorted for each query.

            games_t = np.transpose( games_selected )
            nGames = dataset.Data( 'homes', target_segment ).shape[0]
            #if stakes[ target_segment ].shape != tf.TensorShape([]) : nGames = stakes[ target_segment ].shape[0]
            games_unselected_t = np.array( [ [ id for id in range( nGames ) if id not in games_t[ queryPlace, : ] ] for queryPlace in range( len( modelBase.task.queries ) ) ] )
            games_unselected = np.transpose( games_unselected_t ) # Remainig rowIds for each query. eg: ( 45, 13)

            return games_selected, games_unselected

        
        games_selected, games_unselected = select_games_of_higher_score( scores_by_game )

        del homes_tv, aways_tv, happened_tv, stake_tv, profit_tv, stake_bin_tv
        del homes_v, aways_v, happened_v, stake_v, profit_v, stake_bin_v
        del homes_t, aways_t, happened_t, stake_t, profit_t, stake_bin_t


        return scores_by_game, games_selected, games_unselected
