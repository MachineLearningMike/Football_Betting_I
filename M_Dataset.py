import tensorflow as tf
import numpy as np
import pandas as pd

from M_Football import *

from M_Config import *

class Dataset() :

    def __init__(self, dataUse = None, data = None, queries = [] ) :

        self.queries = queries

        self.dataUse = dataUse
        self.data = data

        self.viewer = None

        self.Initialize()

    def Initialize( self ) :

        dataUse = self.dataUse
        data = self.data

        #=========================================== Standard columns, and their derivatives.

        array_queries = np.array( self.queries )

        self.train_erows = data['erowsTrain']; self.valid_erows = data['erowsValid']; self.test_erows = data['erowsTest']
        self.train_dates = data['datesTrain']; self.valid_dates = data['datesValid']; self.test_dates = data['datesTest']
        self.train_leagues = data['leaguesTrain']; self.valid_leagues = data['leaguesValid']; self.test_leagues = data['leaguesTest']
        self.nLeagues = data['nLeagues']
        self.leagues_dict = data['leagues_dic']
        self.train_homes = data['homesTrain']; self.valid_homes = data['homesValid']; self.test_homes = data['homesTest']
        self.train_aways = data['awaysTrain']; self.valid_aways = data['awaysValid']; self.test_aways = data['awaysTest']
        self.nTeams = data['nTeams']
        self.teams_dict = data['teams_dic']
        self.train_situation = data['situationTrain']; self.valid_situation = data['situationValid']; self.test_situation = data['situationTest']
        self.nSituation = self.train_situation.shape[ -1 ]
        self.train_odds = data['oddsTrain'][:, array_queries]; self.valid_odds = data['oddsValid'][:, array_queries]; self.test_odds = data['oddsTest'][:, array_queries]
        self.train_rows = data['rowsTrain'].astype( np.float32 ); self.valid_rows = data['rowsValid'].astype( np.float32 ); self.test_rows = data['rowsTest'].astype( np.float32 )

        self.total_erows = np.concatenate( [ self.train_erows, self.valid_erows, self.test_erows ], axis = 0 )
        self.total_dates = np.concatenate( [ self.train_dates, self.valid_dates, self.test_dates ], axis = 0 )
        self.total_leagues = np.concatenate( [ self.train_leagues, self.valid_leagues, self.test_leagues ] , axis = 0 )
        self.total_homes = np.concatenate( [ self.train_homes, self.valid_homes, self.test_homes ], axis = 0 )
        self.total_aways = np.concatenate( [ self.train_aways, self.valid_aways, self.test_aways ], axis = 0 )
        self.total_situation = np.concatenate( [ self.train_situation, self.valid_situation, self.test_situation ], axis = 0 )
        self.total_odds = np.concatenate( [ self.train_odds, self.valid_odds, self.test_odds ], axis = 0 )
        self.total_rows = np.concatenate( [ self.train_rows, self.valid_rows, self.test_rows ], axis = 0 )

        self.inp_leagues = tf.keras.Input( name = 'leagues', shape = (), batch_size = dataUse['.BSize'], dtype = tf.int32 )
        self.inp_homes = tf.keras.Input( name = 'homes', shape = (), batch_size = dataUse['.BSize'], dtype = tf.int32 )
        self.inp_aways = tf.keras.Input( name = 'aways', shape = (), batch_size = dataUse['.BSize'], dtype = tf.int32 )
        self.inp_situation = tf.keras.Input( name = 'situation', shape = ( self.nSituation, ), batch_size = dataUse['.BSize'], dtype = tf.float32 )
        self.inp_odds = tf.keras.Input( name = 'odds', shape = ( len(self.queries), ), batch_size = dataUse['.BSize'], dtype = tf.float32 )
        self.inp_rows = tf.keras.Input( name = 'rows', shape = ( 1,), batch_size = dataUse['.BSize'], dtype = tf.float32 )

        #=========================================== Database-specific columns.

        if super_config['Contest'] :

            self.train_lCountries = data['lCountriesTrain']; self.valid_lCountries = data['lCountriesValid']; self.test_lCountries = data['lCountriesTest']
            self.train_hCountries = data['hCountriesTrain']; self.valid_hCountries = data['hCountriesValid']; self.test_hCountries = data['hCountriesTest']
            self.train_aCountries = data['aCountriesTrain']; self.valid_aCountries = data['aCountriesValid']; self.test_aCountries = data['aCountriesTest']
            self.nCountries = data['nCountries']
            self.train_outcome = data['outcomeTrain']; self.valid_outcome = data['outcomeValid']; self.test_outcome = data['outcomeTest']

            self.total_lCoutries = np.concatenate( [self.train_lCountries, self.valid_lCountries, self.test_lCountries ], axis = 0 )
            self.total_hCoutries = np.concatenate( [self.train_hCountries, self.valid_hCountries, self.test_hCountries ], axis = 0 )
            self.total_aCoutries = np.concatenate( [self.train_aCountries, self.valid_aCountries, self.test_aCountries ], axis = 0 )
            self.total_outcome = np.concatenate( [self.train_outcome, self.valid_outcome, self.test_outcome ], axis = 0 )

            self.inp_lCountries = tf.keras.Input( name = 'lCountries', shape = (), batch_size = dataUse['.BSize'], dtype = tf.int32 )
            self.inp_hCountries = tf.keras.Input( name = 'hCountries', shape = (), batch_size = dataUse['.BSize'], dtype = tf.int32 )
            self.inp_aCountries = tf.keras.Input( name = 'aCountries', shape = (), batch_size = dataUse['.BSize'], dtype = tf.int32 )


            # PENDING ===================================


        elif not super_config['Contest'] :

            self.train_goals = data['goalsTrain']; self.valid_goals = data['goalsValid']; self.test_goals = data['goalsTest']
            self.total_goals = np.concatenate( [self.train_goals, self.valid_goals, self.test_goals ], axis = 0 )

            self.train_happened, self.valid_happened, self.test_happened = self.Get_Happened( self.train_goals, self.valid_goals, self.test_goals )
            self.train_happened, self.valid_happened, self.test_happened = self.train_happened[:, array_queries], self.valid_happened[:, array_queries], self.test_happened[:, array_queries]

            self.train_happened = self.train_happened.astype( np.float32 ); self.valid_happened = self.valid_happened.astype( np.float32 ); self.test_happened = self.test_happened.astype( np.float32 )
            self.total_happened = np.concatenate( [ self.train_happened, self.valid_happened, self.test_happened ], axis = 0 )

            self.train_p_goals = data['p_goalsTrain']; self.valid_p_goals = data['p_goalsValid']; self.test_p_goals = data['p_goalsTest']
            self.total_p_goals = np.concatenate( [ self.train_p_goals, self.valid_p_goals, self.test_p_goals ], axis = 0 )

            self.train_p_happened, self.valid_p_happened, self.test_p_happened = self.Get_Happened( self.train_p_goals, self.valid_p_goals, self.test_p_goals )
            self.train_p_happened, self.valid_p_happened, self.test_p_happened = self.train_p_happened[:, array_queries], self.valid_p_happened[:, array_queries], self.test_p_happened[:, array_queries]
            self.train_p_happened = self.train_p_happened.astype( np.float32 ); self.valid_p_happened = self.valid_p_happened.astype( np.float32 ); self.test_p_happened = self.test_p_happened.astype( np.float32 )
            self.total_p_happened = np.concatenate( [ self.train_p_happened, self.valid_p_happened, self.test_p_happened ], axis = 0 )

            self.inp_p_goals = tf.keras.Input( name = 'p_goals', shape = (2,), batch_size = dataUse['.BSize'], dtype = tf.float32 )
            self.inp_p_happened = tf.keras.Input( name = 'p_happened', shape = (len(self.queries),), batch_size = dataUse['.BSize'], dtype = tf.float32 )
            self.inp_goals = tf.keras.Input( name = 'goals', shape = (2,), batch_size = dataUse['.BSize'], dtype = tf.float32 )


        self.standard_segments = [ 'train', 'valid', 'test' ] # One of the most influential values.

        self.data_segments = self.ArrangeDataSegments()

        return


    def Add_Viewer( self, viewerOutputs ):
        
        self.train_viewerOutput = viewerOutputs['train']
        self.valid_viewerOutput = viewerOutputs['valid']
        self.test_viewerOutput = viewerOutputs['test']
        self.data_segments = self.AddDataSegments( self.data_segments, 'viewOutput', viewerOutputs )

        self.inp_viewerOutput = tf.keras.Input( name = 'viewerOutput', shape = ( viewerOutputs.shape[-1], ), batch_size = dataUse['.BSize'], dtype = tf.int32 )






    def Data( self, feature, segment ) :
        try:
            ret = self.data_segments.loc[ feature, segment ]
        except:
            ret = None
        return ret


    def Checked_OK( self, dataChecker, dictionaries ) :

        def max_error( arr1, arr2 ) :
            max = np.nanmax( np.abs( np.nan_to_num( arr1 ) - np.nan_to_num( arr2 ) ) )
            return max

        def find_unique_name( name, name_variations ) :
            unique_name = None
            for key, variations in name_variations.items() :
                for variation in variations :
                    if name == variation :
                        unique_name = key
                        break
                if unique_name is not None :
                    break
            if unique_name is None :
                unique_name = name
        
            return unique_name


        name_variations = dictionaries['name_variations']
        leagues = dictionaries['leagues']
        teams = dictionaries['teams']

        dc = dataChecker

        checked_ok = True

        for erow, year, month, dayofweek, league, home, away, p_hGoal, p_aGoal, hGoal, aGoal, odds \
            in zip( dc.erow, dc.year, dc.month, dc.dayofweek, dc.league, dc.home, dc.away, dc.p_hGoal, dc.p_aGoal, dc.hGoal, dc.aGoal, dc.odds ) :
            situation = np.array( [ year, month, dayofweek ], dtype = np.float32 )
            year_rows = set( np.where( self.total_situation[:, 0] == year )[0] )
            month_rows = set( np.where( self.total_situation[:, 1] == month )[0] )
            dayofweek_rows = set( np.where( self.total_situation[:, 2] == dayofweek )[0] )
            league = find_unique_name( league, name_variations ); league = leagues[league]
            league_rows = set( np.where( self.total_leagues == league )[0] )
            home = find_unique_name( home, name_variations ); home = teams[home]
            home_rows = set( np.where( self.total_homes == home )[0] )
            away = find_unique_name( away, name_variations ); away = teams[away]
            away_rows = set( np.where( self.total_aways == away )[0] )
            p_hGoal_rows = set( np.where( self.total_p_goals[:, 0] == p_hGoal )[0] )
            p_aGoal_rows = set( np.where( self.total_p_goals[:, 1] == p_aGoal )[0] )
            hGoal_rows = set( np.where( self.total_goals[:, 0] == hGoal )[0] )
            aGoal_rows = set( np.where( self.total_goals[:, 1] == aGoal )[0] )

            rows = year_rows & month_rows & dayofweek_rows & league_rows & home_rows & away_rows & p_hGoal_rows & p_aGoal_rows & hGoal_rows & aGoal_rows
            rows = rows

            if len( rows ) <= 0 : # Ever found any rows?
                checked_ok =False
                print( "?????????? Excel row {} is found at no Excel rows.".format( erow ) )             
                break

            if len( rows ) != 1 : # Found a unique row?
                checked_ok = False
                erows_found = [ str( self.total_erows[ row ] ) for row in sorted( list( rows ) ) ]
                print( "?????????? Excel row {} is found at multiple Excel rows {}".format( erow, erows_found ) )
                break

            row_found = list( rows )[0]
            erow_found = self.total_erows[ row_found ]
            if erow != erow_found : # Found at a different row?
                checked_ok = False
                print( "?????????? Excel row {} is found at a wrong Excel row {}".format( erow, erow_found ) )
                break

            # OK. A unique row was found. The only quesion remaining is are their odds the same?
            if max_error( odds, self.total_odds[ row_found ] ) > 1e-5 : # Once the max_error was 1.9e-7 on the same odds.
                checked_ok = False
                print( "?????????? Excel row {} wid odds of {} was input with a wrong odds {}".format( erow, odds, self.total_odds[ row_found ] ) )
                break

        return checked_ok


    def Get_Happened( self, train_goals, valid_goals, test_goals ) :

        return  Prob_Happened_R_Concat_13( train_goals[:, 0], train_goals[:, 1] ).numpy(), \
                Prob_Happened_R_Concat_13( valid_goals[:, 0], valid_goals[:, 1] ).numpy(), \
                Prob_Happened_R_Concat_13( test_goals[:, 0], test_goals[:, 1] ).numpy()


    def ArrangeDataSegments( self ) :

        data = { \
        'train' : [ self.train_erows, self.train_leagues, self.train_homes, self.train_aways, self.train_situation, self.train_rows, self.train_odds, self.train_happened ], \
        'valid' : [ self.valid_erows, self.valid_leagues, self.valid_homes, self.valid_aways, self.valid_situation, self.valid_rows, self.valid_odds, self.valid_happened ], \
        'test' : [ self.test_erows, self.test_leagues, self.test_homes, self.test_aways, self.test_situation, self.test_rows, self.test_odds, self.test_happened ], \
        }
        indices = [ 'erows', 'leagues', 'homes', 'aways', 'situation', 'rows', 'odds', 'happened' ]

        if super_config['Contest'] :
            data['train'] += [ self.train_lCountries, self.train_hCountries, self.train_aCountries, self.train_outcome ]
            data['valid'] += [ self.valid_lCountries, self.valid_hCountries, self.valid_aCountries, self.valid_outcome ]
            data['test'] += [ self.test_lCountries, self.test_hCountries, self.test_aCountries, self.test_outcome ]
            indices += [ 'lCountries', 'hCountries', 'aCountries', 'outcome' ]

        elif not super_config['Contest'] :
            data['train'] += [ self.train_p_goals, self.train_goals, self.train_p_happened ]
            data['valid'] += [ self.valid_p_goals, self.valid_goals, self.valid_p_happened ]
            data['test'] += [ self.test_p_goals, self.test_goals, self.test_p_happened ]
            indices += [ 'p_goals', 'goals', 'p_happened' ]
    
        dataframe = pd.DataFrame( data, index = indices )
        
        return dataframe


    def AddDataSegments( self, dataframe, index, data ) :

        dataframe.loc[ index ] = [ data['train'], data['valid'], data['test'] ]


        self.train_viewerOutput = viewerOutputs['train']
        self.valid_viewerOutput = viewerOutputs['valid']
        self.test_viewerOutput = viewerOutputs['test']



    def Select_Games( self, segment, game_indices ) :
        df = self.data_segments
        df.loc[ 'leagues', segment ] = df.loc[ 'leagues', segment ][ game_indices ]
        df.loc[ 'homes', segment ] = df.loc[ 'homes', segment ][ game_indices ]
        df.loc[ 'aways', segment ] = df.loc[ 'aways', segment ][ game_indices ]
        df.loc[ 'situation', segment ] = df.loc[ 'situation', segment ][ game_indices ]
        df.loc[ 'rows', segment ] = df.loc[ 'rows', segment ][ game_indices ]
        df.loc[ 'odds', segment ] = df.loc[ 'odds', segment ][ game_indices ]
        df.loc[ 'happened', segment ] = df.loc[ 'happened', segment ][ game_indices ]

        if super_config['Contest'] :
            df.loc[ 'lCountries', segment ] = df.loc[ 'lCountries', segment ][ game_indices ]
            df.loc[ 'hCountries', segment ] = df.loc[ 'hCountries', segment ][ game_indices ]
            df.loc[ 'aCountries', segment ] = df.loc[ 'aCountries', segment ][ game_indices ]
            df.loc[ 'outcome', segment ] = df.loc[ 'outcome', segment ][ game_indices ]
        elif not super_config['Contest'] :
            df.loc[ 'p_goals', segment ] = df.loc[ 'p_goals', segment ][ game_indices ]
            df.loc[ 'p_happened', segment ] = df.loc[ 'p_happened', segment ][ game_indices ]

        return

   
    def Is_Standard_Segment( self, segment ) :
        
        return segment in self.standard_segments


    def Get_Propositional_Margins( self, margin, segment ) :

        dict = {}

        for segment in self.standard_segments :
            leagues = self.data_segments.loc[ 'leagues', segment ]
            dict[segment] = leagues.shape[0]
        
        assert segment in self.standard_segments

        weight = margin / dict[ segment ]

        for segment, nEntries in dict.items() :
            dict[ segment ] = weight * dict[ segment ]

        return dict
