from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import numpy_indexed as npi
from datetime import date, timedelta

import pandas as pd
import pickle
import json
from collections import Counter

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import os
import re

from M_Config import *

dataSchemes = [ \
        #================================================================================================================================================================
        # 0
        { \
        'file_path' : "./Data/Clean_Trim_Data_v1.csv", \
        'leagueCol' : ['League'], \
        'homeCol' : ['Home'], \
        'awayCol' : ['Away'], \
        'goalCols' : ['rHGoal', 'rAGoal'], \
        'envnumericalCols' : ['Year', 'Month', 'Day', 'poHWin', 'poDraw', 'poAWin', 'pHGoal', 'pAGoal'], \
        'oddCols' : ['poHWin', 'poDraw', 'poAWin'], \
        }, \
        #================================================================================================================================================================
        # 1
        { \
        'file_path' : "./Data/Clean_Trim_Data_v2.csv", \
        'leagueCol' : ['League'], \
        'homeCol' : ['Home'], \
        'awayCol' : ['Away'], \
        'goalCols' : ['rHGoal', 'rAGoal'], \
        'numericalCols' : ['Year', 'Month', 'Day', 'pHGoal1', 'pAGoal1', 'pHGoal1Der1', 'pAGoal1Der1', 'pHGoal', 'pAGoal', \
            'pGoalDer1', 'pGoalDer2', 'pHGoalDer1', 'pAGoalDer1', 'poHWin', 'poDraw', 'poAWin', 'poDer1'], \
        'oddCols' : ['poHWin', 'poDraw', 'poAWin'], \
        }, \
        #================================================================================================================================================================
        # 2
        { \
        'file_path' : "../Thomas/Data/Clean_trim_Data_2_ReColumned.csv", \
        'leagueCol' : ['LEAGUE'], \
        'homeCol' : ['HOME'], \
        'awayCol' : ['AWAY'], \
        'goalCols' : ['TSHG', 'TSAG'], \
        'numericalCols' : ['YEAR', 'MONTH', 'DAY', 'PSHG', 'PSAG', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'OHWIN', 'ODRAW', 'OAWIN'], \
        'oddCols' : ['OHWIN', 'ODRAW', 'OAWIN'], \
        }, \
        #================================================================================================================================================================
        # 3
        { \
        'file_path' : "../Thomas/Data/Core_Data_3.csv", \
        'leagueCol' : ['LEAGUE'], \
        'homeCol' : ['HOME'], \
        'awayCol' : ['AWAY'], \
        'goalCols' : ['TSHG', 'TSAG'], \
        'numericalCols' : ['OHWIN', 'ODRAW', 'OAWIN', 'YEAR', 'MONTH', 'DAY'], \
        'oddCols' : ['OHWIN', 'ODRAW', 'OAWIN'], \
        }, \
        #================================================================================================================================================================
        # 4
        { \
        'file_path' : "../Thomas/Data/2020-05-10 GetODDs Refined.csv", \
        'leagueCol' : ['LEAGUE'], \
        'homeCol' : ['HOME'], \
        'awayCol' : ['AWAY'], \
        'goalCols' : ['HGOAL', 'AGOAL'], \
        'numericalCols' : ['O_HWIN', 'O_DRAW', 'O_AWIN', 'O_BOTH', 'O_NO_BOTH', 'O_OVER_15', 'O_UNDER_15', 'O_OVER_25', 'O_UNDER_25', 'O_AH_15_1', 'O_AH_15_2', 'O_1_0', 'O_2_1', 'O_1_1', 'YEAR', 'MONTH', 'DAY', 'NDOW'], \
        'oddCols' : ['O_HWIN', 'O_DRAW', 'O_AWIN', 'O_BOTH', 'O_NO_BOTH', 'O_OVER_15', 'O_UNDER_15', 'O_OVER_25', 'O_UNDER_25', 'O_1_0', 'O_2_1', 'O_1_1' ], \
        }, \
        #================================================================================================================================================================
        # 5
        { \
        'file_path' : "../Thomas/Data/2020-05-10 GetODDs Refined.csv", \
        'leagueCol' : ['LEAGUE'], \
        'homeCol' : ['HOME'], \
        'awayCol' : ['AWAY'], \
        'goalCols' : ['HGOAL', 'AGOAL'], \
        'numericalCols' : ['O_HWIN', 'O_DRAW', 'O_AWIN', 'O_BOTH', 'O_NO_BOTH', 'O_OVER_15', 'O_UNDER_15', 'O_OVER_25', 'O_UNDER_25', 'O_1_0', 'O_2_1', 'O_1_1', 'O_AH_15_1', 'O_AH_15_2', 'YEAR', 'MONTH', 'DAY', 'NDOW'], \
        'oddCols' : ['O_HWIN', 'O_DRAW', 'O_AWIN', 'O_BOTH', 'O_NO_BOTH', 'O_OVER_15', 'O_UNDER_15', 'O_OVER_25', 'O_UNDER_25', 'O_1_0', 'O_2_1', 'O_1_1', 'O_AH_15_1', 'O_AH_15_2' ], \
        }, \
        #================================================================================================================================================================
        # 6
        { \
        'file_path' : "../Thomas/Data/Clean_trim_Data_2_ReColumned_Corrected.csv", \
        'leagueCol' : ['LEAGUE'], \
        'homeCol' : ['HOME'], \
        'awayCol' : ['AWAY'], \
        'goalCols' : ['TSHG', 'TSAG'], \
        'numericalCols' : ['YEAR', 'MONTH', 'DAY', 'PSHG', 'PSAG', 'D1', 'D2', 'D3', 'D4', 'OHWIN', 'ODRAW', 'OAWIN'], \
        'oddCols' : ['OHWIN', 'ODRAW', 'OAWIN'], \
        }, \
        #================================================================================================================================================================
        # 7
        { \
        'file_path' : "../Thomas/Data/2020-05-10 GetODDs Refined.csv", \
        'leagueCol' : ['LEAGUE'], \
        'homeCol' : ['HOME'], \
        'awayCol' : ['AWAY'], \
        'goalCols' : ['HGOAL', 'AGOAL'], \
        'situationCols' : ['YEAR', 'MONTH', 'DAY', 'NDOW'], \
        'oddCols' : ['O_HWIN', 'O_DRAW', 'O_AWIN' ], \
        }, \
        #================================================================================================================================================================
        # 8
        { \
        'file_path' : "../Thomas/Data/2020-05-10 GetODDs Refined.csv", \
        'leagueCol' : ['LEAGUE'], \
        'homeCol' : ['HOME'], \
        'awayCol' : ['AWAY'], \
        'goalCols' : ['HGOAL', 'AGOAL'], \
        'situationCols' : ['YEAR', 'MONTH', 'DAY', 'NDOW'], \
        'oddCols' : ['O_HWIN', 'O_DRAW', 'O_AWIN', 'O_BOTH', 'O_NO_BOTH', 'O_OVER_15', 'O_UNDER_15', 'O_OVER_25', 'O_UNDER_25', 'O_1_0', 'O_2_1', 'O_1_1' ], \
        }, \
        #================================================================================================================================================================
        # 9
        { \
        'file_path' : "2020-05-10 GetODDs Refined.csv", \
        'leagueCol' : ['LEAGUE'], \
        'homeCol' : ['HOME'], \
        'awayCol' : ['AWAY'], \
        'goalCols' : ['HGOAL', 'AGOAL'], \
        'situationCols' : ['YEAR', 'MONTH', 'DAY', 'NDOW'], \
        'oddCols' : ['O_HWIN', 'O_DRAW', 'O_AWIN', 'O_BOTH', 'O_NO_BOTH', 'O_OVER_15', 'O_UNDER_15', 'O_OVER_25', 'O_UNDER_25', 'O_1_0', 'O_2_1', 'O_1_1' ], \
        }, \
        #================================================================================================================================================================
        # 10
        { \
        'file_path' : "GetODDs V.1.0. np - July 26 Refined.csv", \
        'leagueCol' : ['LEAGUE'], \
        'homeCol' : ['HOME'], \
        'awayCol' : ['AWAY'], \
        'goalCols' : ['HGOAL', 'AGOAL'], \
        'p_goalCols' : ['pHGOAL', 'pAGOAL'], \
        'situationCols' : ['YEAR', 'MONTH', 'NDOW'], # remove 'DAY' \
        'oddCols' : ['O_HWIN', 'O_DRAW', 'O_AWIN', 'O_BOTH', 'O_NO_BOTH', 'O_OVER_15', 'O_UNDER_15', 'O_OVER_25', 'O_UNDER_25', 'O_1_0', 'O_2_1', 'O_1_1' ], \
        'maxOdds' : super_config['.M.Odds'], \
        'treat_maxOdds' : 'drop', # or 'replace' \
        '.M.Score' : super_config['.M.Score'], \
        'treat_maxGoals' : 'drop', # or 'replace' \
        }, \
        #================================================================================================================================================================
        # 11
        # The columns of the following column names are either found in the excel file or created by EnhanceReadExcelFileForArrays(.)
         { \
        'past_data_path' : "thomas_past_data Sep 29.xlsx", \
        'past_data_bin_path' : "thomas_past_data Sep 29.bin", \
        'future_data_path' : "thomas_testing_data.xlsx", \
        'past_data_generated' : 'thomas_past_data_generated.xlsx', \
        #--------------------------------------------------------------------------------------- database-specific columns
        'goalCols' : ['hGoal', 'aGoal'], # generated \
        'p_goalCols' : ['pHGoal', 'pAGoal'], # generated \
        #--------------------------------------------------------------------------------------- standart columns
        'leagueCol' : ['League'], # as is \
        'homeCol' : ['Home'], # generated \
        'awayCol' : ['Away'], # generated \
        'oddCols' : ['Home Win', 'Draw', 'Away Win', 'Both teams to score - yes', 'Both teams to score – no', 'O/U (over / under) – over 1.5 goals', \
                'O/U (over / under) – under 1.5 goals', 'O/U (over / under) – over 2.5 goals', 'O/U (over / under) – under 2.5 goals', \
                'CS (correct score) – 1:0', 'CS (correct score) – 2:1', 'CS (correct score) – 1:1', 'CS (correct score) - all others', \
                # all as is, but 'CS (correct score) - all others' must be generated form 1:0, 2:1, and 1:1 socres.
                # Asian Handicap odds are excluded, as significant portion of their values are empty. They are not used as queries.
        ], \
        'situationCols' : ['year', 'month', 'dayofweek'], # generated \
        #-------------------------------------------------------------------------------------------------------------
        'maxOdds' : super_config['.M.Odds'], \
        'treat_maxOdds' : 'replace', # 'drop' or 'replace' \
        '.M.Score' : super_config['.M.Score'], \
        'treat_maxGoals' : 'drop', # 'drop' or 'replace' \
        }, \
        #================================================================================================================================================================
        # 12
        # The columns of the following column names are either found in the excel file or created by EnhanceReadExcelFileForArrays(.)
         { \
        #'past_data_path' : "contest_past_data_partial.xlsx", \
        'past_data_path' : "contest_train_and_valid.xlsx", \
        'past_data_bin_path' : "contest_train_and_valid.bin", \
        'future_data_path' : "contest_testing_data.xlsx", \
        'past_data_generated' : 'contest_past_data_generated.xlsx', \
        #--------------------------------------------------------------------------------------- database-specific columns
        'leagueCountryCol' : ["league_country"], # as is, 'Unknown' - filled \
        'homeCountryCol' : ["home_team_country"], # as is, 'Unknown' - filled \
        'awayCountryCol' : ["away_team_country"], # as is, 'Unknown' - filled \
        'outcomeCol' : ['outcome'], # as is, [0,1,2] -> [-1, 0, 1] \
        #--------------------------------------------------------------------------------------- standard columns
        'leagueCol' : ['league_id'], # as is, srting-ized \
        'homeCol' : ['teams_home_team_id'], # as is, srting-ized \
        'awayCol' : ['teams_away_team_id'], # as is, srting-ized \
        'oddCols' : [ 'winning_odds_home', 'winning_odds_draw', 'winning_odds_away' ], # generated \,
        'situationCols' : ['year', 'month', 'dayofweek', # generated \
                
                'goals_home', 'goals_away', # as is, mean - filled \

                'winning_percent_home',	'winning_percent_draws', 'winning_percent_away', \
                
                #'comparison_forme_home', 'comparison_forme_away', 'comparison_att_home', \
                #'comparison_att_away', 'comparison_def_home', 'comparison_def_away', 'comparison_fish_law_home', 'comparison_fish_law_away', 'comparison_h2h_home', \
                #'comparison_h2h_away', 'comparison_goals_h2h_home', 'comparison_goals_h2h_away', 'teams_home_last_5_matches_forme', 'teams_home_last_5_matches_att', \
                #'teams_home_last_5_matches_def', 'teams_away_last_5_matches_forme', 'teams_away_last_5_matches_att', 'teams_away_last_5_matches_def', \
                #'teams_home_last_5_matches_goals', 'teams_home_last_5_matches_goals_avg', 'teams_home_last_5_matches_goals_against', 'teams_home_last_5_matches_goals_against_avg', \
                #'teams_home_all_last_matches_matchs_matchsPlayed_home', 'teams_home_all_last_matches_matchs_matchsPlayed_away', 'teams_home_all_last_matches_matchs_wins_home', \
                #'teams_home_all_last_matches_matchs_wins_away', 'teams_home_all_last_matches_matchs_draws_home', 'teams_home_all_last_matches_matchs_draws_away', \
                #'teams_home_all_last_matches_matchs_loses_home', 'teams_home_all_last_matches_matchs_loses_away', 'teams_home_all_last_matches_goals_goalsFor_home', \
                #'teams_home_all_last_matches_goals_goalsFor_away', 'teams_home_all_last_matches_goals_goalsAgainst_home', 'teams_home_all_last_matches_goals_goalsAgainst_away', \
                #'teams_home_all_last_matches_goalsAvg_goalsFor_home', 'teams_home_all_last_matches_goalsAvg_goalsFor_away', 'teams_home_all_last_matches_goalsAvg_goalsAgainst_home', \
                #'teams_home_all_last_matches_goalsAvg_goalsAgainst_away', 'teams_home_last_h2h_played_home', 'teams_home_last_h2h_played_away', 'teams_home_last_h2h_wins_home', \
                #'teams_home_last_h2h_wins_away', 'teams_home_last_h2h_draws_home', 'teams_home_last_h2h_draws_away', 'teams_home_last_h2h_loses_home', 'teams_home_last_h2h_loses_away', \
                #'teams_away_last_5_matches_goals', 'teams_away_last_5_matches_goals_avg', 'teams_away_last_5_matches_goals_against', 'teams_away_last_5_matches_goals_against_avg', \
                #'teams_away_all_last_matches_matchs_matchsPlayed_home', 'teams_away_all_last_matches_matchs_matchsPlayed_away', 'teams_away_all_last_matches_matchs_wins_home', \
                #'teams_away_all_last_matches_matchs_wins_away', 'teams_away_all_last_matches_matchs_draws_home', 'teams_away_all_last_matches_matchs_draws_away', \
                #'teams_away_all_last_matches_matchs_loses_home', 'teams_away_all_last_matches_matchs_loses_away', 'teams_away_all_last_matches_goals_goalsFor_home', \
                #'teams_away_all_last_matches_goals_goalsFor_away', 'teams_away_all_last_matches_goals_goalsAgainst_home', 'teams_away_all_last_matches_goals_goalsAgainst_away', \
                #'teams_away_all_last_matches_goalsAvg_goalsFor_home', 'teams_away_all_last_matches_goalsAvg_goalsFor_away', 'teams_away_all_last_matches_goalsAvg_goalsAgainst_home', \
                #'teams_away_all_last_matches_goalsAvg_goalsAgainst_away', 'teams_away_last_h2h_played_home', 'teams_away_last_h2h_played_away', 'teams_away_last_h2h_wins_home', \
                #'teams_away_last_h2h_wins_away', 'teams_away_last_h2h_draws_home', 'teams_away_last_h2h_draws_away', 'teams_away_last_h2h_loses_home', 'teams_away_last_h2h_loses_away', \
        ], \
        #------------------------------------------------------------------------------------------------------------
        'maxOdds' : super_config['.M.Odds'], \
        'treat_maxOdds' : 'replace', # or 'replace' \
        '.M.Score' : super_config['.M.Score'], \
        'treat_maxGoals' : 'drop', # or 'replace' \
        }, \
        #================================================================================================================================================================
        #================================================================================================================================================================
        # 13
        # The columns of the following column names are either found in the excel file or created by EnhanceReadExcelFileForArrays(.)
         { \
        'past_data_path' : "thomas_past_data Nov 09.xlsx", \
        'past_data_bin_path' : "thomas_past_data Nov 09.bin", \
        'future_data_path' : "thomas_testing_data.xlsx", \
        'past_data_generated' : 'thomas_past_data_generated.xlsx', \
        #--------------------------------------------------------------------------------------- database-specific columns
        'goalCols' : ['hGoal', 'aGoal'], # generated \
        'p_goalCols' : ['pHGoal', 'pAGoal'], # generated \
        #--------------------------------------------------------------------------------------- standart columns
        'leagueCol' : ['League'], # as is \
        'homeCol' : ['Home'], # generated \
        'awayCol' : ['Away'], # generated \
        'oddCols' : ['Home Win', 'Draw', 'Away Win', 'Both teams to score - yes', 'Both teams to score – no', 'O/U (over / under) – over 1.5 goals', \
                'O/U (over / under) – under 1.5 goals', 'O/U (over / under) – over 2.5 goals', 'O/U (over / under) – under 2.5 goals', \
                'CS (correct score) – 1:0', 'CS (correct score) – 2:1', 'CS (correct score) – 1:1', 'CS (correct score) - all others', \
                # all as is, but 'CS (correct score) - all others' must be generated form 1:0, 2:1, and 1:1 socres.
                # Asian Handicap odds are excluded, as significant portion of their values are empty. They are not used as queries.
        ], \
        'situationCols' : ['year', 'month', 'dayofweek'], # generated \
        #-------------------------------------------------------------------------------------------------------------
        'maxOdds' : super_config['.M.Odds'], \
        'treat_maxOdds' : 'replace', # 'drop' or 'replace' \
        '.M.Score' : super_config['.M.Score'], \
        'treat_maxGoals' : 'drop', # 'drop' or 'replace' \
        }, \
        #================================================================================================================================================================
        # 14
        # The columns of the following column names are either found in the excel file or created by EnhanceReadExcelFileForArrays(.)
         { \
        'past_data_path' : "thomas_past_data Dec 04.xlsx", \
        'past_data_bin_path' : "thomas_past_data Dec 04.bin", \
        'future_data_path' : "thomas_testing_data.xlsx", \
        'past_data_generated' : 'thomas_past_data_generated.xlsx', \
        #--------------------------------------------------------------------------------------- database-specific columns
        'goalCols' : ['hGoal', 'aGoal'], # generated \
        'p_goalCols' : ['pHGoal', 'pAGoal'], # generated \
        #--------------------------------------------------------------------------------------- standart columns
        'leagueCol' : ['League'], # as is \
        'homeCol' : ['Home'], # generated \
        'awayCol' : ['Away'], # generated \
        'oddCols' : ['Home Win', 'Draw', 'Away Win', 'Both teams to score - yes', 'Both teams to score – no', 'O/U (over / under) – over 1.5 goals', \
                'O/U (over / under) – under 1.5 goals', 'O/U (over / under) – over 2.5 goals', 'O/U (over / under) – under 2.5 goals', \
                'CS (correct score) – 1:0', 'CS (correct score) – 2:1', 'CS (correct score) – 1:1', 'CS (correct score) - all others', \
                # all as is, but 'CS (correct score) - all others' must be generated form 1:0, 2:1, and 1:1 socres.
                # Asian Handicap odds are excluded, as significant portion of their values are empty. They are not used as queries.
        ], \
        'situationCols' : ['year', 'month', 'dayofweek'], # generated \
        #-------------------------------------------------------------------------------------------------------------
        'maxOdds' : super_config['.M.Odds'], \
        'treat_maxOdds' : 'replace', # 'drop' or 'replace' \
        '.M.Score' : super_config['.M.Score'], \
        'treat_maxGoals' : 'drop', # 'drop' or 'replace' \
        }, \
]

#         'past_data_path' : "GetODDs V.1.0. np - July 26 Refined - 2.csv", \

def PrepareData( dataUse, dictionaries = None, from_save = False ) :

        data = None; dataUse_loaded = None

        compiled_db_path = 'Database_Final_Binary_' + str( dataUse['DB.Id'] ) + '_M' + str( dataUse['M.Day'] ).zfill(2) + '.bin'

        if from_save :
                dataPath = compiled_db_path
                compiled_db = LoadBinaryData( dataPath )
                if compiled_db is not None :
                        data, dataUse_updated, dictionaries_updated = compiled_db

        if data is None or dataUse_updated is None or dictionaries_updated is None :

                data = GetCSVData( dataUse ) # 1

                data = CorrectSpells( data, dictionaries ) # 3

                data, dictionaries_updated = Tokenize( data, dictionaries ) # 4

                data = AugmentData( data ) # 5

                data, dictionaries_updated = Select_Major_Games( data, dataUse, dictionaries_updated ) # 6

                data = AddColRowId( data, dataUse ) # 7

                SaveCSVData( 'Final_data_before_split.xlsx', data ) # 2

                data, dataUse_updated = SplitData( data, dataUse ) # 8

                #data = ResampleData( data, dataUse ) # 9

                #data = SelectRelevantGames( data, dataUse ) # 10

                SaveBinaryData( ( data, dataUse_updated, dictionaries_updated ), compiled_db_path ) # 10

        return data, dataUse_updated, dictionaries_updated


# 1
def GetCSVData( dataUse ) :

        database_id = dataUse['DB.Id']

        file_name = dataSchemes[database_id]['past_data_path']
        file_path = os.path.join( super_config['Data'], file_name )

        dataScheme = dataSchemes[database_id]

        data = EnhanceReadExcelFileForArrays( file_path, dataScheme, dataUse )

        return data

# 1.1
def EnhanceReadExcelFileForArrays( file_path, dataScheme, dataUse ) :

        binPath = 'Database_Excel_Binary_' + str(dataUse['DB.Id']) + '.bin'
        bin = LoadBinaryData( binPath )
        if bin is not None : 
                dataFrame = bin
        else :
                dataFrame = pd.read_excel(file_path)
                SaveBinaryData( dataFrame, binPath )

        #========== Replace "" with NaN.
        nan_value = float( "NaN" )
        dataFrame.replace( "", nan_value, inplace=True )
        dataFrame.replace( "N", nan_value, inplace=True )
        dataFrame.replace( "NA", nan_value, inplace=True )

        #============================================================================================ Enhance and refine dataFrame =================================

        erows = [ row for row in range( 2, dataFrame.shape[0] + 2 ) ] # 2, because Excel rows starts with 2 rather than 0.
        dataFrame.insert( 0, 'erows', erows )

        #m_date = super_config['.date_org'] + datetime.timedelta( days = super_config['.M.Day'] )
        #dataUse['C.Date'] = super_config['.date_org'] + datetime.timedelta( days = super_config['.C.Day'] )

        dates = [ date( year, month, day ) for year, month, day in \
                zip( pd.DatetimeIndex( dataFrame['Date'] ).year, pd.DatetimeIndex( dataFrame['Date'] ).month, pd.DatetimeIndex( dataFrame['Date'] ).day ) ]
        dataFrame.insert( 1, 'date', dates )

        dataFrame.sort_values( by = ['date'], ascending = False, inplace = True ) #----------------------------- Sort by Date.

        #==========  Reverse the row order.
        dataFrame = dataFrame.iloc[::-1]

        #--------------------------------------------------------------------------------------------------------------------------

        if super_config['Contest'] : 

                #========== Fill in empty country names.
                dataFrame.loc[ pd.isnull( dataFrame['league_country'] ), 'league_country' ] = 'Unknown'
                dataFrame.loc[ pd.isnull( dataFrame['home_team_country'] ), 'home_team_country' ] = 'Unknown'
                dataFrame.loc[ pd.isnull( dataFrame['away_team_country'] ), 'away_team_country' ] = 'Unknown'

                #========== [0,1,2] -> [-1,0,1]: 2->-1 ( home_win : 1, draw : 0, away_win : -1 )
                dataFrame.loc[ dataFrame['outcome'] == 2, 'outcome' ] = -1

                #========== String-ize league_id, home_id, and away_id
                dataFrame[ 'league_id' ] =  [ str( league_number ) for league_number in dataFrame[ 'league_id' ]  ]
                dataFrame[ 'teams_home_team_id' ] = [ str( home_number ) for home_number in dataFrame[ 'teams_home_team_id' ]  ]
                dataFrame[ 'teams_away_team_id' ] = [ str( away_number ) for away_number in dataFrame[ 'teams_away_team_id' ]  ]

                #========== Add odds for home, draw, and away.
                bfr = super_config['.brate'] # eg. 0.05. bfr == bookie_fee_rate. The percentage of ( 1/p -1 ), or the bookies share in the amount that the bettor actually creates.
                wp = dataFrame['winning_percent_home']; loc = dataFrame.columns.get_loc('winning_percent_home') + 1
                dataFrame.insert( loc, 'winning_odds_home', 100 / ( wp + 0.5 ) * ( 1.0 - bfr ) + bfr )
                wp = dataFrame['winning_percent_draws']; loc = dataFrame.columns.get_loc('winning_percent_draws') + 1
                dataFrame.insert( loc, 'winning_odds_draw', 100 / ( wp + 0.5 ) * ( 1.0 - bfr ) + bfr )
                wp = dataFrame['winning_percent_away']; loc = dataFrame.columns.get_loc('winning_percent_away') + 1
                dataFrame.insert( loc, 'winning_odds_away', 100 / ( wp + 0.5 ) * ( 1.0 - bfr ) + bfr )

                #========== Add year, month, and ndow.
                loc = dataFrame.columns.get_loc( 'fixture_date' )
                dataFrame.insert( loc + 1, 'year', pd.DatetimeIndex( dataFrame['fixture_date'] ).year - 2019  )
                dataFrame.insert( loc + 2, 'month', pd.DatetimeIndex( dataFrame['fixture_date'] ).month - 6 )
                dataFrame.insert( loc + 3, 'dayofweek', pd.DatetimeIndex( dataFrame['fixture_date'] ).dayofweek - 3 )

                #========== Fill in empty goals_*
                dataFrame.loc[ pd.isnull(dataFrame['goals_home']), 'goals_home' ] = dataFrame['goals_home'].mean()
                dataFrame.loc[ pd.isnull(dataFrame['goals_away']), 'goals_away' ] = dataFrame['goals_away'].mean()


        elif not super_config['Contest'] :

                def Pick_Goals( string, delimeter ) :
                        hGoal = aGoal = nan_value
                        if not pd.isnull( string ) : 
                                search = re.search( "\d+\s*" + delimeter + "\s*\d+", string )
                                if search is not None :
                                        match = search.group()
                                        hGoal = int( re.search( "^\d*", match ).group() )
                                        aGoal = int( re.search( "\d*$", match ).group() )
                                else : raise Exception( "hGoal" + delimeter + "aGoal" + " is not well formed in '{}'".format( string ) )
                        return hGoal, aGoal

                def Pick_Teams( string, delimeter1, delimeter2 = None ) : # "\s-\s"
                        home = away = nan_value
                        if not pd.isnull( string ) : 
                                search = re.search( ".*\s+" + delimeter1 + "\s+.*", string )
                                if search is None and delimeter2 is not None :
                                       search = re.search( ".*\s+" + delimeter2 + "\s+.*", string )
                                if search is not None :
                                        match = search.group()
                                        search = re.search( " " + delimeter1 + " ", match )
                                        if search is None and delimeter2 is not None :
                                                search = re.search( " " + delimeter2 + " ", match )
                                        if search is not None :
                                                home = match[ 0 : search.span()[0] ].strip().upper()
                                                away = match[ search.span()[1] : ].strip().upper()
                                        else : raise Exception( "Home " + delimeter1 + "/" + delimeter2 + " Away" + "is not well formed in {}".format( string ) )
                                else: raise Exception( "Home " + delimeter1 + "/" + delimeter2 + " Away" + "is not well formed in '{}'".format( string ) )
                        else : home = nan_value; away = nan_value
                        return home, away


                #========== Generate hGoal and aGoal.
                goal_list = [ Pick_Goals( string, r":" ) for string in dataFrame['Actual Result'] ]
                loc = dataFrame.columns.get_loc('Actual Result')
                dataFrame.insert( loc + 1, "hGoal", [ hGoal for ( hGoal, _ ) in goal_list ] )
                dataFrame.insert( loc + 2, "aGoal", [ aGoal for ( _, aGoal ) in goal_list ] )

                #========== Generate pHGoal and pAGoal.
                goal_list = [ Pick_Goals( string, r"-" ) for string in dataFrame['Predicted Result'] ]
                loc = dataFrame.columns.get_loc('Predicted Result')
                dataFrame.insert( loc + 1, "pHGoal", [ hGoal for ( hGoal, _ ) in goal_list ] )
                dataFrame.insert( loc + 2, "pAGoal", [ aGoal for ( _, aGoal ) in goal_list ] )

                #==========  Capitalize League
                leagues = [ league.upper() for league in dataFrame['League'] ]
                dataFrame.loc[ :,  'League' ] = leagues

                #==========  Generate Home and Away.
                team_list = [ Pick_Teams( string, r"vs", r"-" ) for string in dataFrame['Home Vs Away'] ]
                loc = dataFrame.columns.get_loc('Home Vs Away')
                dataFrame.insert( loc + 1, "Home", [ home for ( home, _ ) in team_list ] )
                dataFrame.insert( loc + 2, "Away", [ away for ( _, away ) in team_list ] )

                #========== Generate 'CS (correct score) - all others', which is an Odds column.

                def Probability( Odds, bfr ) : return 1.0 / ( Odds - bfr )

                bfr = super_config['.brate'] # bfr == bookie_fee_rate. The percentage of ( 1/p -1 ), or the bookies share in the amount that the bettor actually creates.
                p10 = Probability( dataFrame['CS (correct score) – 1:0'], bfr )
                p21 = Probability( dataFrame['CS (correct score) – 2:1'], bfr )
                p11 = Probability( dataFrame['CS (correct score) – 1:1'], bfr )
                p_all_others = 1.0 - ( p10 + p21 + p11 )
                wp = p_all_others * 100
                odds_all_others = 100 / ( wp + 0.5 ) * ( 1.0 - bfr ) + bfr
                assert np.min( odds_all_others ) > 1.0
                loc = dataFrame.columns.get_loc('CS (correct score) – 1:1') + 1
                dataFrame.insert( loc, 'CS (correct score) - all others', odds_all_others )

                #========== Add year, month, and ndow.
                loc = dataFrame.columns.get_loc( 'Date' )
                dataFrame.insert( loc + 1, 'year', pd.DatetimeIndex( dataFrame['Date'] ).year - 2019  )
                dataFrame.insert( loc + 2, 'month', pd.DatetimeIndex( dataFrame['Date'] ).month - 6 )
                dataFrame.insert( loc + 3, 'dayofweek', pd.DatetimeIndex( dataFrame['Date'] ).dayofweek - 3 )

                # NOTE : This is a very early place to drop incomplete data rows. All rows that will not be used are dropped here. What if we delay this drop till team/league names are catalogued?
                dataFrame.dropna( axis = 0, subset = dataScheme['situationCols'] + dataScheme['goalCols'] + dataScheme['p_goalCols'] + dataScheme['oddCols'], inplace=True )

        #------------------------------------------------------------------------ Regularize odds columns.
        if dataScheme['treat_maxOdds'] == 'drop' :
                for oddCol in dataScheme['oddCols']: dataFrame.loc[ dataFrame[oddCol] > dataScheme['maxOdds'] , [oddCol] ] = nan_value
                dataFrame.dropna( axis = 0, subset = dataScheme['oddCols'], inplace=True )
        elif dataScheme['treat_maxOdds'] == 'replace' :
                for oddCol in dataScheme['oddCols']: dataFrame.loc[ dataFrame[oddCol] > dataScheme['maxOdds'] , [oddCol] ] = dataScheme['maxOdds']
        else : raise Exception('Wrong data scheme.')


        if not super_config['Contest'] : # database-specific. ------------------- Regularize goals for thomas database.
                if dataScheme['treat_maxGoals'] == 'drop' :
                        for goalCol in dataScheme['goalCols']: dataFrame.loc[ dataFrame[goalCol] > dataScheme['.M.Score'] , [goalCol] ] = nan_value
                        dataFrame.dropna( axis = 0, subset = dataScheme['goalCols'], inplace=True )
                elif dataScheme['treat_maxGoals'] == 'replace' :
                        for goalCol in dataScheme['goalCols']: dataFrame.loc[ dataFrame[goalCol] > dataScheme['.M.Score'] , [goalCol] ] = dataScheme['.M.Score']
                else : raise Exception('Wrong data scheme.')

        #---------------------------------- Drop unused columns, and optionally save dataFrame for visual examination.

        columns_used = []
        for col in [ 'leagueCol', 'homeCol', 'awayCol', 'oddCols', 'situationCols' ] : columns_used += dataScheme[ col ]
        columns_used += ['erows', 'date']

        if super_config['Contest'] :
                for col in [ 'leagueCountryCol', 'homeCountryCol', 'awayCountryCol', 'outcomeCol' ] : columns_used += dataScheme[ col ]
        elif not super_config['Contest'] :
                for col in [ 'goalCols', 'p_goalCols' ] : columns_used += dataScheme[ col ]

        columns_unused = [ col for col in dataFrame.columns if col not in columns_used ]
        dataFrame.drop( axis = 1, columns = columns_unused ) # PENDING - They are not dropped and remain, when seved to excel.

        # dataFrame.to_excel( dataScheme['past_data_generated'] ) # Save back to a file for visual exam. Takes too long. Mask it for the moment.


        #====================================================================================== Collect np.arrays ===========================================

        #---------------------------------------------------- Standard columns

        erows = np.array( dataFrame[ 'erows' ], dtype = np.int64 ); erows = np.reshape( erows, (-1) )
        dates = np.array( dataFrame[ 'date' ] ); dates = np.reshape( dates, (-1) )
        leagues = np.array( dataFrame[ dataScheme['leagueCol'] ], dtype = np.str ); leagues = np.reshape( leagues, (-1) )
        homes = np.array( dataFrame[ dataScheme['homeCol'] ], dtype = np.str ); homes = np.reshape(homes, (-1) )
        aways = np.array( dataFrame[ dataScheme['awayCol'] ], dtype = np.str ); aways = np.reshape(aways, (-1) )
        odds = np.array( dataFrame[ dataScheme['oddCols'] ], dtype = np.float32 ) # No: odds = np.reshape( odds, (-1) )
        situation = np.array( dataFrame[ dataScheme['situationCols'] ], dtype = np.float32 ) # No: situation = np.reshape( situation, (-1) )

        #---------------------------------------------------- Database-specific columns

        if super_config['Contest'] :

                lCountry = np.array(dataFrame[ dataScheme['leagueCountryCol'] ], dtype = np.str ); lCountry = np.reshape( lCountry, (-1) )
                hCountry = np.array(dataFrame[ dataScheme['homeCountryCol'] ], dtype = np.str ); hCountry = np.reshape( hCountry, (-1) )
                aCountry = np.array(dataFrame[ dataScheme['awayCountryCol'] ], dtype = np.str ); aCountry = np.reshape( aCountry, (-1) )

                outcome = np.array( dataFrame[ dataScheme['outcomeCol'] ], dtype = np.float32 ); outcome = np.reshape( outcome, (-1) )

                data = { 'leagues': leagues, 'homes': homes, 'aways': aways, 'situation': situation, 'odds': odds, 'lCountries': lCountry, 'hCountries': hCountry, 'aCountries' : aCountry, 'outcome' : outcome }
                data[ 'featureNames' ] = [ 'erows', 'leagues', 'homes', 'aways', 'situation', 'odds', 'lCountries', 'hCountries', 'aCountries', 'outcome' ]

        elif not super_config['Contest'] :

                goals = np.array( dataFrame[ dataScheme['goalCols'] ], dtype = np.float32 ) # No: goals = np.reshape( goals, (-1) )
                p_goals = np.array( dataFrame[ dataScheme['p_goalCols'] ], dtype = np.float32 ) # No: p_goals = np.reshape( p_goals, (-1) )

                data = { 'erows': erows, 'dates' : dates, 'leagues': leagues, 'homes': homes, 'aways': aways, 'situation': situation, 'odds': odds, 'goals': goals, 'p_goals': p_goals }
                data[ 'featureNames' ] = [ 'erows', 'dates', 'leagues', 'homes', 'aways', 'situation', 'odds', 'goals', 'p_goals' ]

        return data

# 2
def SaveCSVData( path, data ) :
        #dataframe = pd.DataFrame( data )
        #dataframe.to_excel( path )
        pass

# 3
def CorrectSpells( data, dictionaries ) :

        lenth = len( data['leagues'] )

        variations = dictionaries.get( 'name_variations', None )

        if super_config['Contest'] :
                string_features = [ 'leagues', 'homes', 'aways', 'lCountries', 'hCountries', 'aCountries' ]
        elif not super_config['Contest'] :
                string_features = [ 'leagues', 'homes', 'aways' ]

        for n in range( lenth ) :
                for feature in string_features :
                        data[feature][n] = Regularize( data[feature][n] )                      
                        if variations is not None :  data[feature][n] = Find_Formal_Name( data[feature][n], variations )

        return data


def Regularize( string ):
        string = string.upper()
        string = re.sub( '^\s+', '', string )
        string = re.sub( '\s+$', '', string )
        return string

def Find_Formal_Name( string, variations ) :
        for key, list_val in variations.items() :
                if string in list_val :
                        string = key
        return string

# 4
def Tokenize( data, dictionaries ) :

        #if dictionaries.get( 'leagues', None ) is None : 
        #        string_array = data['leagues']
        #        dictionaries['leagues'] = CreateStringDict( string_array )

        variations = dictionaries.get( 'name_variations', None )
       
        data['leagues'], dictionaries['leagues'] = Translate( data['leagues'], dictionaries['leagues'], variations )
        data['nLeagues'] = len( dictionaries['leagues'] ) # nLeagues will not change even if some leagues will be removed from data. league_ids will spread all over [0, nLeagues).
        data['leagues_dic'] = dictionaries['leagues']

        #if dictionaries.get( 'teams', None ) is None : 
        #        string_array = np.concatenate( [ data['homes'], data['aways'] ], axis = 0 )
        #        dictionaries['teams'] = CreateStringDict( string_array )
       
        data['homes'], dictionaries['teams'] = Translate( data['homes'], dictionaries['teams'], variations )
        data['aways'], dictionaries['teams'] = Translate( data['aways'], dictionaries['teams'], variations )
        data['nTeams'] = len( dictionaries['teams'] ) # nTeams will not change even if some teams will be removed from data. team_ids will spread all over [0, nTeams).
        data['teams_dic'] = dictionaries['teams']

        if super_config['Contest'] :
                data['lCountries'], dictionaries['countries'] = Translate( data['lCountries'], dictionaries['countries'], variations )
                data['hCountries'], dictionaries['countries'] = Translate( data['hCountries'], dictionaries['countries'], variations )
                data['aCountries'], dictionaries['countries'] = Translate( data['aCountries'], dictionaries['countries'], variations )
                data['nCountries'] = len( dictionaries['countries'] ) # nCountries will not change even if some countries will be removed from data. country_ids will spread all over [0, nCountries).
                data['countries_dic'] = dictionaries['countries']

        return data, dictionaries

# 4.1
def Translate( string_array, existing_dict, variations ) :

        #--------------------------------------------------------------------------------------------------------------
        #
        #       Design principle : Preserve existing part of dictionary, and APPEND ONLY.
        #
        #       Do NOT allow a name to change its token from time to time. 
        #       Do not revoke tokens from removed names.
        #       Newly introduced names can only have a newly introduced token.


        index_max = -1
        for string, index in existing_dict.items() :
                if index_max < index : index_max = index
        assert index_max + 1 == len( existing_dict ) # Not very strict restrict. The order of indices is not checked. Note: python dictionary now preserves the order of elements.

        tokenized_list = []
        for string in string_array :
                index = existing_dict.get( string, -1 )
                if index >= 0 :
                        tokenized_list.append( index )
                else: 
                        index = len( existing_dict ) # takes time? but elegant :) There will be few times to do this.
                        string = Regularize( string )
                        if variations is not None : string = Find_Formal_Name( string, variations )
                        existing_dict[ string ] = index
                        tokenized_list.append( index ) # PENDING or NOTE : a place for unknown strings.

        dictionary_updated = existing_dict

        return np.array( tokenized_list ), dictionary_updated

# 5
def AugmentData( data ) :

        return data

# 6
def Select_Major_Games( data, dataUse, dictionaries ) :

        # Premise: data is arranged in ascending order of game ids.

        nGames = len( data['homes'] ); homes = data['homes']; aways = data['aways']
        nTeams = max( max( homes ), max( aways ) ) + 1 # len( set( homes ).union( set( aways ) ) )

        major_teams_list, dictionaries = Get_Major_Clustered_Teams( nGames, nTeams, homes, aways, dataUse, dictionaries )
        major_teams = set( major_teams_list )


        multi_game_teams = set( Get_Multi_Game_Teams( homes, aways, minGames = dataUse['.m.Attends'] ) ) # Attend at least 'minGames' games, either as home or away.
        major_teams = major_teams.intersection( multi_game_teams )
        major_game_ids = [ gameId for gameId in range( len(homes) ) if ( homes[gameId] in major_teams ) and ( aways[gameId] in major_teams ) ]
        major_game_ids.sort()

        def Select_Major( feature ): data[ feature ] = data[ feature ][ major_game_ids ]

        for feature in data['featureNames'] : Select_Major( feature )
        
        return data, dictionaries

# 6.1
def Get_Major_Clustered_Teams( nGames, nTeams, homes, aways, dataUse, dictionaries ) :

        data = np.array( [ 1 ] * nGames ) # a game between 'home' and 'away' teams is a connection/edge between the nodes 'home' and 'away'
        
        graph = csr_matrix( ( data, ( homes, aways ) ), shape = ( nTeams, nTeams ) ) 
        # homes[game_id] : the id of home team that palyed in the game of game_id.

        n_components, labels = connected_components( csgraph = graph, directed = False, return_labels = True )

        unique, counts = np.unique( list( labels ), return_counts = True )
        
        total = np.sum( counts )
        labels_of_major_groups = np.where( counts >= total * dataUse[ 'm.cluster.size' ] )[0]

        indices = []
        if len( labels_of_major_groups ) > 0 :
                indice_set = set()
                for n in range( len( labels_of_major_groups) ) : 
                        indice_set = indice_set.union( set( np.where( labels == labels_of_major_groups[ n ] )[0] ) )
                indices = sorted( list ( indice_set ) )

        # labels[team_id] : the label of the component that the team of team_id belongs to.
        # dictionaries['teams'] = { team_name : team_id }
        unique_counts = np.stack( [ unique, counts ], axis = 0 )
        unique_counts_sorted = unique_counts[ :, np.flip( unique_counts[ 1, : ].argsort() ) ] # sort all, including 'unique' component labels, in the ascending order of 'counts'.
       
        teams_dict = dictionaries['teams']
        component_dict = {}; cluster_id = -1
        for component_id_idx in range( len( unique_counts_sorted[ 0, : ] ) ) : #
                component_id = unique_counts_sorted[ 0, component_id_idx ].item()
                team_list = []
                for label_idx in range( len( labels ) ) : # label_is serves as team_id. labels[team_id] : the label of the component that the team of team_id belongs to
                        if component_id == labels[ label_idx ] :
                                for team_name, team_id in teams_dict.items() :
                                        if team_id == label_idx :
                                                team_list.append( team_name )
                cluster_id += 1
                component_dict[ str( cluster_id ) ] = team_list # Use cluster_id, instead of component_id, beacues component_dict would show items in key order.
        # dictionaries['clusters'] = component_dict

        return indices, dictionaries # ordered list of team ids.


# 6.2
def Get_Multi_Game_Teams( homes, aways, minGames ) : # Teams that attended no game or just a single game. They are of no help.

        unique, counts = np.unique( np.concatenate( [ homes, aways ], axis = -1 ), return_counts = True )
        multi_game_indices = np.where( counts >= minGames )
        multi_game_teams = unique[ multi_game_indices[0] ]

        multi_game_teams.sort()

        return multi_game_teams # ordered list of team ids. 

# 7
def AddColRowId ( data, dataUse ) :

        database_id = dataUse['DB.Id']

        #dataScheme = dataSchemes[database_id]
        #oddCols = dataScheme['oddCols']
        #nOdds = len(oddCols)
        #data['cols'] = np.array( [ [ n for n in range( nOdds ) ] for _ in range( data['odds'].shape[0] ) ] )

        data['rows'] = np.array( range( data['odds'].shape[0] ) )

        data[ 'featureNames' ].append( 'rows' )

        return data

# 8
def SplitData( data, dataUse ) :

        date_org = date( super_config['year.org'], super_config['month.org'], super_config['day_org'] )
        date_model = date_org + timedelta( days = super_config['.M.Day'] )

        rows_train_valid = np.where( data['dates'] < date_model )[0]
        rows_test = np.where( data['dates'] >= date_model )[0]

        shrink = 1.0 * len( rows_train_valid ) / ( dataUse['.nValid'] + dataUse['.nTrain'] )
        nRows_train = int( shrink * dataUse['.nTrain'] )
        assert nRows_train > 0 and nRows_train < len( rows_train_valid )

        # Assumption : data is sorted in dates. If not, sort them immediately after reading them from a file.

        rows_train = rows_train_valid[ : nRows_train ]
        rows_valid = rows_train_valid[ nRows_train : ]
        rows_test = rows_test

        for feature in data['featureNames'] :

                featureData = data[ feature ]
                data[ feature + 'Train' ] = featureData[ rows_train ]
                data[ feature + 'Valid' ] = featureData[ rows_valid ]
                data[ feature + 'Test' ] = featureData[ rows_test ]

                data.pop( feature )

        return data, dataUse


def SplitData_Old_Save( data, dataUse ) :

        date_org = date( super_config['year.org'], super_config['month.org'], super_config['day_org'] )
        date_model = date_org + timedelta( days = super_config['.M.Day'] )

        rows_train_valid = np.where( data['dates'] < date_model )[0]
        shrink = 1.0 * len( rows_train_valid ) / ( dataUse['.nValid'] + dataUse['.nTrain'] )
        nRows_train = int( shrink * dataUse['.nTrain'] )
        assert nRows_train > 0 and nRows_train < len( rows_train_valid )

        # Assumption : data is sorted in dates. If not, sort them immediately after reading them from a file.

        rows_train = rows_train_valid[ : nRows_train ]
        rows_valid = rows_train_valid[ nRows_train : ]

        rows_test = []; erows_test = []
        date_a = date_model
        worth_try = ( len( np.where( data['dates'] >= date_a )[0] ) > 0 )
        while worth_try :
                date_b = date_a + timedelta( days = 1 )
                rows = np.where( np.logical_and( date_a <= data['dates'], data['dates'] < date_b ) )[0]
                rows_test.append( rows ) # whether empty or not.
                erows_test.append( data['erows'][ rows ] )
                date_a = date_b
                worth_try = ( len( np.where( data['dates'] >= date_a )[0] ) > 0 )

        # rows_test is a list of arrays of possibly different lengths, which are row ids.

        for feature in data['featureNames'] :

                featureData = data[ feature ]
                data[ feature + 'Train' ] = featureData[ rows_train ]
                data[ feature + 'Valid' ] = featureData[ rows_valid ]

                data[ feature + 'Test' ] = []
                for rows in rows_test :
                        data[ feature + 'Test' ].append( featureData[ rows ] )

                data.pop( feature )

        return data, dataUse


def ReplaceDataSegment( replace_data, with_data, segment ) :

        for feature in replace_data['featureNames'] :
                key = feature + segment[0].upper() + segment[1:]
                replace_data[ key ] = with_data[ key ]

        return


# 9
def ResampleData( data, dataUse ) :

        if super_config['Contest'] : return data

        def swap_row( segment1, idx1, segment2, idx2 ) :
                for feature in data['featureNames'] :
                        array1 = feature + segment1
                        array2 = feature + segment2
                        row = data[ array1 ][ idx1 ]
                        data[ array1 ][ idx1 ] = data[ array2 ][ idx2 ]
                        data[ array2 ][ idx2 ] = row
                return

        def find_game_with_same_teams_in_decreasing_order( segment, home, away, indices_to_exclude ) :
                homes = data[ 'homes' + segment ]; aways = data[ 'aways' + segment ]
                idx_return = -1
                for idx in range( len(homes)-1, -1, -1 ) :
                        if idx in indices_to_exclude : continue
                        if homes[ idx ] == home and aways[ idx ] == away :
                               idx_return = idx; break
                return idx_return

        def find_game_with_half_same_team_in_decreasing_order( segment, home, away, indices_to_exclude, homes_relevant, aways_relevant ) :
                homes = data[ 'homes' + segment ]; aways = data[ 'aways' + segment ]
                idx_return = -1
                for idx in range( len(homes)-1, -1, -1 ) :
                        if idx in indices_to_exclude : continue
                        if ( homes[ idx ] == home and aways[ idx ] in aways_relevant ) or ( homes[ idx ] in homes_relevant and aways[ idx ] == away ) :
                               idx_return = idx; break
                return idx_return
        
        def find_game_with_relevant_teams_in_decreasing_order( segment, home, away, indices_to_exclude, homes_relevant, aways_relevant ) :
                homes = data[ 'homes' + segment ]; aways = data[ 'aways' + segment ]
                idx_return = -1
                for idx in range( len(homes)-1, -1, -1 ) :
                        if idx in indices_to_exclude : continue
                        if homes[ idx ] in homes_relevant and aways[ idx ] in aways_relevant :
                               idx_return = idx; break
                return idx_return

        def find_any_remaining_game( segment, indices ) :
                indices_seg = list( range( len( data[ 'homes' + segment ] ) ) )
                remaining_indices = [ idx for idx in indices_seg if idx not in indices ]
                idx_return = -1
                if len( remaining_indices ) > 0 : idx_return = remaining_indices[0]
                return idx_return

        # swap_row( 'Valid', 3, 'Test', 5 )
        homes_check = set( data[ 'homesCheck' ] )
        aways_check = set( data[ 'awaysCheck' ] )
        teams_check = homes_check.union( aways_check )

        test_indices_resolved = []
        check_indices_resolved = []
        valid_indices_engaged = []
        train_indices_engaged = []
        
        swap_plans = []

        #================================ Same team pair ===============================

        if find_any_remaining_game( 'Test', test_indices_resolved ) >= 0 : # Check is also not finished.

                #-------- Find Test games that have the same team pair with a Check game - test_indices_resolved
                #-------- Find Check games that have the same team pair with a Test game - check_indices_resolved.
                for check_idx in range( len( data['homesCheck'] ) ) :
                        home = data['homesCheck'][ check_idx ]; away = data['awaysCheck'][ check_idx ]
                        test_idx = find_game_with_same_teams_in_decreasing_order( 'Test', home, away, test_indices_resolved )
                        if test_idx >= 0 :
                                test_indices_resolved.append( test_idx )
                                check_indices_resolved.append( check_idx )

        if find_any_remaining_game( 'Test', test_indices_resolved ) >= 0 : # Check is also not finished.

                #--------- Find all swap plans [ ( 'Test', idx1, 'Valid', idx2 ) ]
                #--------- where game('Valid', idx) has the same team pair with an unresolved Check game
                #--------- and where game('Test', idx) is not resolved yet.
                for check_idx in range( len( data['homesCheck'] ) ) :
                        if check_idx in check_indices_resolved : continue
                        home = data['homesCheck'][ check_idx ]; away = data['awaysCheck'][ check_idx ]
                        valid_idx = find_game_with_same_teams_in_decreasing_order( 'Valid', home, away, valid_indices_engaged )
                        if valid_idx >= 0 :
                                test_idx = find_any_remaining_game( 'Test', test_indices_resolved )
                                if test_idx >= 0 :
                                        swap_plans.append( [ 'Test', test_idx, 'Valid', valid_idx ] )
                                        test_indices_resolved.append( test_idx )
                                        check_indices_resolved.append( check_idx )
                                        valid_indices_engaged.append( valid_idx )
                        else :
                                train_idx = find_game_with_same_teams_in_decreasing_order( 'Train', home, away, train_indices_engaged )
                                if train_idx >= 0 :
                                        test_idx = find_any_remaining_game( 'Test', test_indices_resolved )
                                        if test_idx >= 0 :
                                                swap_plans.append( [ 'Test', test_idx, 'Train', train_idx ] )
                                                test_indices_resolved.append( test_idx )
                                                check_indices_resolved.append( check_idx )
                                                train_indices_engaged.append( train_idx )                                        


        #================================ Half-same team pair ===============================
        
        if find_any_remaining_game( 'Test', test_indices_resolved ) >= 0 : # Check is also not finished.

                #-------- Find Test games that have half-same team pair with a Check game - test_indices_resolved
                #-------- Find Check games that have half-same team pair with a Test game - check_indices_resolved.
                for check_idx in range( len( data['homesCheck'] ) ) :
                        if check_idx in check_indices_resolved : continue
                        home = data['homesCheck'][ check_idx ]; away = data['awaysCheck'][ check_idx ]
                        test_idx = find_game_with_half_same_team_in_decreasing_order( 'Test', home, away, test_indices_resolved, homes_check, aways_check )
                        if test_idx >= 0 :
                                test_indices_resolved.append( test_idx )
                                check_indices_resolved.append( check_idx )


        if find_any_remaining_game( 'Test', test_indices_resolved ) >= 0 : # Check is also not finished.

                #--------- Find all swap plans [ ( 'Test', idx1, 'Valid', idx2 ) ]
                #--------- where game('Valid', idx) has half-same team pair with an unresolved Check game
                #--------- and where game('Test', idx) is not resolved yet.
                for check_idx in range( len( data['homesCheck'] ) ) :
                        if check_idx in check_indices_resolved : continue
                        home = data['homesCheck'][ check_idx ]; away = data['awaysCheck'][ check_idx ]
                        valid_idx = find_game_with_half_same_team_in_decreasing_order( 'Valid', home, away, valid_indices_engaged, homes_check, aways_check )
                        if valid_idx >= 0 :
                                test_idx = find_any_remaining_game( 'Test', test_indices_resolved )
                                if test_idx >= 0 :
                                        swap_plans.append( [ 'Test', test_idx, 'Valid', valid_idx ] )
                                        test_indices_resolved.append( test_idx )
                                        check_indices_resolved.append( check_idx )
                                        valid_indices_engaged.append( valid_idx )
                        else :
                                train_idx = find_game_with_half_same_team_in_decreasing_order( 'Train', home, away, train_indices_engaged, homes_check, aways_check )
                                if train_idx >= 0 :
                                        test_idx = find_any_remaining_game( 'Test', test_indices_resolved )
                                        if test_idx >= 0 :
                                                swap_plans.append( [ 'Test', test_idx, 'Train', train_idx ] )
                                                test_indices_resolved.append( test_idx )
                                                check_indices_resolved.append( check_idx )
                                                train_indices_engaged.append( train_idx )                                        


        #================================ Relevant team pair ===============================

        if find_any_remaining_game( 'Test', test_indices_resolved ) >= 0 : # Check is also not finished.

                #-------- Find Test games that have relevant team pair with a Check game - test_indices_resolved
                #-------- Find Check games that have relevant team pair with a Test game - check_indices_resolved.
                for check_idx in range( len( data['homesCheck'] ) ) :
                        if check_idx in check_indices_resolved : continue
                        home = data['homesCheck'][ check_idx ]; away = data['awaysCheck'][ check_idx ]
                        test_idx = find_game_with_relevant_teams_in_decreasing_order( 'Test', home, away, test_indices_resolved, homes_check, aways_check )
                        if test_idx >= 0 :
                                test_indices_resolved.append( test_idx )
                                check_indices_resolved.append( check_idx )

        if find_any_remaining_game( 'Test', test_indices_resolved ) >= 0 : # Check is also not finished.

                #--------- Find all swap plans [ ( 'Test', idx1, 'Valid', idx2 ) ]
                #--------- where game('Valid', idx) has relevant team pair with an unresolved Check game
                #--------- and where game('Test', idx) is not resolved yet.
                for check_idx in range( len( data['homesCheck'] ) ) :
                        if check_idx in check_indices_resolved : continue
                        home = data['homesCheck'][ check_idx ]; away = data['awaysCheck'][ check_idx ]
                        valid_idx = find_game_with_relevant_teams_in_decreasing_order( 'Valid', home, away, valid_indices_engaged, homes_check, aways_check )
                        if valid_idx >= 0 :
                                test_idx = find_any_remaining_game( 'Test', test_indices_resolved )
                                if test_idx >= 0 :
                                        swap_plans.append( [ 'Test', test_idx, 'Valid', valid_idx ] )
                                        test_indices_resolved.append( test_idx )
                                        check_indices_resolved.append( check_idx )
                                        valid_indices_engaged.append( valid_idx )
                        else :
                                train_idx = find_game_with_half_same_team_in_decreasing_order( 'Train', home, away, train_indices_engaged, homes_check, aways_check )
                                if train_idx >= 0 :
                                        test_idx = find_any_remaining_game( 'Test', test_indices_resolved )
                                        if test_idx >= 0 :
                                                swap_plans.append( [ 'Test', test_idx, 'Train', train_idx ] )
                                                test_indices_resolved.append( test_idx )
                                                check_indices_resolved.append( check_idx )
                                                train_indices_engaged.append( train_idx )                                        


        #======================================= Implement swap_plan

        #----------- PENDING - Remove remaining games from Check and Test.


        for splan in swap_plans :
                swap_row( splan[0], splan[1], splan[2], splan[3] )


        return data



def SelectRelevantGames( data, dataUse ) : # 10

        def Find_Matching_Teams( games_all, teams ) :
                matching_teams = []
                for team in teams :
                        matching_teams += list( games_all[ np.where( games_all[ :, 0 ] == team )[ 0 ], 1 ] )
                        matching_teams += list( games_all[ np.where( games_all[ :, 1 ] == team )[ 0 ], 0 ] )
                matching_teams = set(  matching_teams )
                return  matching_teams

        gamesTrain = np.stack( [ data['homesTrain'], data['awaysTrain'] ], axis = -1 )
        gamesValid = np.stack( [ data['homesValid'], data['awaysValid'] ], axis = -1 )
        gamesCheck = np.stack( [ data['homesCheck'], data['awaysCheck'] ], axis = -1 )

        games_all = np.concatenate( [ gamesTrain, gamesValid, gamesTest ], axis = 0 )

        teams = set( gamesCheck[ :, 0 ] ).union( set( gamesCheck[ :, 1 ] ) )

        for round in range( dataUse[ '.relDepth' ] ) :
                matching_teams = Find_Matching_Teams( games_all, teams )
                teams = teams.union( matching_teams )

        return data


def show1( rowid, data ) :
    print( "{}, {} : {} = {}, odds = {}, situ = {}".format( data['leagues'][rowid], data['homes'][rowid], data['aways'][rowid], data['goals'][rowid], data['odds'][rowid], data['situation'][rowid] ) )



def show2( rowid, data ) :
    def team( teamid ):
        for key, id in data['teams_dic'].items():
            if id == teamid : return key
            else : continue
        return

    def league( teamid ):
        for key, id in data['leagues_dic'].items():
            if id == teamid : return key
            else : continue
        return

    print( "{}, {} : {} = {}, odds = {}, situ = {}".format( league(data['leagues'][rowid]), team(data['homes'][rowid]), team(data['aways'][rowid]), data['goals'][rowid], data['odds'][rowid], data['situation'][rowid] ) )


def show3( rowid, segment, data ) :
    def team( teamid ):
        for key, id in data['teams_dic'].items():
            if id == teamid : return key
            else : continue
        return

    def league( teamid ):
        for key, id in data['leagues_dic'].items():
            if id == teamid : return key
            else : continue
        return

    seg = segment[0].upper() + segment[1:]

    print( "{}, {} : {} = {}, odds = {}, situ = {}".format( league(data['leagues'+seg][rowid]), team(data['homes'+seg][rowid]), team(data['aways'+seg][rowid]), data['goals'+seg][rowid], data['odds'+seg][rowid], data['situation'+seg][rowid] ) )



def CreateStringDict( string_array ) :

        normalized_list = sorted( list ( set( string_array ) ) )

        dict = {}
        for name in normalized_list :
                dict[ name ] = normalized_list.index( name ) # using index(.) is coupled with Transform(.) code.

        return dict

def Get_Chain( team, rows, homes, aways, nTeams ) :

        rescon = []; con = []; flags = []

        for num in range( nTeams ) :
                rescon.append( [] )
                con.append( [] )
                flags.append( True )

        for num in rows :
                num = num.item()
                home = homes[num].item(); away = aways[num].item()
                rescon[ home ].append( num )
                con[ home ].append( aways[num].item() )
                rescon[ away ].append( num )
                con[ aways[num] ].append( home )

        game_queue = [ team.item() ]
        connected_list = []
        while( len( game_queue ) > 0 ) :
                point_team = game_queue.pop(0)
                connected_list.append( point_team )
                flags[ point_team ] = False
                for rival in con[ point_team ] :
                        if flags[ rival ] :
                                flags[ rival ] = False
                                game_queue.append( rival )
        
        game_ids = set()
        for team_id in connected_list :
                for game in rescon[ team_id ] :
                        game_ids.add( game )

        return game_ids



def GetExpertExpansion( basicData, dataScheme ) :

        leagueCol = dataScheme['leagueCol']
        homeCol = dataScheme['homeCol']
        awayCol = dataScheme['awayCol']
        situationCols = dataScheme['situationCols']
        oddCols = dataScheme['oddCols']

        dates = basicData['date']
        leages = basicData['league']
        homes = basicData['homes']
        aways = basicData['aways']
        preds = basicData['preds']
        odds = basicData['odds']

#=============================== Contest =================================================================================

def PrepareData_Contest( dataUse, dictionaries = None, from_save = False ) :

        data = None

        if from_save :
                dataPath = 'Data_Binary_' + str(dataUse['DB.Id']) + '.bin'
                data = LoadBinaryData( dataPath )
                if data != None : return data

        data = GetCSVData( dataUse ) # 1

        SaveCSVData( 'Refined_data_file.csv', data ) # 2

        data = CorrectSpells( data, dictionaries ) # 3

        data, dictionaries_updated = Tokenize( data, dictionaries ) # 4

        data = AugmentData( data ) # 5

        data = Select_Major_Games( data, dataUse ) # 6

        data = AddColRowId( data, dataUse ) # 7

        data = SplitData( data, dataUse ) # 8

        SaveBinaryData( data, 'Data_Binary_' + str(dataUse['DB.Id']) + '.bin' ) # 9

        return data, dictionaries_updated




#=============================== Using NLP tech ============================

def ReadCSVAsSentence( file_path, queries ) :
        dataFrame = pd.read_csv(file_path)

        source, clusters_hw, clusters_dr, clusters_aw = GetSourceSentences( dataFrame )
        target = GetTargetSentences( dataFrame, queries )

        return source, target, clusters_hw, clusters_dr, clusters_aw


def GetSourceSentences( dataFrame ):

        years = np.array(dataFrame['YEAR'], dtype = np.int32); years = np.reshape(years, (-1))       
        months = np.array(dataFrame['MNAME'], dtype = np.str); months = np.reshape(months, (-1))       
        days = np.array(dataFrame['DAY'], dtype = np.int32); days = np.reshape(days, (-1))
        dows = np.array(dataFrame['DOW'], dtype = np.str); dows = np.reshape(dows, (-1))
        leagues = np.array(dataFrame['LEAGUE'], dtype = np.str); leagues = np.reshape(leagues, (-1))       
        homes = np.array(dataFrame['HOME'], dtype = np.str); homes = np.reshape(homes, (-1))
        aways = np.array(dataFrame['AWAY'], dtype = np.str); aways = np.reshape(aways, (-1))
        #psHG = np.array(dataFrame['PSHG'], dtype = np.int32 ); psHG = np.reshape(psHG, (-1))
        #psAG = np.array(dataFrame['PSAG'], dtype = np.int32 ); psAG = np.reshape(psAG, (-1))
        oddsHW = np.array(dataFrame['OHWIN'], dtype = np.float32); oddsHW = np.reshape(oddsHW, (-1))
        oddsDR = np.array(dataFrame['ODRAW'], dtype = np.float32); oddsDR = np.reshape(oddsDR, (-1))
        oddsAW = np.array(dataFrame['OAWIN'], dtype = np.float32); oddsAW = np.reshape(oddsAW, (-1))

        sentences, clusters_hw, clusters_dr, clusters_aw = Textize( years, months, days, dows, leagues, homes, aways, oddsHW, oddsDR, oddsAW  )

        return sentences, clusters_hw, clusters_dr, clusters_aw

def Textize( years, months, days, dows,leagues, homes, aways, oddsHW, oddsDR, oddsAW, clusters_hw = None, clusters_dr = None, clusters_aw = None ):

        years = TexizeNumberArray( array = years, prefix = 'Year_' )              
        days = TexizeNumberArray( array = days, prefix = "day_")
        leagues = NormalizeProperNames( leagues )
        homes = NormalizeProperNames( homes )
        aways = NormalizeProperNames( aways )
        #psHG = TexizeNumberArray( array = psHG, prefix = "psHG_" )
        #psAG = TexizeNumberArray( array = psAG, prefix = "psAG_" )
        oddsHW, clusters_hw = ClusterTextizeNumberArray( array = oddsHW, n_clusters = 100, prefix = "oddsHW_", random_state = 0, clusters = clusters_hw  )
        oddsDR, clusters_dr = ClusterTextizeNumberArray( array = oddsDR, n_clusters = 100, prefix = "oddsDR_", random_state = 0, clusters = clusters_dr  )
        oddsAW, clusters_aw = ClusterTextizeNumberArray( array = oddsAW, n_clusters = 100, prefix = "oddsAW_", random_state = 0, clusters = clusters_aw  )

        sentences = JoinStringTokens(  years, months, days, dows,leagues, homes, aways, oddsHW, oddsDR, oddsAW )

        return sentences, clusters_hw, clusters_dr, clusters_aw

def JoinStringTokens( years, months, days, dows,leagues, homes, aways, oddsHW, oddsDR, oddsAW ) :
        sentences = []

        for n in range( years.shape[-1] ) :
                sentences.append( " ".join ( [ years[n], months[n], days[n], dows[n], leagues[n], homes[n], aways[n], oddsHW[n], oddsDR[n], oddsAW[n] ] )  )

        sentences = np.array( sentences )

        return sentences

def JoinStringTokensList( tokensList, tokenizer ) :
        sentences = []

        for n in range( tokensList[0].shape[-1] ) :
                sentences.append( " ".join ( [ tokens[n] for tokens in tokensList ] )  )

        sentences = np.array( sentences )

        return sentences


#for x, y in list( self.val_dataset.as_numpy_iterator() ) :
        
def RestoreSentences( xFeatures, xTokenizer, yFeatures, yTokenizer, queries ) :

        years = ConvertToStringTokens( xFeatures[ :, 1 ], xTokenizer )
        months = ConvertToStringTokens( xFeatures[ :, 2 ], xTokenizer )
        days = ConvertToStringTokens( xFeatures[ :, 3 ], xTokenizer )
        dows = ConvertToStringTokens( xFeatures[ :, 4 ], xTokenizer )
        leagues = ConvertToStringTokens( xFeatures[ :, 5 ], xTokenizer )
        homes = ConvertToStringTokens( xFeatures[ :, 6 ], xTokenizer )
        aways = ConvertToStringTokens( xFeatures[ :, 7 ], xTokenizer )
        oddsHW = ConvertToStringTokens( xFeatures[ :, 8 ], xTokenizer )
        oddsDR = ConvertToStringTokens( xFeatures[ :, 9 ], xTokenizer )
        oddsAW = ConvertToStringTokens( xFeatures[ :, 10 ], xTokenizer )

        sentences_x = JoinStringTokens( years, months, days, dows, leagues, homes, aways, oddsHW, oddsDR, oddsAW )

        answers = []
        for n in range( len(queries) ) :
                answers.append( ConvertToStringTokens( yFeatures[ :, 1 + n ], yTokenizer) )
        
        sentences_y = JoinStringTokensList( answers, yTokenizer )

        return sentences_x, sentences_y

def ConvertToStringTokens( numberTokens, tokenizer  ) :
        stringTokens = tokenizer.sequences_to_texts( [numberTokens] )[0]
        stringTokens = np.array(stringTokens.split(' '))

        return stringTokens

from M_Football import *

def GetTargetSentences( dataFrame, queries ) :

        hGoals = np.array(dataFrame['TSHG'], dtype = np.int32); hGoals = np.reshape(hGoals, (-1))
        aGoals = np.array(dataFrame['TSAG'], dtype = np.int32); aGoals = np.reshape(aGoals, (-1))

        sentences = []

        for n in range( hGoals.shape[-1] ) :
                sentences.append( " ".join ( [ query(None, hGoals[n], aGoals[n]) for query in queries  ] ) )

        sentences = np.array( sentences )

        return sentences


def NormalizeProperNames( array ) :
        for n in range( array.shape[-1] ) :
                array[n] = array[n].replace( ' ', '_' )

        return array

def TexizeNumberArray( array, prefix  ) :

        predict = array.astype ( np.str )

        prefix = np.full( predict.shape, prefix )

        for n in range (predict.shape[0]) :
                predict[n] = prefix[n] + predict[n]

        return predict

from sklearn.cluster import KMeans

# TextizeNumberArray( array = data['odds'][:, 0], n_clusters = 30, prefix = 'odd_HWin_', random_state = 0 )

def ClusterTextizeNumberArray( array, n_clusters, prefix, random_state, clusters = None  ) :
        new_array = np.copy( array )
        if len(new_array.shape) <= 1 :
                new_array = new_array.reshape(-1, 1)

        if clusters == None :
                clusters = KMeans( n_clusters = n_clusters, random_state = random_state ).fit( new_array )

        predict = clusters.predict( new_array )
        del new_array

        predict = predict.astype( np.str )

        prefix = np.full( predict.shape, prefix )

        for n in range (predict.shape[0]) :
                predict[n] = prefix[n] + predict[n]

        return predict, clusters

# ('Date', 'League', 'Home', 'Away', 'Year', 'Month', 'Day', 'poHWin', 'poDraw', 'poAWin', 'pHGoal', 'pAGoal', 'rHGoal', 'rAGoal'),\

STR_ZONE_START = -1 # assigned its value from outside.
NUM_ZONE_START = -1 # assigned its value from outside.
PO_ZONE_START = -1 # assigned its value from outside.
GOAL_ZONE_START = -1 # assigned its value from outside.

#@tf.function
def PackString( features, goals ):
        allFields = list(features.values()) + [ goals ]

        fields = allFields[STR_ZONE_START : NUM_ZONE_START]
        fields = [ tf.cast(elem, tf.string) for elem in fields ]
        fields = tf.stack( fields, axis = 1 )

        return fields

def PackNumerical( features, goals ):
        allFields = list(features.values()) + [ goals ]

        fields = [ tf.cast(elem, tf.float32) for elem in allFields[NUM_ZONE_START : GOAL_ZONE_START] ] + \
                 [ tf.cast( 1.0 / (elem + 1e-9), tf.float32 ) for elem in allFields[PO_ZONE_START : PO_ZONE_START + 3] ] # poHWin, poDraw, poAWin
        #fields = [ tf.Variable(3.3, tf.float32), tf.Variable(4.5, tf.float32) ]

        fields = tf.stack( fields ) #, axis = -1 )

        return fields

def Packgoals( features, goals ):
        allFields = list(features.values()) + [ goals ]

        fields = allFields[GOAL_ZONE_START :]
        fields = [ tf.cast(elem, tf.float32) for elem in fields ]
        fields = tf.stack( fields ) #, axis = -1 )

        return fields

def Pack( features, goals ):
        #total_fields = list(features.values()) + [ goals ]

        #string1 = tf.cast(total_fields[1], tf.string); string1 = tf.stack( [string1], axis = -1 )
        #string2 = tf.cast(total_fields[2], tf.string); string2 = tf.stack( [string2], axis = -1 )
        #string3 = tf.cast(total_fields[3], tf.string); string3 = tf.stack( [string3], axis = -1 )
        #string4 = tf.cast(total_fields[4], tf.string); string4 = tf.stack( [string4], axis = -1 )

        #fields = total_fields[5:-2]
        #fields = [ tf.cast(elem, tf.float32) for elem in fields ]
        #float_fields = tf.stack( fields, axis = -1 )

        #fields = total_fields[-2:]
        #fields = [ tf.cast(elem, tf.float32) for elem in fields ]
        #goal_fields = tf.stack( fields, axis = -1 )

        #return (string1, string2, string3, string4, float_fields, goal_fields)
        return features, goals

def GetCSVDataset(dScheme, batch_size, **kwargs):

        file_path = dScheme['FILE_PATH']
        csv_columns = dScheme['CSV_COLUMNS']
        goal_column = csv_columns[-1]
        sel_columns = range(len(csv_columns))
        STR_ZONE_START = dScheme['STR_ZONE_START']
        NUM_ZONE_START = dScheme['NUM_ZONE_START']
        PO_ZONE_START = dScheme['PO_ZONE_START']
        GOAL_ZONE_START = dScheme['GOAL_ZONE_START']

        dataset = tf.data.experimental.make_csv_dataset(
                file_path,
                batch_size = batch_size,
                na_value="?",
                num_epochs=1,
                ignore_errors=False,
                shuffle = False,
                header = True,
                column_names = csv_columns,
                select_columns = sel_columns,
                goal_name = goal_column,
                **kwargs)

        ds_string = dataset.map( PackString )
        ds_numerical = dataset.map( PackNumerical )
        ds_lablel = dataset.map( Packgoals )

        leagues = []; tLeague = set(); homes = []; tHome = set(); aways = []; tAway = set()
        strings = np.array(list(ds_string.as_numpy_iterator()))
        for patch in strings:
                for records in patch:
                        leagues.append(records[0]); tLeague.add( records[0] )
                        homes.append(records[1]); tHome.add( records[1] )
                        aways.append(records[2]); tAway.add( records[2] )
        tLeague = list(tLeague); tHome = list(tHome); tAway = list(tAway)

        iLeague = [ tLeague.index(elem) for elem in leagues ]; ds_league = tf.data.Dataset.from_tensor_slices(iLeague).batch(batch_size)
        iHome = [ tHome.index(elem) for elem in homes ]; ds_home = tf.data.Dataset.from_tensor_slices(iHome).batch(batch_size)
        iAway = [ tAway.index(elem) for elem in aways ]; ds_away = tf.data.Dataset.from_tensor_slices(iAway).batch(batch_size)

        for bat in ds_home.take(3):
                for rec in bat: 
                        print( tHome[rec] )

        return (ds_league, len(tLeague), ds_home, len(tHome), ds_away, len(tAway)), ds_numerical, ds_lablel


def RemoveAllInFolder(folder):
        def rm(f):
                if os.path.isdir(f): return os.rmdir(f)
                if os.path.isfile(f): return os.unlink(f)
                raise Exception('must be either file or folder')

        map( rm, (os.path.join( directory,f) for f in os.listdir(folder)) )


def CleanSlotFolder():
        RemoveAllInFolder(SLOT_PATH)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def SaveJsonData(data, path):
        try:
                file = open(path, 'wt+')
                json.dump(data, file, cls = NumpyEncoder)
                file.close()
        except:
                raise Exception("Couldn't open/write/close file:" + path)


def LoadJsonData(path):
        data = None

        try:
                file = open(path, 'rt')
                data = json.load(file)
                file.close()
        except:
                #raise Exception("Couldn't open/read/close file:" + path)
                pass

        return data

def SaveBinaryData(data, path):
        try:
                file = open(path, 'wb+')
                pickle.dump(data, file)
                file.close()
        except:
                raise Exception("Couldn't open/read/save/close file:" + path)

def LoadBinaryData(path):
        data = None

        try:
                file = open(path, 'rb')
                data = pickle.load(file)
                file.close()
        except:
                # raise Exception("Couldn't open/read/close file:" + path)
                pass

        return data

def DeleteFile( path ):
        os.remove( path )
