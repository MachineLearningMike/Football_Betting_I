import numpy as np
import datetime

from M_Config import *

rows = []
# AH Tip1 and AH Tip2 removed.
rows.append( [ 9327, 2019, 1, 1, 'PREMIER LEAGUE', 'Everton', 'Leicester City', 2, 0, 0, 1, 2, 3.6, 4, 1.75, 2, 1.3, 3.5, 1.9, 1.9, 8, 9.5, 7 ] )
rows.append( [ 9323, 2019, 1, 1, 'CHAMPIONSHIP', 'Stoke City', 'Bristol City', 1, 2, 0, 2, 2.2, 3.1, 4, 1.9, 1.8, 1.4, 3, 2.3, 1.61, 6.5, 9.5, 5.5 ] )
rows.append( [ 9303, 2019, 1, 4, 'PRIMERA DIVISION', 'Espanyol', 'Leganes', 1, 0, 1, 0, 2, 3.2, 4.2, 2.2, 1.61, 1.53, 2.5, 2.5, 1.53, 5.5, 9.5, 6 ] )
rows.append( [ 9300, 2019, 1, 5, 'PRIMERA DIVISION', 'Valladolid', 'Rayo Vallecano', 3, 2, 0, 1, 2.15, 3.25, 3.6, 1.9, 1.9, 1.36, 3.2, 2.2, 1.66, 7, 9, 6 ] )
rows.append( [ 9200, 2019, 1, 15, 'FA CUP', 'Luton Town', 'Sheffield Wednesday', 1, 0, 0, 1, 2.25, 3.4, 3.5, 1.8, 1.95, 1.3, 3.4, 2.05, 1.75, 7.5, 9, 6 ] )
rows.append( [ 9100, 2019, 1, 26, 'PRIMERA DIVISION', 'Sevilla', 'Levante', 3, 0, 5, 0, 1.36, 5.5, 8, 1.61, 2.2, 1.11, 6.5, 1.36, 3.2, 11, 8.5, 11 ] )
rows.append( [ 9000, 2019, 2, 2, 'CHAMPIONSHIP', 'Wigan Athletic', 'Queens Park Rangers', 0, 2, 2, 1, 2.37, 3.3, 3.3, 1.8, 1.9, 1.36, 3.2, 2, 1.8, 7.5, 9.5, 6 ] )
rows.append( [ 8900, 2019, 2, 9, 'BUNDESLIGA', 'Borussia Dortmund', 'Hoffenheim', 3, 1, 3, 3, 1.72, 4, 4.5, 1.4, 2.75, 1.11, 6.5, 1.4, 3, 13, 9, 8.5 ] )
rows.append( [ 8800, 2019, 2, 17, 'LIGUE 1', 'Lille', 'Montpellier', 1, 0, 0, 0, 1.85, 3.3, 4.75, 2.1, 1.66, 1.44, 2.75, 2.37, 1.57, 5.5, 9, 6.5 ] )

rows.append( [ 8000, 2019, 4, 22, 'SERIE A', 'Napoli', 'Atalanta Bergamo', 2, 1, 1, 2, 2.1, 3.6, 3.3, 1.61, 2.2, 1.22, 4.33, 1.72, 2.1, 9.5, 8.5, 7 ] )
rows.append( [ 7000, 2019, 8, 3, 'LEAGUE 1', 'Peterborough', 'Fleetwood', 3, 1, 1, 3, 2.37, 3.3, 3.3, 1.95, 1.8, 1.36, 3, 2.2, 1.65, 7, 9.5, 6 ] )
rows.append( [ 6000, 2019, 9, 28, 'PREMIER LEAGUE', 'Wolves', 'Watford', 2, 1, 2, 0, 1.9, 3.4, 4.2, 1.8, 1.95, 1.3, 3.5, 1.9, 1.9, 8, 9.5, 7 ] )
rows.append( [ 5000, 2019, 11, 23, 'LEAGUE 1', 'Shrewsbury', 'Bristol Rovers', 1, 1, 3, 4, 2.1, 3.25, 4.1, 2.05, 1.7, 1.44, 2.62, 2.4, 1.53, 6, 9.5, 6 ] )
rows.append( [ 4001, 2020, 1, 15, 'LIGUE 1', 'Nimes', 'Rennes', 1, 1, 0, 1, 3.6, 3.2, 2.15, 1.95, 1.8, 1.44, 2.75, 2.37, 1.57, 8.5, 13, 6 ] )
rows.append( [ 3000, 2020, 3, 7, 'LEAGUE 1', 'Southend', 'Bristol Rovers', 1, 3, 3, 1, 3, 3.3, 2.4, 1.7, 2.05, 1.28, 3.5, 2, 1.8, 9.5, 11, 6 ] )
rows.append( [ 2050, 2020, 6,28, 'SERIE A', 'Milan', 'AS Roma', 1, 1, 2, 0, 2.45, 3.4, 2.87, 1.61, 2.2, 1.22, 4.33, 1.66, 2.2, 10, 9.5, 6.5 ] )																																												
rows.append( [ 1000, 2020, 8, 16, 'NORWAY ELITESERIEN', 'Brann', 'Mjondalen', 2, 1, 0, 1, 1.6, 3.8, 5.75, 1.7, 2.05, 1.2, 4.33, 1.66, 2.15, 8.5, 8.5, 7.5 ] )
rows.append( [ 2, 2020, 9, 29, 'CHAMPIONS LEAGUE', 'Dynamo Kyiv', 'Gent', 2, 1, 3, 0, 1.95, 4, 3.6, 1.57, 2.25, 1.2, 4.33, 1.6, 2.3, 10, 10,7.5 ] )



class DataChecker() :

    def __init__( self ) :

        def Probability( Odds, bfr ) : return 1.0 / ( Odds - bfr )

        self.erow = []; self.year = []; self.month = []; self.dayofweek = []; self.league = []; self.home = []; self.away = []; self.p_hGoal = []; self.p_aGoal = []; self.hGoal = []; self.aGoal = []; self.odds = []

        for row in rows :
            assert len( row ) == 23
            self.erow.append( row[0] )
            self.year.append( row[1] - 2019 ) # Couples with M_Database.py.
            self.month.append( row[2] - 6 ) # Couples with M_Database.py.
            self.dayofweek.append( datetime( row[1], row[2], row[3] ).weekday() - 3 ) # Couples with M_Database.py.
            self.league.append( row[4].upper() )
            self.home.append( row[5].upper() )
            self.away.append( row[6].upper() )
            self.p_hGoal.append( row[7] )
            self.p_aGoal.append( row[8] )
            self.hGoal.append( row[9] )
            self.aGoal.append( row[10] )

            # This block couples with M_Database.py.
            bfr = super_config['.brate'] # bfr == bookie_fee_rate. The percentage of ( 1/p -1 ), or the bookies share in the amount that the bettor actually creates.
            p10 = Probability( row[-3], bfr )
            p21 = Probability( row[-2], bfr )
            p11 = Probability( row[-1], bfr )
            p_all_others = 1.0 - ( p10 + p21 + p11 )
            wp = p_all_others * 100
            odds_all_others = 100 / ( wp + 0.5 ) * ( 1.0 - bfr ) + bfr
            assert np.min( odds_all_others ) > 1.0

            self.odds.append( row[11:] + [ odds_all_others ] )

        return

    #def Tokenize( self )


dc = DataChecker()
pass

        

