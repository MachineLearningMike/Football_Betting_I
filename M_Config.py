import os
from datetime import datetime, date, timedelta
from M_QGroup import *

super_config = {}

#========================= Make sure main folders exist. ==========================

Root = os. getcwd()
super_config['Root'] = Root
super_config['Data'] = os.path.join( Root, 'Data' )
super_config['History'] = os.path.join( Root, 'History' )
super_config['Models'] = os.path.join( Root, 'Models' )
super_config['Reports'] = os.path.join( Root, 'Reports' )
super_config['Summary'] = os.path.join( Root, 'Summary' )
super_config['CheckPoints'] = os.path.join( Root, 'CheckPoints' )

for folder, path in super_config.items() :

    try:
        if not os.path.exists( path ) or not os.path.isdir( path ) :
            access_rights = 0o755 # =======================
            os.mkdir( path, access_rights ) if not os.path.exists( path ) else None
    except OSError :
        raise Exception( "Couldn't create a folder: {}".format( path ) )

#===================================================================================

super_config['Contest'] = False
super_config['db_from_save'] = True
#-----------------------------------------------------------------------------------
( super_config['year.org'], super_config['month.org'], super_config['day_org'] ) = ( 2020, 10, 9 )
day_org = date( super_config['year.org'], super_config['month.org'], super_config['day_org'] )
#-----------------------------------------------------------------------------------
try: session_time
except NameError: session_time = datetime.now()
#====================================================================================

super_config['.Test'] = True

super_config['.Stage'] = 1

super_config['.Ver'] = 20201205
super_config['.dversion'] = 11


super_config['GPU'] = True
if super_config['Contest'] : super_config['GPU'] = False

super_config['QDic'] = [\
    "HWin", \
    "Draw", \
    "AWin", \
]
if not super_config['Contest'] :
    super_config['QDic'] += [ \
    "Bo_Y", \
    "Bo_N", \
    "O_15", \
    "U_15", \
    "O_25", \
    "U_25", \
    "S_10", \
    "S_21", \
    "S_11", \
    "S_xx", \
    "A_T1", \
    "A_T2", \
    ]

super_config['.brate'] = 0.20

#------------------------------------------------------------------------------------------------------------------------------
# !!!!!!!!  Keep the order that the query_ids appear in, in an ascending order.
#
"""
super_config['QGroups'] = {\
    'g_win': { 'Qs': [ 0, 1, 2 ], 'roots': [ 0, 1, 2 ], 'dummy' : [] }, \
    'g_both': { 'Qs': [ 3,  4 ], 'roots' : [ 3 ], 'dummy' : [] }, \
    'g_15': { 'Qs' : [ 5, 6 ], 'roots' : [5], 'dummy' : [] }, \
    'g_25': { 'Qs' : [ 7, 8 ], 'roots' : [7], 'dummy' : [] }, \
    'g_score': { 'Qs' : [ 9, 10, 11, 12 ], 'roots' : [ 9, 10, 11 ], 'dummy' : [ 12 ] }, \
    } )
"""
super_config['QGroups'] = {\
    #'HWn': { 'Qs': [ 0 ], 'roots': [ 0 ], 'dummy' : [] }, \
    #'Drw': { 'Qs': [ 1 ], 'roots': [ 1 ], 'dummy' : [] }, \
    'AWn': { 'Qs': [ 2 ], 'roots': [ 2 ], 'dummy' : [] }, \
    #'BoY': { 'Qs': [ 3 ], 'roots': [ 3 ], 'dummy' : [] }, \
    #'BoN': { 'Qs': [ 4 ], 'roots': [ 4 ], 'dummy' : [] }, \
    #'O15': { 'Qs': [ 5 ], 'roots': [ 5 ], 'dummy' : [] }, \
    #'U15': { 'Qs': [ 6 ], 'roots': [ 6 ], 'dummy' : [] }, \
    #'O25': { 'Qs': [ 7 ], 'roots': [ 7 ], 'dummy' : [] }, \
    #'U25': { 'Qs': [ 8 ], 'roots': [ 8 ], 'dummy' : [] }, \
    #'S10': { 'Qs': [ 9 ], 'roots': [ 9 ], 'dummy' : [] }, \
    #'S21': { 'Qs': [ 10 ], 'roots': [ 10 ], 'dummy' : [] }, \
    #'S11': { 'Qs': [ 11 ], 'roots': [ 11 ], 'dummy' : [] }, \
    #'Oth': { 'Qs': [ 12 ], 'roots': [ 12 ], 'dummy' : [] }, \
    }

super_q_group = M_QGroup( super_config['QGroups'] )

#---------------------------------------------------------------------------------------------------------------------------------

super_config['.loss_patterns'] = [ \
    { 'xSLD' : 1.0, 'xEpochs' : 1.00 }, # x times Standard Loss Drop, x times nEpochs \
    #{ 'xSLD' : 0.2, 'xEpochs' : 0.30 }, # x times Standard Loss Drop, x times nEpochs \
    #{ 'xSLD' : 0.05, 'xEpochs' : 0.10 }, # x times Standard Loss Drop, x times nEpochs \
]

super_config['.M.Day'] = ( date( 2020, 11, 4 ) - day_org ).days
super_config['.C.Day'] = ( date( 2020, 10, 9 ) - day_org ).days

def Idx_C_Day() : return super_config['.C.Day'] - super_config['.M.Day']

super_config['RecTestEmb'] = False # 

super_config['.NCheck'] = 500 #-------------------------
if not super_config['Contest'] :  super_config['.NCheck'] = 15

super_config['.d.coun'] = 5
super_config['.d.leag'] = 10
super_config['.d.team'] = 20 # considering use of emb, the dim should not be large.
super_config['.relDepth'] = 30 # hop depth of team relations, to include in data.

super_config['.M.Odds'] = 50.0 # Games with more than MaxOdds are dropped from history data or replaced by MaxOddss. Remove compiled database upon change of this.
super_config['.M.Score'] = 10 # Games with more than MaxGoals are dropped from history data or replaced by MaxGoals. Remove compiled database upon change of this.


super_config['.BSize'] = 32

super_config['Margin'] = 100
super_config['const.stake'] = False # PENDING - False has a problem. tf.pow? It works.
super_config['stake'] = 10.0
super_config['stakePow'] = 0.2 # 1.0

super_config['ignore_odds'] = 1.0 # !!!!! PENDING - Formularize it. Very sensitive. The larger, the more freedom from odds's magnetic attraction.
super_config['exacerbate'] = False
super_config['std'] = False

super_config['.cut'] = True
super_config['.CutRate'] = 0.90
super_config['.AbsCut'] = 3 # This number of best models will be effectively available for a check game to choose in each query. This will limit Adverse Contributions.
super_config['.score'] = 'std*std*f1' #'std*std*pre', 'm.acc', 'std*std*f1*f1',  'm.pre*pre', 'std*std*pre*pre'
                        #'f1', 'std*std*f1', 'acc*acc', 'std*std*f1', m.pre', 'm.f1', 'm.std*f1', 'm.f1*f1', 'm.acc*acc'

# Change requires removing account file. Re-building account file takes long.
super_config['bidir'] = True
super_config['valid_only'] = False

super_config['debugging_1'] = '' # 'follow_pred_happened', 'follow_true_happened'

super_config['save_model'] = True

super_config['job_prefix'] = 'mG10C' # original
#super_config['job_prefix'] = 'D1_PH_'
#super_config['job_prefix'] = 'D1_100%_'

super_config['session'] = 'train'  # 'train', 'test', 'average', 'analyze'
super_config['visualize_check'] = False

super_config['no_score'] = 0.0

# Comment out 'happ' and 'profit'. We have to stick to build a pure scores that does NOT know the labels of target segment.
# 'stake' does NOT know the labels of target segment. It comes from models that don't know target unless target is one of 'train' and 'valid'.
# If target segment is 'train', 'valid', or a combination of them, then ok for a quite sophsticated resason.
# You have a chance to explicitly add labels when you extend the views/scores.
super_config['schemes'] = [ \
'odds', 'output', 'stake', \
'pHapp', '_m.pre', 'm.pre', '_m.acc', 'm.acc', '_m.f1', 'm.f1', '_m.std*f1', 'm.std*f1', '_m.pre*pre', \
'm.pre*pre', '_m.acc*acc', 'm.acc*acc', '_acc*acc', 'acc*acc', '_m.f1*f1', 'm.f1*f1', '_m.std*std*f1*f1', \
'm.std*std*f1*f1', '_m.std*f1*f1', 'm.std*f1*f1', '_m.std*std*f1', 'm.std*std*f1', '_std*std', 'std*std', \
'_f1*f1', 'f1*f1', '_std*std*f1', 'std*std*f1', '_std*std*pre', 'std*std*pre', '_std*std*f1*f1', 'std*std*f1*f1', \
'_std*std*pre*pre', 'std*std*pre*pre' ]

super_config['nFramesToShow'] = 20

super_config['segs.outcomes'] = [ 'train', 'valid', 'test' ]

def Report_super_config( config = None ) :
    if config is None : config = super_config

    text = ''
    for key, value in config.items() :
        if key.startswith('.') : text += ( key[1:] + ': ' + str( value )  + ', ' )

    return text

# QueryWise, 

import tensorflow as tf

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
