import tensorflow as tf
import numpy as np

from M_Football import *

from Visulaize import *

xArange = np.arange(-1.0, 4.5, .05)
yArange = np.arange(-1.0, 4.5, .05)

h = tf.keras.layers.Input( shape=(len(xArange), len(yArange) ), name = 'h')
a = tf.keras.layers.Input( shape=(len(xArange), len(yArange) ), name = 'a')

#home = tf.keras.layers.Dense( 1 ) (h)
#away = tf.keras.layers.Dense( 1 ) (a)

sigmoid = tf.keras.activations.sigmoid
linear = tf.keras.activations.linear

def ZFunction( filterSet, filterCore, query_id, x, y ) :
    output = filterSet[query_id] ( filterCore, None, h, a )
    model = tf.keras.models.Model( inputs = [ h, a ], outputs = output )
    z = np.ones_like( x )
    x = np.expand_dims( x, axis = 0 ); y = np.expand_dims( y, axis = 0 )
    z = model.predict( x = { 'h' : x, 'a' : y } )
    z = np.squeeze( z )
    return z  

def XFunction( filterSet, query_id, x, y ) :
    output = filterSet[query_id] ( h, a )
    model = tf.keras.models.Model( inputs = [ h, a ], outputs = output )
    z = np.ones_like( x )
    x = np.expand_dims( x, axis = 0 ); y = np.expand_dims( y, axis = 0 )
    z = model.predict( x = { 'h' : x, 'a' : y } )
    z = np.squeeze( z )
    return z

def QFunction( filterSet, filterCore, query_id, x, y ) :
    output = filterSet[query_id] ( filterCore, h, a )
    model = tf.keras.models.Model( inputs = [ h, a ], outputs = output )
    z = np.ones_like( x )
    x = np.expand_dims( x, axis = 0 ); y = np.expand_dims( y, axis = 0 )
    z = model.predict( x = { 'h' : x, 'a' : y } )
    z = np.squeeze( z )
    return z


def _Zero_For_Fit_Q_00( x, y ) : return ZFunction(Zero_For_Fit_Q, None, 0, x, y)
def _RvsProb_Q_00( x, y ) : return QFunction(RvsProb_Q, sigmoid, 0, x, y)
def _Zero_For_Fit_Q_01( x, y ) : return ZFunction(Zero_For_Fit_Q, None, 1, x, y)
def _RvsProb_Q_01( x, y ) : return QFunction(RvsProb_Q, sigmoid, 1, x, y)
def _Zero_For_Fit_Q_02( x, y ) : return ZFunction(Zero_For_Fit_Q, None, 2, x, y)
def _RvsProb_Q_02( x, y ) : return QFunction(RvsProb_Q, sigmoid, 2, x, y)
def _Zero_For_Fit_Q_03( x, y ) : return ZFunction(Zero_For_Fit_Q, None, 3, x, y)
def _RvsProb_Q_03( x, y ) : return QFunction(RvsProb_Q, sigmoid, 3, x, y)
def _Zero_For_Fit_Q_04( x, y ) : return ZFunction(Zero_For_Fit_Q, None, 4, x, y)
def _RvsProb_Q_04( x, y ) : return QFunction(RvsProb_Q, sigmoid, 4, x, y)
def _Zero_For_Fit_Q_05( x, y ) : return ZFunction(Zero_For_Fit_Q, None, 5, x, y)
def _RvsProb_Q_05( x, y ) : return QFunction(RvsProb_Q, sigmoid, 5, x, y)


"""
PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_Q_00, "Home > 0.5 - Binary" )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Q_00, "Home > 0.5 - Prob" )
PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_Q_01, "Away > 0.5 - Bianry" )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Q_01, "Away > 0.5 - Prob" )
PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_Q_02, "Home > 1.5 - Binary" )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Q_02, "Home > 1.5 - Prob" )
PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_Q_03, "Away > 1.5 - Binary" )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Q_03, "Away > 1.5 - Prob" )
PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_Q_04, "Home > 2.5 - Binary" )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Q_04, "Home > 2.5 - Prob" )
PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_Q_05, "Away > 2.5 - Binary" )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Q_05, "Away > 2.5 - Prob" )
"""


def _Zero_For_Fit_P_00( x, y ) : return ZFunction(Zero_For_Fit_P, None, 0, x, y)
def _RvsProb_P_00( x, y ) : return QFunction(RvsProb_P, sigmoid, 0, x, y)
def _RvsProb_Sketch_P_00( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 0, x, y)

def _Zero_For_Fit_P_01( x, y ) : return ZFunction(Zero_For_Fit_P, None, 1, x, y)
def _RvsProb_P_01( x, y ) : return QFunction(RvsProb_P, sigmoid, 1, x, y)
def _RvsProb_Sketch_P_01( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 1, x, y)

def _Zero_For_Fit_P_02( x, y ) : return ZFunction(Zero_For_Fit_P, None, 2, x, y)
def _RvsProb_P_02( x, y ) : return QFunction(RvsProb_P, sigmoid, 2, x, y)
def _RvsProb_Sketch_P_02( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 2, x, y)

def _Zero_For_Fit_P_03( x, y ) : return ZFunction(Zero_For_Fit_P, None, 3, x, y)
def _RvsProb_P_03( x, y ) : return QFunction(RvsProb_P, sigmoid, 3, x, y)
def _RvsProb_Sketch_P_03( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 3, x, y)

def _Zero_For_Fit_P_04( x, y ) : return ZFunction(Zero_For_Fit_P, None, 4, x, y)
def _RvsProb_P_04( x, y ) : return QFunction(RvsProb_P, sigmoid, 4, x, y)
def _RvsProb_Sketch_P_04( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 4, x, y)

def _Zero_For_Fit_P_05( x, y ) : return ZFunction(Zero_For_Fit_P, None, 5, x, y)
def _RvsProb_P_05( x, y ) : return QFunction(RvsProb_P, sigmoid, 5, x, y)
def _RvsProb_Sketch_P_05( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 5, x, y)

def _Zero_For_Fit_P_06( x, y ) : return ZFunction(Zero_For_Fit_P, None, 6, x, y)
def _RvsProb_P_06( x, y ) : return QFunction(RvsProb_P, sigmoid, 6, x, y)
def _RvsProb_Sketch_P_06( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 6, x, y)

def _Zero_For_Fit_P_07( x, y ) : return ZFunction(Zero_For_Fit_P, None, 7, x, y)
def _RvsProb_P_07( x, y ) : return QFunction(RvsProb_P, sigmoid, 7, x, y)
def _RvsProb_Sketch_P_07( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 7, x, y)

def _Zero_For_Fit_P_08( x, y ) : return ZFunction(Zero_For_Fit_P, None, 8, x, y)
def _RvsProb_P_08( x, y ) : return QFunction(RvsProb_P, sigmoid, 8, x, y)
def _RvsProb_Sketch_P_08( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 8, x, y)

def _Zero_For_Fit_P_09( x, y ) : return ZFunction(Zero_For_Fit_P, None, 9, x, y)
def _RvsProb_P_09( x, y ) : return QFunction(RvsProb_P, sigmoid, 9, x, y)
def _RvsProb_Sketch_P_09( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 9, x, y)

def _Zero_For_Fit_P_10( x, y ) : return ZFunction(Zero_For_Fit_P, None, 10, x, y)
def _RvsProb_P_10( x, y ) : return QFunction(RvsProb_P, sigmoid, 10, x, y)
def _RvsProb_Sketch_P_10( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 10, x, y)

def _Zero_For_Fit_P_11( x, y ) : return ZFunction(Zero_For_Fit_P, None, 11, x, y)
def _RvsProb_P_11( x, y ) : return QFunction(RvsProb_P, sigmoid, 11, x, y)
def _RvsProb_Sketch_P_11( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 11, x, y)

def _Zero_For_Fit_P_12( x, y ) : return ZFunction(Zero_For_Fit_P, None, 12, x, y)
def _RvsProb_P_12( x, y ) : return QFunction(RvsProb_P, sigmoid, 12, x, y)
def _RvsProb_Sketch_P_12( x, y ) : return QFunction(RvsProb_Sketch_P, sigmoid, 12, x, y)


"""
PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_00, "Home > 0.5 - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_00, "Home > 0.5 - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_00, "Home > 0.5 - Prob_Stetch, " )

PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_01, "Away > 0.5 - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_01, "Away > 0.5 - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_01, "Away > 0.5 - Prob_Stetch, " )
"""

"""
PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_02, "Both > 0.5 - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_02, "Both > 0.5 - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_02, "Both > 0.5 - Prob_Stetch, " )


PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_03, "<  2, > 0.5 - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_03, "<  2, > 0.5 - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_03, "<  2, > 0.5 - Prob_Stetch, " )
"""

"""
PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_04, ">  0, > 2.5 - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_04, ">  0, > 2.5 - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_04, ">  0, > 2.5 - Prob_Stetch, " )

PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_05, ">  0, < 2.5 - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_05, ">  0, < 2.5 - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_05, ">  0, < 2.5 - Prob_Stetch, " )

PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_06, ">  0, > 1.5 - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_06, ">  0, > 1.5 - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_06, ">  0, > 1.5 - Prob_Stetch, " )
"""

"""
PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_07, "Home > Away - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_07, "Home > Away - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_07, "Home > Away - Prob_Stetch, " )

PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_08, "Home < Away - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_08, "Home < Away - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_08, "Home < Away - Prob_Stetch, " )

PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_09, "Home = Away - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_09, "Home = Away - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_09, "Home = Away - Prob_Stetch, " )

PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_10, "Score 1 : 0 - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_10, "Score 1 : 0 - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_10, "Score 1 : 0 - Prob_Stetch, " )

PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_11, "Score 1 : 1 - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_11, "Score 1 : 1 - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_11, "Score 1 : 1 - Prob_Stetch, " )

PlotFunctionSurface ( xArange, yArange, _Zero_For_Fit_P_12, "Score 2 : 1 - Binary, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_P_12, "Score 2 : 1 - Prob, " )
PlotFunctionSurface ( xArange, yArange, _RvsProb_Sketch_P_12, "Score 2 : 1 - Prob_Stetch, " )
"""

def _Prob_Not_Happened_R_00( x, y ) : return XFunction( Prob_Not_Happened_R, 0, x, y )
def _Prob_Not_Happened_R_01( x, y ) : return XFunction( Prob_Not_Happened_R, 1, x, y )
def _Prob_Not_Happened_R_02( x, y ) : return XFunction( Prob_Not_Happened_R, 2, x, y )
def _Prob_Not_Happened_R_03( x, y ) : return XFunction( Prob_Not_Happened_R, 3, x, y )
def _Prob_Not_Happened_R_04( x, y ) : return XFunction( Prob_Not_Happened_R, 4, x, y )
def _Prob_Not_Happened_R_05( x, y ) : return XFunction( Prob_Not_Happened_R, 5, x, y )
def _Prob_Not_Happened_R_06( x, y ) : return XFunction( Prob_Not_Happened_R, 6, x, y )
def _Prob_Not_Happened_R_07( x, y ) : return XFunction( Prob_Not_Happened_R, 7, x, y )
def _Prob_Not_Happened_R_08( x, y ) : return XFunction( Prob_Not_Happened_R, 8, x, y )
def _Prob_Not_Happened_R_09( x, y ) : return XFunction( Prob_Not_Happened_R, 9, x, y )
def _Prob_Not_Happened_R_10( x, y ) : return XFunction( Prob_Not_Happened_R, 10, x, y )
def _Prob_Not_Happened_R_11( x, y ) : return XFunction( Prob_Not_Happened_R, 11, x, y )
def _Prob_Not_Happened_R_12( x, y ) : return XFunction( Prob_Not_Happened_R, 12, x, y )
def _Prob_Not_Happened_R_13( x, y ) : return XFunction( Prob_Not_Happened_R, 13, x, y )

def _NotBettingProb_Precise_R_00( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 0, x, y)
def _NotBettingProb_Precise_R_01( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 1, x, y)
def _NotBettingProb_Precise_R_02( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 2, x, y)
def _NotBettingProb_Precise_R_03( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 3, x, y)
def _NotBettingProb_Precise_R_04( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 4, x, y)
def _NotBettingProb_Precise_R_05( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 5, x, y)
def _NotBettingProb_Precise_R_06( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 6, x, y)
def _NotBettingProb_Precise_R_07( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 7, x, y)
def _NotBettingProb_Precise_R_08( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 8, x, y)
def _NotBettingProb_Precise_R_09( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 9, x, y)
def _NotBettingProb_Precise_R_10( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 10, x, y)
def _NotBettingProb_Precise_R_11( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 11, x, y)
def _NotBettingProb_Precise_R_12( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 12, x, y)
def _NotBettingProb_Precise_R_13( x, y ) : return QFunction(NotBettingProb_Precise_R, sigmoid, 13, x, y)


PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_00, "Home__Win - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_00, "Home__Win - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_01, "Game_Draw - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_01, "Game_Draw - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_02, "Away__Win - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_02, "Away__Win - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_03, "Both_Sc_Y - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_03, "Both_Sc_Y - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_04, "Both_Sc_N - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_04, "Both_Sc_N - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_05, "Over__1.5 - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_05, "Over__1.5 - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_06, "Under_1.5 - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_06, "Under_1.5 - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_07, "Over__2.5 - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_07, "Over__2.5 - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_08, "Under_2.5 - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_08, "Under_2.5 - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_09, "Score_1_0 - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_09, "Score_1_0 - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_10, "Score_2_1 - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_10, "Score_2_1 - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_11, "Score_1_1 - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_11, "Score_1_1 - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_12, "AH_1.5_T1 - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_12, "AH_1.5_T1 - Prob, " )

PlotFunctionSurface ( xArange, yArange, _Prob_Not_Happened_R_13, "AH_1.5_T2 - True, " )
PlotFunctionSurface ( xArange, yArange, _NotBettingProb_Precise_R_13, "AH_1.5_T2 - Prob, " )


def Function( transformers_By_Query, function, query_id, x, y ) :
    output = transformers_By_Query[query_id] ( h, a, 'output' )
    model = tf.keras.models.Model( inputs = [ h, a ], outputs = output )
    z = np.ones_like( x )
    x = np.expand_dims( x, axis = 0 ); y = np.expand_dims( y, axis = 0 )
    z = model.predict( x = { 'h' : x, 'a' : y } )
    z = np.squeeze( z )
    return z
