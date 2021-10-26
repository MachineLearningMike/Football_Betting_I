
import tensorflow as tf
import numpy as np

masterSlope = 15.0

linear = tf.keras.activations.linear
sigmoid = tf.keras.activations.sigmoid
def step( x ): return sigmoid( x * 10e7 )

def LinearTensor( function, tensor, alpha, beta, slope, name = None ):
    """
    alphaSlope = tf.constant( alpha * slope, shape = (tensor.shape[-1], tensor.shape[-1]), dtype = tf.float32 )
    betaSlope = tf.constant( beta * slope, shape = (tensor.shape[-1], tensor.shape[-1]), dtype = tf.float32 )

    input_ax = tf.keras.layers.Multiply() ( [alphaSlope, tensor] )
    input_ax_plus_b = tf.keras.layers.Add() ( [input_ax, betaSlope] )

    final = tf.keras.layers.Dense( tensor.shape[-1], activation = function, \
        kernel_initializer = tf.constant_initializer(1.), \
        bias_initializer = tf.constant_initializer(0.), \
    name = name, trainable = False ) \
    ( input_ax_plus_b )
    """
    """
    final = tf.keras.layers.Dense( units = tensor.shape[-1], activation = function, \
        kernel_initializer = tf.keras.initializers.constant( alpha * slope ),  \
        bias_initializer = tf.keras.initializers.constant( beta * slope ), \
        trainable = False, name = name ) ( tensor )
    """
    final = tf.keras.layers.Lambda( lambda x : function ( slope * ( alpha * x + beta  ) ), trainable = False, name = name ) ( tensor )

    return final

queryDic_P = [\
    "Home, > 0.5", \
    "Away, > 0.5", \
    "Both, > 0.5", \
    "<  2, > 0.5", \
    ">  0, > 2.5", \
    ">  0, < 2.5", \
    ">  0, > 1.5", \
    "Home > Away", \
    "Home < Away", \
    "Home = Away", \
    "Score 1 : 0", \
    "Score 1 : 0", \
    "Score 2 : 1"  \
]

queryDic_Q = [\
    "Home > 0.5", \
    "Away > 0.5", \
    "Home > 1.5", \
    "Away > 1.5", \
    "Home > 2.5", \
    "Away > 2.5", \
]

def RvsProb_Q_00( function, home, away, name = None ) : # H > 0.5
    return LinearTensor( function = function, tensor = home, alpha = -1.0, beta = 0.5, slope = masterSlope, name = name )

def RvsProb_Q_01( function, home, away, name = None ) : # A > 0.5
    return LinearTensor( function = function, tensor = away, alpha = -1.0, beta = 0.5, slope = masterSlope, name = name )

def RvsProb_Q_02( function, home, away, name = None ) : # H > 1.5
    return LinearTensor( function = function, tensor = home, alpha = -1.0, beta = 1.5, slope = masterSlope, name = name )

def RvsProb_Q_03( function, home, away, name = None ) : # A > 1.5
    return LinearTensor( function = function, tensor = away, alpha = -1.0, beta = 1.5, slope = masterSlope, name = name )

def RvsProb_Q_04( function, home, away, name = None ) : # H > 2.5
    return LinearTensor( function = function, tensor = home, alpha = -1.0, beta = 2.5, slope = masterSlope, name = name )

def RvsProb_Q_05( function, home, away, name = None ) : # A > 2.5
    return LinearTensor( function = function, tensor = away, alpha = -1.0, beta = 2.5, slope = masterSlope, name = name )

def RvsProb_Q_06( function, home, away, name = None ) : # H > 3.5
    return LinearTensor( function = function, tensor = home, alpha = -1.0, beta = 3.5, slope = masterSlope, name = name )

def RvsProb_Q_07( function, home, away, name = None ) : # A > 3.5
    return LinearTensor( function = function, tensor = away, alpha = -1.0, beta = 3.5, slope = masterSlope, name = name )



def NOT ( tensor, name = None ) : 
    return LinearTensor( function = linear, tensor = tensor, alpha = -1.0, beta = 1.0, slope = 1.0, name = name )

def AND ( tensorA, tensorB, name = None ) :
    return tf.keras.layers.Maximum( name = name ) ( [tensorA, tensorB] )

def AND3 ( tensorA, tensorB, tensorC, name = None ) :
    return tf.keras.layers.Maximum( name = name ) ( [tensorA, tensorB, tensorC ] )   

def AND4 ( tensorA, tensorB, tensorC, tensorD, name = None ) :
    return tf.keras.layers.Maximum( name = name ) ( [tensorA, tensorB, tensorC, tensorD ] )

def OR ( tensorA, tensorB, name = None ) :
    return tf.keras.layers.Minimum( name = name ) ( [tensorA, tensorB] )

def OR3 ( tensorA, tensorB, tensorC, name = None ) :
    return tf.keras.layers.Minimum( name = name ) ( [tensorA, tensorB, tensorC] )

def OR4 ( tensorA, tensorB, tensorC, tensorD, name = None ) :
    return tf.keras.layers.Minimum( name = name ) ( [tensorA, tensorB, tensorC, tensorD] )

def OR5 ( tensorA, tensorB, tensorC, tensorD, tensorE, name = None ) :
    return tf.keras.layers.Minimum( name = name ) ( [tensorA, tensorB, tensorC, tensorD, tensorE] )

def OR6 ( tensorA, tensorB, tensorC, tensorD, tensorE, tensorF, name = None ) :
    return tf.keras.layers.Minimum( name = name ) ( [tensorA, tensorB, tensorC, tensorD, tensorE, tensorF] )

def OR10 ( tensorA, tensorB, tensorC, tensorD, tensorE, tensorF, tensorG, tensorH, tensorI, tensorJ, name = None ) :
    return tf.keras.layers.Minimum( name = name ) ( [tensorA, tensorB, tensorC, tensorD, tensorE, tensorF, tensorG, tensorH, tensorI, tensorJ] )

def OR11 ( tensorA, tensorB, tensorC, tensorD, tensorE, tensorF, tensorG, tensorH, tensorI, tensorJ, tensorK, name = None ) :
    return tf.keras.layers.Minimum( name = name ) ( [tensorA, tensorB, tensorC, tensorD, tensorE, tensorF, tensorG, tensorH, tensorI, tensorJ, tensorK] )

def OR13 ( tensorA, tensorB, tensorC, tensorD, tensorE, tensorF, tensorG, tensorH, tensorI, tensorJ, tensorK, tensorL, tensorM, name = None ) :
    return tf.keras.layers.Minimum( name = name ) ( [tensorA, tensorB, tensorC, tensorD, tensorE, tensorF, tensorG, tensorH, tensorI, tensorJ, tensorK, tensorL, tensorM] )

def UnderHalfOne( queryModelOutput ) :
    return tf.less( queryModelOutput, 0.5 )
def Bool_Not ( boolean ) :
    return tf.logical_not ( boolean )
def Bool_And ( boolA, boolB ) :
    return tf.logical_and ( boolA, boolB )
def Bool_And3 ( boolA, boolB, boolC ) :
    return tf.logical_and ( tf.logical_and( boolA, boolB ), boolC )
def Bool_And4 ( boolA, boolB, boolC, boolD ) :
    return tf.logical_and( tf.logical_and( boolA, boolB ), tf.logical_and( boolC, boolD ) )
def Bool_Or ( boolA, boolB ) :
    return tf.logical_or ( boolA, boolB )
def Bool_Or3 ( boolA, boolB, boolC ) :
    return tf.logical_or( tf.logical_or( boolA, boolB ), boolC )
def Bool_Or4 ( boolA, boolB, boolC, boolD ) :
    return tf.logical_or( tf.logical_or( boolA, boolB ), tf.logical_or( boolC, boolD ) )
def Bool_Or6( boolA, boolB, boolC, boolD, boolE, boolF ) :
    return tf.logical_or( tf.logical_or( tf.logical_or( boolA, boolB ), boolC ), tf.logical_or( tf.logical_or( boolD, boolE ), boolF ) )



def RvsProb_S_00 ( function, home, away, name = None ) :
    return AND ( \
        NOT( RvsProb_Q_00(function, home, away) ), \
        NOT( RvsProb_Q_01( function, home, away) ), \
        name = name \
    )
def UnderHalfOne_S_00 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And( Bool_Not( UnderHalfOne( B1 ) ), Bool_Not( UnderHalfOne( B2 ) ) )

def RvsProb_S_01 ( function, home, away, name = None ) :
    return AND3 ( \
        NOT( RvsProb_Q_00( function, home, away ) ), \
        RvsProb_Q_01( function, home, away ), \
        NOT( RvsProb_Q_03( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_01 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And3( Bool_Not( UnderHalfOne( B1 ) ), UnderHalfOne( B2 ), Bool_Not( UnderHalfOne( B4 ) ) )

def RvsProb_S_02 ( function, home, away, name = None ) :
    return AND3 ( \
            NOT( RvsProb_Q_00( function, home, away ) ), \
            RvsProb_Q_03( function, home, away ), \
            NOT( RvsProb_Q_05( function, home, away ) ), \
            name = name \
    )
def UnderHalfOne_S_02 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And3( Bool_Not( UnderHalfOne( B1 ) ), UnderHalfOne( B4 ), Bool_Not( UnderHalfOne( B6 ) ) )

def RvsProb_S_03 ( function, home, away, name = None ) :
    return AND3 ( \
        NOT( RvsProb_Q_00( function, home, away ) ), \
        RvsProb_Q_05( function, home, away ), \
        NOT( RvsProb_Q_07( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_03 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And( Bool_Not( UnderHalfOne( B1 ) ), UnderHalfOne( B6 ) )

def RvsProb_S_04 ( function, home, away, name = None ) :
    return AND ( \
        NOT( RvsProb_Q_00( function, home, away ) ), \
        RvsProb_Q_07( function, home, away ), \
        name = name \
    )



def RvsProb_S_10 ( function, home, away, name = None ) :
    return AND3 ( \
        RvsProb_Q_00( function, home, away ), \
        NOT( RvsProb_Q_02( function, home, away ) ), \
        NOT( RvsProb_Q_01( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_10 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And3( UnderHalfOne( B1 ), Bool_Not( UnderHalfOne( B3 ) ), Bool_Not( UnderHalfOne( B2 ) ) )

def RvsProb_S_11 ( function, home, away, name = None ) :
    return AND4 ( \
        RvsProb_Q_00( function, home, away ), \
        NOT( RvsProb_Q_02( function, home, away ) ), \
        RvsProb_Q_01( function, home, away ), \
        NOT( RvsProb_Q_03( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_11 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And4( UnderHalfOne( B1 ), Bool_Not( UnderHalfOne( B3 ) ), UnderHalfOne( B2 ), Bool_Not( UnderHalfOne( B4 ) ) )

def RvsProb_S_12 ( function, home, away, name = None ) :
    return AND4 ( \
        RvsProb_Q_00( function, home, away ), \
        NOT ( RvsProb_Q_02( function, home, away ) ), \
        RvsProb_Q_03( function, home, away ), \
        NOT( RvsProb_Q_05( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_12 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And4( UnderHalfOne( B1 ), Bool_Not( UnderHalfOne( B3 ) ), UnderHalfOne( B4 ), Bool_Not( UnderHalfOne( B6 ) ) )

def RvsProb_S_13 ( function, home, away, name = None ) :
    return AND4 ( \
        RvsProb_Q_00( function, home, away ), \
        NOT( RvsProb_Q_02( function, home, away ) ), \
        RvsProb_Q_05( function, home, away ), \
        NOT( RvsProb_Q_07( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_13 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And3( UnderHalfOne( B1 ), Bool_Not( UnderHalfOne( B3 ) ), UnderHalfOne( B6 ) )

def RvsProb_S_14 ( function, home, away, name = None ) :
    return AND3 ( \
        RvsProb_Q_00( function, home, away ), \
        NOT( RvsProb_Q_02( function, home, away ) ), \
        RvsProb_Q_07( function, home, away ), \
        name = name \
    )


def RvsProb_S_20 ( function, home, away, name = None ) :
    return AND3 ( \
        RvsProb_Q_02( function, home, away ), \
        NOT ( RvsProb_Q_04( function, home, away ) ), \
        NOT ( RvsProb_Q_01( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_20 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And3( UnderHalfOne( B3 ), Bool_Not( UnderHalfOne( B5 ) ),  Bool_Not( UnderHalfOne( B2 ) ) )

def RvsProb_S_21 ( function, home, away, name = None ) :
    return AND4 ( \
        RvsProb_Q_02( function, home, away ), \
        NOT( RvsProb_Q_04( function, home, away ) ), \
        RvsProb_Q_01( function, home, away ), \
        NOT( RvsProb_Q_03( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_21 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And4( UnderHalfOne( B3 ), Bool_Not( UnderHalfOne( B5 ) ), UnderHalfOne( B2 ), Bool_Not( UnderHalfOne( B4 ) ) )

def RvsProb_S_22 ( function, home, away, name = None ) :
    return AND4 ( \
        RvsProb_Q_02( function, home, away ), \
        NOT( RvsProb_Q_04( function, home, away ) ), \
        RvsProb_Q_03( function, home, away ), \
        NOT( RvsProb_Q_05( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_22 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And4( UnderHalfOne( B3 ), Bool_Not( UnderHalfOne( B5 ) ), UnderHalfOne( B4 ), Bool_Not( UnderHalfOne( B6 ) ) )

def RvsProb_S_23 ( function, home, away, name = None ) :
    return AND4 ( \
        RvsProb_Q_02( function, home, away ), \
        NOT( RvsProb_Q_04( function, home, away ) ), \
        RvsProb_Q_05( function, home, away ), \
        NOT( RvsProb_Q_07( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_23 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And3( UnderHalfOne( B3 ), Bool_Not( UnderHalfOne( B5 ) ), UnderHalfOne( B6 ) )

def RvsProb_S_24 ( function, home, away, name = None ) :
    return AND3 ( \
        RvsProb_Q_02( function, home, away ), \
        NOT( RvsProb_Q_04( function, home, away ) ), \
        RvsProb_Q_07( function, home, away ), \
        name = name \
    )



def RvsProb_S_30 ( function, home, away, name = None ) :
    return AND3 ( \
        RvsProb_Q_04( function, home, away ), \
        NOT( RvsProb_Q_06( function, home, away ) ), \
        NOT( RvsProb_Q_01( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_30 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And( UnderHalfOne( B5 ), Bool_Not( UnderHalfOne( B2 ) ) )

def RvsProb_S_31 ( function, home, away, name = None ) :
    return AND4 ( \
        RvsProb_Q_04( function, home, away ), \
        NOT( RvsProb_Q_06( function, home, away ) ), \
        RvsProb_Q_01( function, home, away ), \
        NOT( RvsProb_Q_03( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_31 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And3( UnderHalfOne( B5 ), UnderHalfOne( B2 ), Bool_Not( UnderHalfOne( B4 ) ) )

def RvsProb_S_32 ( function, home, away, name = None ) :
    return AND4 ( \
        RvsProb_Q_04( function, home, away ), \
        NOT( RvsProb_Q_06( function, home, away ) ), \
        RvsProb_Q_03( function, home, away ), \
        NOT( RvsProb_Q_05( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_32 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And3( UnderHalfOne( B5 ), UnderHalfOne( B4 ), Bool_Not( UnderHalfOne( B6 ) ) )

def RvsProb_S_33 ( function, home, away, name = None ) :
    return AND4 ( \
        RvsProb_Q_04( function, home, away ), \
        NOT( RvsProb_Q_06( function, home, away ) ), \
        RvsProb_Q_05( function, home, away ), \
        NOT( RvsProb_Q_07( function, home, away ) ), \
        name = name \
    )
def UnderHalfOne_S_33 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And( UnderHalfOne( B5 ), UnderHalfOne( B6 ) )

def RvsProb_S_34 ( function, home, away, name = None ) :
    return AND3 ( \
        RvsProb_Q_04( function, home, away ), \
        NOT( RvsProb_Q_06( function, home, away ) ), \
        RvsProb_Q_07( function, home, away ), \
        name = name \
    )


def RvsProb_S_40 ( function, home, away, name = None ) :
    return AND ( \
        RvsProb_Q_06( function, home, away ), \
        NOT( RvsProb_Q_01( function, home, away ) ), \
        name = name \
    )

def RvsProb_S_41 ( function, home, away, name = None ) :
    return AND3 ( \
        RvsProb_Q_06( function, home, away ), \
        RvsProb_Q_01( function, home, away ), \
        NOT( RvsProb_Q_03( function, home, away ) ), \
        name = name \
    )

def RvsProb_S_42 ( function, home, away, name = None ) :
    return AND3 ( \
        RvsProb_Q_06( function, home, away ), \
        RvsProb_Q_03( function, home, away ), \
        NOT( RvsProb_Q_05( function, home, away ) ), \
        name = name \
    )

def RvsProb_S_43 ( function, home, away, name = None ) :
    return AND3 ( \
        RvsProb_Q_06( function, home, away ), \
        RvsProb_Q_05( function, home, away ), \
        NOT( RvsProb_Q_07( function, home, away ) ), \
        name = name \
    )

def RvsProb_S_44 ( function, home, away, name = None ) :
    return AND ( \
        RvsProb_Q_06( function, home, away ), \
        RvsProb_Q_07( function, home, away ), \
        name = name \
    )


def RvsProb_P_00 ( function, home, away, name = None ) :
    return RvsProb_Q_00( function, home, away )

def UnderhalfOne_Q_00 ( B1, B2, B3, B4, B5, B6 ) :
    return UnderHalfOne( B1 )

def RvsProb_P_01 ( function, home, away, name = None ) :
    return RvsProb_Q_01( function, home, away, name )

def UnderhalfOne_Q_01 ( B1, B2, B3, B4, B5, B6 ) :
    return UnderHalfOne( B2 )


def RvsProb_P_02 ( function, home, away, name = None ) :
    return AND ( \
        RvsProb_Q_00( function, home, away ), \
        RvsProb_Q_01( function, home, away ), \
        name = name \
    )

def UnderhalfOne_Q_02 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And( UnderHalfOne( B1 ), UnderHalfOne( B2 ) )

def RvsProb_P_03 ( function, home, away, name = None ) : # Not both scores
    return OR ( \
        NOT( RvsProb_Q_00( function, home, away ) ), \
        NOT( RvsProb_Q_01( function, home, away ) ), \
        name = name \
    )

def UnderhalfOne_Q_03 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_Or( Bool_Not( UnderHalfOne( B1 ) ), Bool_Not( UnderHalfOne( B2 ) ) )

def RvsProb_P_04 ( function, home, away, name = None ) :
    return OR ( \
        RvsProb_Q_04( function, home, away ), \
        RvsProb_Q_05( function, home, away ), \
        name = name \
    )

def UnderhalfOne_Q_04 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_Or( UnderHalfOne( B5 ), UnderHalfOne( B6 ) )
	
def RvsProb_P_05 ( function, home, away, name = None ) :
    return OR ( \
        NOT( RvsProb_Q_04( function, home, away ) ), \
        NOT( RvsProb_Q_05( function, home, away ) ), \
        name = name \
    )

def UnderhalfOne_Q_05 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_Or( Bool_Not( UnderHalfOne( B5 ) ), Bool_Not( UnderHalfOne( B6 ) ) )

def RvsProb_P_06 ( function, home, away, name = None ) :
    return OR ( \
        RvsProb_Q_02( function, home, away ), \
        RvsProb_Q_03( function, home, away ), \
        name = name \
    )

def UnderhalfOne_Q_07 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_Or( UnderHalfOne( B3 ), UnderHalfOne( B4 ) )

def RvsProb_P_07 ( function, home, away, name = None ) : # home win
    return OR10 ( \
        RvsProb_S_10( function, home, away ), \
        RvsProb_S_20( function, home, away ), \
        RvsProb_S_21( function, home, away ), \
        RvsProb_S_30( function, home, away ), \
        RvsProb_S_31( function, home, away ), \
        RvsProb_S_32( function, home, away ), \
        RvsProb_S_40( function, home, away ), \
        RvsProb_S_41( function, home, away ), \
        RvsProb_S_42( function, home, away ), \
        RvsProb_S_43( function, home, away ), \
        name = name \
    )

def UnderhalfOne_Q_08 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_Or6( \
        UnderhalfOne_S_10 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_20 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_21 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_30 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_31 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_32 ( B1, B2, B3, B4, B5, B6 ), \
    )

def RvsProb_P_08 ( function, home, away, name = None ) : # Away win
    return OR10 ( \
        RvsProb_S_01( function, home, away ), \
        RvsProb_S_02( function, home, away ), \
        RvsProb_S_12( function, home, away ), \
        RvsProb_S_03( function, home, away ), \
        RvsProb_S_13( function, home, away ), \
        RvsProb_S_23( function, home, away ), \
        RvsProb_S_04( function, home, away ), \
        RvsProb_S_14( function, home, away ), \
        RvsProb_S_24( function, home, away ), \
        RvsProb_S_34( function, home, away ), \
        name = name \
    )

def UnderhalfOne_Q_09 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_Or6( \
        UnderhalfOne_S_01 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_02 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_12 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_03 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_13 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_23 ( B1, B2, B3, B4, B5, B6 ), \
    )

def RvsProb_P_09 ( function, home, away, name = None ) : # Draw
    return OR5 ( \
        RvsProb_S_00( function, home, away ), \
        RvsProb_S_11( function, home, away ), \
        RvsProb_S_22( function, home, away ), \
        RvsProb_S_33( function, home, away ), \
        RvsProb_S_44( function, home, away ), \
        name = name \
    )

def UnderhalfOne_Q_10 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_Or4( \
        UnderhalfOne_S_00 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_11 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_22 ( B1, B2, B3, B4, B5, B6 ), \
        UnderhalfOne_S_33 ( B1, B2, B3, B4, B5, B6 ), \
    )

def RvsProb_P_10 ( function, home, away, name = None ) :
    return AND3 ( \
        RvsProb_Q_00( function, home, away ), \
        NOT( RvsProb_Q_02( function, home, away ) ), \
        NOT( RvsProb_Q_01( function, home, away ) ), \
        name = name \
    )
def UnderhalfOne_Q_11 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And3( UnderHalfOne( B1 ), Bool_Not( UnderHalfOne( B3 ) ), Bool_Not( UnderHalfOne( B2 ) ) )

def RvsProb_P_11 ( function, home, away, name = None ) :
    return AND4 ( \
        RvsProb_Q_00( function, home, away ), \
        NOT( RvsProb_Q_02( function, home, away ) ), \
        RvsProb_Q_01( function, home, away ), \
        NOT( RvsProb_Q_03( function, home, away ) ), \
        name = name \
    )

def UnderhalfOne_Q_12 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_And4( UnderHalfOne( B1 ), Bool_Not( UnderHalfOne( B3 ) ), UnderHalfOne( B2 ), Bool_Not( UnderHalfOne( B4 ) ) )  

def RvsProb_P_12 ( function, home, away, name = None ) :
    return AND4 ( \
        RvsProb_Q_02( function, home, away ), \
        NOT( RvsProb_Q_04( function, home, away ) ), \
        RvsProb_Q_01( function, home, away ), \
        NOT( RvsProb_Q_03( function, home, away ) ), \
        name = name \
    )

def UnderhalfOne_Q_13 ( B1, B2, B3, B4, B5, B6 ) :
    return Bool_AND4( UnderHalfOne( B3 ), Bool_Not( UnderHalfOne( B5 ) ), UnderHalfOne( B2 ), Bool_Not( UnderHalfOne( B4 ) ) )  


def RvsProb_P_14 ( function, home, away, name = None ) : # SUM > 1.5
    return OR ( \
        OR11 ( \
        RvsProb_S_02( function, home, away ), \
        RvsProb_S_03( function, home, away ), \
        RvsProb_S_04( function, home, away ), \
        RvsProb_S_11( function, home, away ), \
        RvsProb_S_12( function, home, away ), \
        RvsProb_S_13( function, home, away ), \
        RvsProb_S_14( function, home, away ), \
        RvsProb_S_20( function, home, away ), \
        RvsProb_S_21( function, home, away ), \
        RvsProb_S_22( function, home, away ), \
        RvsProb_S_23( function, home, away ), \
        ),\
        OR11 ( \
        RvsProb_S_24( function, home, away ), \
        RvsProb_S_30( function, home, away ), \
        RvsProb_S_31( function, home, away ), \
        RvsProb_S_32( function, home, away ), \
        RvsProb_S_33( function, home, away ), \
        RvsProb_S_34( function, home, away ), \
        RvsProb_S_40( function, home, away ), \
        RvsProb_S_41( function, home, away ), \
        RvsProb_S_42( function, home, away ), \
        RvsProb_S_43( function, home, away ), \
        RvsProb_S_44( function, home, away ), \
        ), \
        name = name \
    )

#def RvsProb_P_14 ( function, home, away, name = None ) : # SUM > 1.5
#    return 1.0 - RvsProb_P_15 ( function, home, away, name )

def RvsProb_Sketch_P_14_Shift( function, home, away, name = None ) : # SUM > 1.5
    H =  LinearTensor( function = linear, tensor = home, alpha = 1.0, beta = 0.0, slope = 1. )
    _A = LinearTensor( function = linear, tensor = away, alpha = 1.0, beta = 0.0, slope = 1. )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = LinearTensor( function = function, tensor = H_A, alpha = -1.0, beta = (-1.0) * (-1.0) * ( 1.5 + 0.5 ), slope = masterSlope, name = name )   # 0.5

    return final

def RvsProb_P_15 ( function, home, away, name = None ) : # SUM < 1.5
    return OR3 ( \
        RvsProb_S_00( function, home, away ), \
        RvsProb_S_10( function, home, away ), \
        RvsProb_S_01( function, home, away ), \
        name = name \
    )

def RvsProb_Sketch_P_15_Shift( function, home, away, name = None ) : # SUM < 1.5
    H =  LinearTensor( function = linear, tensor = home, alpha = 1.0, beta = 0.0, slope = 1. )
    _A = LinearTensor( function = linear, tensor = away, alpha = 1.0, beta = 0.0, slope = 1. )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = LinearTensor( function = function, tensor = H_A, alpha = 1.0, beta = (-1.0) * (1.0) * ( 1.5 - 0.5 ), slope = masterSlope, name = name )   # 0.5

    return final


def RvsProb_P_16 ( function, home, away, name = None ) : # SUM > 2.5
    return OR10 ( \
        RvsProb_S_03( function, home, away ), \
        RvsProb_S_12( function, home, away ), \
        RvsProb_S_13( function, home, away ), \
        RvsProb_S_21( function, home, away ), \
        RvsProb_S_22( function, home, away ), \
        RvsProb_S_23( function, home, away ), \
        RvsProb_S_30( function, home, away ), \
        RvsProb_S_31( function, home, away ), \
        RvsProb_S_32( function, home, away ), \
        RvsProb_S_33( function, home, away ), \
        name = name \
    )

#def RvsProb_P_16 ( function, home, away, name = None ) : # SUM > 2.5
#    return 1.0 - RvsProb_P_17 ( function, home, away, name )

def RvsProb_Sketch_P_16_Shift( function, home, away, name = None ) : # SUM > 2.5
    H =  LinearTensor( function = linear, tensor = home, alpha = 1.0, beta = 0.0, slope = 1. )
    _A = LinearTensor( function = linear, tensor = away, alpha = 1.0, beta = 0.0, slope = 1. )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = LinearTensor( function = function, tensor = H_A, alpha = -1.0, beta = (-1.0) * (-1.0) * ( 2.5 + 0.5 ), slope = masterSlope, name = name )   # 0.5

    return final

def RvsProb_P_17 ( function, home, away, name = None ) : # SUM < 2.5
    return OR6 ( \
        RvsProb_S_00( function, home, away ), \
        RvsProb_S_01( function, home, away ), \
        RvsProb_S_02( function, home, away ), \
        RvsProb_S_10( function, home, away ), \
        RvsProb_S_11( function, home, away ), \
        RvsProb_S_20( function, home, away ), \
        name = name \
    )

def RvsProb_Sketch_P_17_Shift( function, home, away, name = None ) : # SUM < 2.5
    H =  LinearTensor( function = linear, tensor = home, alpha = 1.0, beta = 0.0, slope = 1. )
    _A = LinearTensor( function = linear, tensor = away, alpha = 1.0, beta = 0.0, slope = 1. )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = LinearTensor( function = function, tensor = H_A, alpha = 1.0, beta = (-1.0) * (1.0) * ( 2.5 - 0.5 ), slope = masterSlope, name = name )   # 0.5

    return final




def Scores_Fit_Q_00(h, a) :
    return tf.math.greater( h, 0.5 )

def Scores_Fit_Q_01(h, a) :
    return tf.math.greater( a, 0.5 )

def Scores_Fit_Q_02(h, a) :
    return tf.math.greater( h, 1.5 )

def Scores_Fit_Q_03(h, a) :
    return tf.math.greater( a, 1.5 )

def Scores_Fit_Q_04(h, a) :
    return tf.math.greater( h, 2.5 )

def Scores_Fit_Q_05(h, a) :
    return tf.math.greater( a, 2.5 )

def Scores_Fit_Q_06(h, a) :
    return tf.math.greater( h, 3.5 )

def Scores_Fit_Q_07(h, a) :
    return tf.math.greater( a, 3.5 )

def Scores_Fit_Q_08(h, a) :
    return tf.math.greater( h, 4.5 )

def Scores_Fit_Q_09(h, a) :
    return tf.math.greater( a, 4.5 )


def Zero_For_Fit_Score_Q_00(dummy, odds, h, a) :
    return 1.0 - tf.cast( Scores_Fit_Q_00(h,a), tf.float32 )

def Zero_For_Fit_Score_Q_01(dummy, odds, h, a) :
    return 1.0 - tf.cast( Scores_Fit_Q_01(h,a), tf.float32 )

def Zero_For_Fit_Score_Q_02(dummy, odds, h, a) :
    return 1.0 - tf.cast( Scores_Fit_Q_02(h,a), tf.float32 )

def Zero_For_Fit_Score_Q_03(dummy, odds, h, a) :
    return 1.0 - tf.cast( Scores_Fit_Q_03(h,a), tf.float32 )

def Zero_For_Fit_Score_Q_04(dummy, odds, h, a) :
    return 1.0 - tf.cast( Scores_Fit_Q_04(h,a), tf.float32 )

def Zero_For_Fit_Score_Q_05(dummy, odds, h, a) :
    return 1.0 - tf.cast( Scores_Fit_Q_05(h,a), tf.float32 )

def Zero_For_Fit_Score_Q_06(dummy, odds, h, a) :
    return 1.0 - tf.cast( Scores_Fit_Q_06(h,a), tf.float32 )

def Zero_For_Fit_Score_Q_07(dummy, odds, h, a) :
    return 1.0 - tf.cast( Scores_Fit_Q_07(h,a), tf.float32 )

def Zero_For_Fit_Score_Q_08(dummy, odds, h, a) :
    return 1.0 - tf.cast( Scores_Fit_Q_08(h,a), tf.float32 )

def Zero_For_Fit_Score_Q_09(dummy, odds, h, a) :
    return 1.0 - tf.cast( Scores_Fit_Q_09(h,a), tf.float32 )


def ConcatnateDenseOne(dummyFunc, home, away) :
    final = None
    
    if home != None and away != None :
        final = tf.keras.layers.concatenate( [home, away], axis = -1 )
    elif home != None :
        final = home
    elif  away != None :
        final = away
    else :
        raise Exception('Attend here! Concat')

    return tf.keras.layers.Dense( units = 1 ) ( final )


def RvsProb_Sketch_P_00( function, home, away, name = None ) :
    final = LinearTensor( function = function, tensor = home, alpha = -1.0, beta = 0.5, slope = masterSlope, name = name )

    return final

def RvsProb_Sketch_P_01( function, home, away, name = None ) : # Away, > 0.5
    final = LinearTensor( function = function, tensor = away, alpha = -1.0, beta = 0.5, slope = masterSlope, name = name )

    return final

def RvsProb_Sketch_P_02( function, home, away, name = None ) : # Both, > 0.5
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        LinearTensor( function = function, tensor = home, alpha = -1.0, beta = 0.5, slope = masterSlope ), \
        LinearTensor( function = function, tensor = away, alpha = -1.0, beta = 0.5, slope = masterSlope ) \
    ] )

    return final

def RvsProb_Sketch_P_03( function, home, away, name = None ) : # <= 1, > 0.5
    final = tf.keras.layers.Multiply( name = name ) ( [ \
     	LinearTensor( function = function, tensor = home, alpha = 1.0, beta = -0.5, slope = masterSlope ), \
		LinearTensor( function = function, tensor = away, alpha = 1.0, beta = -0.5, slope = masterSlope ) \
    ] )

    return final

def RvsProb_Sketch_P_04( function, home, away, name = None ) : # >= 1, > 2.5
    final = tf.keras.layers.Multiply( name = name ) ( [ \
        LinearTensor( function = function, tensor = home, alpha = -1.0, beta = 2.5, slope = masterSlope ), \
		LinearTensor( function = function, tensor = away, alpha = -1.0, beta = 2.5, slope = masterSlope ) \
    ] )

    return final

def RvsProb_Sketch_P_05( function, home, away, name = None ) : # >= 1, < 2.5
    final = tf.keras.layers.Multiply( name = name ) ( [ \
        LinearTensor( function = function, tensor = home, alpha = 1.0, beta = -2.5, slope = masterSlope ), \
		LinearTensor( function = function, tensor = away, alpha = 1.0, beta = -2.5, slope = masterSlope ) \
    ] )

    return final

def RvsProb_Sketch_P_06( function, home, away, name = None ) : # >= 1, > 1.5
    final = tf.keras.layers.Multiply ( name = name ) ( [ \
        LinearTensor( function = function, tensor = home, alpha = -1.0, beta = 1.5, slope = masterSlope ), \
		LinearTensor( function = function, tensor = away, alpha = -1.0, beta = 1.5, slope = masterSlope ) \
    ] )

    return final

def RvsProb_Sketch_P_07( function, home, away, name = None ) : # Home > Away
    H =  LinearTensor( function = linear, tensor = home, alpha = 1.0, beta = 0.0, slope = masterSlope )
    _A = LinearTensor( function = linear, tensor = away, alpha = -1.0, beta = 0.0, slope = masterSlope )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = LinearTensor( function = function, tensor = H_A, alpha = -1.0, beta = 0.0, slope = 1.0, name = name )   # 0.5

    return final


def RvsProb_Sketch_P_07_Shift( function, home, away, name = None ) : # Home > Away
    H =  LinearTensor( function = linear, tensor = home, alpha = 1.0, beta = 0.0, slope = 1. )
    _A = LinearTensor( function = linear, tensor = away, alpha = -1.0, beta = 0.0, slope = 1. )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = LinearTensor( function = function, tensor = H_A, alpha = -1.0, beta = (-1.0) * (-1.0) * ( 1.0 ), slope = masterSlope, name = name )   # 0.5

    return final

def RvsProb_Sketch_P_08( function, home, away, name = None ) : # Home < Away
    H =  LinearTensor( function = linear, tensor = home, alpha = 1.0, beta = 0.0, slope = masterSlope )
    _A = LinearTensor( function = linear, tensor = away, alpha = -1.0, beta = 0.0, slope = masterSlope )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = LinearTensor( function = function, tensor = H_A, alpha = 1.0, beta = 0.0, slope = 1.0, name = name )  # -0.5

    return final

def RvsProb_Sketch_P_08_Shift( function, home, away, name = None ) : # Home < Away
    H =  LinearTensor( function = linear, tensor = home, alpha = 1.0, beta = 0.0, slope = 1. )
    _A = LinearTensor( function = linear, tensor = away, alpha = -1.0, beta = 0.0, slope = 1. )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = LinearTensor( function = function, tensor = H_A, alpha = 1.0, beta = (-1.0) * (1.0) * ( -1.0 ), slope = masterSlope, name = name )  # -0.5

    return final

def RvsProb_Sketch_P_09( function, home, away, name = None ) : # Home = Away
    H =  LinearTensor( function = linear, tensor = home, alpha = 1.0, beta = 0.0, slope = masterSlope )
    _A = LinearTensor( function = linear, tensor = away, alpha = -1.0, beta = 0.0, slope = masterSlope )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = tf.keras.layers.Add( name = name ) ( [ \
        LinearTensor( function = function, tensor = H_A, alpha = -1.0, beta = (-0.5), slope = 1.0 ), # -0.5
	    LinearTensor( function = function, tensor = H_A, alpha = 1.0, beta = -0.5, slope = 1.0 ) # -0.5
    ] )

    return final


def RvsProb_Sketch_P_10( function, home, away, name = None ) : # Score 1 : 0
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        LinearTensor( function = function, tensor = home, alpha = -1.0, beta = 0.5, slope = masterSlope ), \
        LinearTensor( function = function, tensor = home, alpha = 1.0, beta = -1.5, slope = masterSlope ), \
        LinearTensor( function = function, tensor = away, alpha = -1.0, beta = -0.5, slope = masterSlope ), \
		LinearTensor( function = function, tensor = away, alpha = 1.0, beta = -0.5, slope = masterSlope )
    ] )
 
    return final

def RvsProb_Sketch_P_11( function, home, away, name = None ) : # Score 1 : 1
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        LinearTensor( function = function, tensor = home, alpha = -1.0, beta = 0.5, slope = masterSlope ), \
        LinearTensor( function = function, tensor = home, alpha = 1.0, beta = -1.5, slope = masterSlope ), \
        LinearTensor( function = function, tensor = away, alpha = -1.0, beta = 0.5, slope = masterSlope ), \
		LinearTensor( function = function, tensor = away, alpha = 1.0, beta = -1.5, slope = masterSlope )
    ] )

    return final


def RvsProb_Sketch_P_12( function, home, away, name = None ) : # Score 2 : 1
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        LinearTensor( function = function, tensor = home, alpha = -1.0, beta = 1.5, slope = masterSlope ), \
        LinearTensor( function = function, tensor = home, alpha = 1.0, beta = -2.5, slope = masterSlope ), \
        LinearTensor( function = function, tensor = away, alpha = -1.0, beta = 0.5, slope = masterSlope ), \
		LinearTensor( function = function, tensor = away, alpha = 1.0, beta = -1.5, slope = masterSlope )
    ] )

    return final


def Scores_Fit_P_00(h, a) : # for original Query 1: Home team scores.
    return tf.math.greater( h, 0.5 )

def Scores_Fit_P_01(h, a) : # for original Query 2: Away team scores.
    return tf.math.greater( a, 0.5 )

def Scores_Fit_P_02(h, a) : # for original Query 3: Both teams score.
    return  tf.math.logical_and( tf.math.greater( h, 0.5 ), tf.math.greater( a, 0.5 ) )

def Scores_Fit_P_03(h, a) : # for original Query 4: At most 1 team scores.
    return   tf.math.logical_not( tf.math.logical_and( tf.math.greater( h, 0.5 ), tf.math.greater( a, 0.5 ) ) )

def Scores_Fit_P_04(h, a) : # for original Query 5: At least 1 team scores more than 2 goals.
    return  tf.math.logical_or( tf.math.greater( h, 2.5 ), tf.math.greater( a, 2.5 ) )

def Scores_Fit_P_05(h, a) : # for original Query 6: At least 1 team scores less than 3 goals.
    return  tf.math.logical_or( tf.math.less( h, 2.5 ), tf.math.less( a, 2.5 ) )

def Scores_Fit_P_06(h, a) : # for original Query 7: At leas 1 team scores more than 1 goal.
    return  tf.math.logical_or( tf.math.greater( h, 1.5 ), tf.math.greater( a, 1.5 ) )

def Scores_Fit_P_07(h, a) : # for original Query 8: Home team wins.
    return tf.math.greater( tf.round(h), tf.round(a) )

def Scores_Fit_P_08(h, a) : # for original Query 9: Away team wins.
    return tf.math.less( tf.round(h), tf.round(a) )

def Scores_Fit_P_09(h, a) : # for original Query 10: The match draws.
    return tf.math.equal( tf.round(h), tf.round(a) )

def Scores_Fit_P_10(h, a) : # for original Query 12: The score is 1:0.
    return tf.math.logical_and( tf.math.equal( tf.round(h), 1.), tf.math.equal( tf.round(a), 0. ) )

def Scores_Fit_P_11(h, a) : # for original Query 13: The score is 1:1.
    return tf.math.logical_and( tf.math.equal( tf.round(h), 1.), tf.math.equal( tf.round(a), 1. ) )

def Scores_Fit_P_12(h, a) : # for original Query 14: The score is 2:1.
    return tf.math.logical_and( tf.math.equal( tf.round(h), 2.), tf.math.equal( tf.round(a), 1. ) )

def Scores_Fit_P_13(h, a) : # for original Query 14: The score is 2:1.
    return tf.math.logical_not( tf.math.logical_or( tf.math.logical_or( Scores_Fit_P_10(h, a), Scores_Fit_P_11(h, a) ), Scores_Fit_P_12(h, a) ) )


def Zero_For_Fit_Score_P_00(dummy, odds, h, a) : # for original Query 1: Home team scores.
    ret = 1. - tf.cast( Scores_Fit_P_00(h,a), tf.float32 )
    return ret

def Zero_For_Fit_Score_P_01(dummy, odds, h, a) : # for original Query 2: Away team scores.
    return 1. - tf.cast( Scores_Fit_P_01(h,a), tf.float32 )

def Zero_For_Fit_Score_P_02(dummy, odds, h, a) : # for original Query 3: Both teams score.
    return 1. - tf.cast( Scores_Fit_P_02(h,a), tf.float32 )

def Zero_For_Fit_Score_P_03(dummy, odds, h, a) : # for original Query 4: At most 1 team scores.
    return 1. - tf.cast( Scores_Fit_P_03(h,a), tf.float32 )

def Zero_For_Fit_Score_P_04(dummy, odds, h, a) : # for original Query 5: At least 1 team scores more than 2 goals.
    return 1. - tf.cast( Scores_Fit_P_04(h,a), tf.float32 )

def Zero_For_Fit_Score_P_05(dummy, odds, h, a) : # for original Query 6: At least 1 team scores less than 3 goals.
    return 1. - tf.cast( Scores_Fit_P_05(h,a), tf.float32 )

def Zero_For_Fit_Score_P_06(dummy, odds, h, a) : # for original Query 7: At leas 1 team scores more than 1 goal.
    return 1. - tf.cast( Scores_Fit_P_06(h,a), tf.float32 )

def Zero_For_Fit_Score_P_07(dummy, odds, h, a) : # for original Query 8: Home team wins.
    return 1. - tf.cast( Scores_Fit_P_07(h,a), tf.float32 )

def Zero_For_Fit_Score_P_08(dummy, odds, h, a) : # for original Query 9: Away team wins.
    return 1. - tf.cast( Scores_Fit_P_08(h,a), tf.float32 )

def Zero_For_Fit_Score_P_09(dummy, odds, h, a) : # for original Query 10: The match draws.
    return 1. - tf.cast( Scores_Fit_P_09(h,a), tf.float32 )

def Zero_For_Fit_Score_P_10(dummy, odds, h, a) : # for original Query 12: The score is 1:0.
    return 1. - tf.cast( Scores_Fit_P_10(h,a), tf.float32 )

def Zero_For_Fit_Score_P_11(dummy, odds, h, a) : # for original Query 13: The score is 1:1.
    return 1. - tf.cast( Scores_Fit_P_11(h,a), tf.float32 )

def Zero_For_Fit_Score_P_12(dummy, odds, h, a) : # for original Query 14: The score is 2:1.
    return 1. - tf.cast( Scores_Fit_P_12(h,a), tf.float32 )

def Zero_For_Fit_Score_P_13(dummy, odds, h, a) : # for original Query 14: The score is 2:1.
    return 1. - tf.cast( Scores_Fit_P_13(h,a), tf.float32 )


Q_Label_Tokens = [ \
    [ 'H_>_05', 'H_<_05' ], \
    [ 'A_>_05', 'A_<_05' ], \
    [ 'H_>_15', 'H_<_15' ], \
    [ 'A_>_15', 'A_<_15' ], \
    [ 'H_>_25', 'H_<_25' ], \
    [ 'A_>_25', 'A_<_25' ], \
    [ 'H_>_35', 'H_<_35' ], \
    [ 'A_>_35', 'A_<_35' ], \
    [ 'H_>_45', 'H_<_45' ], \
    [ 'A_>_45', 'A_<_45' ], \
]


def Q_Label_Token_Q_00(dummyFunc, h, a) :
    h = tf.cast(h, tf.float32); a = tf.cast(a, tf.float32)
    return Q_Label_Tokens [0] [ 1 - tf.cast( Scores_Fit_Q_00(h,a), tf.int32 ) ]
def Q_Label_Token_Q_01(dummyFunc, h, a) :
    h = tf.cast(h, tf.float32); a = tf.cast(a, tf.float32)
    return Q_Label_Tokens [1] [ 1 - tf.cast( Scores_Fit_Q_01(h,a), tf.int32 ) ]
def Q_Label_Token_Q_02(dummyFunc, h, a) :
    h = tf.cast(h, tf.float32); a = tf.cast(a, tf.float32)
    return Q_Label_Tokens [2] [ 1 - tf.cast( Scores_Fit_Q_02(h,a), tf.int32 ) ]
def Q_Label_Token_Q_03(dummyFunc, h, a) :
    h = tf.cast(h, tf.float32); a = tf.cast(a, tf.float32)
    return Q_Label_Tokens [3] [ 1 - tf.cast( Scores_Fit_Q_03(h,a), tf.int32 ) ]
def Q_Label_Token_Q_04(dummyFunc, h, a) :
    h = tf.cast(h, tf.float32); a = tf.cast(a, tf.float32)
    return Q_Label_Tokens [4] [ 1 - tf.cast( Scores_Fit_Q_04(h,a), tf.int32 ) ]
def Q_Label_Token_Q_05(dummyFunc, h, a) :
    h = tf.cast(h, tf.float32); a = tf.cast(a, tf.float32)
    return Q_Label_Tokens [5] [ 1 - tf.cast( Scores_Fit_Q_05(h,a), tf.int32 ) ]
def Q_Label_Token_Q_06(dummyFunc, h, a) :
    h = tf.cast(h, tf.float32); a = tf.cast(a, tf.float32)
    return Q_Label_Tokens [6] [ 1 - tf.cast( Scores_Fit_Q_06(h,a), tf.int32 ) ]
def Q_Label_Token_Q_07(dummyFunc, h, a) :
    h = tf.cast(h, tf.float32); a = tf.cast(a, tf.float32)
    return Q_Label_Tokens [7] [ 1 - tf.cast( Scores_Fit_Q_07(h,a), tf.int32 ) ]
def Q_Label_Token_Q_08(dummyFunc, h, a) :
    h = tf.cast(h, tf.float32); a = tf.cast(a, tf.float32)
    return Q_Label_Tokens [8] [ 1 - tf.cast( Scores_Fit_Q_08(h,a), tf.int32 ) ]
def Q_Label_Token_Q_09(dummyFunc, h, a) :
    h = tf.cast(h, tf.float32); a = tf.cast(a, tf.float32)
    return Q_Label_Tokens [9] [ 1 - tf.cast( Scores_Fit_Q_09(h,a), tf.int32 ) ]


# The order of appearence is important.
Q_Label_Token_Q = [ Q_Label_Token_Q_00, Q_Label_Token_Q_01, Q_Label_Token_Q_02, Q_Label_Token_Q_03, Q_Label_Token_Q_04, Q_Label_Token_Q_05, \
    Q_Label_Token_Q_06, Q_Label_Token_Q_07, Q_Label_Token_Q_08, Q_Label_Token_Q_09 ]


def Q_Label_Token(dummyFunc, h, a) :
    return " ".join( [ tokenFunc(dummyFunc, float(h), float(a) ) for tokenFunc in Q_Label_Token_Q  ] )

#test = Q_Label_Token ( None, 2, 0 )


def Text_Token_Q_00( dummyFunc, h, a ) :
    final = 'H_<_05'
    if h > 0.5 : final = 'H_>_05'
    return final

def Text_Token_Q_01( dummyFunc, h, a ) :
    final = 'A_<_05'
    if a > 0.5 : final = 'A_>_05'
    return final

def Text_Token_Q_02( dummyFunc, h, a ) :
    final = 'H_<_15'
    if h > 1.5 : final = 'H_>_15'
    return final

def Text_Token_Q_03( dummyFunc, h, a ) :
    final = 'A_<_15'
    if a > 1.5 : final = 'A_>_15'
    return final

def Text_Token_Q_04( dummyFunc, h, a ) :
    final = 'H_<_25'
    if h > 2.5 : final = 'H_>_25'
    return final

def Text_Token_Q_05( dummyFunc, h, a ) :
    final = 'A_<_25'
    if a > 2.5 : final = 'A_>_25'
    return final

def Text_Token_Q_06( dummyFunc, h, a ) :
    final = 'H_Under35'
    if h > 3.5 : final = 'H_>_35'
    return final

def Text_Token_Q_07( dummyFunc, h, a ) :
    final = 'A_<_35'
    if a > 3.5 : final = 'A_>_35'
    return final

def Text_Token_Q_08( dummyFunc, h, a ) :
    final = 'H_<_45'
    if h > 4.5 : final = 'H_>_45'
    return final

def Text_Token_Q_09( dummyFunc, h, a ) :
    final = 'A_<_45'
    if a > 4.5 : final = 'A_>_45'
    return final

#========================================

def Text_Token_Q_10( dummyFunc, h, a ) :
    final = 'H_>_05'
    if h < 0.5 : final = 'H_<_05'
    return final

def Text_Token_Q_11( dummyFunc, h, a ) :
    final = 'A_>_05'
    if a < 0.5 : final = 'A_<_05'
    return final

def Text_Token_Q_12( dummyFunc, h, a ) :
    final = 'H_>_15'
    if h < 1.5 : final = 'H_<_15'
    return final

def Text_Token_Q_13( dummyFunc, h, a ) :
    final = 'A_>_15'
    if a < 1.5 : final = 'A_<_15'
    return final

def Text_Token_Q_14( dummyFunc, h, a ) :
    final = 'H_>_25'
    if h < 2.5 : final = 'H_<_25'
    return final

def Text_Token_Q_15( dummyFunc, h, a ) :
    final = 'A_>_25'
    if a < 2.5 : final = 'A_<_25'
    return final

def Text_Token_Q_16( dummyFunc, h, a ) :
    final = 'H_Over35'
    if h < 3.5 : final = 'H_<_35'
    return final

def Text_Token_Q_17( dummyFunc, h, a ) :
    final = 'A_>_35'
    if a < 3.5 : final = 'A_<_35'
    return final

def Text_Token_Q_18( dummyFunc, h, a ) :
    final = 'H_>_45'
    if h < 4.5 : final = 'H_<_45'
    return final

def Text_Token_Q_19( dummyFunc, h, a ) :
    final = 'A_>_45'
    if a < 4.5 : final = 'A_<_45'
    return final

#===========================================

def Text_Score_Q_00( dummyFunc, h, a ) :
    final = 'Score_Not_00'
    if h == 0 and a == 0 : final = 'Score_00'
    return final

def Text_Score_Q_01( dummyFunc, h, a ) :
    final = 'Score_Not_01'
    if h == 0 and a == 1 : final = 'Score_01'
    return final

def Text_Score_Q_02( dummyFunc, h, a ) :
    final = 'Score_Not_02'
    if h == 0 and a == 2 : final = 'Score_02'
    return final

def Text_Score_Q_03( dummyFunc, h, a ) :
    final = 'Score_Not_03'
    if h == 0 and a == 3 : final = 'Score_03'
    return final

def Text_Score_Q_04( dummyFunc, h, a ) :
    final = 'Score_Not_04'
    if h == 0 and a == 4 : final = 'Score_04'
    return final

def Text_Score_Q_05( dummyFunc, h, a ) :
    final = 'Score_Not_05'
    if h == 0 and a == 5 : final = 'Score_05'
    return final
#---------------------------------------------------

def Text_Score_Q_10( dummyFunc, h, a ) :
    final = 'Score_Not_10'
    if h == 1 and a == 0 : final = 'Score_10'
    return final

def Text_Score_Q_11( dummyFunc, h, a ) :
    final = 'Score_Not_11'
    if h == 1 and a == 1 : final = 'Score_11'
    return final

def Text_Score_Q_12( dummyFunc, h, a ) :
    final = 'Score_Not_12'
    if h == 1 and a == 2 : final = 'Score_12'
    return final

def Text_Score_Q_13( dummyFunc, h, a ) :
    final = 'Score_Not_13'
    if h == 1 and a == 3 : final = 'Score_13'
    return final

def Text_Score_Q_14( dummyFunc, h, a ) :
    final = 'Score_Not_14'
    if h == 1 and a == 4 : final = 'Score_14'
    return final

def Text_Score_Q_15( dummyFunc, h, a ) :
    final = 'Score_Not_15'
    if h == 1 and a == 5 : final = 'Score_15'
    return final
#--------------------------------------------------

def Text_Score_Q_20( dummyFunc, h, a ) :
    final = 'Score_Not_20'
    if h == 2 and a == 0 : final = 'Score_20'
    return final

def Text_Score_Q_21( dummyFunc, h, a ) :
    final = 'Score_Not_21'
    if h == 2 and a == 1 : final = 'Score_21'
    return final

def Text_Score_Q_22( dummyFunc, h, a ) :
    final = 'Score_Not_22'
    if h == 2 and a == 2 : final = 'Score_22'
    return final

def Text_Score_Q_23( dummyFunc, h, a ) :
    final = 'Score_Not_23'
    if h == 2 and a == 3 : final = 'Score_23'
    return final

def Text_Score_Q_24( dummyFunc, h, a ) :
    final = 'Score_Not_24'
    if h == 2 and a == 4 : final = 'Score_24'
    return final

def Text_Score_Q_25( dummyFunc, h, a ) :
    final = 'Score_Not_25'
    if h == 2 and a == 5 : final = 'Score_25'
    return final
#-------------------------------------------------

def Text_Score_Q_30( dummyFunc, h, a ) :
    final = 'Score_Not_30'
    if h == 3 and a == 0 : final = 'Score_30'
    return final

def Text_Score_Q_31( dummyFunc, h, a ) :
    final = 'Score_Not_31'
    if h == 3 and a == 1 : final = 'Score_31'
    return final

def Text_Score_Q_32( dummyFunc, h, a ) :
    final = 'Score_Not_32'
    if h == 3 and a == 2 : final = 'Score_32'
    return final

def Text_Score_Q_33( dummyFunc, h, a ) :
    final = 'Score_Not_33'
    if h == 3 and a == 3 : final = 'Score_33'
    return final

def Text_Score_Q_34( dummyFunc, h, a ) :
    final = 'Score_Not_34'
    if h == 3 and a == 4 : final = 'Score_34'
    return final

def Text_Score_Q_35( dummyFunc, h, a ) :
    final = 'Score_Not_35'
    if h == 3 and a == 5 : final = 'Score_35'
    return final
#--------------------------------------------------

def Text_Score_Q_40( dummyFunc, h, a ) :
    final = 'Score_Not_40'
    if h == 4 and a == 0 : final = 'Score_40'
    return final

def Text_Score_Q_41( dummyFunc, h, a ) :
    final = 'Score_Not_41'
    if h == 4 and a == 1 : final = 'Score_41'
    return final

def Text_Score_Q_42( dummyFunc, h, a ) :
    final = 'Score_Not_42'
    if h == 4 and a == 2 : final = 'Score_42'
    return final

def Text_Score_Q_43( dummyFunc, h, a ) :
    final = 'Score_Not_43'
    if h == 4 and a == 3 : final = 'Score_43'
    return final

def Text_Score_Q_44( dummyFunc, h, a ) :
    final = 'Score_Not_44'
    if h == 4 and a == 4 : final = 'Score_44'
    return final

def Text_Score_Q_45( dummyFunc, h, a ) :
    final = 'Score_Not_45'
    if h == 4 and a == 5 : final = 'Score_45'
    return final
#----------------------------------------------

def Text_Score_Q_50( dummyFunc, h, a ) :
    final = 'Score_Not_50'
    if h == 5 and a == 0 : final = 'Score_50'
    return final

def Text_Score_Q_51( dummyFunc, h, a ) :
    final = 'Score_Not_51'
    if h == 5 and a == 1 : final = 'Score_51'
    return final

def Text_Score_Q_52( dummyFunc, h, a ) :
    final = 'Score_Not_52'
    if h == 5 and a == 2 : final = 'Score_52'
    return final

def Text_Score_Q_53( dummyFunc, h, a ) :
    final = 'Score_Not_53'
    if h == 5 and a == 3 : final = 'Score_53'
    return final

def Text_Score_Q_54( dummyFunc, h, a ) :
    final = 'Score_Not_54'
    if h == 5 and a == 4 : final = 'Score_54'
    return final

def Text_Score_Q_55( dummyFunc, h, a ) :
    final = 'Score_Not_55'
    if h == 5 and a == 5 : final = 'Score_55'
    return final

#============================================================================================

Text_Token_Q = [ \
	Text_Token_Q_00, Text_Token_Q_01, Text_Token_Q_02, Text_Token_Q_03, Text_Token_Q_04, \
	Text_Token_Q_05, Text_Token_Q_06, Text_Token_Q_07, Text_Token_Q_08, Text_Token_Q_09, \
	Text_Token_Q_10, Text_Token_Q_11, Text_Token_Q_12, Text_Token_Q_13, Text_Token_Q_14, \
	Text_Token_Q_15, Text_Token_Q_16, Text_Token_Q_17, Text_Token_Q_18, Text_Token_Q_19, \
]


Text_Score_Q = [ \
	Text_Score_Q_00, Text_Score_Q_01, Text_Score_Q_02, Text_Score_Q_03, Text_Score_Q_04, Text_Score_Q_05, \
	Text_Score_Q_10, Text_Score_Q_11, Text_Score_Q_12, Text_Score_Q_13, Text_Score_Q_14, Text_Score_Q_15, \
	Text_Score_Q_20, Text_Score_Q_21, Text_Score_Q_22, Text_Score_Q_23, Text_Score_Q_24, Text_Score_Q_25, \
	Text_Score_Q_30, Text_Score_Q_31, Text_Score_Q_32, Text_Score_Q_33, Text_Score_Q_34, Text_Score_Q_35, \
	Text_Score_Q_40, Text_Score_Q_41, Text_Score_Q_42, Text_Score_Q_43, Text_Score_Q_44, Text_Score_Q_45, \
	Text_Score_Q_50, Text_Score_Q_51, Text_Score_Q_52, Text_Score_Q_53, Text_Score_Q_54, Text_Score_Q_55, \
]




Text_Token_Q = [ Text_Token_Q_00, Text_Token_Q_01, Text_Token_Q_02, Text_Token_Q_03, Text_Token_Q_04, Text_Token_Q_05, Text_Token_Q_06, Text_Token_Q_07 ]
Text_Token_Q = [ Text_Token_Q_00, Text_Token_Q_01 ]

#==============================
Zero_For_Fit_Q = [ Zero_For_Fit_Score_Q_00, Zero_For_Fit_Score_Q_01, Zero_For_Fit_Score_Q_02, \
	Zero_For_Fit_Score_Q_03, Zero_For_Fit_Score_Q_04, Zero_For_Fit_Score_Q_05, Zero_For_Fit_Score_Q_06, Zero_For_Fit_Score_Q_07 ]

RvsProb_Q = [ RvsProb_Q_00, RvsProb_Q_01, RvsProb_Q_02, RvsProb_Q_03, RvsProb_Q_04, RvsProb_Q_05, RvsProb_Q_06, RvsProb_Q_07 ]
#==============================


ConcatnateDenseOne_Q = [ ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne ]



Zero_For_Fit_P = [ Zero_For_Fit_Score_P_00, Zero_For_Fit_Score_P_01, Zero_For_Fit_Score_P_02, Zero_For_Fit_Score_P_03, \
    Zero_For_Fit_Score_P_04, Zero_For_Fit_Score_P_05, Zero_For_Fit_Score_P_06, Zero_For_Fit_Score_P_07, Zero_For_Fit_Score_P_08, \
    Zero_For_Fit_Score_P_09, Zero_For_Fit_Score_P_10, Zero_For_Fit_Score_P_11, Zero_For_Fit_Score_P_12, \
    Zero_For_Fit_Score_Q_02, Zero_For_Fit_Score_Q_03, Zero_For_Fit_Score_Q_04, Zero_For_Fit_Score_Q_05, Zero_For_Fit_Score_Q_06, Zero_For_Fit_Score_Q_07 ]


RvsProb_P = [ RvsProb_P_00, RvsProb_P_01, RvsProb_P_02, RvsProb_P_03, RvsProb_P_04, RvsProb_P_05, RvsProb_P_06, \
	RvsProb_P_07, RvsProb_P_08, RvsProb_P_09, RvsProb_P_10,RvsProb_P_11, RvsProb_P_12, \
    RvsProb_Q_02, RvsProb_Q_03, RvsProb_Q_04, RvsProb_Q_05 ]


RvsProb_Sketch_P = [ RvsProb_Sketch_P_00, RvsProb_Sketch_P_01, RvsProb_Sketch_P_02, RvsProb_Sketch_P_03, RvsProb_Sketch_P_04, \
		RvsProb_Sketch_P_05, RvsProb_Sketch_P_06, RvsProb_Sketch_P_07, RvsProb_Sketch_P_08, RvsProb_Sketch_P_09, \
		RvsProb_Sketch_P_10, RvsProb_Sketch_P_11, RvsProb_Sketch_P_12, \
        RvsProb_Q_02, RvsProb_Q_03, RvsProb_Q_04, RvsProb_Q_05 ]

ConcatnateDenseOne_P = [ ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, \
    ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, \
    ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne, ConcatnateDenseOne ]


# Assume: Odds: HomeWin, Draw, AwayWin, BothScoresYes, BothScoresNo, 

def Happened_R_00( h, a ): # "Home__Win"
    return tf.greater(h, a + 0.5 )

def Happened_R_01( h, a ): # "Game_Draw"
    return tf.logical_and( tf.greater(h, a - 0.5 ), tf.less(h, a + 0.5) )

def Happened_R_02( h, a ): # "Away__Win"
    return tf.less(h, a - 0.5 )

def Happened_R_03( h, a ): # "Both_Sc_Y"
    return tf.logical_and( tf.greater(h, 0.5), tf.greater(a, 0.5) )

def Happened_R_04( h, a ): # "Both_Sc_N"
    return tf.logical_not( Happened_R_03( h, a ) )

def Happened_R_05( h, a ): # "Over__1.5"
    return tf.greater(h + a, 1.5 )

def Happened_R_06( h, a ): # "Under_1.5"
    return tf.logical_not( Happened_R_05( h, a ) )

def Happened_R_07( h, a ): # "Over__2.5"
    return tf.greater(h + a, 2.5 )

def Happened_R_08( h, a ): # "Under_2.5"
    return tf.logical_not( Happened_R_07( h, a ) )

def Happened_R_09( h, a ): # "Score_1_0"
    return tf.logical_and( tf.logical_and( tf.greater(h, 0.5 ), tf.less(h, 1.5) ), tf.logical_and( tf.greater(a, -0.5 ), tf.less(a, 0.5) ) )

def Happened_R_10( h, a ): # "Score_2_1"
    return tf.logical_and( tf.logical_and( tf.greater(h, 1.5 ), tf.less(h, 2.5) ), tf.logical_and( tf.greater(a, 0.5 ), tf.less(a, 1.5) ) )

def Happened_R_11( h, a ): # "Score_1_1"
    return tf.logical_and( tf.logical_and( tf.greater(h, 0.5 ), tf.less(h, 1.5) ), tf.logical_and( tf.greater(a, 0.5 ), tf.less(a, 1.5) ) )

def Happened_R_12( h, a ): # "Score_Oth"
    return tf.logical_not( tf.logical_or( tf.logical_or( Happened_R_09( h, a ), Happened_R_10( h, a )  ), Happened_R_11( h, a ) ) )

def Happened_R_13( h, a ): # "AH_1.5_T1"  
    return tf.greater( h - 1.5, a )

def Happened_R_14( h, a ): # "AH_1.5_T2"
    return tf.logical_not( tf.greater( h - 1.5, a ) )

Happened_R = [ Happened_R_00, Happened_R_01, Happened_R_02, Happened_R_03, Happened_R_04, Happened_R_05, Happened_R_06, Happened_R_07, \
    Happened_R_08, Happened_R_09, Happened_R_10, Happened_R_11, Happened_R_12, Happened_R_13, Happened_R_14 ]


def Odds_Happened_R_00( core, row, odds, h, a ): # "Home__Win"
    return tf.stack( [ row, odds[:, 0], tf.cast( Happened_R_00( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_01( core, row, odds, h, a ): # "Game_Draw"
    happened = tf.equal(h, a)
    return tf.stack( [ row, odds[:, 1], tf.cast( Happened_R_01( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_02( core, row, odds, h, a ): # "Away__Win"
    return tf.stack( [ row, odds[:, 2], tf.cast( Happened_R_02( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_03( core, row, odds, h, a ): # "Both_Sc_Y"
    return tf.stack( [ row, odds[:, 3], tf.cast( Happened_R_03( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_04( core, row, odds, h, a ): # "Both_Sc_N"
    return tf.stack( [ row, odds[:, 4], tf.cast( Happened_R_04( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_05( core, row, odds, h, a ): # "Over__1.5"
    return tf.stack( [ row, odds[:, 5], tf.cast( Happened_R_05( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_06( core, row, odds, h, a ): # "Under_1.5"
    return tf.stack( [ row, odds[:, 6], tf.cast( Happened_R_06( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_07( core, row, odds, h, a ): # "Over__2.5"
    return tf.stack( [ row, odds[:, 7], tf.cast( Happened_R_07( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_08( core, row, odds, h, a ): # "Under_2.5"
    return tf.stack( [ row, odds[:, 8], tf.cast( Happened_R_08( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_09( core, row, odds, h, a ): # "Score_1_0"
    return tf.stack( [ row, odds[:, 9], tf.cast( Happened_R_09( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_10( core, row, odds, h, a ): # "Score_2_1"
    return tf.stack( [ row, odds[:, 10], tf.cast( Happened_R_10( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_11( core, row, odds, h, a ): # "Score_1_1"
    return tf.stack( [ row, odds[:, 11], tf.cast( Happened_R_11( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_12( core, row, odds, h, a ): # "Score_Oth"
    return tf.stack( [ row, odds[:, 12], tf.cast( Happened_R_12( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_13( core, row, odds, h, a ): # "AH_1.5_T1"  
    return tf.stack( [ row, odds[:, 13], tf.cast( Happened_R_13( h, a ), tf.float32 ) ], axis = -1 )

def Odds_Happened_R_14( core, row, odds, h, a ): # "AH_1.5_T2"
    return tf.stack( [ row, odds[:, 14], tf.cast( Happened_R_14( h, a ), tf.float32 ) ], axis = -1 )


def BettingProb_R_ ( core, home, away ):

    input = None
    if home != None and away == None :
        input = home
    elif home == None and away != None :
        input = away
    elif home != None and away != None :
        input = tf.keras.layers.concatenate( [home, away], axis = -1 )
    else:
        raise Exception('Attend here: home and away are both None.')

    probability = tf.keras.layers.Dense( 1, activation = 'sigmoid', name = 'probability' ) (input)

    dummy = tf.ones_like( probability, tf.float32 )
    final = tf.keras.layers.concatenate( [ probability, dummy ], axis = -1 )

    tf.debugging.assert_all_finite( final, '\n ---------------------- non found. Place01' )

    return final

Odds_Happened_R = [ Odds_Happened_R_00, Odds_Happened_R_01, Odds_Happened_R_02, Odds_Happened_R_03, Odds_Happened_R_04, Odds_Happened_R_05, \
    Odds_Happened_R_06, Odds_Happened_R_07, Odds_Happened_R_08, Odds_Happened_R_09, Odds_Happened_R_10, Odds_Happened_R_11, Odds_Happened_R_12, Odds_Happened_R_13, Odds_Happened_R_14 ]

BettingProb_R = [ BettingProb_R_, BettingProb_R_, BettingProb_R_, BettingProb_R_, BettingProb_R_, BettingProb_R_, \
    BettingProb_R_, BettingProb_R_, BettingProb_R_, BettingProb_R_, BettingProb_R_, BettingProb_R_, BettingProb_R_, BettingProb_R_ ]


def Odds_Not_Happened_R_00( core, row, odds, h, a ): # "Home__Win"
    return tf.stack( [ row, odds[:, 0], tf.cast( tf.logical_not( Happened_R_00( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_01( core, row, odds, h, a ): # "Game_Draw"
    return tf.stack( [ row, odds[:, 1], tf.cast( tf.logical_not( Happened_R_01( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_02( core, row, odds, h, a ): # "Away__Win"
    return tf.stack( [ row, odds[:, 2], tf.cast( tf.logical_not( Happened_R_02( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_03( core, row, odds, h, a ): # "Both_Sc_Y"
    return tf.stack( [ row, odds[:, 3], tf.cast( tf.logical_not( Happened_R_03( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_04( core, row, odds, h, a ): # "Both_Sc_N"
    return tf.stack( [ row, odds[:, 4], tf.cast( tf.logical_not( Happened_R_04( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_05( core, row, odds, h, a ): # "Over__1.5"
    return tf.stack( [ row, odds[:, 5], tf.cast( tf.logical_not( Happened_R_05( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_06( core, row, odds, h, a ): # "Under_1.5"
    return tf.stack( [ row, odds[:, 6], tf.cast( tf.logical_not( Happened_R_06( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_07( core, row, odds, h, a ): # "Over__2.5"
    return tf.stack( [ row, odds[:, 7], tf.cast( tf.logical_not( Happened_R_07( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_08( core, row, odds, h, a ): # "Under_2.5"
    return tf.stack( [ row, odds[:, 8], tf.cast( tf.logical_not( Happened_R_08( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_09( core, row, odds, h, a ): # "Score_1_0"
    return tf.stack( [ row, odds[:, 9], tf.cast( tf.logical_not( Happened_R_09( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_10( core, row, odds, h, a ): # "Score_2_1"
    return tf.stack( [ row, odds[:, 10], tf.cast( tf.logical_not( Happened_R_10( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_11( core, row, odds, h, a ): # "Score_1_1"
    return tf.stack( [ row, odds[:, 11], tf.cast( tf.logical_not( Happened_R_11( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_12( core, row, odds, h, a ): # "Score_Oth"
    return tf.stack( [ row, odds[:, 12], tf.cast( tf.logical_not( Happened_R_12( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_13( core, row, odds, h, a ): # "AH_1.5_T1" 
    return tf.stack( [ row, odds[:, 13], tf.cast( tf.logical_not( Happened_R_13( h, a ) ), tf.float32 ) ], axis = -1 )

def Odds_Not_Happened_R_14( core, row, odds, h, a ): # "AH_1.5_T2"
    return tf.stack( [ row, odds[:, 14], tf.cast( tf.logical_not( Happened_R_14( h, a ) ), tf.float32 ) ], axis = -1 )


Odds_Not_Happened_R = [ Odds_Not_Happened_R_00, Odds_Not_Happened_R_01, Odds_Not_Happened_R_02, Odds_Not_Happened_R_03, Odds_Not_Happened_R_04, \
    Odds_Not_Happened_R_05, Odds_Not_Happened_R_06, Odds_Not_Happened_R_07, Odds_Not_Happened_R_08, Odds_Not_Happened_R_09, Odds_Not_Happened_R_10, \
    Odds_Not_Happened_R_11, Odds_Not_Happened_R_12, Odds_Not_Happened_R_13, Odds_Not_Happened_R_14 \
    ]


def Prob_Not_Happened_R_00( h, a ): # "Home__Win"
    return tf.cast( tf.logical_not( Happened_R_00( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_01( h, a ): # "Game_Draw"
    return tf.cast( tf.logical_not( Happened_R_01( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_02( h, a ): # "Away__Win"
    return tf.cast( tf.logical_not( Happened_R_02( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_03( h, a ): # "Both_Sc_Y"
    return tf.cast( tf.logical_not( Happened_R_03( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_04( h, a ): # "Both_Sc_N"
    return tf.cast( tf.logical_not( Happened_R_04( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_05( h, a ): # "Over__1.5"
    return tf.cast( tf.logical_not( Happened_R_05( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_06( h, a ): # "Under_1.5"
    return tf.cast( tf.logical_not( Happened_R_06( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_07( h, a ): # "Over__2.5"
    return tf.cast( tf.logical_not( Happened_R_07( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_08( h, a ): # "Under_2.5"
    return tf.cast( tf.logical_not( Happened_R_08( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_09( h, a ): # "Score_1_0"
    return tf.cast( tf.logical_not( Happened_R_09( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_10( h, a ): # "Score_2_1"
    return tf.cast( tf.logical_not( Happened_R_10( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_11( h, a ): # "Score_1_1"
    return tf.cast( tf.logical_not( Happened_R_11( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_12( h, a ): # "Score_1_1"
    return tf.cast( tf.logical_not( Happened_R_12( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_13( h, a ): # "AH_1.5_T1" 
    return tf.cast( tf.logical_not( Happened_R_13( h, a ) ), tf.float32 )

def Prob_Not_Happened_R_14( h, a ): # "AH_1.5_T2"
    return tf.cast( tf.logical_not( Happened_R_14( h, a ) ), tf.float32 )


Prob_Not_Happened_R = [ Prob_Not_Happened_R_00, Prob_Not_Happened_R_01, Prob_Not_Happened_R_02, Prob_Not_Happened_R_03, Prob_Not_Happened_R_04, \
    Prob_Not_Happened_R_05, Prob_Not_Happened_R_06, Prob_Not_Happened_R_07, Prob_Not_Happened_R_08, Prob_Not_Happened_R_09, Prob_Not_Happened_R_10, \
    Prob_Not_Happened_R_11, Prob_Not_Happened_R_12, Prob_Not_Happened_R_13, Prob_Not_Happened_R_14 \
    ]


def Prob_Happened_R_00( h, a ): # "Home__Win"
    return tf.cast( Happened_R_00( h, a ), tf.int64 )

def Prob_Happened_R_01( h, a ): # "Game_Draw"
    return tf.cast( Happened_R_01( h, a ), tf.int64 )

def Prob_Happened_R_02( h, a ): # "Away__Win"
    return tf.cast( Happened_R_02( h, a ), tf.int64 )

def Prob_Happened_R_03( h, a ): # "Both_Sc_Y"
    return tf.cast( Happened_R_03( h, a ), tf.int64 )

def Prob_Happened_R_04( h, a ): # "Both_Sc_N"
    return tf.cast( Happened_R_04( h, a ), tf.int64 )

def Prob_Happened_R_05( h, a ): # "Over__1.5"
    return tf.cast( Happened_R_05( h, a ), tf.int64 )

def Prob_Happened_R_06( h, a ): # "Under_1.5"
    return tf.cast( Happened_R_06( h, a ), tf.int64 )

def Prob_Happened_R_07( h, a ): # "Over__2.5"
    return tf.cast( Happened_R_07( h, a ), tf.int64 )

def Prob_Happened_R_08( h, a ): # "Under_2.5"
    return tf.cast( Happened_R_08( h, a ), tf.int64 )

def Prob_Happened_R_09( h, a ): # "Score_1_0"
    return tf.cast( Happened_R_09( h, a ), tf.int64 )

def Prob_Happened_R_10( h, a ): # "Score_2_1"
    return tf.cast( Happened_R_10( h, a ), tf.int64 )

def Prob_Happened_R_11( h, a ): # "Score_1_1"
    return tf.cast( Happened_R_11( h, a ), tf.int64 )

def Prob_Happened_R_12( h, a ): # "Score_Oth"
    return tf.cast( Happened_R_12( h, a ), tf.int64 )

def Prob_Happened_R_13( h, a ): # "AH_1.5_T1" 
    return tf.cast( Happened_R_13( h, a ), tf.int64 )

def Prob_Happened_R_14( h, a ): # "AH_1.5_T2"
    return tf.cast( Happened_R_14( h, a ), tf.int64 )


Prob_Happened_R = [ Prob_Happened_R_00, Prob_Happened_R_01, Prob_Happened_R_02, Prob_Happened_R_03, Prob_Happened_R_04, \
    Prob_Happened_R_05, Prob_Happened_R_06, Prob_Happened_R_07, Prob_Happened_R_08, Prob_Happened_R_09, Prob_Happened_R_10, \
    Prob_Happened_R_11, Prob_Happened_R_12, Prob_Happened_R_13, Prob_Happened_R_14 \
    ]

def Prob_Happened_R_Concat_13( hGoal, aGoal, name = None ) :
    tensorList = [ Prob_Happened_R[query] ( hGoal, aGoal ) for query in range(13) ]

    tensor = tf.stack( tensorList, axis = -1, name = name )

    return tensor


def Prob_Happened_Contest_home_win( outcome ) : # outcome.shape = [n, 1]
    return tf.cast( tf.logical_and( 0.5 < outcome, outcome < 1.5 ), dtype = tf.float32 )

def Prob_Happened_Contest_draw( outcome ) : # outcome.shape = [n, 1]
    return tf.cast( tf.logical_and( -0.5 < outcome, outcome < 0.5 ), dtype = tf.float32 )

def Prob_Happened_Contest_away_win( outcome ) : # outcome.shape = [n, 1]
    return tf.cast( tf.logical_and( -1.5 < outcome, outcome < -0.5 ), dtype = tf.float32 )


def Prob_Happened_Contest( outcome ) :
    result = tf.stack( [ Prob_Happened_Contest_home_win( outcome ), Prob_Happened_Contest_draw( outcome ), Prob_Happened_Contest_away_win( outcome ) ], axis = -1 )
    return result


def NotBettingProb_Precise_R_00 ( core, home, away, name = None ): # "Home__Win"
    final = OR( \
        RvsProb_P_07( core, home, away, name = name ), \
        RvsProb_Sketch_P_07_Shift( core, home, away, name = name ), \
    )
    return final

def NotBettingProb_Precise_R_01 ( core, home, away, name = None ): # "Game_Draw"
    final = RvsProb_P_09 ( core, home, away, name = name )
    return final

def NotBettingProb_Precise_R_02 ( core, home, away, name = None ): # "Away__Win"
    final = OR( \
        RvsProb_P_08( core, home, away, name = name ), \
        RvsProb_Sketch_P_08_Shift( core, home, away, name = name ), \
    )
    return final

def NotBettingProb_Precise_R_03 ( core, home, away, name = None ): # "Both_Sc_Y"
    final = RvsProb_P_02 ( core, home, away, name = name )
    return final

def NotBettingProb_Precise_R_04 ( core, home, away, name = None ): # "Both_Sc_N"
    final = RvsProb_P_03 ( core, home, away, name = name )
    return final

def NotBettingProb_Precise_R_05 ( core, home, away, name = None ): # "Over__1.5"
    final = OR( \
        RvsProb_P_14( core, home, away, name = name ), \
        RvsProb_Sketch_P_14_Shift( core, home, away, name = name ), \
    )
    return final

def NotBettingProb_Precise_R_06 ( core, home, away, name = None ): # "Under_1.5"
    final = OR( \
        RvsProb_P_15( core, home, away, name = name ), \
        RvsProb_Sketch_P_15_Shift( core, home, away, name = name ), \
    )
    return final

def NotBettingProb_Precise_R_07 ( core, home, away, name = None ): # "Over__2.5"
    final = OR( \
        RvsProb_P_16( core, home, away, name = name ), \
        RvsProb_Sketch_P_16_Shift( core, home, away, name = name ), \
    )
    return final

def NotBettingProb_Precise_R_08 ( core, home, away, name = None ): # "Under_2.5"
    final = OR( \
        RvsProb_P_17( core, home, away, name = name ), \
        RvsProb_Sketch_P_17_Shift( core, home, away, name = name ), \
    )
    return final

def NotBettingProb_Precise_R_09 ( core, home, away, name = None ): # "Score_1_0"
    final = RvsProb_P_10 ( core, home, away, name = name )
    return final

def NotBettingProb_Precise_R_10 ( core, home, away, name = None ): # "Score_2_1"
    final = RvsProb_P_12 ( core, home, away, name = name )
    return final

def NotBettingProb_Precise_R_11 ( core, home, away, name = None ): # "Score_1_1"
    final = RvsProb_P_11 ( core, home, away, name = name )
    return final

def NotBettingProb_Precise_R_12 ( core, home, away, name = None ): # "Score_Oth" # PENDING correct it.
    final = RvsProb_P_11 ( core, home, away, name = name )
    return final

def NotBettingProb_Precise_R_13 ( core, home, away, name = None ): # "AH_1.5_T1" 
    return NotBettingProb_Precise_R_00 ( core, home - 1.0, away, name = name )

def NotBettingProb_Precise_R_14 ( core, home, away, name = None ): # "AH_1.5_T2"
    return 1.0 - NotBettingProb_Precise_R_00 ( core, home - 1.0, away, name )

NotBettingProb_Precise_R = [ NotBettingProb_Precise_R_00, NotBettingProb_Precise_R_01, NotBettingProb_Precise_R_02, NotBettingProb_Precise_R_03, \
    NotBettingProb_Precise_R_04, NotBettingProb_Precise_R_05, NotBettingProb_Precise_R_06, NotBettingProb_Precise_R_07, NotBettingProb_Precise_R_08, \
    NotBettingProb_Precise_R_09, NotBettingProb_Precise_R_10, NotBettingProb_Precise_R_11, NotBettingProb_Precise_R_12, NotBettingProb_Precise_R_13, NotBettingProb_Precise_R_14 ]


def Prob_R___ ( core, prob_logits, name = None ):

    return core( prob_logits )

Prob_R = [ Prob_R___, Prob_R___, Prob_R___, Prob_R___, \
    Prob_R___, Prob_R___, Prob_R___, Prob_R___, Prob_R___, \
    Prob_R___, Prob_R___, Prob_R___, Prob_R___, Prob_R___, Prob_R___ ]


def Prob_R_Concat_13 ( core, prob_logits, name = None ):

    #tensorList = [ Prob_R[query_id] ( core, prob_logits[:, query_id], name = name ) for query_id in range(12) ]
    #tensor = tf.stack( tensorList, axis = -1, name = name )

    tensor = core( prob_logits, name = name )

    return tensor