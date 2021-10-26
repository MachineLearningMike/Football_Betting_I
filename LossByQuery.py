import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

threshold = 1e-7 # 
epsilon = 1e-7

#======================== The following functions must be differentiable, as they participate in loss function.

def Metrics_ReLu_01(h, a) : # for original Query 1: Home team scores.
    return tf.keras.activations.relu( - h + 0.5, alpha = 0.0, max_value = None, threshold = 0)

def Metrics_ReLu_02(h, a) : # for original Query 2: Away team scores.
    return tf.keras.activations.relu( - a + 0.5, alpha = 0.0, max_value = None, threshold = 0)

def Metrics_ReLu_03(h, a) : # for original Query 3: Both teams score.
    return tf.keras.activations.relu( - h + 0.5, alpha = 0.0, max_value = None, threshold = 0) + \
        tf.keras.activations.relu( - a + 0.5, alpha = 0.0, max_value = None, threshold = 0)

def Metrics_ReLu_04(h, a) : # for original Query 4: At most 1 team scores.
    return tf.keras.activations.relu( h - 0.5, alpha = 0.0, max_value = None, threshold = 0) * \
        tf.keras.activations.relu( a - 0.5, alpha = 0.0, max_value = None, threshold = 0)

def Metrics_ReLu_05(h, a) : # for original Query 5: At least 1 team scores more than 2 goals.
    return tf.keras.activations.relu( - h + 2.5, alpha = 0.0, max_value = None, threshold = 0) * \
        tf.keras.activations.relu( - a + 2.5, alpha = 0.0, max_value = None, threshold = 0)

def Metrics_ReLu_06(h, a) : # for original Query 6: At least 1 team scores less than 3 goals.
    return tf.keras.activations.relu( h - 2.5, alpha = 0.0, max_value = None, threshold = 0) * \
        tf.keras.activations.relu( a - 2.5, alpha = 0.0, max_value = None, threshold = 0)

def Metrics_ReLu_07(h, a) : # for original Query 7: At leas 1 team scores more than 1 goal.
    return tf.keras.activations.relu( - h + 1.5, alpha = 0.0, max_value = None, threshold = 0) * \
        tf.keras.activations.relu( - a + 1.5, alpha = 0.0, max_value = None, threshold = 0)

def Metrics_ReLu_08(h, a) : # for original Query 8: Home team wins.
    # Wrong. Think of rounding.
    return tf.keras.activations.relu( - ( h - a ) + 0.5, alpha = 0.0, max_value = None, threshold = 0)

def Metrics_ReLu_09(h, a) : # for original Query 9: Away team wins.
    # Wrong. Think of rounding.
    return tf.keras.activations.relu( - ( a - h ) + 0.5, alpha = 0.0, max_value = None, threshold = 0)

def Metrics_ReLu_10(h, a) : # for original Query 10: The match draws.
    # Wrong. Think of rounding.
    return tf.keras.activations.relu( ( h - a ) - 0.5, alpha = 0.0, max_value = None, threshold = 0) \
        + tf.keras.activations.relu( ( a - h ) - 0.5, alpha = 0.0, max_value = None, threshold = 0)

def Metrics_ReLu_11(h, a) : # for original Query 10: The score.
    return tf.abs( h - a )

def Metrics_ReLu_12(h, a) : # for original Query 12: The score is 1:0.
    return tf.keras.activations.relu( - h + 0.5, alpha = 0.0, max_value = None, threshold = 0) \
        + tf.keras.activations.relu( h - 1.5, alpha = 0.0, max_value = None, threshold = 0) \
        + tf.keras.activations.relu( - a - 0.5, alpha = 0.0, max_value = None, threshold = 0) \
        + tf.keras.activations.relu( a - 0.5, alpha = 0.0, max_value = None, threshold = 0)

def Metrics_ReLu_13(h, a) : # for original Query 13: The score is 1:1.
    return tf.keras.activations.relu( - h + 0.5, alpha = 0.0, max_value = None, threshold = 0) \
        + tf.keras.activations.relu( h - 1.5, alpha = 0.0, max_value = None, threshold = 0) \
        + tf.keras.activations.relu( - a + 0.5, alpha = 0.0, max_value = None, threshold = 0) \
        + tf.keras.activations.relu( a - 1.5, alpha = 0.0, max_value = None, threshold = 0)

def Metrics_ReLu_14(h, a) : # for original Query 14: The score is 2:1.
    return tf.keras.activations.relu( - h + 1.5, alpha = 0.0, max_value = None, threshold = 0) \
        + tf.keras.activations.relu( h - 2.5, alpha = 0.0, max_value = None, threshold = 0) \
        + tf.keras.activations.relu( - a + 0.5, alpha = 0.0, max_value = None, threshold = 0) \
        + tf.keras.activations.relu( a - 1.5, alpha = 0.0, max_value = None, threshold = 0)



sigmoid_slop = 15.

def Metrics_Sigmoid_01(h, a) : # for original Query 1: Home team scores.
    return tf.keras.activations.sigmoid( ( - h + 0.5 ) * sigmoid_slop )

def Metrics_Sigmoid_02(h, a) : # for original Query 2: Away team scores.
    return tf.keras.activations.sigmoid( ( - a + 0.5 ) * sigmoid_slop )

def Metrics_Sigmoid_03(h, a) : # for original Query 3: Both teams score.
    return tf.math.maximum ( \
        tf.keras.activations.sigmoid( ( - h + 0.5 ) * sigmoid_slop ), \
        tf.keras.activations.sigmoid( ( - a + 0.5 ) * sigmoid_slop ) \
        )

def Metrics_Sigmoid_04(h, a) : # for original Query 4: At most 1 team scores.
    return tf.keras.activations.sigmoid( ( h - 0.5 ) * sigmoid_slop ) * \
        tf.keras.activations.sigmoid( ( a - 0.5 ) * sigmoid_slop )

def Metrics_Sigmoid_05(h, a) : # for original Query 5: At least 1 team scores more than 2 goals.
    return tf.keras.activations.sigmoid( ( - h + 2.5 ) * sigmoid_slop ) * \
        tf.keras.activations.sigmoid( ( - a + 2.5 ) * sigmoid_slop )

def Metrics_Sigmoid_06(h, a) : # for original Query 6: At least 1 team scores less than 3 goals.
    return tf.keras.activations.sigmoid( ( h - 2.5 ) * sigmoid_slop ) * \
        tf.keras.activations.sigmoid( ( a - 2.5 ) * sigmoid_slop )

def Metrics_Sigmoid_07(h, a) : # for original Query 7: At leas 1 team scores more than 1 goal.
    return tf.keras.activations.sigmoid( ( - h + 1.5 ) * sigmoid_slop ) * \
        tf.keras.activations.sigmoid( ( - a + 1.5 ) * sigmoid_slop )

def Metrics_Sigmoid_08(h, a) : # for original Query 8: Home team wins.
    # Wrong. Think of rounding.
    return tf.keras.activations.sigmoid( ( - ( h - a ) ) * sigmoid_slop )

def Metrics_Sigmoid_09(h, a) : # for original Query 9: Away team wins.
    # Wrong. Think of rounding.
    return tf.keras.activations.sigmoid( ( - ( a - h ) ) * sigmoid_slop )

def Metrics_Sigmoid_10(h, a) : # for original Query 10: The match draws.
    # Wrong. Think of rounding.
    return tf.keras.activations.sigmoid( ( ( h - a ) - 0.5 ) * sigmoid_slop ) \
        + tf.keras.activations.sigmoid( ( ( a - h ) - 0.5 ) * sigmoid_slop )

def Metrics_Sigmoid_11(h, a) : # for original Query 10: The score.
   return tf.math.maximum ( tf.keras.activations.sigmoid( ( - h + 0.5 ) * sigmoid_slop ),
        tf.keras.activations.sigmoid( ( - a + 0.5 ) * sigmoid_slop ) )

def Metrics_Sigmoid_12(h, a) : # for original Query 12: The score is 1:0.
    return tf.math.maximum( \
        tf.math.maximum( tf.keras.activations.sigmoid( ( - h + 0.5 ) * sigmoid_slop ), \
        tf.keras.activations.sigmoid( ( h - 1.5 ) * sigmoid_slop ) ), \
        tf.math.maximum( tf.keras.activations.sigmoid( ( - a - 0.5 ) * sigmoid_slop ), \
        tf.keras.activations.sigmoid( ( a - 0.5 ) * sigmoid_slop ) ) \
        )

def Metrics_Sigmoid_13(h, a) : # for original Query 13: The score is 1:1.
    return tf.math.maximum ( \
        tf.math.maximum( tf.keras.activations.sigmoid( ( - h + 0.5 ) * sigmoid_slop ), \
        tf.keras.activations.sigmoid( ( h - 1.5 ) * sigmoid_slop ) ), \
        tf.math.maximum( tf.keras.activations.sigmoid( ( - a + 0.5 ) * sigmoid_slop ), \
        tf.keras.activations.sigmoid( ( a - 1.5 ) * sigmoid_slop ) ) \
        )

def Metrics_Sigmoid_14(h, a) : # for original Query 14: The score is 2:1.
    return tf.math.maximum( \
        tf.math.maximum( tf.keras.activations.sigmoid( ( - h + 1.5 ) * sigmoid_slop ), \
        tf.keras.activations.sigmoid( ( h - 2.5 ) * sigmoid_slop ) ), \
        tf.math.maximum( tf.keras.activations.sigmoid( ( - a + 0.5 ) * sigmoid_slop ), \
        tf.keras.activations.sigmoid( ( a - 1.5 ) * sigmoid_slop ) ) \
        )

Metrics_Sigmoid_By_Query = [ Metrics_Sigmoid_01, Metrics_Sigmoid_02, Metrics_Sigmoid_03, Metrics_Sigmoid_04,
        Metrics_Sigmoid_05, Metrics_Sigmoid_06, Metrics_Sigmoid_07, Metrics_Sigmoid_08, Metrics_Sigmoid_09,
        Metrics_Sigmoid_10, Metrics_Sigmoid_11, Metrics_Sigmoid_12, Metrics_Sigmoid_13, Metrics_Sigmoid_14 ]



def Input_Alpha_X_plus_B_with_Slope( function, input_x, alpha, beta, slope, name = None ):
    """
    alphaSlope = tf.constant( alpha * slope, shape = (input_x.shape[-1], input_x.shape[-1]), dtype = tf.float32 )
    betaSlope = tf.constant( beta * slope, shape = (input_x.shape[-1], input_x.shape[-1]), dtype = tf.float32 )

    input_ax = tf.keras.layers.Multiply() ( [alphaSlope, input_x] )
    input_ax_plus_b = tf.keras.layers.Add() ( [input_ax, betaSlope] )

    final = tf.keras.layers.Dense( input_x.shape[-1], activation = function, \
        kernel_initializer = tf.constant_initializer(1.), \
        bias_initializer = tf.constant_initializer(0.), \
    name = name, trainable = False ) \
    ( input_ax_plus_b )
    """
    """
    final = tf.keras.layers.Dense( units = input_x.shape[-1], activation = function, \
        kernel_initializer = tf.keras.initializers.constant( alpha * slope ),  \
        bias_initializer = tf.keras.initializers.constant( beta * slope ), \
        trainable = False, name = name ) ( input_x )
    """
    final = tf.keras.layers.Lambda( lambda x : function ( slope * ( alpha * x + beta  ) ), trainable = False, name = name ) ( input_x )

    return final
 

def Input_Alpha_X_plus_Beta_Y( alpha, input_x, beta, input_y, name = None ):

    final = tf.keras.layers.Lambda( lambda x : ( alpha * x[0] + beta * x[1]  ), trainable = False, name = name ) ( [input_x, input_y] )

    return final

slope = 15.
linear = tf.keras.activations.linear

def Metrics_Flex_01( function, home, away, name = None ) :
    final = Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = -1.0, beta = 0.5, slope = slope, name = name )

    return final

def Metrics_Flex_02( function, home, away, name = None ) : # Away, > 0.5
    final = Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = -1.0, beta = 0.5, slope = slope, name = name )

    return final

def Metrics_Flex_03( function, home, away, name = None ) : # Both, > 0.5
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = -1.0, beta = 0.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = -1.0, beta = 0.5, slope = slope ) \
    ] )

    return final

def Metrics_Flex_04( function, home, away, name = None ) : # <= 1, > 0.5
    final = tf.keras.layers.Multiply( name = name ) ( [ \
     	Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = 1.0, beta = -0.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = 1.0, beta = -0.5, slope = slope ) \
    ] )

    return final

def Metrics_Flex_05( function, home, away, name = None ) : # >= 1, > 2.5
    final = tf.keras.layers.Multiply( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = -1.0, beta = 2.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = -1.0, beta = 2.5, slope = slope ) \
    ] )

    return final

def Metrics_Flex_06( function, home, away, name = None ) : # >= 1, < 2.5
    final = tf.keras.layers.Multiply( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = 1.0, beta = -2.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = 1.0, beta = -2.5, slope = slope ) \
    ] )

    return final

def Metrics_Flex_07( function, home, away, name = None ) : # >= 1, > 1.5
    final = tf.keras.layers.Multiply ( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = -1.0, beta = 1.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = -1.0, beta = 1.5, slope = slope ) \
    ] )

    return final

def Metrics_Flex_08( function, home, away, name = None ) : # Home > Away
    H =  Input_Alpha_X_plus_B_with_Slope( function = linear, input_x = home, alpha = 1.0, beta = 0.0, slope = slope )
    _A = Input_Alpha_X_plus_B_with_Slope( function = linear, input_x = away, alpha = -1.0, beta = 0.0, slope = slope )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = Input_Alpha_X_plus_B_with_Slope( function = function, input_x = H_A, alpha = -1.0, beta = 0.5, slope = 1.0, name = name )   

    return final

def Metrics_Flex_09( function, home, away, name = None ) : # Home < Away
    H =  Input_Alpha_X_plus_B_with_Slope( function = linear, input_x = home, alpha = 1.0, beta = 0.0, slope = slope )
    _A = Input_Alpha_X_plus_B_with_Slope( function = linear, input_x = away, alpha = -1.0, beta = 0.0, slope = slope )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = Input_Alpha_X_plus_B_with_Slope( function = function, input_x = H_A, alpha = 1.0, beta = -0.5, slope = 1.0, name = name )  

    return final

def Metrics_Flex_10( function, home, away, name = None ) : # Home = Away
    H =  Input_Alpha_X_plus_B_with_Slope( function = linear, input_x = home, alpha = 1.0, beta = 0.0, slope = slope )
    _A = Input_Alpha_X_plus_B_with_Slope( function = linear, input_x = away, alpha = -1.0, beta = 0.0, slope = slope )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = tf.keras.layers.Add( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = H_A, alpha = -1.0, beta = -0.5, slope = 1.0 ),
	    Input_Alpha_X_plus_B_with_Slope( function = function, input_x = H_A, alpha = 1.0, beta = -0.5, slope = 1.0 )
    ] )

    return final

def Metrics_Flex_11( function, home, away, name = None ) : # Pin - Point
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = -1.0, beta = 0.5, slope = slope ),
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = -1.0, beta = 0.5, slope = slope )
    ] )

    return final

def Metrics_Flex_12( function, home, away, name = None ) : # Score 1 : 0
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = -1.0, beta = 0.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = 1.0, beta = -1.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = -1.0, beta = -0.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = 1.0, beta = -0.5, slope = slope )
    ] )
 
    return final

def Metrics_Flex_13( function, home, away, name = None ) : # Score 1 : 1
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = -1.0, beta = 0.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = 1.0, beta = -1.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = -1.0, beta = 0.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = 1.0, beta = -1.5, slope = slope )
    ] )

    return final


def Metrics_Flex_14( function, home, away, name = None ) : # Score 2 : 1
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = -1.0, beta = 1.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = home, alpha = 1.0, beta = -2.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = -1.0, beta = 0.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = function, input_x = away, alpha = 1.0, beta = -1.5, slope = slope )
    ] )

    return final


Metrics_Flex_By_Query = [ Metrics_Flex_01, Metrics_Flex_02, Metrics_Flex_03, Metrics_Flex_04, Metrics_Flex_05, \
		Metrics_Flex_06, Metrics_Flex_07, Metrics_Flex_08, Metrics_Flex_09, Metrics_Flex_10, \
		Metrics_Flex_11, Metrics_Flex_12, Metrics_Flex_13, Metrics_Flex_14 ]



sigmoid = tf.keras.activations.sigmoid

def Metrics_Half_Flex_01( home, away, name = None ) :
    final = Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = -1.0, beta = 0.5, slope = slope, name = name )

    return final

def Metrics_Half_Flex_02( home, away, name = None ) : # Away, > 0.5
    final = Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = -1.0, beta = 0.5, slope = slope, name = name )

    return final

def Metrics_Half_Flex_03( home, away, name = None ) : # Both, > 0.5
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = -1.0, beta = 0.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = -1.0, beta = 0.5, slope = slope ) \
    ] )

    return final

def Metrics_Half_Flex_04( home, away, name = None ) : # <= 1, > 0.5
    final = tf.keras.layers.Multiply( name = name ) ( [ \
     	Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = 1.0, beta = -0.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = 1.0, beta = -0.5, slope = slope ) \
    ] )

    return final

def Metrics_Half_Flex_05( home, away, name = None ) : # >= 1, > 2.5
    final = tf.keras.layers.Multiply( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = -1.0, beta = 2.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = -1.0, beta = 2.5, slope = slope ) \
    ] )

    return final

def Metrics_Half_Flex_06( home, away, name = None ) : # >= 1, < 2.5
    final = tf.keras.layers.Multiply( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = 1.0, beta = -2.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = 1.0, beta = -2.5, slope = slope ) \
    ] )

    return final

def Metrics_Half_Flex_07( home, away, name = None ) : # >= 1, > 1.5
    final = tf.keras.layers.Multiply ( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = -1.0, beta = 1.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = -1.0, beta = 1.5, slope = slope ) \
    ] )

    return final

def Metrics_Half_Flex_08( home, away, name = None ) : # Home > Away
    H =  Input_Alpha_X_plus_B_with_Slope( function = linear, input_x = home, alpha = 1.0, beta = 0.0, slope = slope )
    _A = Input_Alpha_X_plus_B_with_Slope( function = linear, input_x = away, alpha = -1.0, beta = 0.0, slope = slope )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = H_A, alpha = -1.0, beta = 0.5, slope = 1.0, name = name )   

    return final

def Metrics_Half_Flex_09( home, away, name = None ) : # Home < Away
    H =  Input_Alpha_X_plus_B_with_Slope( function = linear, input_x = home, alpha = 1.0, beta = 0.0, slope = slope )
    _A = Input_Alpha_X_plus_B_with_Slope( function = linear, input_x = away, alpha = -1.0, beta = 0.0, slope = slope )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = H_A, alpha = 1.0, beta = -0.5, slope = 1.0, name = name )  

    return final

def Metrics_Half_Flex_10( home, away, name = None ) : # Home = Away
    H =  Input_Alpha_X_plus_B_with_Slope( function = linear, input_x = home, alpha = 1.0, beta = 0.0, slope = slope )
    _A = Input_Alpha_X_plus_B_with_Slope( function = linear, input_x = away, alpha = -1.0, beta = 0.0, slope = slope )
    H_A = tf.keras.layers.Add() ( [ H, _A ] )
    final = tf.keras.layers.Add( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = H_A, alpha = -1.0, beta = -0.5, slope = 1.0 ),
	    Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = H_A, alpha = 1.0, beta = -0.5, slope = 1.0 )
    ] )

    return final

def Metrics_Half_Flex_11( home, away, name = None ) : # Pin - Point
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = -1.0, beta = 0.5, slope = slope ),
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = -1.0, beta = 0.5, slope = slope )
    ] )

    return final

def Metrics_Half_Flex_12( home, away, name = None ) : # Score 1 : 0
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = -1.0, beta = 0.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = 1.0, beta = -1.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = -1.0, beta = -0.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = 1.0, beta = -0.5, slope = slope )
    ] )
 
    return final

def Metrics_Half_Flex_13( home, away, name = None ) : # Score 1 : 1
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = -1.0, beta = 0.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = 1.0, beta = -1.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = -1.0, beta = 0.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = 1.0, beta = -1.5, slope = slope )
    ] )

    return final


def Metrics_Half_Flex_14( home, away, name = None ) : # Score 2 : 1
    final = tf.keras.layers.Maximum( name = name ) ( [ \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = -1.0, beta = 1.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = home, alpha = 1.0, beta = -2.5, slope = slope ), \
        Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = -1.0, beta = 0.5, slope = slope ), \
		Input_Alpha_X_plus_B_with_Slope( function = sigmoid, input_x = away, alpha = 1.0, beta = -1.5, slope = slope )
    ] )

    return final


Metrics_Half_Flex_By_Query = [ Metrics_Half_Flex_01, Metrics_Half_Flex_02, Metrics_Half_Flex_03, Metrics_Half_Flex_04, Metrics_Half_Flex_05, \
		Metrics_Half_Flex_06, Metrics_Half_Flex_07, Metrics_Half_Flex_08, Metrics_Half_Flex_09, Metrics_Half_Flex_10, \
		Metrics_Half_Flex_11, Metrics_Half_Flex_12, Metrics_Half_Flex_13, Metrics_Half_Flex_14 ]

#======================== The following functions must be differentiable, as they participate in loss function.

relu_slopt = 15.


def Metrics_Sigmoid_ReLu_01(h, a) : # for original Query 1: Home team scores.
    return tf.keras.activations.sigmoid( Metrics_ReLu_01(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_02(h, a) : # for original Query 2: Away team scores.
    return tf.keras.activations.sigmoid( Metrics_ReLu_02(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_03(h, a) : # for original Query 3: Both teams score.
    return tf.keras.activations.sigmoid( Metrics_ReLu_03(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_04(h, a) : # for original Query 4: At most 1 team scores.
    return tf.keras.activations.sigmoid( Metrics_ReLu_04(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_05(h, a) : # for original Query 5: At least 1 team scores more than 2 goals.
    return tf.keras.activations.sigmoid( Metrics_ReLu_05(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_06(h, a) : # for original Query 6: At least 1 team scores less than 3 goals.
    return tf.keras.activations.sigmoid( Metrics_ReLu_06(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_07(h, a) : # for original Query 7: At leas 1 team scores more than 1 goal.
    return tf.keras.activations.sigmoid( Metrics_ReLu_07(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_08(h, a) : # for original Query 8: Home team wins.
    return tf.keras.activations.sigmoid( Metrics_ReLu_08(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_09(h, a) : # for original Query 9: Away team wins.
    return tf.keras.activations.sigmoid( Metrics_ReLu_09(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_10(h, a) : # for original Query 10: The match draws.
    return tf.keras.activations.sigmoid( Metrics_ReLu_10(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_11(h, a) : # for original Query 10: The score.
    return tf.keras.activations.sigmoid( Metrics_ReLu_11(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_12(h, a) : # for original Query 12: The score is 1:0.
    return tf.keras.activations.sigmoid( Metrics_ReLu_12(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_13(h, a) : # for original Query 13: The score is 1:1.
    return tf.keras.activations.sigmoid( Metrics_ReLu_13(h, a) * relu_slopt )

def Metrics_Sigmoid_ReLu_14(h, a) : # for original Query 14: The score is 2:1.
    return tf.keras.activations.sigmoid( Metrics_ReLu_14(h, a) * relu_slopt )

#======================== The following functions don't have to be differentiable, as they dont't participate in loss function.

def Metrics_Bool_01(h, a) : # for original Query 1: Home team scores.
    return tf.math.greater( h, 0.5 )

def Metrics_Bool_02(h, a) : # for original Query 2: Away team scores.
    return tf.math.greater( a, 0.5 )

def Metrics_Bool_03(h, a) : # for original Query 3: Both teams score.
    return  tf.math.logical_and( tf.math.greater( h, 0.5 ), tf.math.greater( a, 0.5 ) )

def Metrics_Bool_04(h, a) : # for original Query 4: At most 1 team scores.
    return   tf.math.logical_not( tf.math.logical_and( tf.math.greater( h, 0.5 ), tf.math.greater( a, 0.5 ) ) )

def Metrics_Bool_05(h, a) : # for original Query 5: At least 1 team scores more than 2 goals.
    return  tf.math.logical_or( tf.math.greater( h, 2.5 ), tf.math.greater( a, 2.5 ) )

def Metrics_Bool_06(h, a) : # for original Query 6: At least 1 team scores less than 3 goals.
    return  tf.math.logical_or( tf.math.less( h, 2.5 ), tf.math.less( a, 2.5 ) )

def Metrics_Bool_07(h, a) : # for original Query 7: At leas 1 team scores more than 1 goal.
    return  tf.math.logical_or( tf.math.greater( h, 1.5 ), tf.math.greater( a, 1.5 ) )

def Metrics_Bool_08(h, a) : # for original Query 8: Home team wins.
    return tf.math.greater( tf.round(h), tf.round(a) )

def Metrics_Bool_09(h, a) : # for original Query 9: Away team wins.
    return tf.math.less( tf.round(h), tf.round(a) )

def Metrics_Bool_10(h, a) : # for original Query 10: The match draws.
    return tf.math.equal( tf.round(h), tf.round(a) )

def Metrics_Bool_11(h, a) : # for original Query 10: The score.
    return tf.math.logical_and( tf.math.greater( tf.round(h), .5), tf.math.greater( tf.round(a), .5 )  )

def Metrics_Bool_12(h, a) : # for original Query 12: The score is 1:0.
    return tf.math.logical_and( tf.math.equal( tf.round(h), 1.), tf.math.equal( tf.round(a), 0. ) )

def Metrics_Bool_13(h, a) : # for original Query 13: The score is 1:1.
    return tf.math.logical_and( tf.math.equal( tf.round(h), 1.), tf.math.equal( tf.round(a), 1. ) )

def Metrics_Bool_14(h, a) : # for original Query 14: The score is 2:1.
    return tf.math.logical_and( tf.math.equal( tf.round(h), 2.), tf.math.equal( tf.round(a), 1. ) )


Metrics_Bool_By_Query = [ Metrics_Bool_01, Metrics_Bool_02, Metrics_Bool_03, Metrics_Bool_04, Metrics_Bool_05, Metrics_Bool_06, Metrics_Bool_07, \
	Metrics_Bool_08, Metrics_Bool_09, Metrics_Bool_10, Metrics_Bool_11, Metrics_Bool_12, Metrics_Bool_13, Metrics_Bool_14 ]


def Metrics_Category_01(h, a) : # for original Query 1: Home team scores.
    return tf.cast( Metrics_Bool_01(h,a), tf.float32 )

def Metrics_Category_02(h, a) : # for original Query 2: Away team scores.
    return tf.cast( Metrics_Bool_02(h,a), tf.float32 )

def Metrics_Category_03(h, a) : # for original Query 3: Both teams score.
    return tf.cast( Metrics_Bool_03(h,a), tf.float32 )

def Metrics_Category_04(h, a) : # for original Query 4: At most 1 team scores.
    return tf.cast( Metrics_Bool_04(h,a), tf.float32 )

def Metrics_Category_05(h, a) : # for original Query 5: At least 1 team scores more than 2 goals.
    return tf.cast( Metrics_Bool_05(h,a), tf.float32 )

def Metrics_Category_06(h, a) : # for original Query 6: At least 1 team scores less than 3 goals.
    return tf.cast( Metrics_Bool_06(h,a), tf.float32 )

def Metrics_Category_07(h, a) : # for original Query 7: At leas 1 team scores more than 1 goal.
    return tf.cast( Metrics_Bool_07(h,a), tf.float32 )

def Metrics_Category_08(h, a) : # for original Query 8: Home team wins.
    return tf.cast( Metrics_Bool_08(h,a), tf.float32 )

def Metrics_Category_09(h, a) : # for original Query 9: Away team wins.
    return tf.cast( Metrics_Bool_09(h,a), tf.float32 )

def Metrics_Category_10(h, a) : # for original Query 10: The match draws.
    return tf.cast( Metrics_Bool_10(h,a), tf.float32 )

def Metrics_Category_11(h, a) : # for original Query 10: The score.
    return tf.cast( Metrics_Bool_11(h,a), tf.float32 )  #tf.stack( [h, a], axis = 1 ) # ====== This is special. 

def Metrics_Category_12(h, a) : # for original Query 12: The score is 1:0.
    return tf.cast( Metrics_Bool_12(h,a), tf.float32 )

def Metrics_Category_13(h, a) : # for original Query 13: The score is 1:1.
    return tf.cast( Metrics_Bool_13(h,a), tf.float32 )

def Metrics_Category_14(h, a) : # for original Query 14: The score is 2:1.
    return tf.cast( Metrics_Bool_14(h,a), tf.float32 )

# Label filter and its output dimension.
Metrics_Category_By_Query = [ [Metrics_Category_01, 1], [Metrics_Category_02, 1], [Metrics_Category_03, 1], [Metrics_Category_04, 1], \
    [Metrics_Category_05, 1], [Metrics_Category_06, 1], [Metrics_Category_07, 1], [Metrics_Category_08, 1], [Metrics_Category_09, 1], \
    [Metrics_Category_10, 1], [Metrics_Category_11, 2], [Metrics_Category_12, 1], [Metrics_Category_13, 1], [Metrics_Category_14,1] ]


def Metrics_Reverse_Category_01(h, a) : # for original Query 1: Home team scores.
    ret = 1. - tf.cast( Metrics_Bool_01(h,a), tf.float32 )
    return ret

def Metrics_Reverse_Category_02(h, a) : # for original Query 2: Away team scores.
    return 1. - tf.cast( Metrics_Bool_02(h,a), tf.float32 )

def Metrics_Reverse_Category_03(h, a) : # for original Query 3: Both teams score.
    return 1. - tf.cast( Metrics_Bool_03(h,a), tf.float32 )

def Metrics_Reverse_Category_04(h, a) : # for original Query 4: At most 1 team scores.
    return 1. - tf.cast( Metrics_Bool_04(h,a), tf.float32 )

def Metrics_Reverse_Category_05(h, a) : # for original Query 5: At least 1 team scores more than 2 goals.
    return 1. - tf.cast( Metrics_Bool_05(h,a), tf.float32 )

def Metrics_Reverse_Category_06(h, a) : # for original Query 6: At least 1 team scores less than 3 goals.
    return 1. - tf.cast( Metrics_Bool_06(h,a), tf.float32 )

def Metrics_Reverse_Category_07(h, a) : # for original Query 7: At leas 1 team scores more than 1 goal.
    return 1. - tf.cast( Metrics_Bool_07(h,a), tf.float32 )

def Metrics_Reverse_Category_08(h, a) : # for original Query 8: Home team wins.
    return 1. - tf.cast( Metrics_Bool_08(h,a), tf.float32 )

def Metrics_Reverse_Category_09(h, a) : # for original Query 9: Away team wins.
    return 1. - tf.cast( Metrics_Bool_09(h,a), tf.float32 )

def Metrics_Reverse_Category_10(h, a) : # for original Query 10: The match draws.
    return 1. - tf.cast( Metrics_Bool_10(h,a), tf.float32 )

def Metrics_Reverse_Category_11(h, a) : # for original Query 10: The score.
    return 1. - tf.cast( Metrics_Bool_11(h,a), tf.float32 ) # 1. - tf.stack( [h, a], axis = 1 ) # ===================== This is special.

def Metrics_Reverse_Category_12(h, a) : # for original Query 12: The score is 1:0.
    return 1. - tf.cast( Metrics_Bool_12(h,a), tf.float32 )

def Metrics_Reverse_Category_13(h, a) : # for original Query 13: The score is 1:1.
    return 1. - tf.cast( Metrics_Bool_13(h,a), tf.float32 )

def Metrics_Reverse_Category_14(h, a) : # for original Query 14: The score is 2:1.
    return 1.0 - tf.cast( Metrics_Bool_14(h,a), tf.float32 )

# Label filter and its output dimension.
Metrics_Reverse_Category_By_Query = [ Metrics_Reverse_Category_01, Metrics_Reverse_Category_02, Metrics_Reverse_Category_03, Metrics_Reverse_Category_04, \
    Metrics_Reverse_Category_05, Metrics_Reverse_Category_06, Metrics_Reverse_Category_07, Metrics_Reverse_Category_08, Metrics_Reverse_Category_09, \
    Metrics_Reverse_Category_10, Metrics_Reverse_Category_11, Metrics_Reverse_Category_12, Metrics_Reverse_Category_13, Metrics_Reverse_Category_14 ]



#======================== The following functions must be differentiable, as they are loss functions.


LOSS_FUNCTION = tf.keras.losses.MeanAbsoluteError()

def Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu ):
    y_true = tf.transpose(y_true)
    y_pred = tf.transpose(y_pred)
    return LOSS_FUNCTION( Metrics_ReLu( y_pred[0], y_pred[1] ), Metrics_ReLu( y_true[0], y_true[1] ) )


def Loss_On_Metrics_ReLu_01( y_true, y_pred ):
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_01 )

def Loss_On_Metrics_ReLu_02( y_true, y_pred ):
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_02 )

def Loss_On_Metrics_ReLu_03( y_true, y_pred ):
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_03 )

def Loss_On_Metrics_ReLu_04( y_true, y_pred ):
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_04 )

def Loss_On_Metrics_ReLu_05( y_true, y_pred ):
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_05 )

def Loss_On_Metrics_ReLu_06( y_true, y_pred ):
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_06 )

def Loss_On_Metrics_ReLu_07( y_true, y_pred ):
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_07 )

def Loss_On_Metrics_ReLu_08( y_true, y_pred ): 
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_08 )

def Loss_On_Metrics_ReLu_09( y_true, y_pred ): 
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_01 )

def Loss_On_Metrics_ReLu_10( y_true, y_pred ): 
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_10 )

def Loss_On_Metrics_ReLu_11( y_true, y_pred ): # Loss_On_Metrics_ReLu_ for Query 01.
    y_true = tf.transpose(y_true)
    y_pred = tf.transpose(y_pred)
    relu = tf.keras.activations.relu( - (y_pred[0] - y_true[0]) + 0.5, alpha = 0.0, max_value = None, threshold = 0) + \
        tf.keras.activations.relu( (y_pred[1] - y_true[1]) - 0.5, alpha = 0.0, max_value = None, threshold = 0)
    return relu.numpy()

def Loss_On_Metrics_ReLu_12( y_true, y_pred ):
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_12 )

def Loss_On_Metrics_ReLu_13( y_true, y_pred ):
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_13 )

def Loss_On_Metrics_ReLu_14( y_true, y_pred ):
    return Loss_On_Metrics_ReLu_Template( y_true, y_pred, Metrics_ReLu_14 )


Loss_On_Metrics_ReLu_By_Query = [Loss_On_Metrics_ReLu_01, Loss_On_Metrics_ReLu_02, Loss_On_Metrics_ReLu_03, Loss_On_Metrics_ReLu_04,\
	Loss_On_Metrics_ReLu_05, Loss_On_Metrics_ReLu_06, Loss_On_Metrics_ReLu_07, Loss_On_Metrics_ReLu_08, Loss_On_Metrics_ReLu_09,\
	Loss_On_Metrics_ReLu_10, Loss_On_Metrics_ReLu_11, Loss_On_Metrics_ReLu_12, Loss_On_Metrics_ReLu_13, Loss_On_Metrics_ReLu_14 ]





