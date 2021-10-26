import tensorflow as tf
import numpy as np
import math

import os
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.patches as patches

def Get_Bet_Worth_Target_V1( min_oh_1, threshold_bet_worth, full_oh_1, full_bet_worth, x ) : #odds, happened_true ) :
    """
    threshold_bet_worth : threshold_prob in (0.0, 1.0)
    full_bet_worth : full_prob in (threshold_bet_worth, 1.0)
    min_oh_1 : (0.0, inf)
    full_oh_1 : (min_oh_1, inf)
    """
    def logit( x ) : return tf.math.log( x / ( 1.0 - x ) )
    
    alpha = ( logit( full_bet_worth ) - logit ( threshold_bet_worth ) ) / ( full_oh_1 - min_oh_1 )

    off = min_oh_1 - logit( threshold_bet_worth ) / alpha

    # x = odds * happened_true - 1.0
    bet_worth_target = tf.math.sigmoid( alpha * ( x - off ) )
    
    return bet_worth_target


def Get_Stake_On_Bet_Worth_V1( bet_worth, max_stake, threshold_bet_worth ) :

    binary_decision = tf.nn.relu( bet_worth - threshold_bet_worth ) / (bet_worth - threshold_bet_worth) #tf.cast( tf.greater( bet_worth, threshold_bet_worth ), tf.float32 )
    #stake = binary_decision * ( max_stake - 1 ) / ( 1.0 - threshold_bet_worth ) * bet_worth + ( 1.0 - max_stake * threshold_bet_worth ) / ( 1.0 + threshold_bet_worth )
    stake = binary_decision  * ( max_stake - 1 ) / ( 1.0 - threshold_bet_worth ) * bet_worth + ( 1.0 - max_stake * threshold_bet_worth ) / ( 1.0 + threshold_bet_worth )

    return stake


def Get_Bet_Worth_Target_V2( x ) : #odds, happened_true ) :

    bet_worth_target = tf.nn.relu( x )
    
    return bet_worth_target


def Get_Stake_On_Bet_Worth_V2( bet_worth, max_stake, full_oh_1, full_bet_worth ) :

    def logit( x ) : return tf.math.log( x / ( 1.0 - x ) )

    off = logit( 1.0 / max_stake ) * full_bet_worth / ( logit( 1.0 / max_stake ) - logit( full_bet_worth ) )
    alpha = - logit( 1.0 / max_stake ) / off
    stake = max_stake * tf.math.sigmoid( alpha * ( bet_worth - off ) )

    return stake

def Get_Ax_Size( ax ):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    #width *= fig.dpi
    #height *= fig.dpi
    return width, height

def Sigmoid( x ):
    return 1 / (1 + math.e ** -x)

def Sigmoid_Intense( profit ) :
    a = - 30; b = 30
    x = ( ( profit - a ) / ( b - a ) - 0.5 ) * 7.0
    return Sigmoid( x )

def Sigmoid_Intense_Positive( profit, b ) :
    a = 0
    x = ( ( profit - a ) / ( b - a ) ) * 3.5
    return ( Sigmoid( x ) - 0.5 ) * 2

def Build_Intense_Rect( ax, profit, mature ) :
    if profit >= 0 :
        intense = Sigmoid_Intense_Positive( profit, mature )
        r = intense; g = 1.0; b = 0.0
        a = 1
        color = ( r, g, b, a )
    else :
        intense = Sigmoid_Intense_Positive( - profit, mature )
        r = 0.0; g = 1.0 - intense; b = 0.0
        a = 1
        color = ( r, g, b, a )
    rsize = 0.1
    width, height = Get_Ax_Size( ax )
    rect = patches.Rectangle( ( 0, height/2 ), width * rsize, height * rsize, linewidth = 1, edgecolor = color, facecolor = color, fill = True )

    return rect

"""
sp = 0.3
fp = 0.95
fx = 2.5
profit_on_bet = np.arange(-1.5, fx + 0.5, 0.05 )
betting_prob_target = Get_Bet_Worth_Target_1( sp, fp, fx, profit_on_bet )
"""

min_oh_1 = 0.5
threshold_bet_worth = 0.5
full_oh_1 = 2.5
full_bet_worth = 0.9
max_stake = 2.0


profit_on_bet = np.arange(-1.5, full_oh_1 + 0.5, 0.05, dtype = np.float32 )
bet_worth_target = Get_Bet_Worth_Target_V1( min_oh_1, threshold_bet_worth, full_oh_1, full_bet_worth, profit_on_bet )
stake_target = Get_Stake_On_Bet_Worth_V1( bet_worth_target, max_stake, threshold_bet_worth )
#bet_worth_target = Get_Bet_Worth_Target_V2( profit_on_bet )
#stake_target = Get_Stake_On_Bet_Worth_V2( bet_worth_target, max_stake, full_oh_1, full_bet_worth )

relu = tf.nn.relu( profit_on_bet )

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_title("prob_target per Profit_on_bet")
ax1.set_ylabel('bet_worth_target')
ax1.set_xlabel('profit_on_unit_bet = o*h-1')
#ax1.set_yticks([beta, gamma, 0.0, 0.5, 1.0], minor=False)
ax1.set_yticks([ threshold_bet_worth, full_bet_worth, 0.0, 0.5, 1.0 ], minor = False )
ax1.set_xticks([ -1.0, 0.0, min_oh_1, full_oh_1, 1.0], minor=True)
ax1.yaxis.grid(True, which='major')
ax1.xaxis.grid(True, which='minor')
ax1.grid(True)

ax1.plot( profit_on_bet, bet_worth_target, profit_on_bet, stake_target )

rect = Build_Intense_Rect( ax1, profit = 20, mature = 30 )
ax1.add_patch( rect )

#ax1.plot( profit_on_bet, stake_target )
ax1.legend( ['bet_worth_target', 'stake_target' ] )

plt.show()


