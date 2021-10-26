import tensorflow as tf
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# evenly sampled time at 200ms intervals


r1 = 2.0
beta = -0.2
gamma = -0.8

r0 = - beta * r1 / ( 1.0 - beta )
slope = ( 1.0 - beta ) / ( r1 + 1e-7 )
alpha =   - ( gamma - beta ) / ( slope + 1e-7 ) 

profit_on_bet = np.arange(-1.5, r1 + 0.5, 0.1)
intermediary = tf.keras.activations.relu( x = slope * profit_on_bet , alpha = alpha ) + beta
betting_prob_target = tf.keras.activations.sigmoid( 3.5 * intermediary )


fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_title("prob_target per Profit_on_bet")
ax1.set_ylabel('prob_target')
ax1.set_xlabel('profit_on_bet')
#ax1.set_yticks([beta, gamma, 0.0, 0.5, 1.0], minor=False)
ax1.set_yticks([beta, gamma, 0.0, 0.5, 1.0], minor=False)
ax1.set_xticks([-1.0, 0.0, 1.0], minor=True)
ax1.yaxis.grid(True, which='major')
ax1.xaxis.grid(True, which='minor')
ax1.grid(True)

ax1.plot( profit_on_bet, betting_prob_target, profit_on_bet, intermediary )

plt.show()
