from turtle import width
import numpy as np
from _FuncSurface import *
import matplotlib.pyplot as plt

def fun_crossentropy(p, q):
    return - p * np.log(q) - (1-p) * np.log(1-q)

def activator(x):
    return 1 / ( 1 + np.exp(-x) )
    
def x1(x, y): return activator( x + y )
def x2(x, y): return activator(- x + 0.5 )
def y1(x, y): return activator( x - y )

def fun_neural(x, y):
    return x1(x, y) + y1(x, y) + x2(x, y)

fig = plt.figure(figsize=(12,6))

x = np.arange(-4, 4, 0.3)
y = np.arange(-4, 4, 0.3)

ax = fig.add_subplot(121, projection='3d')
title = "Contours: sigmoid(x+y), sigmoid(x-y), sigmoid(-x+0.5)"
Plot3DSurface( fig, ax, x, "x", y, "y", x1, title, with_contour=True )
Plot3DSurface( fig, ax, x, "x", y, "y", x2, title, with_contour=False )
Plot3DSurface( fig, ax, x, "x", y, "y", y1, title, with_contour=True )

ax.plot(x, np.zeros_like(x), np.zeros_like(x), c='r')
ax.plot(np.zeros_like(y), y, np.zeros_like(y), c='r')

ax = fig.add_subplot(122, projection='3d')
Plot3DSurface( fig, ax, x, "x", y, "y", fun_neural, "Contours: Sum", with_contour=True )
ax.plot(x, np.zeros_like(x), np.zeros_like(x), c='r')
ax.plot(np.zeros_like(y), y, np.zeros_like(y), c='r')

plt.show()