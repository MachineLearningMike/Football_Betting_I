from turtle import width
import numpy as np
from _FuncSurface import *
import matplotlib.pyplot as plt

def fun_crossentropy(p, q):
    return - p * np.log(q) - (1-p) * np.log(1-q)

def sigmoid(x):
    return x #1 / ( 1 + np.exp(-x) )

def x1(x, y):
    return sigmoid( x + y )

def y1(x, y):
    return sigmoid( x - y )

def fun_neural(x, y):
    return x1(x, y) + y1(x, y)

fig = plt.figure(figsize=(12,6))

x = np.arange(-4, 4, 0.05)
y = np.arange(-4, 4, 0.05)

ax = fig.add_subplot(121, projection='3d')
Plot3DSurface( fig, ax, x, "x", y, "y", x1, "Contours: x+y, x-y", with_contour=True )

# ax = fig.add_subplot(132, projection='3d')
Plot3DSurface( fig, ax, x, "x", y, "y", y1, "Contours: x+y, x-y", with_contour=True )
ax.plot(x, np.zeros_like(x), np.zeros_like(x), c='r')
ax.plot(np.zeros_like(y), y, np.zeros_like(y), c='r')

ax = fig.add_subplot(122, projection='3d')
Plot3DSurface( fig, ax, x, "x", y, "y", fun_neural, "Contours: (x+y) + (x-y)", with_contour=True )
ax.plot(x, np.zeros_like(x), np.zeros_like(x), c='r')
ax.plot(np.zeros_like(y), y, np.zeros_like(y), c='r')

plt.show()