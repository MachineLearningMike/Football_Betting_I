import numpy as np
from _FuncSurface import *
import matplotlib.pyplot as plt

def fun_crossentropy(p, q):
    return - p * np.log(q) - (1-p) * np.log(1-q)

fig = plt.figure(figsize=(12,6))

p = np.arange(0.01, 0.99, 0.05)
q = np.arange(0.01, 0.99, 0.05)
ax = fig.add_subplot(111, projection='3d')
Plot3DSurface( fig, ax, p, "p", q, "q", fun_crossentropy, "CE = -p*np.log(q)-(1-p)*np.log(1-q)", with_contour=False )

p = np.arange(0.01, 0.99, 0.05)
q = p
ax.plot(p, q, zs=0, zdir='z', c='g', label='p=q')
ax.plot(p, q, fun_crossentropy(p, q), c='g')

q = 0.5 * np.ones_like(p, dtype=float)
ax.plot(p, q, zs=0, zdir='z', c='b', label='p=q')
ax.plot(p, q, fun_crossentropy(p, q), c='b')

q = 0.2 * np.ones_like(p, dtype=float)
ax.plot(p, q, zs=0, zdir='z', c='r', label='p=q')
ax.plot(p, q, fun_crossentropy(p, q), c='r')

ax.plot(q, p, zs=0, zdir='z', c='g', label='p=q')
ax.plot(q, p, fun_crossentropy(q, p), c='g')

q = 0.8 * np.ones_like(p, dtype=float)
ax.plot(q, p, zs=0, zdir='z', c='g', label='p=q')
ax.plot(q, p, fun_crossentropy(q, p), c='g')

plt.show()