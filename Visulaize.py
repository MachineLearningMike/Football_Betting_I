
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def PlotFunctionSurface( x, y, func, title ) :

    X,Y = np.meshgrid(x,y)
    Z = func( X, Y )

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    #surf = ax.plot_surface(X, Y, Z, rstride=10, cstride=10, alpha=0.8, cmap=cm.coolwarm) # ocean)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.ocean)
    cset = ax.contourf(X, Y, Z, zdir='x', offset=np.min(X), cmap=cm.ocean)
    cset = ax.contourf(X, Y, Z, zdir='y', offset=np.min(Y), cmap=cm.ocean)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.set_title(title)
    ax.set_xlabel('x', c='r')
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylabel('y', c='r')
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlabel('z ', c='r')
    ax.set_zlim(np.min(Z), np.max(Z))
    
    plt.show()

