
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def Plot3DSurface(fig, ax, x, labelx, y, labely, func, title, with_contour = True ) :

    X,Y = np.meshgrid(x,y)
    Z = func( X, Y )

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.7, linewidth=0, antialiased=True)
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    if with_contour:
        cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.ocean, alpha=0.6)
        # cset = ax.contourf(X, Y, Z, zdir='x', offset=np.min(X), cmap=cm.ocean)
        # cset = ax.contourf(X, Y, Z, zdir='y', offset=np.min(Y), cmap=cm.ocean)

    ax.set_title(title)
    ax.set_xlabel(labelx, c='r')
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylabel(labely, c='r')
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlabel('fun', c='r')
    ax.set_zlim(np.min(Z), np.max(Z))

