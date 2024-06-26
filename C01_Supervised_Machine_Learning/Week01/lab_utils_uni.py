#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from lab_utils_common import dlblue, dlorange, dldarkred, dlmagenta, dlpurple, dlcolors

def plt_house_x(X, y, f_wb=None, ax=None):
    ''' Plot house with aXis '''
    if not ax:
        fig, ax = plt.subplots(1,1)
    ax.scatter(X, y, marker='x', c='r', label="Actual Value")
    ax.set_title("Housing Prices")
    ax.set_ylabel('Price in 1000s of dollars')
    ax.set_xlabel('Size (1000sqft)')
    if f_wb is not None:
        ax.plot(X, f_wb, c=dlblue, label="Our Prediction")
    ax.legend()

def plt_gradients(x_train, y_train, f_compute_cost, f_compute_gradient):
    fig,ax = plt.subplots(1,2,figsize=(12,4))
    fix_b = 100

    w_array = np.linspace(0, 400, 50)
    cost = np.zeros_like(w_array)

    for i in range(len(w_array)):
        tmp_w = w_array[i]
        cost[i] = f_compute_cost(x_train, y_train, tmp_w, fix_b)
    ax[0].plot(w_array, cost,linewidth=1)
    ax[0].set_title("Cost vs w, with gradient, b set to 100")
    ax[0].set_ylabel('Cost')
    ax[0].set_xlabel('w')

    # plot lines for fixed b = 100
    for tmp_w in [100,200,300]:
        fix_b = 100
        dj_dw,dj_db = f_compute_gradient(x_train, y_train, tmp_w, fix_b)
        j = f_compute_cost(x_train, y_train, tmp_w, fix_b)
        add_line(dj_dw, tmp_w, j, 30, ax[0])

    #===============
    # Second Subplot
    #===============

    tmp_b, tmp_w = np.meshgrid(np.linspace(-200,200,10), np.linspace(-100,600,10))
    U = np.zeros_like(tmp_w)
    V = np.zeros_like(tmp_b)
    for i in range(tmp_w.shape[0]):
        for j in range(tmp_w.shape[1]):
            U[i][j], V[i][j] = f_compute_gradient(x_train, y_train, tmp_w[i][j], tmp_b[i][j])
    X = tmp_w
    Y = tmp_b
    n = -2
    color_array = np.sqrt(((V-n)/2)**2 + ((U-n)/2)**2)
    ax[1].set_title("Gradient show in quiver plot")
    Q = ax[1].quiver(X,Y,U,V,color_array,units='width',)
    ax[1].quiverkey(Q,0.9,0.9,2,r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')
    ax[1].set_xlabel("w"); ax[1].set_ylabel("b")

def add_line(dj_dx, x1, y1, d, ax):
    x = np.linspace(x1-d,x1+d,50)
    y = dj_dx*(x-x1) + y1
    ax.scatter(x1,y1,color=dlblue,s=50)
    ax.plot(x,y,'--',c=dldarkred,zorder=10,linewidth=1)
    xoff = 30 if x1 == 200 else 10
    ax.annotate(r"$\frac{\partial J}{\partial w}$ =%d" % dj_dx, fontsize=14,
        xy=(x1, y1), xycoords='data',
        xytext=(xoff, 10), textcoords='offset points',
        arrowprops=dict(arrowstyle="->"),
        horizontalalignment='left', verticalalignment='top')
