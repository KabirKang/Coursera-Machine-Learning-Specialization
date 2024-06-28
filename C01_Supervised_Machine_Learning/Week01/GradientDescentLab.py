#!/usr/bin/env python3
import math, copy
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_house_x, plt_gradients
plt.style.use('ggplot')

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1/(2*m) * cost

    return total_cost

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar): model parameters
    Returns:
      dj_dw (scalar): Gradient of the cost w.r.t. the parameters w
      dj_db (scalar): Gradient of the cost w.r.t. the parameter b
    """

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]

        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iters,cost_function,gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking num_iters gradient steps with learning rate alpha.

    Args:
    x (ndarray (m,)): Data, m examples
    y (ndarray (m,)): target values
    w_in,b_in (scalar): initial values of model parameters
    alpha (float): learning rate
    num_iters (int): number of iterations to run gradient descent
    cost_function: function to call to produce cost
    gradient_function: function to call to produce gradient

    Returns:
    w (scalar): Updated value of parameter after running gradient descent
    b (scalar): Updated value of parameter after running gradient descent
    J_history (List): history of cost values
    p_history (List): History of parameters [w,b]
    """

    J_history = []
    p_history = []
    w = w_in
    b = b_in

    for i in range(num_iters):
        # for num_iter iterations we want to descend
        dj_dw, dj_db = gradient_function(x,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        total_cost = compute_cost(x,y,w,b)

        if i<100000:
            J_history.append(total_cost)
            p_history.append([w,b])

        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w,b,J_history,p_history

x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value
plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt.show()

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# Plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()
