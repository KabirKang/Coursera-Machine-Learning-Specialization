import numpy as np
import matplotlib.pyplot as plt
print(plt.style.available)
plt.style.use('ggplot')

# x_train is the input variable (size in 1000 square feet)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train: {x_train}")
print(f"y_train: {y_train}")

m = len(x_train)
print(f"Number of training examples is: {m}")

for i in range(m):
    x_i = x_train[i]
    y_i = y_train[i]
    print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples
      w,b (scalar): model parameters
    Returns:
      f_wb (ndarray, (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

w = 200
b = 100
tmp_f_wb = compute_model_output(x_train, w, b)

# Plot the prediction
plt.plot(x_train, tmp_f_wb, c='b', label="Our Prediction")

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Housing Prices")
plt.ylabel("Price (in 1000s of dollars)")
plt.xlabel("Size (1000 sqft)")
plt.legend()
plt.show()
