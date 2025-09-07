import matplotlib.pyplot as plt 

""" 
    1. In LeakyReLU a small constant(0.01) is introduced which controls the slop for negative values
    2. LeakyReLU(x) = x if x > 0 ; αx if x <= 0
    
    3. It solves the dying ReLU problem{i.e where neurons permanently output 0 and stop learning}.
    4. Negative Inputs are completely ignored; they are scaled down.
    5. This makes the network more robust while keeping computational effiiciency.
"""

def leaky_relu(x, alpha = 0.01):
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha = 0.01):
    return 1 if x > 0 else alpha

x_values = [x * 0.01 for x in range(-100, 101)]
y_values = [leaky_relu(x) for x in x_values]

plt.plot(x_values, y_values)
plt.title("Leaky ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("Leaky ReLU(x)")
plt.grid(True)
plt.show() 