import math 
import matplotlib.pyplot as plt 

""" 
    1. Tanh transforms any real-valued number into a value between -1 and 1.
    2. Tanh centers the data at 0, which helps the model during training by reducing bias and imporving convergence
    3. For large positive(x), tanh(x)---> 1;
    4. For large negative(x), tanh(x)---> -1;
    5. Derivative(tanh(x)) = 1 - (tanh(x))^2
"""


def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t * t

#Generate values from -5 to 5
x_values = [x * 0.1 for x in range(-50, 51)]
y_values = [tanh(x) for x in x_values]

plt.plot(x_values, y_values)
plt.title("Tanh Activation Function")
plt.xlabel("x")
plt.ylabel("tanh(x)") 
plt.grid(True)
plt.show()    