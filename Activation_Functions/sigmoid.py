import math 
import matplotlib.pyplot as plt


''' 
    1. Sigmoid Function maps any real-based values in a range between 0 and 1.
    2. When x is very large,σ(x)--->1;
    3. When x is very small, σ(x)---->0
    4. Derivative(σ(x)) = σ(x)(1 - σ(x))
'''

def sigmoid(x):
    '''
        Compute the Sigmoid of X 
    '''
    return 1 / ( 1 + math.exp(-x))

def sigmoid_derivative(x):
    """ 
        Compute the derivative of the sigmoid function at x. Important for Backpropagation
    """
    
    sx = sigmoid(x)
    return sx * (1 - sx)

x_values = [x * 0.1 for x in range(-50,51)]
y_values = [sigmoid(x) for x in x_values]

plt.plot(x_values, y_values)
plt.title("Sigmoid Activation Function")
plt.xlabel("x")
plt.ylabel("σ(x)")
plt.grid(True)
plt.show()
