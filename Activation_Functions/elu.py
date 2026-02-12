import matplotlib.pyplot as plt 
import math 

""" 
    Exponential Linear Unit 
    
    1. ELU(x) = x if x > 0 ; alpha(e^x - 1) if x <= 0
    2. For positive inputs it behaves like a ReLU
    3. For negative inputs it smoothly decays towards - alpha insted of being clipped to 0 like ReLU.
    4. The smooth curve for negative inputs helps in reducing bias shifts and improving convergence.
    5. It's useful in deep networks where better gradient flow is required. 
    
    6. derivative(elu(x)) = 1 if x > 0; alpha * e^x if x <= 0
"""

def elu(x, alpha = 1.0):
    return x if x > 0 else  alpha* (math.exp(x) - 1)

def elu_derivative(x, alpha = 1.0):
    return 1 if x > 0 else alpha*(math.exp(x)) 


x_values = [x * 0.1 for x in range(-100, 101)]
y_values = [elu(x) for x in x_values]

plt.plot(x_values, y_values)
plt.title("ELU Activation Function")
plt.xlabel("x")
plt.ylabel("ELU(x)")
plt.grid(True)
plt.show() 

