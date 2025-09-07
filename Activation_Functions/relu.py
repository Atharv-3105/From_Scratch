import matplotlib.pyplot as plt 


""" 
    Rectified Linear Unit
    1. ReLU(x) = max(0, x)
    2. If x >= 0; ReLU(x) = x
    3. If x < 0; ReLU(x) = 0
    4. Derivative(ReLU(x)) = 1 if x > 0; 0 if x <= 0
    
    5. ReLU is computationally efficient and helps models converge faster
    6. It introduces sparsity in the activations because all negative values are mapped to zero.
    7. It avoids the vanishing gradient problem common with sigmoid or tanh in deeper networks.
    
    8. It can suffer from dying ReLU problem, where neurons stop learning if their output is always zero.
"""

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0


x_values = [x * 0.1 for x in range(-50,51)]
y_values = [relu(x) for x in x_values]

plt.plot(x_values, y_values)
plt.title("ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.grid(True)
plt.show()
