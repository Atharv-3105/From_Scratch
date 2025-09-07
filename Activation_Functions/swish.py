import math 

""" 
    1. Swish(x) = x * σ(x) = x / (1 + e^(-x)) = x * sigmoid(x)
    
    2. For large positive values of x; σ(x) ~~ 1 so Swish(x) ~~ x; similar to ReLU
    3. For large negative values of x; σ(x) ~~ 0 so Swish(x) ~~ 0; but not clipped{meaning not exactly 0 but very close to 0}
    
    4. It's smooth and differentiable everywhere, unlike ReLU which has a sharp corner at 0.
    5. It has been emprirically shown to imporve training and generalization in Deep Neural Nets.  
    
    6. derivative(Swish(x)) = 
"""