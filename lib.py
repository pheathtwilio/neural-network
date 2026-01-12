import math

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def neuron_forward(x, w, b):
    """
    x: list of floats, length n - input
    w: list of floats, length n - weights
    b: float - bias
    returns: (z, a)
    """
    z = 0.0
    print("zip(x, w) ", list(zip(x, w)))
    for xi, wi in zip(x, w): # all zip does is puts the first input with the first weight ad infinitum
        z += wi * xi
    z += b
    print("compute a weighted sum z = sum(wi, xi) + b = ", z)
    a = sigmoid(z)
    print("activation function ", a) # converts a real number (z) into a 1 or 0
    return z, a

def mse_loss(y, a):
    return (y - a) ** 2