import math

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def forward(x, w, b):
    """
    Forward pass and calculate the sigmoid for a given list of inputs, weights and bias
    x: list of floats, length n - input
    w: list of floats, length n - weights
    b: float - bias
    returns: (z, a)
    """
    z = 0.0

    print("zip(x, w) ", list(zip(x, w)))
    for xi, wi in zip(x, w): # zip puts the first input with the first weight ad infinitum
        z += wi * xi
    z += b

    print("compute a weighted sum z = sum(wi, xi) + b = ", z)
    a = sigmoid(z)

    print("activation function ", a) # converts a real number (z) into a 1 or 0
    return z, a

def backward(x, y, a):
    """
    Backward pass, determine how should each weight and the bias change to make the loss smaller
    x: list of floats, length n - input
    y: float, the target
    a: the prediction (the output of the function after sigmoid)
    returns dL_dw (list of gradients), dL_db (a single gradient for the bias)
    """

    # if a > y, the (a - y) is positive, the loss will decrease if a goes down, if a < y then (a - y) is negative, the loss will decrease if a goes up
    dL_da = 2.0 * (a - y) # multiplied by 2 because we square for the loss facto
    print("the direction dL_da is ", dL_da)
    
    # if a is near 0.5 sigmoid is sensitive (bigger slope), is a is near 0 or 1 so changing z barely changes a
    da_dz = a * (1.0 - a)
    print("how much the prediction changes if the weighted sum changes")

    # how much does the loss change if we change z?
    dL_dz = dL_da * da_dz
    print("The loss changes by ", dL_dz, " if we change z")

    # produce a gradient for each weight e.g. which way should we change it to reduce loss
    dL_dw = [dL_dz * xi for xi in x]
    print("The gradients for each weight", list(dL_dw))

    # determine the gradient for the bias
    dL_db = dL_dz
    print("the gradient for the bias is ", dL_db)

    return dL_dw, dL_db

def update(w, b, dL_dw, dL_db, lr):
    """
    Update the weights and bias limited by the learning rate (steps to get to near zero loss)
    w: list of current weights
    b: current bias
    dL_dw: list of gradients for each weight, should be same length as w
    dL_db: gradient for bias
    lr: learning rate
    returns new list of weights and bias
    """

    # Update weights - for each weight wi subtract the learning rate from the gradient dwi, zip creates a list of weights with each gradient pair 
    w = [wi - lr * dwi for wi, dwi in zip(w, dL_dw)]

    # update bias - update the bias by multiplying the learning rate by the bias gradient
    b = b - lr * dL_db
    return w, b

def mse(y, a):
    """
    return the mean squared error of the difference between the target y and the activation function a
    """
    return (y - a) ** 2

