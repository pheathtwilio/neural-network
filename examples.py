from lib import forward, mse

"""
Simple first pass Nueron Example

list of inputs x
list of weights w (can be random on first pass)
bias b, adjustable offset this ensures the activation function doesn't pass through the origin
weighted sum z += wi * xi + b, the dot product of the vector
activation function a = sigmoid(z) - this ensures that the neuron handles non-linear values, could be ReLU also
y is typically the target, so loss is calculated from y and a to determine how far off from the result that we are

"""
def first_pass(x, w, b, y):

    z, a = forward(x, w, b)
    loss = mse(y, a)

    print("z =", z)
    print("a = sigmoid(z) =", a)
    print("loss =", loss)

    return z, a, loss