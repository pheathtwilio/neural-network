# from examples import first_pass
import random
from lib import forward, backward, update, mse

if __name__ == "__main__":
    
    random.seed(0)

    # training set
    x = [1.0, 0.0]
    y = 1.0

    # random initialization
    w = [random.uniform(-1, 1) for _ in range(len(x))]
    b = 0.0
    lr = 0.5

    # before update
    _, a1 = forward(x, w, b)
    loss1 = mse(y, a1)

    # compute gradients and updates
    dL_dw, dL_db = backward(x, y, a1)
    w2, b2 = update(w, b, dL_dw, dL_db, lr)

    # after update
    _, a2 = forward(x, w2, b2)
    loss2 = mse(y, a2)

    print("\n\n\nBefore:")
    print(" w = ", w, "b = ", b)
    print(" pred = ", a1, "loss = ", loss1)

    print("\nGradients:")
    print(" dL_dw = ", dL_dw)
    print(" dL_db = ", dL_db)

    print("\nAfter one update:")
    print(" w = ", w2, " b = ", b2)
    print(" pred = ", a2, " loss = ", loss2)

