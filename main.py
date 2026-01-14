# from examples import first_pass
import random
from lib import forward, backward, update, mse, train, predict

# Forward and Backward Propagation
# if __name__ == "__main__":
    
#     random.seed(0)

#     # training set
#     x = [1.0, 0.0]
#     y = 1.0

#     # random initialization
#     w = [random.uniform(-1, 1) for _ in range(len(x))]
#     b = 0.0
#     lr = 0.5

#     # before update
#     _, a1 = forward(x, w, b)
#     loss1 = mse(y, a1)

#     # compute gradients and updates
#     dL_dw, dL_db = backward(x, y, a1)
#     w2, b2 = update(w, b, dL_dw, dL_db, lr)

#     # after update
#     _, a2 = forward(x, w2, b2)
#     loss2 = mse(y, a2)

#     print("\n\n\nBefore:")
#     print(" w = ", w, "b = ", b)
#     print(" pred = ", a1, "loss = ", loss1)

#     print("\nGradients:")
#     print(" dL_dw = ", dL_dw)
#     print(" dL_db = ", dL_db)

#     print("\nAfter one update:")
#     print(" w = ", w2, " b = ", b2)
#     print(" pred = ", a2, " loss = ", loss2)

# Training against the AND dataset and predicting the value
if __name__ == "__main__":
    random.seed(42)

    # AND truth table - will succeed
    dataset = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 0.0),
        ([1.0, 0.0], 0.0),
        ([1.0, 1.0], 1.0),
    ]

    # XOR truth table will fail because XOR is not linearly seperable
    # dataset = [
    #     ([0.0, 0.0], 0.0),
    #     ([0.0, 1.0], 1.0),
    #     ([1.0, 0.0], 1.0),
    #     ([1.0, 1.0], 0.0),
    # ]

    w, b = train(dataset, epochs=2000, lr=0.5)

    print("\nFinal Test:")
    for x, y in dataset:
        a, yhat = predict(x, w, b)
        print(f"x={x} y={y} pred_prob={a:.4f} pred_class={yhat}")