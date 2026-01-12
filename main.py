from examples import first_pass

if __name__ == "__main__":
    
    # sample inputs, weights, bias and target for example first pass
    x = [1.0, 0.0]
    w = [0.2, -0.1]
    b = 0.0
    y = 1.0 # the target, the actual correct answer

    z, a, loss = first_pass(x, w, b, y)