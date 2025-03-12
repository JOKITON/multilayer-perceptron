""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_mb_sig_seed():
    import config
    from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD
    from config import EPOCHS_MINI_BATCH_2, LS_SIGMOID_0, N_LAYERS, BATCH_SIZE
    from preprocessing import get_train_val_pd
    from batch import get_batches, shuffle_batches, get_val_batches
    from activations import sigmoid, relu, der_sigmoid, der_relu, leaky_relu, der_leaky_relu, tanh, der_tanh, softmax, der_softmax
    from loss import f_r2score
    import numpy as np
    from setup import setup_layers

    EPOCHS = EPOCHS_MINI_BATCH_2
    LAYER_SHAPE = LS_SIGMOID_0
    LEARNING_RATE *= 2

    # Normalize the data
    X_train, y_train, X_val, y_val = get_train_val_pd()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_val = y_val.to_numpy().reshape(-1, 1)

    b_epoch = 0
    b_acc = 0
    seed = np.random.randint(0, 1000000000)
    layers = setup_layers(sigmoid, der_sigmoid, LAYER_SHAPE, seed)

    activations = [None] * N_LAYERS
    train_x, train_y = get_batches(X_train, y_train, BATCH_SIZE)

    for epoch in range(EPOCHS):
        # Forward propagation
        if epoch % STEP_SIZE == 0:
            LEARNING_RATE *= DECAY_RATE

        if (epoch > (EPOCHS / 1.5) and epoch % 5 == 0):
            input_train = X_train
            for i in range(N_LAYERS):
                activations[i], _ = layers[i].forward(input_train)
                input_train = activations[i]
            acc_train = f_r2score(y_train, activations[-1])
                
            input_train = X_val
            for i in range(N_LAYERS):
                activations[i], _ = layers[i].forward(input_train)
                input_train = activations[i]
            acc_val = f_r2score(y_val, activations[-1])

            if (acc_train + acc_val > b_acc):
                b_epoch = epoch
                b_acc = acc_train + acc_val

        batch_X, batch_Y = get_val_batches(train_x, train_y, layers, epoch)
        for i in range(N_LAYERS):
            activations[i], _ = layers[i].forward(batch_X)
            batch_X = activations[i]

        for i in reversed(range(N_LAYERS)):
            layers[i].backward(batch_Y, LEARNING_RATE)

    return seed, b_acc, b_epoch