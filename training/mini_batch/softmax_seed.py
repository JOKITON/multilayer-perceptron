""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_mb_soft_seed():
    from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD
    from config import EPOCHS_MINI_BATCH_2, LS_SOFTMAX_0, N_LAYERS, BATCH_SIZE
    import numpy as np
    from batch import get_batches, shuffle_batches, get_val_batches
    from preprocessing import get_train_val_pd
    from activations import softmax, der_softmax
    from setup import setup_layers
    from loss import f_r2score

    EPOCHS = EPOCHS_MINI_BATCH_2
    LAYER_SHAPE = LS_SOFTMAX_0
    LEARNING_RATE *= 2

    # Normalize the data
    X_train, y_train, X_val, y_val = get_train_val_pd()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_val = y_val.to_numpy().reshape(-1, 1)

    b_epoch = 0
    b_acc = 0
    seed = np.random.randint(0, 1000000000)
    layers = setup_layers(softmax, der_softmax, LAYER_SHAPE, seed)

    train_x, train_y = get_batches(X_train, y_train, BATCH_SIZE)

    activations = [None] * N_LAYERS
    acc_train, acc_val = 0, 0

    y_train_softmax = np.zeros((y_train.shape[0], 2))
    y_train_softmax[np.arange(y_train.shape[0]), y_train.flatten()] = 1

    y_val_softmax = np.zeros((y_val.shape[0], 2))
    y_val_softmax[np.arange(y_val.shape[0]), y_val.flatten()] = 1

    for epoch in range(EPOCHS):
        # Forward propagation
        if epoch % STEP_SIZE == 0:
            LEARNING_RATE *= DECAY_RATE
        
        if (epoch % 15 == 0 and epoch != 0):
            train_x, train_y = shuffle_batches(train_x, train_y)
            
        if (epoch > (EPOCHS / 1.5) and epoch % 5 == 0):
            input_train = X_train
            for i in range(N_LAYERS):
                activations[i], _ = layers[i].forward(input_train)
                input_train = activations[i]
            acc_train = f_r2score(y_train_softmax, activations[-1])
                
            input_train = X_val
            for i in range(N_LAYERS):
                activations[i], _ = layers[i].forward(input_train)
                input_train = activations[i]
            acc_val = f_r2score(y_val_softmax, activations[-1])
            
            if (acc_train + acc_val > b_acc):
                b_epoch = epoch
                b_acc = acc_train + acc_val

        batch_X, batch_Y = get_val_batches(train_x, train_y, layers, epoch)
        for i in range(N_LAYERS):
            activations[i], _ = layers[i].forward(batch_X)
            batch_X = activations[i]

        for i in reversed(range(N_LAYERS)):
            if (i == N_LAYERS - 1):
                y_true_one_hot = np.zeros((batch_Y.shape[0], 2))
                y_true_one_hot[np.arange(batch_Y.shape[0]), batch_Y.flatten()] = 1
                input_y = y_true_one_hot
            else:
                input_y = batch_Y
            layers[i].backward(input_y, LEARNING_RATE)

    return seed, b_acc, b_epoch
