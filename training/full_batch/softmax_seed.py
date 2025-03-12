""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_fb_soft_seed():
    from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD
    from config import EPOCHS_FBATCH_2, LS_SOFTMAX_0, N_LAYERS
    from preprocessing import get_train_val_pd
    from activations import softmax, der_softmax
    from loss import f_r2score
    import numpy as np
    from setup import setup_layers

    EPOCHS = EPOCHS_FBATCH_2
    LAYER_SHAPE = LS_SOFTMAX_0
    LEARNING_RATE *= 7.5

    # Normalize the data
    X_train, y_train, X_val, y_val = get_train_val_pd()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_val = y_val.to_numpy().reshape(-1, 1)

    b_epoch = 0
    b_acc = 0
    seed = np.random.randint(0, 1000000000)
    layers = setup_layers(softmax, der_softmax, LAYER_SHAPE, seed)
    
    soft_y_train = np.zeros((y_train.shape[0], 2))
    soft_y_train[np.arange(y_train.shape[0]), y_train.flatten()] = 1

    soft_y_val = np.zeros((y_val.shape[0], 2))
    soft_y_val[np.arange(y_val.shape[0]), y_val.flatten()] = 1

    activations = [None] * N_LAYERS

    for epoch in range(EPOCHS):
        # Forward propagation
        if epoch % STEP_SIZE == 0:
            LEARNING_RATE *= DECAY_RATE

        if (epoch > (EPOCHS / 1.5) and epoch % 3 == 0):
            input_train = X_train
            for i in range(N_LAYERS):
                activations[i], _ = layers[i].forward(input_train)
                input_train = activations[i]
            acc_train = f_r2score(soft_y_train, activations[-1])
                
            input_train = X_val
            for i in range(N_LAYERS):
                activations[i], _ = layers[i].forward(input_train)
                input_train = activations[i]
            acc_val = f_r2score(soft_y_val, activations[-1])
            
            if (acc_train + acc_val > b_acc):
                b_epoch = epoch
                b_acc = acc_train + acc_val

        train_input = X_train
        for i in range(N_LAYERS):
            activations[i], _ = layers[i].forward(train_input)
            train_input = activations[i]

        for i in reversed(range(N_LAYERS)):
            if (i == N_LAYERS - 1):
                soft_y_true = np.zeros((y_train.shape[0], 2))
                soft_y_true[np.arange(y_train.shape[0]), y_train.flatten()] = 1
                input_y = soft_y_true
            else:
                input_y = y_train
            layers[i].backward(input_y, LEARNING_RATE)

    return seed, b_acc, b_epoch
