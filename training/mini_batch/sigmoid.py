""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_mb_sig(plt=None, plt_it=0, plt_ct=2, plt_show=True, plt_ret=False):
    from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD
    from config import EPOCHS_MINI_BATCH_2, LS_SIGMOID_0, N_LAYERS, BATCH_SIZE, SEED_MB_SIG
    from config import Style, RESET_ALL
    from preprocessing import get_train_val_pd
    from activations import sigmoid, der_sigmoid
    from plot import Plot
    from setup import setup_layers
    from evaluate import print_preds
    from batch import get_batches, shuffle_batches, get_val_batches
    import json

    EPOCHS = EPOCHS_MINI_BATCH_2
    LAYER_SHAPE = LS_SIGMOID_0
    COUNT_PLOT = plt_ct
    LEARNING_RATE *= 2

    # Normalize the data
    X_train, y_train, X_val, y_val = get_train_val_pd()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_val = y_val.to_numpy().reshape(-1, 1)
    
    with open(SEED_MB_SIG, 'r') as file:
        data = json.load(file)
        seed = int(data['seed'])
        EPOCHS = int(data['epoch'])

    layers = setup_layers(sigmoid, der_sigmoid, LAYER_SHAPE, seed)

    activations = [None] * N_LAYERS

    if (plt == None):
        plt = Plot(COUNT_PLOT)
    plt.set_plot_config("Training", "indigo", plt_it, "-", EPOCHS)
    plt.set_error_data(X_train, y_train, layers, plt_it)
    plt.set_plot_config("Validation", "peru", plt_it + 1, "--", EPOCHS)
    plt.set_error_data(X_val, y_val, layers, plt_it + 1)

    train_x, train_y = get_batches(X_train, y_train, BATCH_SIZE)

    for epoch in range(EPOCHS):
        # Forward propagation
        if epoch % STEP_SIZE == 0:
            LEARNING_RATE *= DECAY_RATE

        plt.set_error_data(X_train, y_train, layers, plt_it)
        plt.set_error_data(X_val, y_val, layers, plt_it + 1)
        acc_train, mse_train, _, ce_train = plt.get_error_data(plt_it)
        
        if (epoch % 100 == 0):
            print(f"Epoch {epoch}: " + Style.DIM + "MSE", f"[{mse_train:.3f}] "
                + "R2", f" [{acc_train:.3f}]", "CE", f"[{ce_train:.3f}]" + RESET_ALL)

        batch_X, batch_Y = get_val_batches(train_x, train_y, layers, epoch)
        for i in range(N_LAYERS):
            activations[i], _ = layers[i].forward(batch_X)
            batch_X = activations[i]

        for i in reversed(range(N_LAYERS)):
            layers[i].backward(batch_Y, LEARNING_RATE)

    if (plt_show):
        plt.plot_acc_epochs()
        plt.plot_loss_epochs()

    print_preds(layers, X_train, y_train, 1)
    print_preds(layers, X_val, y_val, 2)
    
    if (plt_ret):
        return layers, plt
    else:
        return layers
