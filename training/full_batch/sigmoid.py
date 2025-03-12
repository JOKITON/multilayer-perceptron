""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_fb_sig(plt=None, plt_it=0, plt_ct=2, plt_show=True, plt_ret=False):
    from config import LEARNING_RATE, STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD
    from config import EPOCHS_FBATCH_3, LS_SIGMOID_0, N_LAYERS, SEED_FB_SIG
    from config import Style, RESET_ALL
    from preprocessing import get_train_val_pd
    from activations import sigmoid, der_sigmoid
    from plot import Plot
     
    from setup import setup_layers
    from evaluate import print_preds
    import json
    
    EPOCHS = EPOCHS_FBATCH_3
    LAYER_SHAPE = LS_SIGMOID_0
    LEARNING_RATE *= 2.5
    COUNT_PLOT = plt_ct

    # Normalize the data
    X_train, y_train, X_val, y_val = get_train_val_pd()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_val = y_val.to_numpy().reshape(-1, 1)
    
    with open(SEED_FB_SIG, 'r') as file:
        data = json.load(file)
        seed = int(data['seed'])
        epochs = int(data['epoch'])
        EPOCHS = epochs

    layers = setup_layers(sigmoid, der_sigmoid, LAYER_SHAPE, seed)

    activations = [None] * N_LAYERS

    if (plt == None):
        plt = Plot(COUNT_PLOT)
    plt.set_error_data(X_train, y_train, layers, plt_it)
    plt.set_plot_config("Training", "blue", plt_it, "-", EPOCHS)
    plt.set_error_data(X_val, y_val, layers, plt_it + 1)
    plt.set_plot_config("Validation", "orange", plt_it + 1, "--", EPOCHS)

    for epoch in range(EPOCHS):
        if epoch % STEP_SIZE == 0:
            LEARNING_RATE *= DECAY_RATE

        plt.set_error_data(X_train, y_train, layers, plt_it)
        plt.set_error_data(X_val, y_val, layers, plt_it + 1)
        acc_train, mse_train, _, ce_train = plt.get_error_data(plt_it)
        
        early = plt.bool_early_stop(plt_it + 1, epoch)
        if (early == True):
            print("Early stopping...")
            break

        if (epoch % 100 == 0):
            print(f"Epoch {epoch}: " + Style.DIM + "MSE", f"[{mse_train:.3f}] "
                + "R2", f" [{acc_train:.3f}]", "CE", f"[{ce_train:.3f}]" + RESET_ALL)

        # Forward propagation
        train_input = X_train
        for i in range(N_LAYERS):
            activations[i], _ = layers[i].forward(train_input)
            train_input = activations[i]

        # Backward propagation
        for i in reversed(range(N_LAYERS)):
            layers[i].backward(y_train, LEARNING_RATE)

    if (plt_show):
        plt.plot_acc_epochs()
        plt.plot_loss_epochs()

    print_preds(layers, X_train, y_train, 1)
    print_preds(layers, X_val, y_val, 2)
    
    if (plt_ret):
        return layers, plt
    else:
        return layers
