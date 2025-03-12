""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_single_layer():
    from config import STEP_SIZE, DECAY_RATE, CONVERGENCE_THRESHOLD
    from config import Style, RESET_ALL
    from preprocessing import get_train_val_pd
    from loss import f_cross_entropy, f_mse, f_mae, f_r2score
    from activations import sigmoid, der_sigmoid
    from tests.propagation import stepforward, compute_gradient
    import numpy as np
    from tests.single_layer_plot import plot_acc, plot_ce

    # Hyperparameters
    LEARNING_RATE = 0.01
    CONVERGENCE_THRESHOLD = 1e-10
    EPOCHS = 50000
    COUNT_PLOT = 2

    # Normalize the data
    X_train, y_train, X_val, y_val = get_train_val_pd()
    y_train = y_train.to_numpy().reshape(-1)
    y_val = y_val.to_numpy().reshape(-1)

    # Initialize weights and bias
    weights = np.random.randn(X_train.shape[1]) * np.sqrt(1 / X_train.shape[1])
    bias = 0

    mse_train, acc_train, ce_train = [] * EPOCHS, [] * EPOCHS, [] * EPOCHS
    mse_val, acc_val, ce_val = [] * EPOCHS, [] * EPOCHS, [] * EPOCHS
    actv = 0
    prev_loss = None
    epoch = EPOCHS
    for i in range(EPOCHS):
        # Forward pass
        actv_train, _ = stepforward(X_train, weights, bias)  # Get activated outputs (sigmoid)

        mse_train.append(f_mse(y_train, actv_train))
        acc_train.append(f_r2score(y_train, actv_train))
        ce_train.append(f_cross_entropy(y_train, actv_train))

        actv_val = stepforward(X_val, weights, bias)[0]
        mse_val.append(f_mse(y_val, actv_val))
        acc_val.append(f_r2score(y_val, actv_val))
        ce_val.append(f_cross_entropy(y_val, actv_val))

        if (i % 100 == 0):
            print(f"Epoch {i}: " + Style.DIM + "MSE", f"[{mse_train[-1]:.3f}] "
                + "R2", f" [{acc_train[-1]:.3f}]", "CE", f"[{ce_train[-1]:.3f}]" + RESET_ALL)

        if (len(ce_val) > 2 and ce_val[-2] < ce_val[-1]):
            print("Early stopping...")
            epoch = i + 1
            break

        # Backward pass (gradient computation)
        weights, bias = compute_gradient(X_train, y_train, actv_train, weights, bias)

    print("Training complete.\n")

    actv = np.round(actv)
    print(f"Training set:\n\t" + Style.BRIGHT + "MSE", f"[{mse_train[-1]:.3f}] "
        + "R2", f" [{acc_train[-1]:.3f}]", "CE", f"[{ce_train[-1]:.3f}]" + RESET_ALL)

    # Validation Phase
    predictions = stepforward(X_val, weights, bias)[0]  # Use activated outputs
    predictions = np.round(predictions)  # Round for binary classification

    # val metrics
    print(f"Validation set:\n\t" + Style.BRIGHT + "MSE", f"[{mse_val[-1]:.3f}] "
        + "R2", f" [{acc_val[-1]:.3f}]", "CE", f"[{ce_val[-1]:.3f}]" + RESET_ALL)

    plot_acc(acc_train, acc_val, epoch)
    plot_ce(ce_train, ce_val, epoch)
