from config import N_LAYERS
from loss import f_loss, f_mse, f_mae, f_r2score, f_cross_entropy
from colorama import Fore, Back, Style
from config import RESET_ALL

def make_preds(layers, X, y):
    import numpy as np

    activations = [None] * N_LAYERS
    input_train = X
    for i in range(N_LAYERS):
        activations[i], _ = layers[i].forward(input_train)
        input_train = activations[i]

    """ activations[-1] = np.round(activations[-1]) """
    # MSE, MAE, r2 score
    mse = f_mse(y, activations[-1])
    mae = f_mae(y, activations[-1])
    r2 = f_r2score(y, activations[-1])
    cross_entropy = f_cross_entropy(y, activations[-1])

    return mse, mae, r2, cross_entropy

def print_preds(layers, X, y, bool_print):
    mse, mae, r2, ce = make_preds(layers, X, y)

    print()
    if (bool_print == 1):
        print(Fore.LIGHTGREEN_EX + Style.BRIGHT + "👉🏼  Training set: " + RESET_ALL)
    elif (bool_print == 2):
        print(Fore.LIGHTCYAN_EX + Style.BRIGHT + "👉🏼  Validation set: " + RESET_ALL)
    elif (bool_print == 3):
        print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "👉🏼  The whole dataset: " + RESET_ALL)
    # print(f"MAE: {mae}")
    print(f"\tMSE: {mse:.5f}")
    print(f"\tCross-Entropy: {ce:.5f}")
    print(f"\tR2 Score: {r2:.3f}")
    print()
