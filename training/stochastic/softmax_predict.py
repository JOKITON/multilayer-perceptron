""" Program that creates a Multilayer Perceptron model to detect type of cancer cells. """

def init_st_soft_pred(layers):
    from config import N_LAYERS, RESET_ALL, Fore, Style
    from plot import Plot
     
    from split_data import preprocess_data
    import numpy as np
    
    # Make predictions based on the whole dataset
    pred_X, pred_y = preprocess_data()
    pred_y = pred_y.to_numpy().reshape(-1, 1)
    
    soft_y_pred = np.zeros((pred_y.shape[0], 2))
    soft_y_pred[np.arange(pred_y.shape[0]), pred_y.flatten()] = 1

    print("\n└─> Choose an option: ")
    print("\t[1]" + Style.BRIGHT + " Use training & validation dataset " + RESET_ALL)
    print("\t[2]" + Style.BRIGHT + " Use the whole dataset" + RESET_ALL)
    print("\t[3]" + Style.BRIGHT + " Predict a single row from 'data.csv' " + RESET_ALL)
    print("\t[4]" + Style.BRIGHT + " Go back ↑" + RESET_ALL)
    while 1:
        dec_type = input(Style.BRIGHT + Fore.LIGHTCYAN_EX + "└─> " + Fore.RESET)
        if (dec_type.isdigit() == False):
            print(Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
            continue
        else:
            dec_type = int(dec_type)
            if (dec_type == 1):
                make_pred(layers)
            elif (dec_type == 2):
                make_whole_pred(pred_X, soft_y_pred, layers)
            elif (dec_type == 3):
                make_single_pred(pred_X, soft_y_pred, layers)
            elif (dec_type == 4):
                return

def make_pred(layers):
    from evaluate import print_preds
    from preprocessing import get_train_val_pd
    import numpy as np

    X_train, y_train, X_val, y_val = get_train_val_pd()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_val = y_val.to_numpy().reshape(-1, 1)
    
    y_train_softmax = np.zeros((y_train.shape[0], 2))
    y_train_softmax[np.arange(y_train.shape[0]), y_train.flatten()] = 1

    y_val_softmax = np.zeros((y_val.shape[0], 2))
    y_val_softmax[np.arange(y_val.shape[0]), y_val.flatten()] = 1

    print_preds(layers, X_train, y_train_softmax, 1)
    print_preds(layers, X_val, y_val_softmax, 2)

def make_whole_pred(pred_X, pred_y, layers):
    from config import N_LAYERS
    from evaluate import print_preds
    
    activations = [None] * N_LAYERS

    # Forward propagation
    train_input = pred_X
    for i in range(N_LAYERS):
        activations[i], _ = layers[i].forward(train_input)
        train_input = activations[i]

    print_preds(layers, pred_X, pred_y, 3)

def make_single_pred(pred_X, pred_y, layers):
    from config import N_LAYERS, RESET_ALL, Fore, Style
    import numpy as np

    activations = [None] * N_LAYERS

    print(RESET_ALL + "\t\n└─> Please, input a row from the raw dataset (" + Fore.LIGHTWHITE_EX + Style.BRIGHT
        + "only numbers" + RESET_ALL + "): ")
    while 1:
        main_str = input("\n└─>")
        if (main_str.isdigit() == False):
            print(Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
            continue
        else:
            main_str = int(main_str)
            if (main_str < 2 or main_str > len(pred_X) + 1):
                print(Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
            else:
                break

    # Forward propagation
    train_input = pred_X
    for i in range(N_LAYERS):
        activations[i], _ = layers[i].forward(train_input)
        train_input = activations[i]

    guess = 0
    to_guess = main_str - 2
    guess = activations[-1][to_guess]

    # We need to round the value to make a prediction
    guess = np.round(guess)

    if (pred_y[to_guess][0] == guess[0] and pred_y[to_guess][1] == guess[1]):
        print("Correct prediction!")
    else:
        print("Incorrect prediction...")
    if (guess[0] > 0.5):
        guess = "M"
    else:
        guess = "B"
    print(f"[{to_guess + 2}]: {guess}\n")

    # print_preds(layers, pred_X, pred_y, 3)
