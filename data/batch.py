

import numpy as np
from config import N_LAYERS
from loss import f_cross_entropy, f_r2score

def get_batches(X_train, y_train, BATCH_SIZE):
    """Shuffle and create random batches of data.
    
    Parameters:
    X_train     - Training data, shape (n_samples, n_features)
    y_train     - Training labels, shape (n_samples,)
    BATCH_SIZE  - Size of each batch
    
    Returns:
    batches_x   - List of batches for input features
    batches_y   - List of corresponding batches for labels
    """
    # Ensure inputs are NumPy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Shuffle data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Split data into batches
    batches_x = []
    batches_y = []
    for i in range(0, X_train.shape[0], BATCH_SIZE):
        X_batch = X_train[i:i + BATCH_SIZE]
        y_batch = y_train[i:i + BATCH_SIZE]
        batches_x.append(X_batch)
        batches_y.append(y_batch)
    
    return batches_x, batches_y

def shuffle(X, y):
    """Shuffle the data.
    
    Parameters:
    X - Feature data
    y - Corresponding labels
    
    Returns:
    Shuffled features and labels
    """
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

def shuffle_batches(train_x, train_y):
    """Shuffle the batches of training data.
    
    Parameters:
    train_x - List of batches for features
    train_y - List of corresponding batches for labels
    
    Returns:
    Shuffled batches for features and labels
    """
    combined = list(zip(train_x, train_y))
    np.random.shuffle(combined)
    shuffled_x, shuffled_y = zip(*combined)
    shuffled_x, shuffled_y = list(shuffled_x), list(shuffled_y)
    return list(shuffled_x), list(shuffled_y)

def get_val_batches(train_x, train_y, layers, epoch):
    length = (len(train_x))
    loss_cmp = 0
    for it in range(length):
        batch_X = train_x[it]
        batch_Y = train_y[it]

        activations = [None] * N_LAYERS
        for i in range(N_LAYERS):
            activations[i], output = layers[i].forward(batch_X)
            batch_X = activations[i]

        loss = f_cross_entropy(batch_Y, activations[-1])
        loss = np.abs(loss)
        if (loss == 0):
            loss_cmp = it
        elif (loss > loss_cmp):
            loss_cmp = it
    return train_x[loss_cmp], train_y[loss_cmp]

def get_val_values(X_train, y_train, layers):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    val_X = np.zeros((1, 32))

    loss_cmp = 0
    for it in range(X_train.shape[0]):
        val_X = X_train[it]
        val_Y = y_train[it]

        activations = [None] * N_LAYERS
        for i in range(N_LAYERS):
            activations[i], output = layers[i].forward(val_X)
            val_X = activations[i]

        loss = f_r2score(val_Y, activations[-1])
        loss = np.abs(loss)
        if (loss == 0):
            loss_cmp = it
        elif (loss > loss_cmp):
            loss_cmp = it
    val_X = X_train[loss_cmp].reshape(1, 30)
    return val_X, y_train[loss_cmp]

def get_stochastic(X_train, y_train):
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Randomly select an index
    idx = np.random.randint(0, X_train.shape[0])

    # Get the corresponding feature and label
    opt_train_X = X_train[idx].reshape(1, -1)  # Ensure shape (1, n_features)
    opt_train_Y = y_train[idx]  # Scalar or one-hot encoded label

    return opt_train_X, opt_train_Y
