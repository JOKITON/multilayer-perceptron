""" config.py """

import os
import sys
from colorama import Fore, Back, Style

# Set the current working directory two directories above
PWD = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

# Add paths to the system path
DF_UTILS = os.path.join(PWD, 'data/')
UTILS = os.path.join(PWD, 'utils/')
MODEL = os.path.join(PWD, 'model/')
LAYER = os.path.join(PWD, 'layer/')
PLOT = os.path.join(PWD, 'plots/')
TRAINING_FB = os.path.join(PWD, 'training/full_batch/')
TRAINING_MB = os.path.join(PWD, 'training/mini_batch/')
TRAINING_ST = os.path.join(PWD, 'training/stochastic/')

sys.path.append(DF_UTILS)
sys.path.append(UTILS)
sys.path.append(MODEL)
sys.path.append(LAYER)
sys.path.append(PLOT)

# Define basic parameters for the model
LEARNING_RATE = 0.035
DECAY_RATE = 0.995
STEP_SIZE = 100

CONVERGENCE_THRESHOLD = 1e-5

# Define the number of epochs for each training method
EPOCHS_FBATCH_1 = 300
EPOCHS_FBATCH_2 = 1000
EPOCHS_FBATCH_3 = 2000

EPOCHS_MINI_BATCH_1 = 500
EPOCHS_MINI_BATCH_2 = 1500
EPOCHS_MINI_BATCH_3 = 3000

EPOCHS_STOCHASTIC_1 = 500
EPOCHS_STOCHASTIC_2 = 1000
EPOCHS_STOCHASTIC_3 = 10000

# Mini-batch size
BATCH_SIZE = 32

# Nesterov momentum coefficient
MOMENTUM = 0.9

# Regularization strength
LAMBDA_REG = 0.01

# General model parameters
N_FEATURES = 30
N_NEURONS = 32
N_HIDDEN_LAYERS = 2
N_LAYERS = 2 + N_HIDDEN_LAYERS

# Leak ReLU + Sigmoid
LS_SIGMOID_0 = [[N_FEATURES, 4], [4, 3], [3, 3], [3, 1]]
LS_SIGMOID_1 = [[N_FEATURES, 32], [32, 20], [20, 10], [10, 1]]
LS_SIGMOID_2 = [[N_FEATURES, 32], [32, 20], [20, 20], [20, 1]]
LS_SIGMOID_3 = [[N_FEATURES, 32], [32, 32], [32, 32], [32, 1]]
LS_SIGMOID_4 = [[N_FEATURES, 256], [256, 128], [128, 64], [64, 1]]

# Leaky ReLU + Softmax
LS_SOFTMAX_0 = [[N_FEATURES, 4], [4, 3], [3, 3], [3, 2]]
LS_SOFTMAX_1 = [[N_FEATURES, 32], [32, 20], [20, 10], [10, 2]]
LS_SOFTMAX_2 = [[N_FEATURES, 32], [32, 20], [20, 20], [20, 2]]
LS_SOFTMAX_3 = [[N_FEATURES, 32], [32, 32], [32, 32], [32, 2]]
LS_SOFTMAX_4 = [[N_FEATURES, 256], [256, 128], [128, 64], [64, 2]]

# Define paths to datasets
ML_PLOT_PATH = os.path.join(PWD,
	'plots/multilayer-perceptron/')

ML_CLEAN_DATASET = os.path.join(PWD,
	'data/raw/data.csv')

ML_PROCCESED = os.path.join(PWD,
	'data/processed/')

ML_TRAIN_X = os.path.join(PWD,
	'data/processed/X_train.csv')

ML_TRAIN_Y = os.path.join(PWD,
	'data/processed/y_train.csv')

ML_VALIDATION_X = os.path.join(PWD,
	'data/processed/X_val.csv')

ML_VALIDATION_Y = os.path.join(PWD,
	'data/processed/y_val.csv')

SEED_MB_SOFT = os.path.join(PWD,
	'data/seeds/seed_mb_soft.json')

SEED_MB_SIG = os.path.join(PWD,
	'data/seeds/seed_mb_sig.json')

SEED_FB_SIG = os.path.join(PWD,
	'data/seeds/seed_fb_sig.json')

SEED_FB_SOFT = os.path.join(PWD,
	'data/seeds/seed_fb_soft.json')

SEED_ST_SIG = os.path.join(PWD,
	'data/seeds/seed_st_sig.json')

SEED_ST_SOFT = os.path.join(PWD,
	'data/seeds/seed_st_soft.json')

RESET_ALL = Fore.RESET + Back.RESET + Style.RESET_ALL

YES_NO = Fore.GREEN + Style.BRIGHT + " (yes" + RESET_ALL
YES_NO += " / " + Fore.RED + Style.BRIGHT + "no): " + RESET_ALL + Style.BRIGHT
