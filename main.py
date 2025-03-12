""" Main program for the multilayer perceptron project. """
import config
from layer.dense import DenseLayer
from model.seed import seed_finder
from colorama import Fore, Style
from split_data import assemble_data
from config import RESET_ALL
from config import SEED_MB_SOFT, SEED_MB_SIG, SEED_FB_SIG, SEED_FB_SOFT, SEED_ST_SOFT, SEED_ST_SIG

from training.full_batch.sigmoid import init_fb_sig
from training.full_batch.softmax import init_fb_soft
from training.full_batch.sigmoid_seed import init_fb_sig_seed
from training.full_batch.softmax_seed import init_fb_soft_seed
from training.full_batch.sigmoid_predict import init_fb_sig_pred
from training.full_batch.softmax_predict import init_fb_soft_pred

from training.mini_batch.sigmoid import init_mb_sig
from training.mini_batch.softmax import init_mb_soft
from training.mini_batch.sigmoid_seed import init_mb_sig_seed
from training.mini_batch.softmax_seed import init_mb_soft_seed
from training.mini_batch.sigmoid_predict import init_mb_sig_pred
from training.mini_batch.softmax_predict import init_mb_soft_pred

from training.stochastic.sigmoid import init_st_sig
from training.stochastic.softmax import init_st_soft
from training.stochastic.sigmoid_seed import init_st_sig_seed
from training.stochastic.softmax_seed import init_st_soft_seed
from training.stochastic.sigmoid_predict import init_st_sig_pred
from training.stochastic.softmax_predict import init_st_soft_pred

from tests.single_layer import init_single_layer

def get_model_funcs(model_type, f_actv):
    if model_type == 1: # Full batch
        if f_actv == 1:
            return init_fb_soft, init_fb_soft_seed, init_fb_soft_pred, SEED_FB_SOFT
        if f_actv == 2:
            return init_fb_sig, init_fb_sig_seed, init_fb_sig_pred, SEED_FB_SIG
    if model_type == 2: # Mini-batch
        if f_actv == 1: # Softmax
            return init_mb_soft, init_mb_soft_seed, init_mb_soft_pred, SEED_MB_SOFT
        if f_actv == 2: # Sigmoid
            return init_mb_sig, init_mb_sig_seed, init_mb_sig_pred, SEED_MB_SIG
    if model_type == 3: # Stochastic
        if f_actv == 1:
            return init_st_soft, init_st_soft_seed, init_st_soft_pred, SEED_ST_SOFT
        if f_actv == 2:
            return init_st_sig, init_st_sig_seed, init_st_sig_pred, SEED_ST_SIG
    return None, None, None, None

def handle_dec_type(model_type, f_actv, dec_type, layers):
    init, init_seed, init_pred, seed = get_model_funcs(model_type, f_actv)
    if dec_type == 1:
        layers = init()
    elif dec_type == 2:
        seed_finder(init_seed, SEED_FB_SOFT)
        return 1, layers
    elif dec_type == 3 and layers:
        init_pred(layers)
    elif dec_type == 3 and layers is None:
        print(Fore.LIGHTRED_EX + Style.DIM
            + "You need to train the model first. Please try again." + RESET_ALL)
    else:
        print(Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
    return 0, layers

def train_fb(f_actv):
    """ Full-batch training program. """
    layers : DenseLayer = None
    while 1:
        print("\n└─> Choose an option: ")
        print("\t[1]" + Style.BRIGHT + " Train " + RESET_ALL)
        print("\t[2]" + Style.BRIGHT + " Seed finder " + RESET_ALL)
        print("\t[3]" + Style.BRIGHT + " Predict " + RESET_ALL)
        dec_type = input(Style.BRIGHT + Fore.LIGHTCYAN_EX + "└─> " + Fore.RESET)
        print(RESET_ALL)
        if dec_type.isdigit() is False:
            print(Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
        else:
            dec_type = int(dec_type)
            ret, layers = handle_dec_type(1, f_actv, dec_type, layers)
            if ret == 1:
                break

def train_mb(f_actv):
    """ Mini-batch training program. """
    layers : DenseLayer = None
    while 1:
        print("\n└─> Choose an option: ")
        print("\t[1]" + Style.BRIGHT + " Train " + RESET_ALL)
        print("\t[2]" + Style.BRIGHT + " Seed finder " + RESET_ALL)
        print("\t[3]" + Style.BRIGHT + " Predict " + RESET_ALL)
        dec_type = input(Style.BRIGHT + Fore.LIGHTCYAN_EX + "└─> " + Fore.RESET)
        print(RESET_ALL)
        if dec_type.isdigit() is False:
            print(Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
        else:
            dec_type = int(dec_type)
            ret, layers = handle_dec_type(2, f_actv, dec_type, layers)
            if ret == 1:
                break

def train_st(f_actv):
    """ Stochastic training program. """
    layers : DenseLayer = None
    while 1:
        print("\n└─> Choose an option: ")
        print("\t[1]" + Style.BRIGHT + " Train " + RESET_ALL)
        print("\t[2]" + Style.BRIGHT + " Seed finder " + RESET_ALL)
        print("\t[3]" + Style.BRIGHT + " Predict " + RESET_ALL)
        dec_type = input(Style.BRIGHT + Fore.LIGHTCYAN_EX + "└─> " + Fore.RESET)
        print(RESET_ALL)
        if dec_type.isdigit() is False:
            print(Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
        else:
            dec_type = int(dec_type)
            ret, layers = handle_dec_type(3, f_actv, dec_type, layers)
            if ret == 1:
                break

def choose_model():

    print("\n└─> How should data be fed into the network?")
    print("\t[1]" + Style.BRIGHT + " Whole-batch " + RESET_ALL)
    print("\t[2]" + Style.BRIGHT + " Mini-batch " + RESET_ALL)
    print("\t[3]" + Style.BRIGHT + " Stochastic" + RESET_ALL)
    print("\t[4]" + Style.BRIGHT + " Go back ↑" + RESET_ALL)
    while 1:
        str2 = input(Style.BRIGHT + Fore.LIGHTCYAN_EX + "└─> " + Fore.RESET)
        print(RESET_ALL)
        if str2.isdigit():
            if int(str2) > 0 and int(str2) < 5:
                ds_type = int(str2)
                if ds_type == 1:
                    func = train_fb
                elif ds_type == 2:
                    func = train_mb
                elif ds_type == 3:
                    func = train_st
                elif ds_type == 4:
                    return 1
                break
        else:
            print(
                Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
        continue

    print("\n└─> Which activation function do you want to use?")
    print("\t[1]" + Style.BRIGHT + " Softmax " + RESET_ALL)
    print("\t[2]" + Style.BRIGHT + " Sigmoid " + RESET_ALL)
    print("\t[3]" + Style.BRIGHT + " Go back ↑" + RESET_ALL)
    while 1:
        str3 = input(Style.BRIGHT + Fore.LIGHTCYAN_EX + "└─> " + Fore.RESET)
        print(RESET_ALL)
        if str3.isdigit():
            f_actv = int(str3)
            if f_actv in (1, 2):
                func(f_actv)
                break
            if f_actv == 3:
                return 1
        print(
            Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
        continue
    return 0

def plot_graph():
    while 1:
        print("\n└─> Choose an option: ")
        print("\t[1]" + Style.BRIGHT + " FB Softmax vs. MB Softmax " + RESET_ALL)
        print("\t[2]" + Style.BRIGHT + " FB Sigmoid vs MB Softmax " + RESET_ALL)
        print("\t[3]" + Style.BRIGHT + " ST Sigmoid vs MB Sigmoid " + RESET_ALL)
        print("\t[4]" + Style.BRIGHT + " Single-layer perceptron " + RESET_ALL)
        print("\t[5]" + Style.BRIGHT + " Go back ↑" + RESET_ALL)
        dec_type = input(Style.BRIGHT + Fore.LIGHTCYAN_EX + "└─> " + Fore.RESET)
        print(RESET_ALL)
        if dec_type.isdigit() is False:
            print(Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
        else:
            dec_type = int(dec_type)
            if dec_type == 1:
                _, plt = init_fb_soft(plt=None, plt_it=0, plt_ct=4, plt_show=False, plt_ret=True)
                _ = init_mb_soft(plt=plt, plt_it=2, plt_show=True, plt_ret=False)
            elif dec_type == 2:
                _, plt = init_fb_sig(plt=None, plt_it=0, plt_ct=4, plt_show=False, plt_ret=True)
                _ = init_mb_soft(plt=plt, plt_it=2, plt_show=True, plt_ret=False)
            elif dec_type == 3:
                _, plt = init_st_sig(plt=None, plt_it=0, plt_ct=4, plt_show=False, plt_ret=True)
                _ = init_mb_sig(plt=plt, plt_it=2, plt_show=True, plt_ret=False)
            elif dec_type == 4:
                init_single_layer()
            elif dec_type == 5:
                return 1
            else:
                print(Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
                continue
    return 0

def main():
    print("\n└─> Choose an option: (" + Fore.LIGHTWHITE_EX + Style.BRIGHT
                + "only numbers" + RESET_ALL + "): ")
    print("\t[1]" + Style.BRIGHT + " Assemble " + RESET_ALL + "data" + RESET_ALL)
    print("\t[2]" + Style.BRIGHT + " Choose " + RESET_ALL + "the model" + RESET_ALL)
    print("\t[3]" + Fore.MAGENTA + " Bonus " + RESET_ALL + "options")
    while 1:
        str1 = input(Style.BRIGHT + Fore.LIGHTCYAN_EX + "└─> " + Fore.RESET)
        print(RESET_ALL)
        if str1.isdigit() is True:
            option = int(str1)

            if option == 1:
                assemble_data()
            elif option == 2:
                ret = choose_model()
                if ret == 0:
                    break
                print("\n└─> Choose an option: (" + Fore.LIGHTWHITE_EX + Style.BRIGHT
                    + "only numbers" + RESET_ALL + "): ")
                print("\t[1]" + Style.BRIGHT + " Split " + RESET_ALL + "data" + RESET_ALL)
                print("\t[2]" + Style.BRIGHT + " Train " + RESET_ALL + "the model" + RESET_ALL)
                print("\t[3]" + Fore.MAGENTA + " Bonus " + RESET_ALL + "options")
            elif option == 3:
                ret = plot_graph()
                if ret == 0:
                    break
                print("\n└─> Choose an option: (" + Fore.LIGHTWHITE_EX + Style.BRIGHT
                    + "only numbers" + RESET_ALL + "): ")
                print("\t[1]" + Style.BRIGHT + " Split " + RESET_ALL + "data" + RESET_ALL)
                print("\t[2]" + Style.BRIGHT + " Train " + RESET_ALL + "the model" + RESET_ALL)
                print("\t[3]" + Fore.MAGENTA + " Bonus " + RESET_ALL + "options")
        else:
            print(
                Fore.RED + Style.DIM + "Invalid input. Please try again." + RESET_ALL)
        continue

main()
