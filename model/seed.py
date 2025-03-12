import json
from colorama import Fore, Style, Back
from config import RESET_ALL

def seed_finder(init_model, SEED_FILE):
	print(Fore.LIGHTCYAN_EX + Style.DIM + "(Exit the program to stop the seed finder with CTRL + C)" + RESET_ALL)
	b_acc = 0
	with open(SEED_FILE, 'r') as file:
		if file.read(1):
			file.seek(0)
			data = json.load(file)
			seed = int(data['seed'])
			b_epoch = int(data['epoch'])
			b_acc = float(data['acc'])
	while True:
		seed, ret_acc, b_epoch = init_model()
		if (ret_acc > b_acc):
			b_acc = ret_acc
			print(f"New best accuracy: {b_acc} at epoch {b_epoch}, seed: {seed}")
			with open(SEED_FILE, 'w') as file:
				json.dump({'seed': seed, 'epoch': b_epoch, 'acc': b_acc}, file)
