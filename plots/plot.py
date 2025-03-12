import config
from config import N_LAYERS
from loss import f_loss, f_mse, f_mae, f_r2score, f_cross_entropy
import matplotlib.pyplot as plt

class Plot:
	acc = None

	mse = None

	mae = None

	def __init__(self, data_count):
		self.acc = [[] for _ in range(data_count)]

		self.mse = [[] for _ in range(data_count)]

		self.mae = [[] for _ in range(data_count)]
  
		self.ce = [[] for _ in range(data_count)]
  
		# Plot configurations
		self.label = ["" for _ in range(data_count)]
		self.color = ["" for _ in range(data_count)]
		self.linestyle = ["" for _ in range(data_count)]
  
		self.epochs = [0 for _ in range(data_count)]

	def set_plot_config(self, data_type, color, pos, linestyle, epochs):
		self.label[pos] = (data_type)
		self.color[pos] = (color)
		self.linestyle[pos] = (linestyle)
		self.epochs[pos] = (epochs)

	def set_error_data(self, X, y, layers, pos):
		# Append the data
		self.append_f_mae(X, y, layers, pos)
		self.append_f_mse(X, y, layers, pos)
		self.append_f_r2score(X, y, layers, pos)
		self.append_f_cross_entropy(X, y, layers, pos)

	def get_error_data(self, pos):
		return self.acc[pos][-1], self.mse[pos][-1], self.mae[pos][-1], self.ce[pos][-1]

	def bool_early_stop(self, pos, epoch, patience=5):
		if (epoch < 100):
			return False
		if len(self.ce[pos]) > patience:
			recent_losses = self.ce[pos][-patience:]
			if all(recent_losses[i] <= recent_losses[i + 1] for i in range(len(recent_losses) - 1)):
				self.epochs[pos] = epoch + 1
				self.epochs[pos - 1] = epoch + 1
				print("Early stopping...")
				return True
		return False

	def append_f_mse(self, X, y, layers, pos):
		mse_data = self.get_plot_data(X, y, layers, "mse")
		self.mse[pos].append(mse_data)

	def append_f_mae(self, X, y, layers, pos):
		mae_data = self.get_plot_data(X, y, layers, "mae")
		self.mae[pos].append(mae_data)

	def append_f_r2score(self, X, y, layers, pos):
		r2_data = self.get_plot_data(X, y, layers, "r2")
		self.acc[pos].append(r2_data)

	def append_f_cross_entropy(self, X, y, layers, pos):
		ce_data = self.get_plot_data(X, y, layers, "cross_entropy")
		self.ce[pos].append(ce_data)

	def get_plot_data(self, X, y, layers, loss_type):
		activations = [None] * N_LAYERS

		train_input = X
		for i in range(N_LAYERS):
			activations[i], _ = layers[i].forward(train_input)
			train_input = activations[i]
		""" import numpy as np
  		activations[-1] = np.round(activations[-1]) """
		if (loss_type == "r2"):
			loss = f_r2score(y, activations[-1])
		elif (loss_type == "mse"):
			loss = f_mse(y, activations[-1])
		elif(loss_type == "mae"):
			loss = f_mae(y, activations[-1])
		elif(loss_type == "cross_entropy"):
			loss = f_cross_entropy(y, activations[-1])
		return loss

	def plot_acc_epochs(self):
		""" Plot the accuracy over the epochs. """
		
		# Plotting the graph
		plt.figure(figsize=(20, 12))
		for acc, label, color, linestyle, epochs in zip(self.acc, self.label, self.color, self.linestyle, self.epochs):
			epochs = list(range(1, epochs + 2))
			plt.plot(epochs, acc, label=label + " loss", color=color, marker="", linestyle=linestyle)
		plt.ylim(0, 1)

		# Adding labels, title, and legend
		plt.xlabel("Epochs", fontsize=14)
		plt.ylabel("Accuracy", fontsize=14)
		plt.title("Learning Curves", fontsize=16)
		plt.legend(fontsize=12)
		plt.grid(True)

		# Show the plot
		plt.show()

	def plot_loss_epochs(self):
		""" Plot the loss over the epochs. """
		
		# Plotting the graph
		plt.figure(figsize=(20, 12))
		for mse, label, color, linestyle, epochs in zip(self.mse, self.label, self.color, self.linestyle, self.epochs):
			epochs = list(range(1, epochs + 2))
			plt.plot(epochs, mse, label=label + " loss", color=color, marker="", linestyle=linestyle)
		plt.ylim(0, 0.7)

		# Adding labels, title, and legend
		plt.xlabel("Epochs", fontsize=14)
		plt.ylabel("Loss", fontsize=14)
		plt.title("Learning Curves", fontsize=16)
		plt.legend(fontsize=12)
		plt.grid(True)

		# Show the plot
		plt.show()