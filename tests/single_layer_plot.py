import matplotlib.pyplot as plt

def plot_acc(acc_train, acc_val, EPOCHS):
    # Plotting the graph
    plt.figure(figsize=(20, 12))
    epochs = list(range(1, EPOCHS + 1))
    plt.plot(epochs, acc_train, label="Training loss", color="lawngreen", marker="", linestyle="-")
    plt.plot(epochs, acc_val, label="Validation loss", color="steelblue", marker="", linestyle="--")
    plt.ylim(0, 1)

    # Adding labels, title, and legend
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Learning Curves", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_ce(ce_train, ce_val, EPOCHS):
    # Plotting the graph
    plt.figure(figsize=(20, 12))
    epochs = list(range(1, EPOCHS + 1))
    plt.plot(epochs, ce_train, label="Training loss", color="lawngreen", marker="", linestyle="-")
    plt.plot(epochs, ce_val, label="Validation loss", color="steelblue", marker="", linestyle="--")
    plt.ylim(0, 1)

    # Adding labels, title, and legend
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Learning Curves", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Show the plot
    plt.show()
