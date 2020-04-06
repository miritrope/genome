import matplotlib.pyplot as plt


# visualize the loss
def plot_results(plotme, n_epoch, pic_name):
    print('Plotting results')
    fig = plt.figure(figsize=(10, 8))
    label = ['Training Loss', 'Validation Loss']
    for i in range(len(plotme)):
        plt.plot(range(1, len(plotme[i]) + 1), plotme[i], label=label[i])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.xlim(0, n_epoch)
    plt.ylim(0, 3)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(pic_name, bbox_inches='tight')
    print('Saved picture: ', pic_name)

