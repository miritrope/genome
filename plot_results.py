import matplotlib.pyplot as plt


# visualize the loss
def plot_results(exp_name, plotme, batch_size, tests_acc):
    print('Plotting results')
    fig = plt.figure(figsize=(10, 8))
    label = ['Training Loss', 'Validation Loss']
    for i in range(len(plotme)):
        plt.plot(range(1, len(plotme[i]) + 1), plotme[i], label=label[i])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.xlim(0, 1000)
    plt.ylim(0, 3)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    pic_name = exp_name + '.png'

    fig.savefig(pic_name, bbox_inches='tight')
    print('Saved picture: ', pic_name)

