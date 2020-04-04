import matplotlib.pyplot as plt


# visualize the loss
def plot_results(exp_name, plotme, batch_size, tests_acc):
    print('Plotting results')
    fig = plt.figure(figsize=(10, 8))

    for i in range(len(plotme)):
        lab = str(batch_size[i]) + ' '
        lab += f' {tests_acc[i]:.1f}'
        plt.plot(range(1, len(plotme[i]) + 1), plotme[i], label=lab)

    plt.xlabel('Epoch')
    plt.ylabel('Time')

    plt.xlim(0, 200)
    plt.ylim(0, 20)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    pic_name = exp_name + '.png'

    fig.savefig(pic_name, bbox_inches='tight')
    print('Saved picture: ', pic_name)

