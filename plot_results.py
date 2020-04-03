import pickle as pl
import matplotlib.pyplot as plt
import numpy as np

# visualize the loss
def plot_results(plotme, batch_size, use_embed_layer, tests_acc):
    print('Plotting results')
    fig = plt.figure(figsize=(10, 8))
    lenses = []
    max_vals = []

    for i in range(len(plotme)):
        lab = str(batch_size[i]) + ' '
        lab += f' {tests_acc[i]:.1f}'
        plt.plot(range(1, len(plotme[i]) + 1), plotme[i], label=lab)
        lenses.append(len(plotme[i]))
        max_vals.append(np.max(plotme[i]))

    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.xlim(0, np.max(lenses) + 1)
    plt.ylim(0, np.max(max_vals) + 1)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    pic_name = f' t_f:{use_embed_layer}' + '.png'

    fig.savefig(pic_name, bbox_inches='tight')
    print('Saved picture: ', pic_name)


if __name__ == '__main__':
    file_name = '2020-04-03_tests_res.pkl'
    with open(file_name, 'rb') as f:
        results = pl.load(f)

    fa256, fa128, fa64, fa32, tr256, tr128, tr64, tr32 = results
