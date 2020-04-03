import pickle as pl
import matplotlib.pyplot as plt


# visualize the loss
def plot_results(train_losses, valid_losses, batch_size, use_embed_layer, test_acc):
    print('Plotting results')
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    # find position of lowest validation loss
    minposs = valid_losses.index(min(valid_losses)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_losses) + 1)  # consistent scale
    plt.ylim(0, max(train_losses + valid_losses))  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    pic_name = f' bs:{batch_size}' + f' t_f:{use_embed_layer}' + f' acc:{test_acc:.1f}' + '.png'
    fig.savefig(pic_name, bbox_inches='tight')


if __name__ == '__main__':
    file_name = '2020-04-03_tests_res.pkl'
    with open(file_name, 'rb') as f:
        results = pl.load(f)

    fa256, fa128, fa64, fa32, tr256, tr128, tr64, tr32 = results
    # pick which experiment you want by assigning exp
    exp1 = fa128

    # extract the arguments
    batch_size = exp[0]
    train_losses = exp[1]
    valid_losses = exp[2]
    test_acc = exp[3]

    # if str(exp) == str(fa256) or str(fa128) or str(fa64) or str(fa32):
    #     plot_results(train_losses, valid_losses, batch_size, 'False', test_acc)
    # else:
    #     plot_results(train_losses, valid_losses, batch_size, 'True', test_acc)
