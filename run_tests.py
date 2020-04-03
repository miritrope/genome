import learn_model as lm
import sys
import os
import pickle
from datetime import date
import plot_results as pr


def blockPrint():
    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__


batch_sizes = [256, 128, 64, 32]
n_epochs = 500
patience = 50
fold = 1
use_embed_layer = False
results = []


blockPrint()
for task in batch_sizes: results.append(lm.execute(fold, task, n_epochs, patience, use_embed_layer))

enablePrint()
for r in results: print(r, '\n')

# save the results
today = date.today()
with open(str(today) + '_tests_res.pkl', 'wb') as f:
    pickle.dump(results, f)
print('Saved data file: ', str(today) + '_tests_res.pkl')

train_accs = list(r[2] for r in results)
test_accs = list(r[3] for r in results)

# visualize the loss
# return [0. train_losses, 1. valid_losses, 2. train_accs, 3. test_acc, 4 epoch_times, 5 train_time]
pr.plot_results(train_accs, batch_sizes, use_embed_layer, test_accs)
