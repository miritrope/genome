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
n_epochs = 200
patience = 50
fold = 1
use_embed_layer = True
results = []


blockPrint()
for task in batch_sizes: results.append(lm.execute(fold, task, n_epochs, patience, use_embed_layer))

enablePrint()
for r in results: print(r, '\n')

# save the results
today = date.today()
if use_embed_layer:
    experiment = ' with aux net results'
else:
    experiment = ' without aux net results'

res_file_name = str(today) + experiment
with open(res_file_name + '.pkl', 'wb') as f:
    pickle.dump(results, f)
print('Saved results data file: ', res_file_name + '.pkl')

# return [0. train_losses, 1. valid_losses, 2. train_accs, 3. valid_accs, 4. test_acc, 5. epoch_times, 5 train_time]
train_accs = list(r[2] for r in results)
valid_accs = list(r[3] for r in results)
test_accs = list(r[4] for r in results)


# visualize the loss
pr.plot_results(experiment, valid_accs, batch_sizes, use_embed_layer, test_accs)
