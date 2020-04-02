import learn_model as lm
import sys
import os
import pickle
from datetime import date

def blockPrint():
    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__


batch_size = [256, 128, 64, 32]
n_epochs = 800
patience = 50
fold = 1

results = []
blockPrint()
for task in batch_size: results.append(lm.execute(fold, task, n_epochs, patience, False))
for task in batch_size: results.append(lm.execute(fold, task, n_epochs, patience, True))

enablePrint()
for r in results: print(r, '\n')

# save the results
today = date.today()
with open(today + 'tests_res.pkl', 'wb') as f:
    pickle.dump(results, f)
