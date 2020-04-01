import learn_model as lm
import sys
import os

def blockPrint():
    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__


batch_size = [80, 64, 128]
n_epochs = 300
patience = 50

res_bs = []
blockPrint()
for task in batch_size: res_bs.append(lm.execute(task, n_epochs, patience, True))

res_f = []
res_f.append(lm.execute(batch_size[0], n_epochs, patience, True))
res_f.append(lm.execute(batch_size[0], n_epochs, patience, False))

enablePrint()
# printing
print('distinguish between batch size\n')
for r in res_bs: print(r, '\n')
print('distinguish between emb flag\n')
for r in res_f: print(r, '\n')

