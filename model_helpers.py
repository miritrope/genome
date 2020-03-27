import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ???W=Uniform(encoder_net_init)
class feat_emb_net(nn.Module):
    def __init__(self, n_feats, n_hidden_u, n_hidden_t_enc):
        super(feat_emb_net, self).__init__()
        self.hidden_1 = nn.Linear(n_feats, n_hidden_u)
        self.hidden_2 = nn.Linear(n_hidden_u, n_hidden_t_enc)

    def forward(self, x):
        x = torch.tanh(self.hidden_1(x))
        y_pred = torch.tanh(self.hidden_2(x))
        return y_pred


class discrim_net(nn.Module):
    def __init__(self, embedding, n_feats, n_hidden_u, n_hidden_t_enc, n_targets):
        super(discrim_net, self).__init__()

        self.batchNorm1 = nn.BatchNorm1d(n_hidden_u)
        self.batchNorm2 = nn.BatchNorm1d(n_hidden_t_enc)

        self.dropOut = nn.Dropout(0.5)
        #n_feats = 315, hidden1 weight shape is: 100 x 315, embedding shape is: 315 x 100
        self.hidden_1 = nn.Linear(n_feats, n_hidden_u)
        with torch.no_grad():
            self.hidden_1.weight.copy_(embedding)
        self.hidden_2 = nn.Linear(n_hidden_u, n_hidden_t_enc)
        self.hidden_3 = nn.Linear(n_hidden_t_enc, n_targets)

    def forward(self, x):
        x = torch.relu(self.hidden_1(x))# input_discrim size = 80,315
        x = self.dropOut(self.batchNorm1(x))
        x = self.dropOut(self.batchNorm2(self.hidden_2(x)))
        y = F.softmax(self.hidden_3(x), dim=1)

        return y


#     # discrim_net = InputLayer((batch_size, n_feats), input_var_sup)
#     # discrim_net = DenseLayer(discrim_net, n_hidden_t_enc,
#     #                          W=embedding, nonlinearity=rectify)
#     # discrim_net = BatchNormLayer(discrim_net)
#     # discrim_net = DropoutLayer(discrim_net)
#     # discrim_net = DenseLayer(discrim_net, num_units=hid)
#     # discrim_net = BatchNormLayer(discrim_net)
#     # discrim_net = DropoutLayer(discrim_net)
#     # discrim_net = DenseLayer(discrim_net, num_units=n_targets,
#     #     nonlinearity=eval(disc_nonlinearity))
#
#     return discrim_net


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        # input_discrim size = 80,315
        yhat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step