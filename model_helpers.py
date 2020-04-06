import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, embedding, n_feats, n_hidden_u, n_hidden_t_enc, n_targets, dropout_sizes):
        super(discrim_net, self).__init__()

        self.batchNorm1 = nn.BatchNorm1d(n_hidden_u)
        self.batchNorm2 = nn.BatchNorm1d(n_hidden_t_enc)

        self.dropOut1 = nn.Dropout(dropout_sizes[0])
        self.dropOut2 = nn.Dropout(dropout_sizes[1])

        self.hidden_1 = nn.Linear(n_feats, n_hidden_u)
        if len(embedding):
            with torch.no_grad():
                self.hidden_1.weight.copy_(embedding)
        self.hidden_2 = nn.Linear(n_hidden_u, n_hidden_t_enc)
        self.hidden_3 = nn.Linear(n_hidden_t_enc, n_targets)

    def forward(self, x):
        # input_discrim size: batch_size, n_feats
        x = torch.relu(self.hidden_1(x))
        x = self.dropOut1(self.batchNorm1(x))
        x = self.dropOut2(self.batchNorm2(self.hidden_2(x)))
        y = F.softmax(self.hidden_3(x), dim=1)

        return y

