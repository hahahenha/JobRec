import torch
from torch import nn
import torch.nn.functional as F
from torch_sparse import spspmm, spmm

class HGNN(nn.Module):
    def __init__(self, in_channels, out_channels, ncount, device, top_k):
        super(HGNN, self).__init__()
        self.TOP_K = top_k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        self.device = device
        self.rnn = nn.RNN(self.out_channels, self.ncount)
        self.layer1 = HGNN_layer(self.in_channels, self.out_channels, self.ncount, self.device)
        self.layer2 = HGNN_layer(self.in_channels, self.out_channels, self.ncount, self.device)
        self.dense = nn.Linear(self.out_channels, self.TOP_K).double()
        self.item_factors = nn.Embedding(self.ncount, self.out_channels)

        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, phi1, phi1_inv, phi2, phi2_inv, fea, joblst):
        # print(phi1.shape)

        outs = []
        embs = []
        for i in range(phi1.shape[0]):
            job_indices = joblst[i]
            filtered_indices = job_indices[job_indices != -1]

            emb1 = self.item_factors(filtered_indices)

            emb2, _ = self.rnn(emb1)
            emb3 = emb2[-1]
            embs.append(emb3)

            phi_tmp = phi1[i]
            phi_inverse_tmp = phi1_inv[i]
            feature_tmp = fea[i]

            phi_indices = phi_tmp._indices()
            phi_values = phi_tmp._values()

            phi_inverse_indices = phi_inverse_tmp._indices()
            phi_inverse_values = phi_inverse_tmp._values()

            out1 = self.layer1.forward(phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, feature_tmp)

            phi_tmp = phi2[i]
            phi_inverse_tmp = phi2_inv[i]

            phi_indices = phi_tmp._indices()
            phi_values = phi_tmp._values()

            phi_inverse_indices = phi_inverse_tmp._indices()
            phi_inverse_values = phi_inverse_tmp._values()

            out2 = self.layer2.forward(phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, feature_tmp)

            output = out1 + out2
            outs.append(output)

        res = torch.stack(outs, dim=0)
        Emb = torch.stack(embs, dim=0)

        res2 = F.gelu(self.dense.forward(res))
        res3 = res2.permute(0,2,1)
        Emb2 = F.gelu(Emb.unsqueeze(1).repeat(1,self.TOP_K,1))
        res4 = res3 * Emb2

        return F.sigmoid(res4.permute(0,2,1))



class HGNN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, ncount, device):
        super(HGNN_layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        self.device = device

        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels)).double().to(self.device)
        self.diagonal_weight_indices = torch.LongTensor([[node for node in range(self.ncount)],
                                                         [node for node in range(self.ncount)]])

        self.diagonal_weight_indices = self.diagonal_weight_indices.to(self.device)
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.ncount, 1))

        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.9, 1.1)
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, features, dropout=None):
        rescaled_phi_indices, rescaled_phi_values = spspmm(phi_indices,
                                                           phi_values,
                                                           self.diagonal_weight_indices,
                                                           self.diagonal_weight_filter.view(-1),
                                                           self.ncount,
                                                           self.ncount,
                                                           self.ncount)

        phi_product_indices, phi_product_values = spspmm(rescaled_phi_indices,
                                                         rescaled_phi_values,
                                                         phi_inverse_indices,
                                                         phi_inverse_values,
                                                         self.ncount,
                                                         self.ncount,
                                                         self.ncount)

        filtered_features = torch.mm(features, self.weight_matrix)

        localized_features = spmm(phi_product_indices,
                                  phi_product_values,
                                  self.ncount,
                                  self.ncount,
                                  filtered_features)

        if dropout is not None:
            localized_features = torch.nn.functional.dropout(torch.nn.functional.relu(localized_features),
                                                           training=self.training,
                                                           p=dropout)

        return localized_features

    def forward_sp(self, phi_indices, phi_values, phi_inverse_indices,
                phi_inverse_values, feature_indices, feature_values, dropout=0.8):
        rescaled_phi_indices, rescaled_phi_values = spspmm(phi_indices,
                                                           phi_values,
                                                           self.diagonal_weight_indices,
                                                           self.diagonal_weight_filter.view(-1),
                                                           self.ncount,
                                                           self.ncount,
                                                           self.ncount)

        phi_product_indices, phi_product_values = spspmm(rescaled_phi_indices,
                                                         rescaled_phi_values,
                                                         phi_inverse_indices,
                                                         phi_inverse_values,
                                                         self.ncount,
                                                         self.ncount,
                                                         self.ncount)

        filtered_features = spmm(feature_indices,
                                 feature_values,
                                 self.ncount,
                                 self.in_channels,
                                 self.weight_matrix)

        localized_features = spmm(phi_product_indices,
                                  phi_product_values,
                                  self.ncount,
                                  self.ncount,
                                  filtered_features)

        dropout_features = torch.nn.functional.dropout(torch.nn.functional.relu(localized_features),
                                                       training=self.training,
                                                       p=dropout)
        return dropout_features