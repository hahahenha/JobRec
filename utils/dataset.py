import os
import random
import torch
import networkx as nx
import pickle as pkl
from torch.utils.data import Dataset, DataLoader

from utils.preprocess.Preprocessing import Preprocessing
import warnings
warnings.filterwarnings("ignore")

def transform_list(input_list, k):
    unique_list = []
    [unique_list.append(x) for x in input_list if x not in unique_list]
    if len(unique_list) >= k:
        output_list = unique_list[-k:]
    else:
        output_list = [-1] * (k - len(unique_list)) + unique_list
    return output_list

class JobDataset(Dataset):
    def __init__(self,
                 root="/individual/hanxiao/sample",
                 mode="train",
                 top_k = 10,
                 order=3, clip_num=0.0,val_prop=0.2,k_job=100, k_person=500):
        self.k = k_job
        self.TOP_k = top_k
        self.k_person = 500
        self.order = order
        self.clip_num = clip_num
        all_samples = []
        count = 0
        self.start_from_zero = False
        self.root = root
        sample_list = os.listdir(self.root)  # source ：文件夹路径
        for sample_file in sample_list:
            if os.path.splitext(sample_file)[1] == ".pkl" and os.path.basename(sample_file)[
                                                            0:11] == 'datasample_':  # 后缀是tif, 前11个字符是datasample_的文件
                num = int(os.path.basename(sample_file)[:-4][11:])
                if num > count:
                    count = num
                sample = os.path.join(self.root, sample_file)
                all_samples.append(sample)  # 完整的路径
        # start from 0
        count += 1
        self.memory = {}

        train_sample = int(count * (1-val_prop))
        if mode == "train":
            self.samples = all_samples[:train_sample]
            self.count = len(self.samples)
            print('train sample count:', self.count)
#             print('train sample:', self.samples)
        elif mode == "test":
            self.samples = all_samples[train_sample:]
            self.count = len(self.samples)
            print('test sample count:', self.count)
#             print('test sample:', self.samples)
        elif mode == "inference":
            self.samples = all_samples
            self.count = len(self.samples)

    def __getitem__(self, index):
        while(True):
            if index in self.memory.keys():
                return self.memory[index]
            
            f = open(self.samples[index], 'rb')
            A1, A2, Fea, joblst, label, geek_id = pkl.load(f)
            N = len(Fea)
            f.close()
            A1 = A1.tocoo()
            G1 = nx.from_scipy_sparse_array(A1)
            # print('start preprocessing')
            try:
                preprocessing = Preprocessing(G1, approximation_order=self.order, tolerance=self.clip_num)
                preprocessing.calculate_all()
            except Exception as e:
                print('err', e)
                index = (index + 1) % self.count
                continue

            phi1_indices = torch.LongTensor(preprocessing.phi_matrices[0].nonzero())
            phi1_values = torch.FloatTensor(preprocessing.phi_matrices[0][preprocessing.phi_matrices[0].nonzero()]).view(-1)
            phi1_inverse_indices = torch.LongTensor(preprocessing.phi_matrices[1].nonzero())
            phi1_inverse_values = torch.FloatTensor(preprocessing.phi_matrices[1][preprocessing.phi_matrices[1].nonzero()]).view(-1)

            # Construct the sparse matrix
            sparse1 = torch.sparse_coo_tensor(phi1_indices, phi1_values, torch.Size([N, N]))
            sparse1_inverse = torch.sparse_coo_tensor(phi1_inverse_indices, phi1_inverse_values, torch.Size([N, N]))

            A2 = A2.tocoo()
            G2 = nx.from_scipy_sparse_array(A2)
            try:
                preprocessing = Preprocessing(G2, approximation_order=self.order, tolerance=self.clip_num)
                preprocessing.calculate_all()
            except Exception as e:
                print('err', e)
                index = (index + 1) % self.count
                continue

            phi2_indices = torch.LongTensor(preprocessing.phi_matrices[0].nonzero())
            phi2_values = torch.FloatTensor(preprocessing.phi_matrices[0][preprocessing.phi_matrices[0].nonzero()]).view(-1)
            phi2_inverse_indices = torch.LongTensor(preprocessing.phi_matrices[1].nonzero())
            phi2_inverse_values = torch.FloatTensor(preprocessing.phi_matrices[1][preprocessing.phi_matrices[1].nonzero()]).view(-1)

            # Construct the sparse matrix
            sparse2 = torch.sparse_coo_tensor(phi2_indices, phi2_values, torch.Size([N, N]))
            sparse2_inverse = torch.sparse_coo_tensor(phi2_inverse_indices, phi2_inverse_values, torch.Size([N, N]))
            
#             print('11111#####11111')
            data = {"phi1": sparse1,
                    "phi1_inverse": sparse1_inverse,
                    "phi2": sparse2,
                    "phi2_inverse": sparse2_inverse,
                    "Fea":Fea,
                    "joblst":torch.LongTensor(transform_list(joblst, self.k)),
                    "label":torch.LongTensor(label),
                    "geek_id":geek_id
                    }
            self.memory[index] = data

            return data

    def __len__(self):
        return self.count


if __name__ == '__main__':
    jobdata = JobDataset()
    dataloader = DataLoader(jobdata, batch_size=2, shuffle=True, num_workers=0)

    for i, batch in enumerate(dataloader):
        print(i)
        print(batch)
        break