import os
import sys
import random
import numpy as np
import pandas as pd
import pickle as pkl
import datetime

os.environ["HADOOP_USER_NAME"] = "xxx"


K_job = 500
K_person = 1000  # number of person classes ### same as the K in 16.data_xxx.ipynb
numTopics = 1000  
test_days = 60
ddays = 7
LIST_CNT = 30
TOP_K = 20
start_date = sys.argv[1]
end_date = sys.argv[2]
path = sys.argv[3]
file_path = 'file://' + path
city_short = sys.argv[4]

def generate_train_test_dates(start_date, end_date):
    start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    date_list_train = [(start + datetime.timedelta(days=x)).strftime('%Y-%m-%d') 
                 for x in range(0, (end - start).days + 1 - test_days)]
    date_list_test = [(start + datetime.timedelta(days=x)).strftime('%Y-%m-%d') 
                 for x in range((end - start).days + 1 - test_days, (end - start).days + 1)]
    
    return date_list_train, date_list_test

dates_train, dates_test = generate_train_test_dates(start_date, end_date)
print('dates tain:', dates_train)
print('dates test:', dates_test)

# For training data
dict_train = {}
with open(f'{path}/train/relation/dict.pkl', 'rb') as f:
    dict_train = pkl.load(f)
print('count train:', len(dict_train))

# For testing data
dict_test = {}
with open(f'{path}/test/relation/dict.pkl', 'rb') as f:
    dict_test = pkl.load(f)
    
from scipy.sparse import coo_matrix, linalg

def A_cal(row, col, values, N, M, deg_n, deg_e, w_e):
    H = coo_matrix((values, (row, col)), shape=(N, M)).tocsc()
    row1 = range(N)
    col1 = range(N)
    Dn = coo_matrix((deg_n, (row1, col1)), shape=(N, N)).tocsc()
    deg_n_inv = []
    for i in deg_n:
        deg_n_inv.append(1.0/(i+0.000001))
    Dn_inv = coo_matrix((deg_n_inv, (row1, col1)), shape=(N, N)).tocsc()
    row2 = range(M)
    col2 = range(M)
    De = coo_matrix((deg_e, (row2, col2)), shape=(M, M)).tocsc()
    deg_e_inv = []
    for i in deg_e:
        deg_e_inv.append(1.0/i)
    De_inv = coo_matrix((deg_e_inv, (row2, col2)), shape=(M, M)).tocsc()
    We = coo_matrix((w_e, (row2, col2)), shape=(M, M)).tocsc()
    tmp = H.dot(We)
    tmp2 = De_inv.dot(H.T)
    A = tmp.dot(tmp2)
#     print(A.shape)
    return A

# For training data
kk = 0
if not os.path.exists(f'{path}/train/sample'):
    os.mkdir(f'{path}/train/sample')
for a in range(K_person):
    b = dict_train[a]
    if a % 100 == 0:
        print('person', a, ':', b)
    if b == 0:
        continue
    
    # b!=0
    # cc = random.randint(0,b-1)
    for cc in range(b):
        
        # row: different sessions for the same person class, col: item list
        joblst = []
        label = -1

        job_relation_dict = {}
        for i in range(K_job):
            job_relation_dict[i] = []
        
        # two views of hypergraphs
        # 同一用户class，不同session，每一session构成一条超边
        H1_row = []
        H1_col = []
        H1_val = []
        De1 = []
        Dv1 = [0 for i in range(K_job)]

        # 同一类用户class，当前Item对后一所有出现的Item构建超边
        H2_row = []
        H2_col = []
        H2_val = []
        De2 = []
        Dv2 = [0 for i in range(K_job)]
    
        cnt = 0
        geek_id = -1
        for c in range(b):
            file = f'{path}/train/relation/relation_'+str(a)+'_'+str(c)+'.csv'
            df = pd.read_csv(file)

            job_list = df['pred_job'].values.tolist()
            if cc == c:
                geek_id = str(df['geek_id'].values[0])
                label_job = job_list[-TOP_K:] 
                job_list = job_list[:-TOP_K]
            joblst = job_list
            label = label_job

            tmp_lst = []
            tmp_cnt = 0
            last = -1

            for jobID in job_list:
                if last >= 0 and jobID not in job_relation_dict[last]:
                    job_relation_dict[last].append(jobID)
                last = jobID


                if jobID not in tmp_lst:
                    tmp_lst.append(jobID)
                    tmp_cnt += 1
                    Dv1[jobID] += 1
                    H1_row.append(jobID)
                    H1_col.append(cnt)
                    H1_val.append(1.0)
            De1.append(tmp_cnt)
            cnt += 1

        cnt = 0
        for key, vallst in job_relation_dict.items():
            for value in vallst:
                H2_row.append(value)
                H2_col.append(cnt)
                H2_val.append(1.0)
                Dv2[value] += 1
            num = len(vallst)
            if num > 0:
                De2.append(num)
                cnt += 1
        N1 = len(Dv1)
        # print('\tN1:', N1)
        M1 = len(De1)
        # print('\tM1:', M1)
        We1 = []
        for i in range(M1):
            We1.append(1.0)

        A1 = A_cal(H1_row, H1_col, H1_val, N1, M1, Dv1, De1, We1)


        N2 = len(Dv2)
        # print('\tN2:', N2)
        M2 = len(De2)
        # print('\tM2:', M2)
        We2 = []
        for i in range(M2):
            We2.append(1.0)

        A2 = A_cal(H2_row, H2_col, H2_val, N2, M2, Dv2, De2, We2)
        clustercenters = []
        with open(f'{path}/models/Clustercenters_occ.pkl', 'rb') as f:
            clustercenters = pkl.load(f)
        Fea = np.array(clustercenters)

        with open(f'{path}/train/sample/datasample_'+str(kk)+'.pkl', 'wb') as f:
            data = (A1, A2, Fea, joblst, label, geek_id)
            if a % 100 == 0:
                print('\t', geek_id)
            pkl.dump(data, f)
            kk += 1
kk_train = kk

# For testing data
kk = 0
if not os.path.exists(f'{path}/test/sample'):
    os.mkdir(f'{path}/test/sample')
for a in range(K_person):
    b = dict_test[a]
    if a % 100 == 0:
        print('person', a, ':', b)
    if b == 0:
        continue

    # cc = random.randint(0,b-1)
    for cc in range(b):
        
        # row: different sessions for the same person class, col: item list
        joblst = []
        label = -1

        job_relation_dict = {}
        for i in range(K_job):
            job_relation_dict[i] = []
        
        # two views of hypergraphs
        H1_row = []
        H1_col = []
        H1_val = []
        De1 = []
        Dv1 = [0 for i in range(K_job)]

        H2_row = []
        H2_col = []
        H2_val = []
        De2 = []
        Dv2 = [0 for i in range(K_job)]
    
        cnt = 0
        geek_id = -1
        for c in range(b):
            file = f'{path}/test/relation/relation_'+str(a)+'_'+str(c)+'.csv'
            df = pd.read_csv(file)

            job_list = df['pred_job'].values.tolist()
            if cc == c:
                geek_id = str(df['geek_id'].values[0])
                label_job = job_list[-TOP_K:] 
                job_list = job_list[:-TOP_K]
            joblst = job_list
            label = label_job

            tmp_lst = []
            tmp_cnt = 0
            last = -1

            for jobID in job_list:
                if last >= 0 and jobID not in job_relation_dict[last]:
                    job_relation_dict[last].append(jobID)
                last = jobID


                if jobID not in tmp_lst:
                    tmp_lst.append(jobID)
                    tmp_cnt += 1
                    Dv1[jobID] += 1
                    H1_row.append(jobID)
                    H1_col.append(cnt)
                    H1_val.append(1.0)
            De1.append(tmp_cnt)
            cnt += 1

        cnt = 0
        for key, vallst in job_relation_dict.items():
            for value in vallst:
                H2_row.append(value)
                H2_col.append(cnt)
                H2_val.append(1.0)
                Dv2[value] += 1
            num = len(vallst)
            if num > 0:
                De2.append(num)
                cnt += 1


        N1 = len(Dv1)
        # print('\tN1:', N1)
        M1 = len(De1)
        # print('\tM1:', M1)
        We1 = []
        for i in range(M1):
            We1.append(1.0)

        A1 = A_cal(H1_row, H1_col, H1_val, N1, M1, Dv1, De1, We1)
        # print('\tA1 shape:', A1.shape)


        N2 = len(Dv2)
        # print('\tN2:', N2)
        M2 = len(De2)
        # print('\tM2:', M2)
        We2 = []
        for i in range(M2):
            We2.append(1.0)

        A2 = A_cal(H2_row, H2_col, H2_val, N2, M2, Dv2, De2, We2)    
        # print('\tA2 shape:', A2.shape)

        # print('\tjob list:', joblst)
        # print('\tjob list len:', len(joblst))
        # print('\tnext job ID:', label)

        clustercenters = []
        with open(f'{path}/models/Clustercenters_occ.pkl', 'rb') as f:
            clustercenters = pkl.load(f)
        Fea = np.array(clustercenters)
        # print('\tocc features shape:', Fea.shape)

        with open(f'{path}/test/sample/datasample_'+str(kk)+'.pkl', 'wb') as f:
            data = (A1, A2, Fea, joblst, label, geek_id)
            if a % 100 == 0:
                print('\t', geek_id, ',', joblst, ',', label)
            pkl.dump(data, f)
            kk += 1
kk_test = kk

print('total train sample count:', kk_train)
print('total test sample count:', kk_test)