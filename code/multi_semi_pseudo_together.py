import multiprocessing
import tqdm
import glob
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def func_read_one(argv=None, feature_root=None, name=None):
    feature_root, name = argv
    feature_dir = glob.glob(os.path.join(feature_root, name+'*'))
    feature_path = feature_dir[0]
    feature = []
    if feature_path.endswith('.npy'):
        single_feature = np.load(feature_path)
        single_feature = single_feature.squeeze()
        feature.append(single_feature)
    else:
        facenames = os.listdir(feature_path)
        for facename in sorted(facenames):
            facefeat = np.load(os.path.join(feature_path, facename))
            feature.append(facefeat)

    single_feature = np.array(feature).squeeze(0)
    if len(single_feature) == 0:
        print ('feature has errors!!')
    # elif len(single_feature.shape) == 2:
    #     single_feature = np.mean(single_feature, axis=0)
    return single_feature
def read_data_multiprocess(feature_root=None,mode=None,dataset=None,semi_num=None,lab_num=None,retrain=None):
    '''
    feature_root contain all
    mode is used to determined which csv to use
    '''
    import random

    path = f'./{mode}_emo2_{dataset}.csv' if dataset != 'iemocap' else f'./{mode}_{dataset}.csv' # test_ami.csv
    df = pd.read_csv(path)
    names, vals, emos, idxs = [], [], [], []
    random.seed(1111)
    lab_index = random.sample(range(len(df)), lab_num)
    k = 0.
    for idx, row in df.iterrows():
        if idx in lab_index:
            names.append(row['name'])
            vals.append(row.get('val',-1))
            emos.append(row.get('emo',-1))
            idxs.append(k)
            k+=1
    if retrain:
        path = './ami_pesudo.csv'
        df = pd.read_csv(path)
        random.seed(1111)
        semi_index = random.sample(range(len(df)),semi_num)
        non_audio = ['ES2007b_B_131179', 'ES2007b_B_150557', 'ES2007b_B_150920', 'ES2007b_B_151748']
        k = -1
        for idx, row in df.iterrows():
            if row['name'] in non_audio:
                pass
            elif idx in semi_index:
                names.append(row['name'])
                vals.append(row['val'])
                emos.append(row['emo'])
                idxs.append(k)
                k-=1

    params = []
    for ii, name in tqdm.tqdm(enumerate(names)):
        params.append((feature_root, name))
    with multiprocessing.Pool(processes=8) as pool:
        features = list(tqdm.tqdm(pool.imap(func_read_one, params), total=len(params)))
    feature_dim = features[0].shape[-1]
    print(f'Input feature {feature_root} ===> dim is {feature_dim}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    name2feats, name2vals, name2emos, name2idx = {}, {}, {}, {}
    for ii in range(len(names)):
        name2feats[names[ii]] = features[ii]
        name2vals[names[ii]] = vals[ii]
        name2emos[names[ii]] = emos[ii]
        name2idx[names[ii]] = idxs[ii]


    return name2feats,(name2vals,name2emos,name2idx),feature_dim

class MERDataset(Dataset):
    def __init__(self, audio_root, text_root, video_root, data_type, dataset, semi_num=None, lab_num=None, retrain=None):
        assert data_type in ['train', 'valid', 'test']
        self.name2audio, (self.name2labels, self.name2emos, self.name2idx), self.adim = read_data_multiprocess(audio_root, mode=data_type, dataset=dataset, semi_num=semi_num, lab_num=lab_num, retrain=retrain)
        self.name2text, (self.name2labels, self.name2emos, self.name2idx), self.tdim = read_data_multiprocess(text_root, mode=data_type, dataset=dataset, semi_num=semi_num, lab_num=lab_num, retrain=retrain)
        self.name2video, (self.name2labels, self.name2emos, self.name2idx), self.vdim = read_data_multiprocess(video_root, mode=data_type, dataset=dataset, semi_num=semi_num, lab_num=lab_num, retrain=retrain)
        self.names = [name for name in self.name2audio if 1 == 1]


    def __getitem__(self, index):
        name = self.names[index]
        return  torch.FloatTensor(self.name2audio[name]), \
                torch.FloatTensor(self.name2text[name]), \
                torch.FloatTensor(self.name2video[name]), \
                self.name2labels[name], \
                self.name2emos[name], \
                self.name2idx[name]


    def __len__(self):
        return len(self.names)

    def get_featDim(self):
        print(f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

def pad_tensor(vec, pad, dim, f_dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    vec = vec.view(-1, f_dim)
    if vec.shape[0] > pad -2:
        vec = vec[: pad - 2]
    pad_size = list(vec.shape)#[798,1024]
    # if pad > vec.size(dim):
    cls_sep = torch.zeros(1, f_dim)
    pad_size[dim] = pad - 2 - vec.size(dim)#[800-20,1024]
    return torch.cat([cls_sep, vec, cls_sep, torch.zeros(*pad_size)], dim=dim)

def act_len(vec, f_dim, max_len):
    vec = vec.view(-1, f_dim)
    
    if vec.shape[0]+2 <= max_len:
        return vec.shape[0]+2
    else:
        return max_len

# def collate_func(batch_dic):

#     fea_a_batch, fea_t_batch, fea_v_batch = [],[],[]
#     fea_a_len, fea_t_len, fea_v_len = [], [], []
#     label_batch = []

#     for i in range(len(batch_dic)):  # 分别提取批样本中的feature、label、id、length信息
#         fea_a_len.append(act_len(batch_dic[i][0],f_dim=1024))
#         fea_t_len.append(act_len(batch_dic[i][1],f_dim=1024))
#         fea_v_len.append(act_len(batch_dic[i][2],f_dim=256))
#         fea_a_batch.append(pad_tensor(batch_dic[i][0], pad=800, dim=0, f_dim=1024))
#         fea_t_batch.append(pad_tensor(batch_dic[i][1], pad=200, dim=0, f_dim=1024))
#         fea_v_batch.append(pad_tensor(batch_dic[i][2], pad=200, dim=0, f_dim=256))
#         label_batch.append(batch_dic[i][-1])
#     fea_a_batch = torch.stack(fea_a_batch,dim=0)
#     fea_t_batch = torch.stack(fea_t_batch,dim=0)
#     fea_v_batch = torch.stack(fea_v_batch,dim=0)
#     label_batch = torch.FloatTensor(label_batch)
#     fea_a_len = torch.FloatTensor(fea_a_len)
#     fea_t_len = torch.FloatTensor(fea_t_len)
#     fea_v_len = torch.FloatTensor(fea_v_len)
#     return fea_a_batch,fea_a_len,fea_t_batch,fea_t_len,fea_v_batch,fea_v_len,label_batch

if __name__ == '__main__':
    # train_dataset = MERDataset(audio_root='D:\比赛dataset\mosi_feature\hubertlarge\hubert-large-FRA',text_root='D:\比赛dataset\mosi_feature\\train\\text\\roberta_large\\roberta_large-4-FRA',video_root='D:\比赛dataset\mosi_feature\openface\downsample_fabnet',data_type='train',dataset='mosi')
    # train_loader = DataLoader(train_dataset,batch_size=32,num_workers=0,pin_memory=True,collate_fn=collate_func)
    # for a,a_len,t,t_len,v,v_len,val in train_loader:
    #     print(a_len)
    #     print(a.shape)
    #     print(t.shape)
    #     print(val)
    # train_unlabel_dataset = MERDataset(audio_root='D:\比赛dataset\AMI\hubert_audio\hubert-FRA',
    #                                    text_root='D:\比赛dataset\AMI\\text\\roberta\\roberta-4-FRA',
    #                                    video_root='D:\比赛dataset\AMI\downsample_fabnet', data_type='train',
    #                                    dataset='ami', semi_num=642)
    import datetime
    print(str(datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')))
