from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler
import random
from utils import *
from sklearn.metrics import classification_report
from multi_semi_pseudo_together import *
from tqdm import tqdm
from global_configs import DEVICE
import torch.optim as optim
from model import CMFM
from metric import *
import datetime
from logger import getLogger
import argparse
import time 

class Solver(object):
    def __init__(self, model: nn.Module, train_loader, dev_loader, test_loader, test_ami_loader, optimizer, tune_params, args):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.test_ami_loader = test_ami_loader
        self.optimizer = optimizer
        self.tp = tune_params
        self.args = args
        self.reg_func = MSELoss()
        self.cls_func = CELoss()
    def train_epoch(self):
        self.model.train()
        tr_loss = 0
        nb_tr_steps = 0

        for step,batch in enumerate(tqdm(self.train_loader)):
            batch_lab = tuple(t.to(DEVICE) for t in batch)
            audio, audio_len, text, text_len, visual, visual_len, val, emo, idx = batch_lab

            p_val, p_emo = self.model(visual,text,audio)

            loss = self.tp.get('val_r') * self.reg_func(p_val[idx >= 0],val[idx >= 0]) +  self.tp.get('cls_r') * self.cls_func(p_emo[idx >= 0],emo[idx >= 0]) + self.tp.get('cls_semi_r') * self.cls_func(p_emo[idx < 0],emo[idx < 0])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            nb_tr_steps += 1
            tr_loss += loss.item()
            
        return tr_loss / nb_tr_steps

    def test(self, test = False, test_ami = False):
        loader = self.test_loader if test else self.dev_loader
        if test_ami:
            self.model.load_state_dict(torch.load(os.path.join(f'./model_{self.args.dataset}', 'model.pt')))
            loader = self.test_ami_loader
        self.model.eval()
        all_val,all_p_val,all_emo,all_p_emo,all_c_emo = [],[],[],[],[]
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = tuple(t.to(DEVICE) for t in batch)
                audio,audio_len,text,text_len,visual,visual_len,val,emo,_ = batch
                p_val,p_emo = self.model(visual,text,audio,training = False)

                p_val = p_val.detach().cpu().numpy().flatten()
                con_emo = np.max(p_emo.detach().cpu().numpy(), axis=1).flatten()
                p_emo = np.argmax(p_emo.detach().cpu().numpy(), axis=1).flatten() # bsz,7 --> bsz 按行
                val = val.detach().cpu().numpy().flatten()
                emo = emo.detach().cpu().numpy().flatten()


                all_p_val.extend(p_val)
                all_val.extend(val)

                all_p_emo.extend(p_emo)
                all_emo.extend(emo)

                all_c_emo.extend(con_emo)


            all_p_val = np.array(all_p_val)
            all_val = np.array(all_val)
            all_p_emo = np.array(all_p_emo)
            all_emo = np.array(all_emo)
            all_c_emo = np.array(all_c_emo)

        # emo2id6 = {'neu':0.,'fru':1.,'ang':2., 'hap':3., 'exc':4., 'sad':5.}
        if test_ami:
            return all_p_val, all_p_emo, all_c_emo
        elif self.args.dataset != 'iemocap':
            result = eval_regression(all_p_val,all_val)
        else:
            class_name = ['neu','fru','ang', 'hap', 'exc', 'sad']
            result = classification_report(all_emo, all_p_emo, target_names=class_name, output_dict=True) # true,pred
        return result



    # def calculate_result(self):
    #     preds_score, ts_pred, as_pred, labels_score = self.test(sel)
    #     ## evaluate on dimensional labels
    #     val_preds = preds_score
    #     val_labels = labels_score
    #     save_results = eval_regression(val_preds,val_labels)
    #     resultst = eval_regression(ts_pred,val_labels)
    #     resultsa = eval_regression(as_pred,val_labels)
    #     save_results['final_score'] = overall_metric(save_results['Non0_F1_score'],save_results['MAE'])

    #     return save_results,resultst,resultsa


    def train(self, retrain=False):
        train_losses, result_dev, result_test = list(),list(),list()

        best_score = 1e-8 if self.args.dataset == 'iemocap' else 1e+8

        

        for epoch_i in range(self.tp.get('epoch')):
            train_loss = self.train_epoch()

            valid_results = self.test()

            test_results = self.test(test=True)

            train_losses.append(train_loss)
            result_dev.append(valid_results)
            result_test.append(test_results)



            '''打印训练集损失'''
            print(
                "epoch:{}, train_loss:{:.4f}".format(
                    epoch_i, train_loss,
                )
            )

            if self.args.dataset != 'iemocap':
                overall = valid_results['MAE']
                print("验证集指标:   current Has0_acc_2:{:.4f}, current Has0_F1_score:{:.4f}, current Non0_acc_2:{:.4f}, current Non0_F1_score:{:.4f}, current Mult_acc_5:{:.4f}, current Mult_acc_7:{:.4f}, current MAE:{:.4f}, current Corr:{:.4f}".format(
                        valid_results['Has0_acc_2'], valid_results['Has0_F1_score'], valid_results['Non0_acc_2'],
                        valid_results['Non0_F1_score'],valid_results["Mult_acc_5"],valid_results["Mult_acc_7"],valid_results["MAE"],valid_results["Corr"]
                    )
                )
            
                print("测试集指标:  current Has0_acc_2:{:.4f}, current Has0_F1_score:{:.4f}, current Non0_acc_2:{:.4f}, current Non0_F1_score:{:.4f}, current Mult_acc_5:{:.4f}, current Mult_acc_7:{:.4f}, current MAE:{:.4f},, current Corr:{:.4f}".format(
                        test_results['Has0_acc_2'], test_results['Has0_F1_score'], test_results['Non0_acc_2'],
                        test_results['Non0_F1_score'],test_results["Mult_acc_5"],test_results["Mult_acc_7"],test_results["MAE"],test_results["Corr"],
                    )
                )
                if overall < best_score:
                    best_score = overall
                    best_epoch = epoch_i
                    results = test_results
                    if not os.path.exists(f'./model_{self.args.dataset}'):
                        os.makedirs(f'./model_{self.args.dataset}')
                    if not retrain:
                        torch.save(self.model.state_dict(), os.path.join(f'./model_{self.args.dataset}', 'model.pt'))
                print(self.tp)
                print("最佳测试集指标:  Has0_acc_2:{:.4f}, current Has0_F1_score:{:.4f}, current Non0_acc_2:{:.4f}, current Non0_F1_score:{:.4f}, current Mult_acc_5:{:.4f}, current Mult_acc_7:{:.4f}, current MAE:{:.4f},, current Corr:{:.4f}".format(
                        results['Has0_acc_2'], results['Has0_F1_score'], results['Non0_acc_2'],
                        results['Non0_F1_score'],results["Mult_acc_5"],results["Mult_acc_7"],results["MAE"],results["Corr"],
                    )
                )
                print('patience',epoch_i-best_epoch)
                # if epoch_i - best_epoch >= 5:
                #     break
                print('best_epoch',best_epoch)
            else:
                overall = valid_results['accuracy']
                print(self.tp)

                print("当前验证集指标:  acc:{:.4f}, macro_avg_prec:{:.4f}, macro_avg_recall:{:.4f}, macro_avg_f1:{:.4f}, weighted_avg_prec:{:.4f}, weighted_avg_recall:{:.4f}, weighted_avg_f1:{:.4f}".format(
                            valid_results['accuracy'], valid_results['macro avg']['precision'], valid_results['macro avg']['recall'], valid_results['macro avg']['f1-score'], valid_results['weighted avg']['precision'], valid_results['weighted avg']['recall'], valid_results['weighted avg']['f1-score'], 
                ))
                print("当前测试集指标:  acc:{:.4f}, macro_avg_prec:{:.4f}, macro_avg_recall:{:.4f}, macro_avg_f1:{:.4f}, weighted_avg_prec:{:.4f}, weighted_avg_recall:{:.4f}, weighted_avg_f1:{:.4f}".format(
                            test_results['accuracy'], test_results['macro avg']['precision'], test_results['macro avg']['recall'], test_results['macro avg']['f1-score'], test_results['weighted avg']['precision'], test_results['weighted avg']['recall'], test_results['weighted avg']['f1-score'], 
                ))
                if overall > best_score:
                    best_score = overall
                    best_epoch = epoch_i
                    results = test_results
                print("最佳测试集指标:  acc:{:.4f}, macro_avg_prec:{:.4f}, macro_avg_recall:{:.4f}, macro_avg_f1:{:.4f}, weighted_avg_prec:{:.4f}, weighted_avg_recall:{:.4f}, weighted_avg_f1:{:.4f}".format(
                            results['accuracy'], results['macro avg']['precision'], results['macro avg']['recall'], results['macro avg']['f1-score'], results['weighted avg']['precision'], results['weighted avg']['recall'], results['weighted avg']['f1-score'], 
                ))

                print('patience',epoch_i-best_epoch)
                # if epoch_i - best_epoch >= 5:
                #     break
                print('best_epoch',best_epoch)

        return train_losses, results

def collate_func(batch_dic):

    fea_a_batch, fea_t_batch, fea_v_batch = [],[],[]
    fea_a_len, fea_t_len, fea_v_len = [], [], []
    label_batch = []
    emo_batch = []
    idx = []
    dataset = 'mosei'
    (t_dim,v_dim,a_dim) = (1024,64,64) if dataset == 'iemocap' else (1024,256,1024)
    (t_len,v_len,a_len) = (100,32,158) if dataset == 'iemocap' else (100,50,800)
    for i in range(len(batch_dic)):  # 分别提取批样本中的feature、label、id、length信息
        fea_a_len.append(act_len(batch_dic[i][0],f_dim=a_dim,max_len=800))
        fea_v_len.append(act_len(batch_dic[i][2],f_dim=v_dim,max_len=50)) 
        fea_t_len.append(act_len(batch_dic[i][1],f_dim=t_dim,max_len=100))
        fea_a_batch.append(pad_tensor(batch_dic[i][0], pad=a_len, dim=0, f_dim=a_dim))
        fea_t_batch.append(pad_tensor(batch_dic[i][1], pad=t_len, dim=0, f_dim=t_dim))
        fea_v_batch.append(pad_tensor(batch_dic[i][2], pad=v_len, dim=0, f_dim=v_dim))
        label_batch.append(batch_dic[i][-3])
        emo_batch.append(batch_dic[i][-2])
        idx.append(batch_dic[i][-1])
    fea_a_batch = torch.stack(fea_a_batch,dim=0)
    fea_t_batch = torch.stack(fea_t_batch,dim=0)
    fea_v_batch = torch.stack(fea_v_batch,dim=0)
    label_batch = torch.FloatTensor(label_batch)
    emo_batch = torch.FloatTensor(emo_batch)
    idx = torch.FloatTensor(idx)
    fea_a_len = torch.FloatTensor(fea_a_len)
    fea_t_len = torch.FloatTensor(fea_t_len)
    fea_v_len = torch.FloatTensor(fea_v_len)
    return fea_a_batch,fea_a_len,fea_t_batch,fea_t_len,fea_v_batch,fea_v_len,label_batch,emo_batch,idx


def prep_for_training(tune_params,args):

    model = CMFM(tune_params,args)
    total_para = 0
    for param in model.parameters():
        total_para += np.prod(param.size())
    print('total parameter for the model: ', total_para)
    model.to(DEVICE)
    return model


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """

    print("Seed: {}".format(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_params(seed):
    tune_params = {'epoch': 20, 'train_bsz': 32, 'valid_bsz': 32, 'test_bsz': 32, 'lr': 1e-4, 'wd': 1e-4, 'cls_r': 0.4, 'val_r': 0.6
                   }
    random.seed(seed)
    tune_params['hidden_dim'] = random.choice([128,256,512])
    tune_params['cls_r'] = random.choice([0.5,0.6,0.7,0.8,0.9,1])
    tune_params['val_r'] = 1 - tune_params['cls_r']
    if tune_params['hidden_dim'] == 128:
        tune_params['layers']='128,64'
    elif tune_params['hidden_dim'] == 256:
        tune_params['layers'] ='256,128'
    else:
        tune_params['layers'] = '512,256'
    tune_params['sharing_num'] = random.choice([2,4])
    tune_params['t_encoder_layer'] = tune_params['a_encoder_layer'] = tune_params['v_encoder_layer'] = random.choice([1,2,4])
    tune_params['t_num_heads'] = tune_params['a_num_heads'] = tune_params['v_num_heads'] = random.choice([1, 2, 4, 8])
    tune_params['t_attn_dropout'] = tune_params['v_attn_dropout'] = tune_params['a_attn_dropout'] = tune_params['t_relu_dropout'] = tune_params['a_relu_dropout'] = tune_params['v_relu_dropout'] = tune_params['t_res_dropout'] = tune_params['a_res_dropout'] = tune_params['v_res_dropout'] = random.choice([0.1, 0.2, 0.3])

    return tune_params


def main(args):

    if args.dataset == 'mosi':
        audio_root = '/root/autodl-tmp/ljh/experiment/data/mosi/together/audio'
        if args.use_bert:
            text_root = '/root/autodl-tmp/ljh/experiment/data/mosi/together/text_bert/bert-base-uncased-4-FRA'
        elif args.use_bert_large:
            text_root = '/root/autodl-tmp/ljh/experiment/data/mosi/together/text_bert_large/bert-large-uncased-4-FRA'
        else:
            text_root = '/root/autodl-tmp/ljh/experiment/data/mosi/together/text'
        video_root = '/root/autodl-tmp/ljh/experiment/data/mosi/together/visual'
        train_dataset = MERDataset(audio_root,text_root,video_root,
                                        data_type='train', dataset='mosi', lab_num=args.lab_num, semi_num=3236)
        valid_dataset = MERDataset(audio_root,text_root,video_root,
                                   data_type='valid',
                                dataset='mosi', lab_num=229)
        test_dataset = MERDataset(audio_root,text_root,video_root,
                                   data_type='test',
                                dataset='mosi', lab_num=686)
        test_ami_dataset = MERDataset(audio_root,text_root,video_root, 
                                  data_type='test',
                                dataset='ami', lab_num=3236)
        
    elif args.dataset == 'iemocap':
        audio_root = '../data/iemocap/audio'
        text_root = '../data/iemocap/text_wrong'
        video_root = '../data/iemocap/visual'
        train_dataset = MERDataset(audio_root,text_root,video_root,
                                   data_type='train', dataset='iemocap', lab_num=5237, semi_num=0)
        valid_dataset = MERDataset(audio_root,text_root,video_root, 
                                   data_type='valid',
                                dataset='iemocap', lab_num=521)
        test_dataset = MERDataset(audio_root,text_root,video_root, 
                                  data_type='test',
                                dataset='iemocap', lab_num=1622)
        
    else:
        audio_root = '/root/autodl-tmp/ljh/experiment/data/mosei/hubertlarge_with_incorrect/hubert-FRA'
        if args.use_bert:
            text_root = '/root/autodl-tmp/ljh/experiment/data/mosei/bert_base/bert-base-uncased-4-FRA'
        elif args.use_bert_large:
            text_root = '/root/autodl-tmp/ljh/experiment/data/mosei/bert_large/bert-large-uncased-4-FRA'
        else:
            text_root = '/root/autodl-tmp/ljh/experiment/data/mosei/roberta/roberta-4-FRA'
        video_root = '/root/autodl-tmp/ljh/experiment/data/mosei/down_fabnet'
        train_dataset = MERDataset(audio_root,text_root,video_root,
                                   data_type='train', dataset='mosei', lab_num=16326, semi_num=0)
        valid_dataset = MERDataset(audio_root,text_root,video_root, 
                                   data_type='valid',
                                dataset='mosei', lab_num=1871)
        test_dataset = MERDataset(audio_root,text_root,video_root, 
                                  data_type='test',
                                dataset='mosei', lab_num=4659)



    seeds = [100]
    log_path = os.path.join(f'./trainlog_{args.dataset}',time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
    logger = getLogger(log_path)
    his_params = list()
    for i in range(500):

        # tune_params = get_params(i)

        tune_params = {'epoch': 30, 'train_bsz': 32, 'valid_bsz': 32, 'test_bsz': 32, 
                       'lr': 0.0001, 'wd': 0.0001, 'cls_r': 0.4, 'val_r': 0.6, 'cls_semi_r': 0.2,
                       'hidden_dim': 128, 'layers': '128,64', 'sharing_num': 2, 
                       't_encoder_layer': 1, 'a_encoder_layer': 1, 'v_encoder_layer': 1, 
                       't_num_heads': 1, 'a_num_heads': 1, 'v_num_heads': 1, 
                       't_attn_dropout': 0.3, 'v_attn_dropout': 0.3, 'a_attn_dropout': 0.3, 't_relu_dropout': 0.3, 'a_relu_dropout': 0.3, 'v_relu_dropout': 0.3, 't_res_dropout': 0.3, 'a_res_dropout': 0.3, 'v_res_dropout': 0.3}

        
        his_params.append(tune_params)
        duplicate_flag = False
        if i>=1:
            for his_p in his_params[:-1]:
                if his_p == tune_params:
                    duplicate_flag = True
                    break
        if duplicate_flag:
            continue
        
        
        for seed in seeds:
            set_random_seed(seed)
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=tune_params['train_bsz'],shuffle=True, num_workers=0, pin_memory=True,collate_fn=collate_func)
            dev_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=tune_params['valid_bsz'],shuffle=True, num_workers=0, pin_memory=True,collate_fn=collate_func)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=tune_params['test_bsz'],shuffle=False, num_workers=0, pin_memory=True,collate_fn=collate_func)
            test_ami_loader = torch.utils.data.DataLoader(test_ami_dataset, batch_size=tune_params['test_bsz'],shuffle=False, num_workers=0, pin_memory=True,collate_fn=collate_func)

            model = prep_for_training(tune_params,args)
            optimizer = optim.Adam(model.parameters(), lr=tune_params['lr'], weight_decay=tune_params['wd'])
            solver = Solver(model, train_loader, dev_loader, test_loader, test_ami_loader, optimizer, tune_params, args)
            
            train_losses, results = solver.train()
            tune_params['lab_num'] = args.lab_num
            logger.info(tune_params)
            if args.dataset == 'iemocap':
                logger.info("acc:{:.4f}\tmacro_avg_prec:{:.4f}\tmacro_avg_recall:{:.4f}\tmacro_avg_f1:{:.4f}\tweighted_avg_prec:{:.4f}\tweighted_avg_recall:{:.4f}\tweighted_avg_f1:{:.4f}".format(
                            results['accuracy'], results['macro avg']['precision'], results['macro avg']['recall'], results['macro avg']['f1-score'], results['weighted avg']['precision'], results['weighted avg']['recall'], results['weighted avg']['f1-score'], 
                ))
            else:
                logger.info("Has0_acc_2:{:.4f}\tHas0_F1_score:{:.4f}\tNon0_acc_2:{:.4f}\tNon0_F1_score:{:.4f}\tMult_acc_5:{:.4f}\tMult_acc_7:{:.4f}\tMAE:{:.4f}\tCorr:{:.4f}\n".format(
                        results['Has0_acc_2'], results['Has0_F1_score'], results['Non0_acc_2'],
                        results['Non0_F1_score'],results["Mult_acc_5"],results["Mult_acc_7"],results["MAE"],results["Corr"],
                    ))
            if args.retrain:
                p_val, p_emo, c_emo = solver.test(test=True,test_ami=args.retrain) # bsz
                name = open_csv('./test_emo2_ami.csv')
                p_c_emo = np.column_stack((p_emo,c_emo)) # bsz,2
                top_k_indices = get_top_k_indices(p_c_emo, k = 40) # top k
                select_name = [name[i] for i in top_k_indices]
                save_csv(select_name,p_emo[top_k_indices],p_val[top_k_indices]) # ami_pesudo
                train_dataset = MERDataset(audio_root,text_root,video_root,data_type='train', dataset='mosi', lab_num=args.lab_num, semi_num=len(top_k_indices), retrain=args.retrain)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=tune_params['train_bsz'],shuffle=True, num_workers=0, pin_memory=True,collate_fn=collate_func)
                model.load_state_dict(torch.load(os.path.join(f'./model_{args.dataset}', 'model.pt')))
                solver = Solver(model, train_loader, dev_loader, test_loader, test_ami_loader, optimizer, tune_params, args)
                tune_params['epoch'] = 10
                train_losses, results = solver.train(retrain=args.retrain) # 传入retrain的原因是保证不去保存retrain后的模型
                logger.info("Has0_acc_2:{:.4f}\tHas0_F1_score:{:.4f}\tNon0_acc_2:{:.4f}\tNon0_F1_score:{:.4f}\tMult_acc_5:{:.4f}\tMult_acc_7:{:.4f}\tMAE:{:.4f}\tCorr:{:.4f}\n".format(
                        results['Has0_acc_2'], results['Has0_F1_score'], results['Non0_acc_2'],
                        results['Non0_F1_score'],results["Mult_acc_5"],results["Mult_acc_7"],results["MAE"],results["Corr"],
                    ))
            

      

    # import pickle 
    # output = open('r.pkl','wb')
    # pickle.dump(r, output)
    # output = open('tp.pkl','wb')
    # pickle.dump(tp, output)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mosi')
    parser.add_argument("--retrain", action='store_true')

    parser.add_argument("--lab_num", type=int, default=1284)
    parser.add_argument("--inter", type=str, default='inter') # if inter=global, it is tranditional attention
    parser.add_argument("--use_bert", action='store_true')
    parser.add_argument("--use_bert_large", action='store_true')

    args = parser.parse_args()
    main(args)