import numpy as np
from torch import nn
import torch.nn.functional as F
from cross_transformer import TransformerEncoder
import torch
from timm.models.layers import trunc_normal_
from transformers import AutoModel
from clusterv2 import merge

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.05)

class gate(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.A_affline = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_affline = nn.Linear(embed_dim, embed_dim, bias=True)
    def forward(self,intra_f,inter_f):
        weight_G = torch.sigmoid(self.A_affline(inter_f) + self.V_affline(intra_f))
        combined_x = inter_f * weight_G + intra_f * (1 - weight_G)
        return combined_x
    
class CMFM(nn.Module):
    def __init__(self, params, args):
        super().__init__()
        self.tp = params
        self.d_l = self.tp.get('hidden_dim')
        self.args = args
        self.sharing_num = self.tp['sharing_num']
        if args.inter != 'global':
            self.cmfm1 = self.get_network(layers=1, num_heads=self.tp['t_num_heads'],attn_dropout=self.tp['t_attn_dropout'],
                                        relu_dropout=self.tp['t_relu_dropout'],
                                        res_dropout=self.tp['t_res_dropout'],inter=False)
            self.cmfm2 = self.get_network(layers=1, num_heads=self.tp['t_num_heads'],attn_dropout=self.tp['t_attn_dropout'],
                                        relu_dropout=self.tp['t_relu_dropout'],
                                        res_dropout=self.tp['t_res_dropout'],inter=True)
            self.cmfm_intra = nn.ModuleList([self.cmfm1 for _ in range(self.sharing_num)])
            self.cmfm_inter = nn.ModuleList([self.cmfm2 for _ in range(self.sharing_num)])
            self.gate1 = gate(self.d_l)
            self.gate2 = gate(self.d_l)
        else:
            self.cmfm1 = self.get_network(layers=1, num_heads=self.tp['t_num_heads'],attn_dropout=self.tp['t_attn_dropout'],
                                        relu_dropout=self.tp['t_relu_dropout'],
                                        res_dropout=self.tp['t_res_dropout'],inter='global')
            self.cmfm = nn.ModuleList([self.cmfm1 for _ in range(self.sharing_num)])
            

        self.token_type_embeddings = nn.Embedding(3, self.d_l)
        self.token_type_embeddings.apply(init_weights)

        if self.args.dataset == 'iemocap':
            self.pos_embed = nn.Parameter(torch.zeros(1, 292, self.d_l))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, 252, self.d_l))
        trunc_normal_(self.pos_embed, std=0.02)
        self.val_token = nn.Parameter(torch.zeros(1, 1, self.d_l))
        trunc_normal_(self.val_token, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_l))
        trunc_normal_(self.cls_token, std=0.02)

        if self.args.dataset == 'iemocap':
            self.proj_1vv = nn.Conv1d(64, self.d_l, stride = 1, kernel_size=3, padding=1, bias=False)
            self.proj_1tt = nn.Conv1d(1024, self.d_l, stride = 1, kernel_size=3, padding=1, bias=False)
            self.proj_1aa = nn.Conv1d(64, self.d_l, stride = 1, kernel_size=3, padding=1, bias=False)
            self.proj_2aa = nn.Conv1d(self.d_l, self.d_l, stride = 1, kernel_size=3, padding=1, bias=False)

        else:

            self.proj_1vv = nn.Conv1d(256, self.d_l, stride = 1, kernel_size=3, padding=1, bias=False)

            if self.args.use_bert:
                self.proj_1tt = nn.Conv1d(768, self.d_l, stride = 1, kernel_size=3, padding=1, bias=False)
            else: # bert_large and roberta_large are the same
                self.proj_1tt = nn.Conv1d(1024, self.d_l, stride = 1, kernel_size=3, padding=1, bias=False)

            self.proj_1aa = nn.Conv1d(1024, self.d_l, stride = 2, kernel_size=4, padding=1, bias=False)
            self.proj_2aa = nn.Conv1d(self.d_l, self.d_l, stride = 4, kernel_size=4, padding=1, bias=False)

        

        self.all_layers = []
        layers = self.tp['layers']
        layers = list(map(lambda x: int(x), layers.split(',')))
        input_dim = self.tp['hidden_dim']
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            input_dim = layers[i]

        if self.args.dataset == 'iemocap':
            output_dim1, output_dim2 = 1, 6
        else:
            output_dim1, output_dim2 = 1, 7
        self.module = nn.Sequential(*self.all_layers)
        self.fc_out1 = nn.Linear(layers[-1], output_dim1)
        self.fc_out2 = nn.Linear(layers[-1], output_dim2)


    def get_network(self, layers=None, num_heads=None,attn_dropout = None,relu_dropout=None, res_dropout=None, inter=None):
        return TransformerEncoder(
                embed_dim=self.d_l,
                embed_dropout=0.2,
                num_heads=num_heads,
                layers=layers,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=False,
                inter=inter
            )
    def forward(
            self,
            visual,
            text,
            audio,
            training = True
    ):

        if self.args.dataset == 'iemocap':
            text = F.dropout(text, p=0.5, training=training)
            audio = F.dropout(audio, p=0.5, training=training)
            visual = F.dropout(visual, p=0.5, training=training) # bsz,len,dim
        else:
            text = F.dropout(text, p=0.5, training=training)
            audio = F.dropout(audio, p=0.5, training=training)
            visual = F.dropout(visual, p=0.5, training=training) # bsz,len,dim
        
        

        text = text.permute(0,2,1)
        audio = audio.permute(0,2,1)
        visual = visual.permute(0,2,1) 

        text = self.proj_1tt(text).permute(2,0,1) # bsz,dim,len-->len,bsz,dim
        audio = self.proj_2aa(self.proj_1aa(audio)).permute(2,0,1)
        visual = self.proj_1vv(visual).permute(2,0,1)

        bsz, t_len, a_len, v_len = text.shape[1],text.shape[0],audio.shape[0],visual.shape[0]
        audio, text, visual = (
            audio + self.token_type_embeddings(torch.zeros(bsz, a_len).long().cuda()).permute(1, 0, 2),
            text + self.token_type_embeddings(torch.ones(bsz, t_len).long().cuda()).permute(1, 0, 2),  # bsz,t,dim
            visual + self.token_type_embeddings(2 * torch.ones(bsz, v_len).long().cuda()).permute(1, 0, 2))  # bsz,t,dim

        val_token = self.val_token.expand(bsz, -1, -1)
        cls_token = self.cls_token.expand(bsz, -1, -1)
        atv = torch.cat([val_token.transpose(0, 1), cls_token.transpose(0, 1), audio, text, visual],dim=0)  # 250,bsz,dim
        if self.args.inter != 'global':
            atv_1 = atv + self.pos_embed.expand(bsz, -1, -1).transpose(0, 1)
            atv_2 = atv + self.pos_embed.expand(bsz, -1, -1).transpose(0, 1)
            for layer in self.cmfm_intra:  
                atv_1,_ = layer(atv_1,atv_1,atv_1)
            for layer in self.cmfm_inter:
                atv_2,_ = layer(atv_2,atv_2,atv_2)
            atv_intra,atv_inter = atv_1,atv_2

            intra_val,intra_emo = atv_intra[0],atv_intra[1]
            inter_val,inter_emo = atv_inter[0],atv_inter[1]

            f_val = self.gate1(intra_val,inter_val)
            f_emo = self.gate2(intra_emo,inter_emo)
        else:
            for layer in self.cmfm:  
                atv,_ = layer(atv,atv,atv)
            
            f_val,f_emo = atv[0],atv[1]

        
        p_val = self.fc_out1(self.module(f_val))
        p_emo = self.fc_out2(self.module(f_emo))

        return p_val,p_emo
