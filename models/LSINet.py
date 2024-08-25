__all__ = ['PatchTST_backbone']

# Cell
import time
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as f
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from sklearn.neighbors import kneighbors_graph
# Cell
import scipy


class TimeInvariant(nn.Module):
    def __init__(self, T, dropout_rate=0.1):
        super(TimeInvariant, self).__init__()
        # print(T)
        # time.sleep(500)
        self.fc = nn.Linear(T, T)
        self.relu = nn.ReLU()
        self.relu=nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(T, T)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout2(x)

        return x

class TimeUpdating(nn.Module):
    def __init__(self, T, dropout_rate=0.1):
        super(TimeUpdating, self).__init__()
        # print(T)
        # time.sleep(500)
        self.fc = nn.Linear(T, T)
        self.relu = nn.ReLU()
        self.relu=nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(T, T)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout2(x)

        return x

class PatchIntegration(nn.Module):
    def __init__(self, C, dropout_rate=0.1):
        super(PatchIntegration, self).__init__()
        self.fc1 = nn.Linear(C, C)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(C, C)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x



def sample_gumbel(shape, eps=1e-20,device=None):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10,device=None):
    sample = sample_gumbel(logits.size(), eps=eps,device=device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10,device=None):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps,device=device)
  if hard:
      shape = logits.size()
      _, k = y_soft.data.max(-1)
      y_hard = torch.zeros(*shape).to(device)
      y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
      y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
  else:
      y = y_soft
  return y

class SSConnectionLearning(nn.Module):
    def __init__(self, args,node_num,sparse_k):
        super(SSConnectionLearning, self).__init__()
        self.args=args
        self.k=sparse_k
        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot

        self.num_nodes=node_num
        off_diag = np.ones([self.num_nodes, self.num_nodes])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(args.device)
        self.rel_send = torch.FloatTensor(rel_send).to(args.device)
        self.fc_out = nn.Linear(args.d_model*2, args.d_model)
        self.fc_cat = nn.Linear(args.d_model, 2)

    def forward(self,feats):

        receivers = torch.matmul(self.rel_rec, feats) #(42849, 207)  torch.Size([207, 100])
        senders = torch.matmul(self.rel_send, feats)
        x = torch.cat([senders, receivers], dim=-1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)
        connection_matrix = gumbel_softmax(x, temperature=0.5, hard=True,device=self.args.device)
        connection_matrix = connection_matrix[:,:, 1].clone().reshape(self.args.n_heads,self.num_nodes, -1)
        connection_prob=x.softmax(-1)[:,:, 1].clone()
        values, indices = torch.topk(connection_prob, self.k,dim=1)
        sparse_labels = torch.zeros_like(connection_prob, dtype=torch.float32)
        sparse_labels.scatter_(1, indices, 1.0).to(self.args.device)
        return connection_matrix,connection_prob,sparse_labels

class MultiheadSIM(nn.Module):
    def __init__(self, PatchNum, C, dropout, args=None):
        super(MultiheadSIM, self).__init__()
        self.args = args
        dropout=0.3
        self.mlp_feat = PatchIntegration(C, dropout)
        self.mlp_feat_rev = PatchIntegration(C, dropout)

        self.PatchNum=PatchNum
        self.patch_idx = torch.arange(self.PatchNum).to(self.args.device)
        self.patch_idx_embed = torch.arange(args.d_model).to(self.args.device)

        ##initialized memory M for learning patch embeddings
        self.memory = nn.Embedding(self.PatchNum, self.args.d_model*self.args.n_heads)
        d_ff=self.args.d_model//2
        d_ff2=self.args.d_model//4
        d_ff3 = self.args.d_model // 8
        self.mlp_memory = nn.Sequential(nn.Linear(self.args.d_model, d_ff),
                                  nn.Linear(d_ff, d_ff2),
                                nn.Linear(d_ff2, self.args.d_model))

        self.sparse_k = int(self.PatchNum*self.PatchNum*args.sparse_rate) #0.01 0.05 0.15   0.95 0.3 0.2
        self.connection_pre=SSConnectionLearning(args,PatchNum,self.sparse_k )
        # self.connection_pre_embed = SSConnectionLearning(args, args.d_model, self.sparse_k)
        # self.knn_metric = 'cosine'
        self.bce_loss=torch.nn.BCELoss()
        self.self_attn = _MultiheadAttention(args.d_model, args.n_heads_sam, args.d_model, args.d_model, attn_dropout=0,
                                             proj_dropout=0, res_attention=False)
        self.d_v=args.d_v
        self.timeinvariant = TimeInvariant(self.PatchNum, dropout)
        self.timeupdating = TimeUpdating(self.PatchNum, dropout)
        # proj_dropout = dropout
        self.W_V = nn.Sequential(nn.Linear(args.d_model, self.d_v * args.n_heads,bias=True), nn.Dropout(dropout))
        self.linear_align = nn.Sequential(nn.Linear(args.n_heads * self.d_v, args.d_model), nn.Dropout(dropout))

    def forward(self, x_or):
        bs = x_or.reshape(-1, self.args.enc_in, self.PatchNum, self.args.d_model).shape[0]
        res_x = x_or
        if self.args.Self_Attention_Mechanism:
            out, _ = self.self_attn(x_or)
            x_p = out.transpose(1, 2)
            x = x_p.transpose(1, 2)
            loss_sparse, sparse_rate=0,torch.tensor(0)

        elif self.args.MSIM:
            x_or_t = self.timeinvariant(x_or.permute(0, 2,1)).permute(0, 2,1)

            ###Sparse Time Patch Propagation
            learned_patch_embedding = self.memory(self.patch_idx)
            learned_patch_embedding = learned_patch_embedding.view(self.args.n_heads, self.PatchNum, -1)
            learned_patch_embedding = self.mlp_memory(learned_patch_embedding)
            connection_matrix, connection_prob, sparse_labels = self.connection_pre(learned_patch_embedding)
            non_zero_count = torch.count_nonzero(connection_matrix, dim=(-2, -1))
            sparse_rate = non_zero_count / (self.PatchNum * self.PatchNum)
            sparse_rate = torch.mean(sparse_rate, dim=0)
            loss_sparse = self.bce_loss(connection_prob, sparse_labels)
            v = self.W_V(x_or_t).view(x_or_t.shape[0], -1, self.args.n_heads, self.d_v).transpose(1, 2)
            x_p=torch.matmul(connection_matrix, v) #torch.Size([896, 16, 64, 8])
            x_p = x_p.transpose(1, 2).contiguous().view( -1,self.PatchNum, self.args.n_heads * self.d_v)

            #Align dimension for residual connections
            x_p=self.linear_align(x_p)
            x = x_p + res_x

            #Time Updating for updating the patch information afther information propagation
            x = self.timeupdating(x.transpose(1, 2)).transpose(1, 2)

        else:
            x_revrse, x_revrse_2, x_reverse_allrand = 0, 0, 0
            x_p = x_or
            x=x_p
            sparse_rate = torch.tensor(0)
            loss_sparse = 0

        res_x = x

        x = self.mlp_feat(x) + res_x + x_or

        return x,loss_sparse,sparse_rate






class Model(nn.Module):
    def __init__(self, configs=None):

        super().__init__()
        self.configs = configs
        d_model = configs.d_model
        head_dropout = configs.head_dropout
        individual = configs.individual
        affine = configs.affine
        subtract_last = configs.subtract_last
        # RevIn
        self.revin = True
        self.patch_len= configs.patch_len
        self.stride=configs.stride

        if self.revin: self.revin_layer = RevIN(self.configs.c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        padding_patch = self.configs.padding_patch
        self.scale_all = [1, 2, 4, 8]
        self.configs.scale_all = self.scale_all
        self.configs.patch_num_all = {}
        self.padding_patch_layer_all = {}
        self.configs.patch_num=(int((self.configs.context_window - self.configs.patch_len) / (self.configs.stride) + 1))
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.configs.stride))
            self.configs.patch_num+=1
        self.head_nf = d_model * self.configs.patch_num
        # Backbone
        self.sti_module = STIModule(d_model=d_model,configs=self.configs)

        # Head
        self.n_vars = self.configs.c_in
        self.individual = individual
        self.head_all = []

        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, self.configs.target_window,
                                     head_dropout=head_dropout, args=configs)
        self.revin = True


    def forward(self, z):  # z: [bs x nvars x seq_len]
        z = z.permute(0, 2, 1)
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)
        # do patching
        if self.configs.padding_patch == 'end':
            z_pad = self.padding_patch_layer(z)
        z_patch = z_pad.unfold(dimension=-1, size=self.patch_len, step=self.stride).permute(0, 1, 3, 2)
        z,loss_inf_all = self.sti_module(z_patch)  # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)  # z: [bs x nvars x target_window]
        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z.permute(0, 2, 1),loss_inf_all


class STIModule(nn.Module):  # i means channel-independent
    def __init__(self,d_model,pe='zeros',learn_pe=True,configs=None):
        super().__init__()
        self.configs = configs
        self.W_P_scale_all = nn.ModuleList()
        self.W_Pos_scale_all = []
        self.patch_align = nn.ModuleList()
        self.W_P_z=nn.Linear(self.configs.patch_len, self.configs.d_model)

        self.W_pos = positional_encoding(pe, learn_pe, self.configs.patch_num, d_model)
        self.encoders = nn.ModuleList()
        for i in range(self.configs.n_msim):
            self.encoders.append(MultiheadSIM(self.configs.patch_num, d_model, 0.1, args=configs))
        if self.configs.resdual_block:
            self.encoders_residual_block = nn.ModuleList()
            for i in range(self.configs.n_msim_residual):
                self.encoders_residual_block.append(MultiheadSIM(self.configs.patch_num, d_model, 0.1, args=configs))

        self.dropout = nn.Dropout(0.1)
    def forward(self, z_patch) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]
        x_scale_all = {}
        n_vars = self.configs.enc_in
        loss_inf_all={}

        z_patch_projected=self.W_P_z(z_patch.permute(0, 1, 3, 2))
        z_patch_projected=torch.reshape(z_patch_projected,
                                           (z_patch_projected.shape[0] * z_patch_projected.shape[1], z_patch_projected.shape[2], z_patch_projected.shape[3]))
        z_patch_projected=self.dropout(z_patch_projected + self.W_pos)
        z_all = {}
        for i in range(self.configs.n_msim):
            if i==0:
                z_all[1],loss_sparse,sparse_rate = self.encoders[i](z_patch_projected)
            else:
                z_all[1], loss_sparse, sparse_rate = self.encoders[i](z_all[1])

        if self.configs.resdual_block:
            z_residual=z_all[1]
            for i in range(self.configs.n_msim_residual):
                z_all[1], loss_sparse, sparse_rate = self.encoders_residual_block[i](z_all[1])
            z_all[1]=z_all[1]+z_residual
        loss_inf_all['loss_sparse']=loss_sparse
        loss_inf_all['sparse_rate'] = sparse_rate
        z = z_all[1]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)
        return z,loss_inf_all


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0, args=None):
        super().__init__()
        self.args = args
        self.individual = args.var_individual
        self.n_vars = n_vars
        self.sp_patch_num = 4
        self.var_decomp = args.var_decomp
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        elif self.var_decomp:
            self.var_sp_num = args.var_sp_num  # 11

            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.var_sp_num):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            # if args.linear_mlp:
            #     self.linear = nn.Sequential(nn.Linear(nf, nf // 2), nn.Linear(nf // 2, target_window))
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        elif self.var_decomp:
            x_out = []
            output_chunks = torch.chunk(x, self.var_sp_num, dim=1)

            for i in range(len(output_chunks)):
                z = self.flattens[i](output_chunks[i])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.cat(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)

        return x


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights

