"""An implementation of DRMMTKS Model."""
# for one query to one POI similarity score calculating
import typing
from matplotlib.pyplot import text
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# from transformers import BertModel
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward, CrossAttention, DotAttention, PreNorm2
import numpy as np

class Attention2(nn.Module):
    """
    Attention module.

    :param input_size: Size of input.
    :param mask: An integer to mask the invalid values. Defaults to 0.

    Examples:
        >>> import torch
        >>> attention = Attention(input_size=10)
        >>> x = torch.randn(4, 5, 10)
        >>> x.shape
        torch.Size([4, 5, 10])
        >>> x_mask = torch.BoolTensor(4, 5)
        >>> attention(x, x_mask).shape
        torch.Size([4, 5])

    """

    def __init__(self, input_size: int = 100):
        """Attention constructor."""
        super().__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)

    def forward(self, x, x_mask):
        """Perform attention on the input."""
        x = self.linear(x).squeeze(dim=-1)
        x = x.masked_fill(x_mask, -float('inf'))
        return F.softmax(x, dim=-1)



class BidirectionalAttention2(nn.Module):
    """Computing the soft attention between two sequence."""

    def __init__(self):
        """Init."""
        super().__init__()

    def forward(self, v1, v1_mask, v2, v2_mask):
        """Forward."""
        similarity_matrix = v1.bmm(v2.transpose(2, 1).contiguous())
        if v1_mask is not None:
            similarity_matrix = similarity_matrix.masked_fill(
                v1_mask.unsqueeze(2), -1e-7)

        v2_v1_attn = F.softmax(similarity_matrix, dim=1)

        if v2_mask is not None:
            similarity_matrix = similarity_matrix.masked_fill(
                v2_mask.unsqueeze(1), -1e-7)
        v1_v2_attn = F.softmax(similarity_matrix, dim=2)

        attended_v1 = v1_v2_attn.bmm(v2)
        attended_v2 = v2_v1_attn.transpose(1, 2).bmm(v1)

        if v1_mask is not None:
            attended_v1.masked_fill_(v1_mask.unsqueeze(2), 0)
        if v2_mask is not None:
            attended_v2.masked_fill_(v2_mask.unsqueeze(2), 0)

        return attended_v1, attended_v2



class PositionwiseFeedForward2(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MultiScaleTransformerEncoder(nn.Module):

    def __init__(self, small_dim = 96, small_depth = 3, small_heads =3, small_dim_head = 32, small_mlp_dim = 384,
                 large_dim = 192, large_depth = 3, large_heads = 3, large_dim_head = 64, large_mlp_dim = 768,
                 cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0.):
        super().__init__()
        self.transformer_enc_small = Transformer(small_dim, small_depth, small_heads, small_dim_head, small_mlp_dim)
        self.transformer_enc_large = Transformer(large_dim, large_depth, large_heads, large_dim_head, large_mlp_dim)

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(large_dim, heads = cross_attn_heads, dim_head = large_dim_head, dropout = dropout)),
                nn.Linear(large_dim, small_dim),
                nn.Linear(small_dim, large_dim),
                PreNorm(small_dim, CrossAttention(small_dim, heads = cross_attn_heads, dim_head = small_dim_head, dropout = dropout)),
            ]))

    def forward(self, xs, xl):

        xs = self.transformer_enc_small(xs)
        xl = self.transformer_enc_large(xl)

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch

            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        return xs, xl

class DotTransformerEncoder(nn.Module):
    
    def __init__(self, small_dim = 96, small_depth = 3, small_heads =3, small_dim_head = 32, small_mlp_dim = 384,
                 large_dim = 192, large_depth = 3, large_heads = 3, large_dim_head = 64, large_mlp_dim = 768,
                 dot_attn_depth = 1, dot_attn_heads = 3, dropout = 0.):
        super().__init__()
        self.transformer_enc_small = Transformer(small_dim, small_depth, small_heads, small_dim_head, small_mlp_dim)
        self.transformer_enc_large = Transformer(large_dim, large_depth, large_heads, large_dim_head, large_mlp_dim)

        self.dot_attn_layers = nn.ModuleList([])
        self.norm1 = nn.LayerNorm(small_dim)
        self.norm2 = nn.LayerNorm(large_dim)
        for _ in range(dot_attn_depth):
            self.dot_attn_layers.append(nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm2(large_dim, DotAttention(large_dim, heads = dot_attn_heads, dim_head = large_dim_head, dropout = dropout)),
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm2(large_dim, DotAttention(large_dim, heads = dot_attn_heads, dim_head = large_dim_head, dropout = dropout))
            ]))

    def forward(self, xs, xl):

        xs = self.transformer_enc_small(xs)
        xl = self.transformer_enc_large(xl)


        for f_sl, g_ls, dot_attn_s, f_ls, g_sl, dot_attn_l in self.dot_attn_layers:

            # Cross Attn for left to right
            cal_q = f_sl(xs)
            cal_out = dot_attn_s(cal_q, xl) + cal_q
            cal_out = g_sl(cal_out)
            xs = cal_out

            # Cross Attn for left to right
            cal_q = f_ls(xl)
            cal_out = dot_attn_l(cal_q, xs) + cal_q
            cal_out = g_ls(cal_out)
            xl = cal_out

        return xs, xl


class QPL(nn.Module):
    
    def __init__(self, model_params, device, small_depth = 3, large_depth = 3, multi_scale_enc_depth=2,
     cross_attn_depth = 2, heads = 4, pool = 'cls', dropout = 0., emb_dropout = 0., scale_dim = 4):
        super(QPL, self).__init__()
        self._params = model_params
        self.device = device

        self._params['embedding_input_dim'] = (
                self._params['embedding'].shape[0]
            )
        self._params['embedding_output_dim'] = (
            self._params['embedding'].shape[1]
        )
        self.embedding  = nn.Embedding.from_pretrained(
            embeddings=torch.Tensor(self._params['embedding']),
            freeze=self._params['embedding_freeze'],
            padding_idx=self._params['padding_idx']
        )

        self.l_proj = nn.Linear(self._params['hidden_size'], self._params['d_emb'])
        self.r_proj = nn.Linear(self._params['hidden_size'], self._params['d_emb'])

        self.pos_embedding_query = nn.Parameter(torch.randn(1, self._params['fix_left_length']+1, self._params['d_emb']))
        self.cls_token_query = nn.Parameter(torch.randn(1, 1, self._params['d_emb']))
        self.dropout_query = nn.Dropout(emb_dropout)

        self.pos_embedding_doc = nn.Parameter(torch.randn(1, self._params['fix_right_length']+1, self._params['d_emb']))
        self.cls_token_doc = nn.Parameter(torch.randn(1, 1, self._params['d_emb']))
        self.dropout_doc = nn.Dropout(emb_dropout)
        
        self.multi_scale_transformers1 = nn.ModuleList([])
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers1.append(MultiScaleTransformerEncoder(small_dim=self._params['d_emb'], small_depth=small_depth,
                                                                              small_heads=heads, small_dim_head=self._params['d_emb']//heads,
                                                                              small_mlp_dim=self._params['d_emb']*scale_dim,
                                                                              large_dim=self._params['d_emb'], large_depth=large_depth,
                                                                              large_heads=heads, large_dim_head=self._params['d_emb']//heads,
                                                                              large_mlp_dim=self._params['d_emb']*scale_dim,
                                                                              cross_attn_depth=cross_attn_depth, cross_attn_heads=heads,
                                                                              dropout=dropout))

        self.multi_scale_transformers2 = nn.ModuleList([])

        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers2.append(MultiScaleTransformerEncoder(small_dim=self._params['d_emb'], small_depth=small_depth,
                                                                              small_heads=heads, small_dim_head=self._params['d_emb']//heads,
                                                                              small_mlp_dim=self._params['d_emb']*scale_dim,
                                                                              large_dim=self._params['d_emb'], large_depth=large_depth,
                                                                              large_heads=heads, large_dim_head=self._params['d_emb']//heads,
                                                                              large_mlp_dim=self._params['d_emb']*scale_dim,
                                                                              cross_attn_depth=cross_attn_depth, cross_attn_heads=heads,
                                                                              dropout=dropout))

        self.cls_token_uq = nn.Parameter(torch.randn(1, 1, self._params['d_emb']))
        self.cls_token_up = nn.Parameter(torch.randn(1, 1, self._params['d_emb']))

        self.multi_scale_transformers3 = nn.ModuleList([])
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers3.append(MultiScaleTransformerEncoder(small_dim=self._params['d_emb'], small_depth=small_depth,
                                                                              small_heads=heads, small_dim_head=self._params['d_emb']//heads,
                                                                              small_mlp_dim=self._params['d_emb']*scale_dim,
                                                                              large_dim=self._params['d_emb'], large_depth=large_depth,
                                                                              large_heads=heads, large_dim_head=self._params['d_emb']//heads,
                                                                              large_mlp_dim=self._params['d_emb']*scale_dim,
                                                                              cross_attn_depth=cross_attn_depth, cross_attn_heads=heads,
                                                                              dropout=dropout))

        self.cls_token_2 = nn.Parameter(torch.randn(1, 1, self._params['d_emb']))
        self.transformer_2 = Transformer(self._params['d_emb'], 5, 2, self._params['d_emb']//2, self._params['d_emb']*2)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.l_norm = nn.LayerNorm(self._params['d_emb'])
        self.r_norm = nn.LayerNorm(self._params['d_emb'])


        self.mlp_head_uqp2 = nn.Sequential(
            nn.LayerNorm(self._params['d_emb']),
            nn.Linear(self._params['d_emb'], 1),
        )

        
        self.latgps_embedding = nn.Embedding(
            num_embeddings=self._params['GridCount_x']+3,
            embedding_dim=self._params['d_emb']//2
        )
        self.longps_embedding = nn.Embedding(
            num_embeddings=self._params['GridCount_y']+3,
            embedding_dim=self._params['d_emb']//2
        )


        self.query_proj = nn.Sequential(
            nn.Linear(self._params['d_emb'], self._params['d_emb']),
            nn.LayerNorm(self._params['d_emb']),
        )
        self.user_embedding = nn.Embedding(num_embeddings=self._params['user_num'], embedding_dim=self._params['d_emb'])
        self.uq_proj1 = nn.Sequential(
            nn.LayerNorm(self._params['d_emb']),
        )
        
        self.dis_linear = nn.Sequential(
                nn.Linear(1, self._params['d_emb']),
                nn.LayerNorm(self._params['d_emb']),
        )
        self.out = nn.Linear(2, 1)


    def forward(self, inputs):
        """Forward."""

        query_input_ids = inputs['text_left']   #B * LQ
        doc_input_ids = inputs['text_right']    #B * LD
        q_mask = (query_input_ids != self._params['mask_value']).float()
        
        d_mask = (doc_input_ids != self._params['mask_value']).float()

        # shape = [B, L, D]
        xl = self.embedding(query_input_ids.long())
        xl = self.l_proj(xl)
        b, n, _ = xl.shape

        cls_token_left = repeat(self.cls_token_query, '() n d -> b n d', b = b)
        q_mask = torch.cat([torch.ones((b, 1)).to(self.device), q_mask], dim=1)
        xl = torch.cat((cls_token_left, xl), dim=1)
        xl += self.pos_embedding_query[:, :(n+1)]
        xl = self.dropout_query(xl)

        # shape = [B, R, D]
        xr = self.embedding(doc_input_ids.long())
        xr = self.r_proj(xr)
        b, n, _ = xr.shape
        self.b = b
        cls_token_right = repeat(self.cls_token_doc, '() n d -> b n d', b = b)
        xr = torch.cat((cls_token_right, xr), dim=1)
        xr += self.pos_embedding_doc[:, :(n+1)]
        xr = self.dropout_doc(xr)

        for multi_scale_transformer in self.multi_scale_transformers1:
            xl, _ = multi_scale_transformer(xl, xl)

        for multi_scale_transformer in self.multi_scale_transformers2:
            xr, _ = multi_scale_transformer(xr, xr)
        
        text_score = self.compute_tok_score_pair(xr, doc_input_ids, xl, query_input_ids, q_mask)

        xl = xl.mean(dim = 1) if self.pool == 'mean' else xl[:, 0]

        # calculate the dis score
        query_lat = inputs['location_left_l'][:, 0].long()
        query_lon = inputs['location_left_l'][:, 1].long()
        poi_lat = inputs['location_right_l'][:, 0].long()
        poi_lon = inputs['location_right_l'][:, 1].long()
        
        q_lat_embs = self.latgps_embedding(query_lat+1)   #B * d_g
        q_lon_embs = self.longps_embedding(query_lon+1)
        p_lat_embs = self.latgps_embedding(poi_lat+1)
        p_lon_embs = self.longps_embedding(poi_lon+1)
        q_lat_embs_1 = self.latgps_embedding(query_lat)
        q_lat_embs_2 = self.latgps_embedding(query_lat+2)
        q_lon_embs_1 = self.longps_embedding(query_lon)
        q_lon_embs_2 = self.longps_embedding(query_lon+2)
        p_lat_embs_1 = self.latgps_embedding(poi_lat)
        p_lat_embs_2 = self.latgps_embedding(poi_lat+2)
        p_lon_embs_1 = self.longps_embedding(poi_lon)
        p_lon_embs_2 = self.longps_embedding(poi_lon+2)

        q_latlon_o1 = torch.cat([q_lat_embs, q_lon_embs], dim=1)
        q_latlon_o2 = torch.cat([q_lat_embs, q_lon_embs_1], dim=1)
        q_latlon_o3 = torch.cat([q_lat_embs, q_lon_embs_2], dim=1)
        q_latlon_o4 = torch.cat([q_lat_embs_1, q_lon_embs], dim=1)
        q_latlon_o5 = torch.cat([q_lat_embs_2, q_lon_embs], dim=1)

        p_latlon_o1 = torch.cat([p_lat_embs, p_lon_embs], dim=1)
        p_latlon_o2 = torch.cat([p_lat_embs, p_lon_embs_1], dim=1)
        p_latlon_o3 = torch.cat([p_lat_embs, p_lon_embs_2], dim=1)
        p_latlon_o4 = torch.cat([p_lat_embs_1, p_lon_embs], dim=1)
        p_latlon_o5 = torch.cat([p_lat_embs_2, p_lon_embs], dim=1)


        q_latlon_o = torch.cat([q_latlon_o1.unsqueeze(1), q_latlon_o2.unsqueeze(1), 
        q_latlon_o3.unsqueeze(1), q_latlon_o4.unsqueeze(1),
        q_latlon_o5.unsqueeze(1)], 1)

        p_latlon_o = torch.cat([p_latlon_o1.unsqueeze(1), p_latlon_o2.unsqueeze(1), 
        p_latlon_o3.unsqueeze(1), p_latlon_o4.unsqueeze(1),
        p_latlon_o5.unsqueeze(1)], 1)

        user_input_ids = inputs['userId'].long()
        u_embs = self.user_embedding(user_input_ids)

        user_vec1 = self.uq_proj1(u_embs)
        cls_token_uq = repeat(self.cls_token_uq, '() n d -> b n d', b = b)
        cls_token_up = repeat(self.cls_token_up, '() n d -> b n d', b = b)
        q_latlon = torch.cat([cls_token_uq, q_latlon_o], dim=1)
        p_latlon = torch.cat([cls_token_up, p_latlon_o], dim=1)
        for multi_scale_transformer in self.multi_scale_transformers3:
            uq, up = multi_scale_transformer(q_latlon, p_latlon)
        uq = uq.mean(dim = 1) if self.pool == 'mean' else uq[:, 0]
        up = up.mean(dim = 1) if self.pool == 'mean' else up[:, 0]

        q_pooled_output = self.query_proj(xl)
        x_dis = inputs['distance']
        x_dis = x_dis.unsqueeze(0)
        x_dis = x_dis.transpose(0, 1)
        x_dis = self.dis_linear(x_dis)


        uqp = torch.cat([user_vec1.unsqueeze(1), uq.unsqueeze(1), q_pooled_output.unsqueeze(1), up.unsqueeze(1), x_dis.unsqueeze(1)], dim=1)
        cls_token_2 = repeat(self.cls_token_2, '() n d -> b n d', b = b)
        
        uqp_2 = torch.cat((cls_token_2, uqp), dim=1)
        uqp_2 = self.transformer_2(uqp_2)
        uqp_2 = uqp_2.mean(dim = 1) if self.pool == 'mean' else uqp_2[:, 0]
        user_region_score = self.mlp_head_uqp2(uqp_2)

        output = text_score.unsqueeze(1) + user_region_score
        return output

    def compute_tok_score_pair(self,
            doc_reps_last_hidden, doc_input_ids,
            qry_reps_last_hidden, qry_input_ids, qry_attention_mask):
        qry_input_ids2 = torch.cat([torch.ones((self.b, 1)).to(self.device), qry_input_ids], dim=1)
        doc_input_ids2 = torch.cat([torch.ones((self.b, 1)).to(self.device), doc_input_ids], dim=1)
        exact_match = qry_input_ids2.unsqueeze(2) == doc_input_ids2.unsqueeze(1)  # B * LQ * LD
        LD = doc_input_ids2.shape[1]
        exact_match = exact_match.float()

        exact_match_m, _ = exact_match.max(dim=2)
        exact_match_m = exact_match_m >= 1
        exact_match1 = exact_match_m.float()
        exact_match2 = exact_match1.unsqueeze(2)
        exact_match2 = exact_match2.repeat(1, 1, LD)

        exact_match3 = (exact_match1 * qry_attention_mask)[:, 1:].sum(-1)
        exact_match3 = exact_match3 >= 1
        exact_match3 = exact_match3.float()

        # qry_reps: B * LQ * d
        # doc_reps: B * LD * d
        qry_reps_last_hidden = self.l_norm(qry_reps_last_hidden)
        doc_reps_last_hidden = self.r_norm(doc_reps_last_hidden)
        
        scores_no_masking = torch.bmm(qry_reps_last_hidden, doc_reps_last_hidden.permute(0, 2, 1))  # B * LQ * LD
        if self._params['pooling'] == 'max':
            scores = scores_no_masking * exact_match2
            tok_scores, _ = scores.max(dim=2)  # B * LQ
        else:
            raise NotImplementedError('%s pooling is not defined' % self._params['pooling'])
        # remove padding and cls token
        tok_scores = (tok_scores * qry_attention_mask).sum(-1)
        return tok_scores
