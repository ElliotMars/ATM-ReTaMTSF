import torch
from torch import nn
from utils.Positional_Embedding import positional_encoding
from utils.Coherence_Compute import vectorized_compute_coherence

class TS_CoherAnalysis(nn.Module):
    def __init__(self, configs):
        super(TS_CoherAnalysis, self).__init__()
        self.configs = configs

    def forward(self, target_series, TS_database):
        nperseg = self.configs.nperseg
        B, C_T, L = target_series.shape
        if self.configs.nref >= TS_database.size(1):
            topk_sequences = TS_database.repeat(1, C_T, 1)
            padding = torch.zeros((B, self.configs.nref - TS_database.size(1), L),
                                  device=topk_sequences.device).repeat(1, C_T, 1)
            topk_sequences = torch.concat([topk_sequences, padding], dim=1)
            return topk_sequences
        #target_series = target_series.view(B * C_T, L)  # Flatten target_series for batch processing
        coherence_scores = vectorized_compute_coherence(target_series, TS_database, nperseg)

        # Reshape coherence_scores back to (B, C_T, 21 - C_T)
        coherence_scores = coherence_scores.view(B, C_T, TS_database.size(1))

        # Select topk coherence scores and their corresponding indices
        _, topk_indices = torch.topk(coherence_scores, k=self.configs.nref, dim=2)

        # Gather the top-k corresponding database sequences
        topk_sequences = torch.gather(TS_database.unsqueeze(1).repeat(1, C_T, 1, 1), 2,
                                      topk_indices.unsqueeze(-1).expand(-1, -1, -1,
                                        TS_database.size(-1))).view(B, C_T*self.configs.nref, L)
        return topk_sequences#[B, C_T*K_T, L]

class ContentSynthesis(nn.Module):
    def __init__(self, configs, input_dim=1, d_model=384, nhead=4):
        super(ContentSynthesis, self).__init__()
        self.ablation_syn = configs.ablation_syn
        self.d_model = d_model

        # input embed
        self.target_embed = nn.Linear(input_dim, d_model)
        self.ref_embed = nn.Linear(input_dim, d_model)

        self.norm = nn.LayerNorm(d_model)  # shared norm

        # aggregation stacks
        self.aggregation_layers = nn.ModuleList([
            AggregationLayer(d_model, nhead) for _ in range(configs.naggregation)
        ])


    def forward(self, target_seq, ref_TS):
        # embedd target series
        target = self.target_embed(target_seq.unsqueeze(-1))  # [B, 1, L, D]
        target = positional_encoding(target)
        target = self.norm(target)

        refs = self.ref_embed(ref_TS.unsqueeze(-1)) #[B, nref, L, D]
        refs = positional_encoding(refs)
        refs = self.norm(refs) # [B, nref, L, D]

        # concatenate target series and reference series
        combined = torch.cat([target, refs], dim=1)  # [B, C_TmulK_text+1, L, D]
        synthesized = combined

        if not self.ablation_syn:
            # multi-layer aggregation
            for layer in self.aggregation_layers:
                synthesized = layer(synthesized)

        return synthesized, torch.cat([target_seq, ref_TS], dim=1), combined


class AggregationLayer(nn.Module):
    """aggregation layer"""

    def __init__(self, d_model, nhead):
        super().__init__()
        # time attention
        self.time_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )

        # content attention
        self.content_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )

        # feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):#x:[B, C_T*(K_T+1), L]

        # dimension adjustment
        B, C_Tmul_K_Tplus1_, L, D = x.shape
        x = x.view(B, C_Tmul_K_Tplus1_ * L, D)  # [B, (C_T*(K_T+1)*L, D]
        B, total_len, D = x.shape

        # ---- step1：time attention ----
        # into [B*(K+1), L, D]
        x_time = x.view(B, -1, L, D)  # [B, C_T*(K_T+1), L, D]
        x_time = x_time.reshape(B * C_Tmul_K_Tplus1_, L, D)

        # time attention
        time_out, _ = self.time_attn(
            query=x_time,
            key=x_time,
            value=x_time
        )
        time_out = self.norm1(x_time + time_out) # [B, C_T*(K_T+1), L, D]

        # into [B, C_T*(K_T+1)*L, D]
        x = time_out.view(B, C_Tmul_K_Tplus1_ * L, D)

        # ---- step2：content attention ----
        # into [B*L, C_T*(K_T+1), D]
        x_content = x.view(B, C_Tmul_K_Tplus1_, L, D)  # [B, C_T*(K_T+1), L, D]
        x_content = x_content.permute(0, 2, 1, 3)  # [B, L, C_T*(K_T+1), D]
        x_content = x_content.reshape(B * L, C_Tmul_K_Tplus1_, D)

        # content attention
        content_out, _ = self.content_attn(
            query=x_content,
            key=x_content,
            value=x_content
        )
        content_out = self.norm2(x_content + content_out)

        # into [B, C_T*(K_T+1)*L, D]
        x = content_out.view(B, L, C_Tmul_K_Tplus1_, D)
        x = x.permute(0, 2, 1, 3).reshape(B, C_Tmul_K_Tplus1_ * L, D)

        # ---- step3：feedforward ----
        x = self.norm3(x + self.ffn(x))
        x = x.reshape(B, C_Tmul_K_Tplus1_, L, D)
        return x

class TextCrossAttention(nn.Module):
    def __init__(self, configs, qt_embedding_dim=384, nd_embedding_dim=384, n_heads=8,
                 self_layer=3, cross_layer=3):
        super(TextCrossAttention, self).__init__()
        self.K_n = configs.nref_text
        self.K_n_qt = configs.qt_ref_text
        self.pred_len = configs.pred_len
        cross_encoder_layer = nn.TransformerDecoderLayer(d_model=nd_embedding_dim,
                                                         nhead=n_heads,
                                                         dropout=configs.dropout_rate,
                                                         dim_feedforward=nd_embedding_dim * 4,
                                                         activation='gelu',
                                                         batch_first=True,
                                                         norm_first=True)
        norm_layer = nn.LayerNorm(nd_embedding_dim, eps=1e-5)
        self.cross_encoder = nn.TransformerDecoder(cross_encoder_layer, cross_layer, norm=norm_layer)

        cross_encoder_layer2 = nn.TransformerDecoderLayer(d_model=nd_embedding_dim,
                                                         nhead=n_heads,
                                                         dropout=configs.dropout_rate,
                                                         dim_feedforward=nd_embedding_dim * 4,
                                                         activation='gelu',
                                                         batch_first=True,
                                                         norm_first=True)
        norm_layer2 = nn.LayerNorm(nd_embedding_dim, eps=1e-5)
        self.cross_encoder2 = nn.TransformerDecoder(cross_encoder_layer2, cross_layer, norm=norm_layer2)

        self_encoder_layer = nn.TransformerDecoderLayer(d_model=qt_embedding_dim,
                                                        nhead=n_heads,
                                                        dropout=configs.dropout_rate,
                                                        dim_feedforward=qt_embedding_dim * 4,
                                                        activation='gelu',
                                                        batch_first=True,
                                                        norm_first=True)
        self_norm_layer = nn.LayerNorm(qt_embedding_dim, eps=1e-5)
        self.self_encoder = nn.TransformerDecoder(self_encoder_layer, self_layer, norm=self_norm_layer)

        for p in self.cross_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.cross_encoder2.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.self_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, qt_emb, des_emb, nd_emb):#qt_emb[B, C_T, H(seq_len), D(384)] nd_emb[B, N(7304), M(1), D(384)]
        ref_news_embed = self.retrival(qt_emb, des_emb, nd_emb)#[B, C_T*K_n, H, D]
        B, C_T, H, D = qt_emb.shape
        _, C_TmulK_n, _, _ = ref_news_embed.shape

        qt_emb = qt_emb.repeat(1, self.K_n, 1, 1).reshape(B*C_T*self.K_n, H, D)
        des_emb = des_emb.repeat(1, self.K_n, H, 1).reshape(B*C_T*self.K_n, H, D)
        ref_news_embed = ref_news_embed.reshape(B*C_TmulK_n, H, D)

        result = self.cross_encoder(tgt=des_emb, memory=ref_news_embed)#[B*C_T*K_n, H, D]
        result = self.cross_encoder2(tgt=qt_emb, memory=result)
        result = self.self_encoder(tgt=result, memory=result)#[B*C_T*K_n, H, D]

        result = result.view(B, C_T*self.K_n, -1, D)

        return result

    def retrival(self, qt_emb, des_emb, nd_emb):
        # input size:
        # qt_emb: [B, C_T, H, D]]
        # des_emb: [B, C_T, 1, D]
        # nd_emb: [B, N, M, D]

        B, C_T, H, D = qt_emb.shape
        _, N, M, _ = nd_emb.shape

        # similarity matrix [B, C_T, H, N]
        similarity = torch.matmul(qt_emb.transpose(1, 2),
                                  nd_emb.transpose(1, 2).transpose(2, 3)).permute(0, 2, 1, 3)

        # get Top-Kn index [B, C_T, H, Kn]
        _, topk_indices = torch.topk(similarity, k=self.K_n_qt, dim=-1, sorted=True)

        # generate index template [B, C_T, H, Kn, 1, 1] ->  [B, C_T, H, Kn, M, D]
        expand_dims = (B, C_T, H, self.K_n_qt, M, D)
        topk_indices = topk_indices.view(B, C_T, H, self.K_n_qt, 1, 1).expand(expand_dims)

        # from nd_emb gather [B, C_T, H, Kn, M, D]
        selected_qt = nd_emb.unsqueeze(1).unsqueeze(2).repeat(1, C_T, self.pred_len, 1, 1, 1).gather(  # add K and H dimension, get M dimension
            dim=3,  # gather on N dimension
            index=topk_indices
        )

        # rearrange into [B, C_T, K_n, H, D]
        selected_qt = selected_qt.squeeze(-2).permute(0, 1, 3, 2, 4)  # [B, C_T, K_n, H, D]

        similarity = torch.matmul(des_emb.transpose(1, 2),
                                  nd_emb.transpose(1, 2).transpose(2, 3)).permute(0, 2, 1, 3)
        _, topk_indices = torch.topk(similarity, k=self.K_n-self.K_n_qt, dim=-1, sorted=True)
        expand_dims = (B, C_T, H, self.K_n-self.K_n_qt, M, D)
        topk_indices = topk_indices.view(B, C_T, 1, self.K_n-self.K_n_qt, 1, 1).expand(expand_dims)
        selected_des = nd_emb.unsqueeze(1).unsqueeze(2).repeat(1, C_T, self.pred_len, 1, 1, 1).gather(  # add K and H dimension, get M dimension
            dim=3,  # gather on N dimension
            index=topk_indices
        )

        selected_des = selected_des.squeeze(-2).permute(0, 1, 3, 2, 4)  # [B, C_T, K_n, H, D]

        selected = torch.cat([selected_qt, selected_des], dim=2)

        return selected.reshape(B, C_T*self.K_n, H, D)

class CrossandOutput(nn.Module):
    def __init__(self, configs, text_embedding_dim=384, temp_embedding_dim=384, n_heads=8,
                 self_layer=3, cross_layer=3, TS_attn_layer=3):
        super(CrossandOutput, self).__init__()

        self.ablation = configs.ablation

        cross_encoder_layer = nn.TransformerDecoderLayer(d_model=temp_embedding_dim,
                                                         nhead=n_heads,
                                                         dropout=configs.dropout_rate,
                                                         dim_feedforward=temp_embedding_dim * 4,
                                                         activation='gelu',
                                                         batch_first=True,
                                                         norm_first=True)
        norm_layer = nn.LayerNorm(temp_embedding_dim, eps=1e-5)
        self.cross_encoder = nn.TransformerDecoder(cross_encoder_layer, cross_layer, norm=norm_layer)

        self_encoder_layer = nn.TransformerDecoderLayer(d_model=text_embedding_dim,
                                                        nhead=n_heads,
                                                        dropout=configs.dropout_rate,
                                                        dim_feedforward=text_embedding_dim * 4,
                                                        activation='gelu',
                                                        batch_first=True,
                                                        norm_first=True)
        self_norm_layer = nn.LayerNorm(text_embedding_dim, eps=1e-5)
        self.self_encoder = nn.TransformerDecoder(self_encoder_layer, self_layer, norm=self_norm_layer)

        TS_self_attn_layer = nn.TransformerDecoderLayer(d_model=temp_embedding_dim,
                                                        nhead=n_heads,
                                                        dropout=configs.dropout_rate,
                                                        dim_feedforward=temp_embedding_dim * 4,
                                                        activation='gelu',
                                                        batch_first=True,
                                                        norm_first=True)
        TS_norm_layer = nn.LayerNorm(temp_embedding_dim, eps=1e-5)
        self.TS_self_attention = nn.TransformerDecoder(TS_self_attn_layer, TS_attn_layer, norm=TS_norm_layer)

        for p in self.cross_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.self_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.TS_self_attention.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.dimension_reducer = DimensionReducer(configs)

        # output linear
        self.mlp = nn.Sequential(
            nn.Linear(temp_embedding_dim, temp_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(temp_embedding_dim * 4, 1)
        )
        self.label_len = configs.label_len
        self.length_reducer = nn.Linear(configs.label_len+configs.pred_len, configs.pred_len)

    def forward(self, text_emb, temp_emb):#text_emb[B, C_T*K_n, H, D] temp_emb[B, C_T*(K_T+1), L, D]
        B, C_Tmul_K_Tplus1_, L, D_temp = temp_emb.shape #C=K_temp_plus1
        _, C_Tmul_K_n, H, D_text = text_emb.shape#C=K_text
        if C_Tmul_K_Tplus1_ > C_Tmul_K_n:
            channel_align = nn.Linear(C_Tmul_K_n, C_Tmul_K_Tplus1_).to(text_emb.device)
            text_emb = channel_align(text_emb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            C_Tmul_K_n = C_Tmul_K_Tplus1_
        elif C_Tmul_K_Tplus1_ < C_Tmul_K_n:
            channel_align = nn.Linear(C_Tmul_K_Tplus1_, C_Tmul_K_n).to(temp_emb.device)
            temp_emb = channel_align(temp_emb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            C_Tmul_K_Tplus1_ = C_Tmul_K_n
        else:
            pass

        temp_emb_out = temp_emb
        text_emb_out = text_emb

        if self.ablation:
            text_emb.zero_()

        label = temp_emb[:, :, -self.label_len:, :].reshape(B * C_Tmul_K_Tplus1_, self.label_len, D_temp)
        label = self.TS_self_attention(tgt=label, memory=label)
        label = label.reshape(B, C_Tmul_K_Tplus1_, self.label_len, D_temp)

        query_emb = torch.cat((label, text_emb), dim=2)#[B, C_T*(K_T+1), L+H, D]

        #temp_emb = self.length_reducer(temp_emb.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        # temp_emb = temp_emb[:, :, -H:, :]
        # temp_emb = self.dimension_reducer(temp_emb)
        # temp_emb = self.mlp(temp_emb).squeeze(-1)
        #
        # return temp_emb, temp_emb_out, text_emb_out

        value_emb = temp_emb.reshape(B * C_Tmul_K_Tplus1_, L, D_temp)
        query_emb = query_emb.reshape(B * C_Tmul_K_Tplus1_, self.label_len + H, D_text)

        # cross attention
        result = self.cross_encoder(tgt=query_emb, memory=value_emb)  # [B*C_T*(K_T+1), L+H, D] text as query
        result = self.self_encoder(tgt=result, memory=result)  # [B*C_T*(K_T+1), L+H, D]

        # reshape the result
        result = result.view(B, C_Tmul_K_Tplus1_, -1, D_temp)

        result = result[:, :, -H:, :]
        #result = self.length_reducer(result.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        result = self.dimension_reducer(result)#[B, C_T, L+H, D]
        result = self.mlp(result).squeeze(-1)#[B, C_T, L+H, 1]->[B, C_T, L+H]

        return result, temp_emb_out, text_emb_out

class DimensionReducer(nn.Module):
    def __init__(self, configs, d_model=384, nhead=8, num_layers=3):
        super().__init__()
        # define Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.K_n = configs.nref_text
        self.K_T = configs.nref
        # channel reducer
        self.channel_reducer = nn.Linear(max(self.K_n, self.K_T+1), 1)

    def forward(self, x):
        """
        input: [B, C_T*(K_T+1), L+H, D]
        output: [B, C_T, L+H, D]
        """
        B, C_Tmul_K_Tplus1_, LplusH, D = x.shape

        # step1: channel dimension preprocess
        x_reshaped = x.permute(0, 2, 1, 3)  # [B, L+H, C_T*(K_T+1)， D]
        x_reshaped = x_reshaped.reshape(B, LplusH * C_Tmul_K_Tplus1_, D)

        # step2: Transformer
        transformer_out = self.transformer(x_reshaped)  # [B, (L+H)*C_T*(K_T+1), D]

        # step3: dimension recover
        # [B, L, C_T*(K_T+1), D]
        recovered = transformer_out.view(B, LplusH, C_Tmul_K_Tplus1_, D)

        # step4: channel reducer
        # [B, C_T*(K_T+1), L+H, D] -> [B, D, L+H, C_T*(K_T+1)]
        recovered = recovered.permute(0, 1, 3, 2).view(B, LplusH, D, -1, max(self.K_n, self.K_T+1))  # [B, D, L+H, C_T, (K_T+1)]

        squeezed = self.channel_reducer(recovered)  # [B, L+H, D, C_T, 1]

        # [B, D, L+H, C_T, 1] -> [B, C_T, L+H, D, 1]
        reduced = squeezed.permute(0, 3, 1, 2, 4).squeeze(-1)  # [B, C_T, L+H, D]
        return reduced