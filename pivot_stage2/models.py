"""Independent all-channel MIMO baselines and modular PIVOT."""
import torch
import math
from torch import nn
import torch.nn.functional as F


class CTSR(nn.Module):
    """History-only Pearson + spectral-amplitude similarity, stable tie order."""
    def __init__(self, dim, top_k=4):
        super().__init__()
        self.top_k = top_k
        self.mix = nn.Linear(dim, dim, bias=False)

    def select(self, x, mask):
        valid = mask.sum(1) >= 2
        count = mask.sum(1).clamp_min(1)
        mean = (x * mask).sum(1) / count
        centered = torch.where(mask, x - mean.unsqueeze(1), 0).transpose(1, 2)
        normed = F.normalize(centered, dim=-1)
        correlation = normed @ normed.transpose(1, 2)
        frequency = F.normalize(torch.fft.rfft(centered.float(), dim=-1).abs(), dim=-1)
        spectral = frequency @ frequency.transpose(1, 2)
        score = 0.5 * (correlation.float() + spectral)
        channels = x.shape[-1]
        allowed = valid.unsqueeze(1) & valid.unsqueeze(2)
        allowed &= ~torch.eye(channels, dtype=torch.bool, device=x.device).unsqueeze(0)
        score = score.masked_fill(~allowed, -torch.inf)
        k = min(self.top_k, max(0, channels - 1))
        # Stable sorting gives smaller channel IDs precedence in exact ties.
        ids = torch.argsort(score, dim=-1, descending=True, stable=True)[..., :k]
        selected_valid = torch.gather(allowed, 2, ids)
        return ids, selected_valid

    def forward(self, x, mask, encoded):
        ids, valid = self.select(x, mask)
        if not ids.shape[-1]:
            return encoded
        batch = torch.arange(x.shape[0], device=x.device)[:, None, None]
        peers = encoded[batch, ids]
        pooled = (peers * valid.unsqueeze(-1)).sum(2) / valid.sum(2).clamp_min(1).unsqueeze(-1)
        return encoded + self.mix(pooled)


class PatchEncoder(nn.Module):
    def __init__(self, lookback, dim=32, patch=16, stride=8, layers=1, heads=4, dropout=0.1):
        super().__init__()
        self.patch, self.stride = min(patch, lookback), min(stride, lookback)
        self.padding = (self.stride - (lookback - self.patch) % self.stride) % self.stride
        count = (lookback + self.padding - self.patch) // self.stride + 1
        self.project = nn.Linear(2 * self.patch, dim)
        self.position = nn.Parameter(torch.randn(1, count, dim) * 0.01)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.mask_embed = nn.Linear(1, dim, bias=False)

    def forward(self, x, mask):
        batch, _, channels = x.shape
        patches = F.pad(x.transpose(1, 2), (self.padding, 0)).unfold(-1, self.patch, self.stride)
        masks = F.pad(mask.float().transpose(1, 2), (self.padding, 0)).unfold(-1, self.patch, self.stride)
        tokens = self.project(torch.cat((patches, masks), -1)).reshape(batch * channels, -1, self.project.out_features)
        tokens = self.transformer(tokens + self.position)
        representation = tokens.mean(1).reshape(batch, channels, -1)
        return representation + self.mask_embed(mask.float().mean(1).unsqueeze(-1))


class Fusion(nn.Module):
    def __init__(self, dim, text_dim=384, heads=4, mode="cross_attention"):
        super().__init__()
        self.mode = mode
        self.project = nn.Linear(text_dim, dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.concat = nn.Linear(2 * dim, dim, bias=False)
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, numeric, text, mask):
        if self.mode == "none":
            return numeric
        batch, channels, k, _ = text.shape
        flat_mask = mask.reshape(batch * channels, k)
        present = flat_mask.any(-1)
        if not present.any():
            return numeric  # Exact identity; zero-padded embeddings add nothing.
        base = numeric.reshape(batch * channels, -1)
        selected_mask = flat_mask[present]
        raw_selected = text.reshape(batch * channels, k, -1)[present]
        raw_selected = torch.where(selected_mask.unsqueeze(-1), raw_selected, 0)
        if not torch.isfinite(raw_selected).all():
            raise ValueError("Nonfinite eligible text embedding")
        selected_text = self.project(raw_selected)
        selected_base = base[present]
        if self.mode in {"mean", "concat"}:
            pooled = (selected_text * selected_mask.unsqueeze(-1)).sum(1) / selected_mask.sum(1).unsqueeze(-1)
            update = self.concat(torch.cat((selected_base, pooled), -1))
        else:
            update = self.attention(selected_base.unsqueeze(1), selected_text, selected_text,
                                    key_padding_mask=~selected_mask, need_weights=False)[0].squeeze(1)
        gate = torch.sigmoid(self.gate(torch.cat((selected_base, update), -1)))
        delta = torch.zeros_like(base)
        delta[present] = gate * update
        return (base + delta).reshape(batch, channels, -1)


class PIVOT(nn.Module):
    def __init__(self, lookback, horizon, channels, dim=32, patch=16, stride=8, layers=1, heads=4,
                 dropout=0.1, use_ctsr=True, channel_top_k=4, fusion="cross_attention", **kwargs):
        super().__init__()
        self.encoder = PatchEncoder(lookback, dim, patch, stride, layers, heads, dropout)
        self.ctsr = CTSR(dim, channel_top_k)
        self.use_ctsr = use_ctsr
        self.fusion = Fusion(dim, heads=heads, mode=fusion)
        self.head = nn.Linear(dim, horizon)

    def forward(self, x_num, x_mask, text_embeddings=None, text_mask=None, **kwargs):
        encoded = self.encoder(x_num, x_mask)
        if self.use_ctsr:
            encoded = self.ctsr(x_num, x_mask, encoded)
        if text_embeddings is not None:
            encoded = self.fusion(encoded, text_embeddings, text_mask)
        return self.head(encoded).transpose(1, 2)


class DLinear(nn.Module):
    def __init__(self, lookback, horizon, channels, kernel=7, **kwargs):
        super().__init__()
        self.kernel = min(kernel, lookback if lookback % 2 else max(1, lookback - 1))
        self.trend = nn.Linear(lookback, horizon)
        self.seasonal = nn.Linear(lookback, horizon)
        self.mask_head = nn.Linear(lookback, horizon, bias=False)

    def forward(self, x_num, x_mask, **kwargs):
        x = x_num.transpose(1, 2)
        trend = F.avg_pool1d(F.pad(x, (self.kernel // 2, self.kernel // 2), mode="replicate"), self.kernel, stride=1)
        return (self.trend(trend) + self.seasonal(x - trend) + self.mask_head(x_mask.float().transpose(1, 2))).transpose(1, 2)


class Naive(nn.Module):
    def __init__(self, horizon, zero=False, zero_values=None):
        super().__init__()
        self.horizon, self.zero = horizon, zero
        self.register_buffer("zero_values", torch.as_tensor(zero_values if zero_values is not None else [0.], dtype=torch.float32))

    def forward(self, x_num, **kwargs):
        result = x_num[:, -1:, :].expand(-1, self.horizon, -1)
        return self.zero_values.to(result).view(1, 1, -1).expand_as(result) if self.zero else result


def build_model(config):
    config = dict(config)
    name = config.pop("name", "pivot")
    if name not in {"zero", "persistence", "dlinear", "patchtst", "pivot"}:
        raise ValueError("Unknown model name")
    if name in {"zero", "persistence"}:
        return Naive(config["horizon"], name == "zero", config.get("zero_values"))
    if name == "dlinear":
        return DLinear(**config)
    if name == "patchtst":
        config.update(use_ctsr=False, fusion="none")
    return PIVOT(**config)
