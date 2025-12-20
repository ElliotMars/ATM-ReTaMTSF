import math
import torch


def positional_encoding(x: torch.Tensor) -> torch.Tensor:
    """
    input size: [B, C, L, D]
    output size: [B, C, L, D]
    """
    batch_size, C, seq_len, d_model = x.shape

    # positional encoding matrix [1, 1, L, D]
    position = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)  # [L, 1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=x.device) *
        (-math.log(10000.0) / d_model)
    )  # [D/2]

    pos_enc = torch.zeros(1, 1, seq_len, d_model, device=x.device)  # initialization
    pos_enc[0, 0, :, 0::2] = torch.sin(position * div_term)  # even dimension sin
    pos_enc[0, 0, :, 1::2] = torch.cos(position * div_term)  # odd dimension cos

    return x + pos_enc  # broadcast to [B, C, L, D]