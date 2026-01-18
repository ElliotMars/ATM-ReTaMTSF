import torch


def compute_lagged_corr(target: torch.Tensor,
                        database: torch.Tensor,
                        max_lag: int = 12,
                        eps: float = 1e-6) -> torch.Tensor:
    """
    Compute max absolute lagged Pearson correlation for each target/database pair.

    Args:
        target: [B, C_T, L]
        database: [B, C_D, L]
        max_lag: maximum lag (both positive and negative)
    Returns:
        scores: [B, C_T, C_D]
    """
    B, C_T, L = target.shape
    _, C_D, L_db = database.shape
    if L != L_db:
        raise ValueError("target and database must have the same length for lagged corr.")
    if L <= 1:
        return torch.zeros((B, C_T, C_D), device=target.device, dtype=target.dtype)

    max_lag = min(max_lag, L - 1)
    if max_lag < 0:
        max_lag = 0

    scores = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = target[..., :lag]  # drop last |lag|
            y = database[..., -lag:]
        elif lag > 0:
            x = target[..., lag:]
            y = database[..., :-lag]
        else:
            x = target
            y = database

        L_lag = x.size(-1)
        if L_lag <= 1:
            continue

        x_centered = x - x.mean(dim=-1, keepdim=True)
        y_centered = y - y.mean(dim=-1, keepdim=True)
        x_std = x_centered.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps  # [B, C_T, 1]
        y_std = y_centered.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps  # [B, C_D, 1]

        corr = torch.einsum("bcl,bdl->bcd", x_centered, y_centered)
        denom = (x_std * y_std.transpose(1, 2)).clamp_min(eps)  # [B, C_T, C_D]
        corr = corr / (L_lag * denom)
        scores.append(corr.abs())

    if not scores:
        return torch.zeros((B, C_T, C_D), device=target.device, dtype=target.dtype)

    return torch.stack(scores, dim=-1).max(dim=-1).values


def compute_partial_corr(target: torch.Tensor,
                         database: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    """
    Compute partial correlation between target and database channels.

    Args:
        target: [B, C_T, L]
        database: [B, C_D, L]
    Returns:
        scores: [B, C_T, C_D]
    """
    B, C_T, L = target.shape
    _, C_D, L_db = database.shape
    if L != L_db:
        raise ValueError("target and database must have the same length for partial corr.")
    if L <= 2:
        return torch.zeros((B, C_T, C_D), device=target.device, dtype=target.dtype)

    data = torch.cat([target, database], dim=1)  # [B, C_total, L]
    centered = data - data.mean(dim=-1, keepdim=True)
    cov = centered @ centered.transpose(-1, -2) / (L - 1 + eps)

    diag = torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype)
    cov = cov + eps * diag
    precision = torch.linalg.pinv(cov)

    d = torch.diagonal(precision, dim1=-2, dim2=-1)
    denom = torch.sqrt(d.unsqueeze(-1) * d.unsqueeze(-2)).clamp_min(eps)
    partial = -precision / denom

    return partial[:, :C_T, C_T:].abs()


def _lag_matrix(series: torch.Tensor, lag: int) -> torch.Tensor:
    """
    Build lagged design matrix for a 1D series.
    series: [L]
    returns: [L-lag, lag]
    """
    L = series.size(0)
    cols = [series[lag - i - 1:L - i - 1] for i in range(lag)]
    return torch.stack(cols, dim=1)


def compute_granger_score(target: torch.Tensor,
                          database: torch.Tensor,
                          lag: int = 3,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    Compute Granger-causality style scores (variance reduction) for each pair.

    Args:
        target: [B, C_T, L]
        database: [B, C_D, L]
        lag: autoregressive order
    Returns:
        scores: [B, C_T, C_D]
    """
    B, C_T, L = target.shape
    _, C_D, L_db = database.shape
    if L != L_db:
        raise ValueError("target and database must have the same length for Granger.")
    if L <= lag + 1 or lag <= 0:
        return torch.zeros((B, C_T, C_D), device=target.device, dtype=target.dtype)

    scores = torch.zeros((B, C_T, C_D), device=target.device, dtype=target.dtype)
    for b in range(B):
        for i in range(C_T):
            x = target[b, i]
            Y = x[lag:]
            Xr = _lag_matrix(x, lag)
            if Xr.numel() == 0:
                continue
            beta_r = torch.linalg.lstsq(Xr, Y).solution
            res_r = Y - Xr @ beta_r
            var_r = res_r.pow(2).mean().clamp_min(eps)

            for j in range(C_D):
                y = database[b, j]
                Xy = _lag_matrix(y, lag)
                Xf = torch.cat([Xr, Xy], dim=1)
                beta_f = torch.linalg.lstsq(Xf, Y).solution
                res_f = Y - Xf @ beta_f
                var_f = res_f.pow(2).mean().clamp_min(eps)
                scores[b, i, j] = torch.log(var_r / var_f)

    return scores
