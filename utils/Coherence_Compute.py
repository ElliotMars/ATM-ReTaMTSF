import torch
import torch.fft as fft


def vectorized_compute_coherence(target: torch.Tensor,
                                 database: torch.Tensor,
                                 nperseg: int = 256) -> torch.Tensor:
    """
    vectorized compute coherence（In Batch parallel）。

    Args:
        target: [B, C_T, L]
        database: [B, 21 - C_T, L]
        nperseg: segment length

    Returns:
        coherence matrix [B, C_T, 21 - C_T]
    """
    B, C_T, L = target.shape  # Now target is flattened: B*C_T, L
    _, k, _ = database.shape  # Database is [B, 21 - C_T, L]

    n_overlap = nperseg // 2
    nseg = (L - nperseg) // (nperseg - n_overlap) + 1

    # segmented
    target_seg = target.unfold(-1, nperseg, nperseg - n_overlap)#[:, :, nseg, :]  # [B, C_T, nseg, nperseg]
    database_seg = database.unfold(-1, nperseg, nperseg - n_overlap)#[..., :, nseg, :]  # [B, 21 - C_T, nseg, nperseg]

    # windowed
    window = torch.hann_window(nperseg, device=target.device)
    target_windowed = target_seg * window  # [B, C_T, nseg, nperseg]
    database_windowed = database_seg * window  # [B, 21 - C_T, nseg, nperseg]

    # FFT
    fft_target = fft.rfft(target_windowed, dim=-1)  # [B, C_T, nseg, nperseg//2+1]
    fft_database = fft.rfft(database_windowed, dim=-1)  # [B, 21 - C_T, nseg, nperseg//2+1]

    # CPSD and APSD
    #Pxy = (fft_target.conj() * fft_database).mean(dim=2)  # [B, 21 - C_T, nperseg//2+1]
    Pxy = torch.einsum('bcns,bkns->bckns', fft_target.conj(), fft_database).mean(dim=3)#[B, C_T, 21 - C_T, nperseg//2+1]
    Pxx = (torch.abs(fft_target) ** 2).mean(dim=2)  # [B, C_T, nperseg//2+1]
    Pyy = (torch.abs(fft_database) ** 2).mean(dim=2)  # [B, 21 - C_T, nperseg//2+1]

    # coherence
    coherence = (torch.abs(Pxy) ** 2) / (Pxx.unsqueeze(2) * Pyy.unsqueeze(1) + 1e-10)  # [B, C_T, 21 - C_T, nperseg//2+1]
    return coherence.mean(dim=-1)  # [B, C_T, 21 - C_T]