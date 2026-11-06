"""Optional GFS control interface. Real GFS experiments remain disabled."""
import numpy as np
import pandas as pd
import torch
from torch import nn


def latest_legal_forecast(records, origin, target_times, lag_hours=6):
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None or lag_hours not in (3, 6, 12):
        raise ValueError("Explicit aware origin and prespecified GFS lag required")
    legal = []
    for record in records:
        if record["origin_kind"] != "model_forecast":
            raise ValueError("Actual future station observations cannot be external forecast predictors")
        run = pd.Timestamp(record["run_time"])
        if run.tzinfo is None:
            raise ValueError("GFS run clock must be aware")
        publication = record.get("verified_publication_time")
        available = pd.Timestamp(publication) if publication else run + pd.Timedelta(hours=lag_hours)
        if available < run:
            raise ValueError("Publication precedes model initialization")
        if available <= origin:
            legal.append((run, record["run_id"], record))
    if not legal:
        return None
    latest = max(item[0] for item in legal)
    run_ids = sorted({item[1] for item in legal if item[0] == latest})
    if len(run_ids) != 1:
        raise ValueError("Concurrent model versions need an explicit fixed version policy")
    selected = [item[2] for item in legal if item[1] == run_ids[0]]
    by_valid = {pd.Timestamp(item["valid_time"]): item for item in selected}
    if len(by_valid) != len(selected):
        raise ValueError("Conflicting duplicate forecast valid timestamps require explicit resolution")
    values, masks = [], []
    features = len(selected[0]["values"])
    for target in target_times:
        target = pd.Timestamp(target)
        if target.tzinfo is None:
            raise ValueError("Unknown numerical timezone cannot be aligned to GFS")
        record = by_valid.get(target)
        raw = np.asarray(record["values"], dtype=float) if record else np.full(features, np.nan)
        masks.append(np.isfinite(raw))
        values.append(np.where(np.isfinite(raw), raw, 0))
    return {"run_id": run_ids[0], "values": np.array(values), "mask": np.array(masks),
            "information_set": "same run/variables/valid points must be shared by raw-numeric and text controls",
            "no_nearest_neighbor_or_truth_interpolation": True}


class GFSNumericalControl(nn.Module):
    def __init__(self, backbone, external_features, channels):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Linear(2 * external_features, channels, bias=False)

    def forward(self, x_num, x_mask, gfs_num, gfs_mask, **kwargs):
        baseline = self.backbone(x_num=x_num, x_mask=x_mask)
        values = torch.where(gfs_mask, gfs_num, 0)
        extra = self.projection(torch.cat((values, gfs_mask.float()), -1))
        return baseline + extra
