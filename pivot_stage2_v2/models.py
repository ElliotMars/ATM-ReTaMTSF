import torch
from torch import nn
from pivot_stage2.models import build_model as base_build

class RawForecastControl(nn.Module):
    def __init__(self,config):
        super().__init__();config=dict(config);config.update(name='pivot',fusion='none')
        self.backbone=base_build(config)
        self.projection=nn.Linear(14,config['channels'],bias=False)
    def forward(self,x_num,x_mask,gfs_num,gfs_mask,**kwargs):
        numeric=self.backbone(x_num=x_num,x_mask=x_mask)
        values=torch.where(gfs_mask,gfs_num,0)
        return numeric+self.projection(torch.cat((values,gfs_mask.float()),-1))

def build_model(config):
    config=dict(config)
    return RawForecastControl(config) if config.get('name')=='gfs_raw' else base_build(config)
