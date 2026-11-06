import torch
from torch import nn
from pivot_stage2.models import build_model as base_build
from pivot_stage2.models import PIVOT

class WeatherPIVOT(PIVOT):
    """Frozen weather semantics plus separately encoded forecast timing."""
    def __init__(self,**config):
        super().__init__(**config)
        self.forecast_time=nn.Linear(3,384,bias=False)
    def forward(self,x_num,x_mask,text_embeddings=None,text_mask=None,text_time_features=None,**kwargs):
        if text_embeddings is not None and text_time_features is not None:
            timing=torch.where(text_mask.unsqueeze(-1),text_time_features,0)
            text_embeddings=text_embeddings+self.forecast_time(timing)
        return super().forward(x_num=x_num,x_mask=x_mask,text_embeddings=text_embeddings,text_mask=text_mask)

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
    if config.get('name')=='gfs_raw':return RawForecastControl(config)
    if config.get('name','pivot')=='pivot':
        config.pop('name',None)
        return WeatherPIVOT(**config)
    return base_build(config)
