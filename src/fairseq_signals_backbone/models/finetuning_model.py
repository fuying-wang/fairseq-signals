import contextlib


from dataclasses import dataclass, field
from typing import Optional
from omegaconf import II

from fairseq_signals_backbone.dataclass import Dataclass
from fairseq_signals_backbone.models.model import BaseModel

@dataclass
class FinetuningConfig(Dataclass):
    model_path: Optional[str] = field(
        default=None, metadata={"help": "path to pretrained model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune pretrained model for this many updates"}
    )

class FinetuningModel(BaseModel):
    def __init__(self, cfg: FinetuningConfig, encoder: BaseModel):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder

        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
    
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict
    
    @classmethod
    def build_model(cls, cfg: FinetuningConfig, task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")
    
    def get_logits(self, net_output, normalize=False, aggregate=False, **kwargs):
        raise NotImplementedError()
    
    def get_targets(self, sample, net_output, **kwargs):
        raise NotImplementedError()
    
    def forward(self, **kwargs):
        raise NotImplementedError()