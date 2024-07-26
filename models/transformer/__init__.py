from dataclasses import dataclass
from models import ModelConfig


@dataclass
class TransformerConfig(ModelConfig):
    architecture: str = "transformer"
    extractor: str = "vgg"
    use_bidirectional: bool = True
    dropout: float = 0.1
    d_model: int = 512
    d_ff: int = 2048
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6


@dataclass
class JointCTCAttentionTransformerConfig(TransformerConfig):
    cross_entropy_weight: float = 0.7
    ctc_weight: float = 0.3
    mask_conv: bool = True
    joint_ctc_attention: bool = True