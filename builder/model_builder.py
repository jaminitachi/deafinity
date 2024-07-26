import torch
import torch.nn as nn
from omegaconf import DictConfig
from astropy.modeling import ParameterError
from dataloader.vocabulary import Vocabulary
from models.transformer.model import AV_Transformer_inference
import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(config,vocab):
        
    input_size = config.audio.n_mels
    input_vid_size = config.video.input_feat

    if config.model.architecture.lower() == 'transformer_inference':
        model = build_av_transformer_inference(
            num_classes=len(vocab),
            input_dim=input_size,
            input_vid_dim=input_vid_size,
            d_model=config.model.d_model,
            d_ff=config.model.d_ff,
            num_heads=config.model.num_heads,
            pad_id=vocab.pad_id,
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
            max_length=config.model.max_len,
            num_encoder_layers=config.model.num_encoder_layers,
            num_decoder_layers=config.model.num_decoder_layers,
            dropout_p=config.model.dropout,
            joint_ctc_attention=config.model.joint_ctc_attention,
            extractor=config.model.extractor,
        )

    print("model parameter ")
    print(count_parameters(model))

    return model

def build_av_transformer_inference(
        num_classes: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        input_vid_dim : int,
        input_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        extractor: str,
        dropout_p: float,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        joint_ctc_attention: bool = False,
        max_length: int = 400,
):
    return AV_Transformer_inference(
        input_vid_dim=input_vid_dim,
        input_dim=input_dim,
        num_classes=num_classes,
        extractor=extractor,
        d_model=d_model,
        d_ff=d_ff,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        encoder_dropout_p=dropout_p,
        decoder_dropout_p=dropout_p,
        pad_id=pad_id,
        sos_id=sos_id,
        eos_id=eos_id,
        max_length=max_length,
        joint_ctc_attention=joint_ctc_attention,
    )




