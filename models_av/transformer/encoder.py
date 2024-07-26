import torch.nn as nn
from torch import Tensor
from typing import Tuple

from models.attention import MultiHeadAttention
from models.encoder import BaseEncoder
from models.transformer.embeddings import PositionalEncoding
from models.transformer.mask import get_attn_pad_mask
from models.transformer.sublayers import PositionwiseFeedForward
from models.modules import Linear, Transpose
import pdb

class TransformerEncoderLayer(nn.Module):

    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.attention_prenorm = nn.LayerNorm(d_model)
        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    def forward(self, inputs: Tensor, self_attn_mask: Tensor = None):
        # pdb.set_trace()
        residual = inputs
        inputs = self.attention_prenorm(inputs)
        outputs, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, attn


class TransformerEncoder(BaseEncoder):
    def __init__(
            self,
            input_dim: int,                         # dimension of feature vector
            extractor: str = 'vgg',                 # convolutional extractor
            d_model: int = 512,                     # dimension of model
            d_ff: int = 2048,                       # dimension of feed forward network
            num_layers: int = 6,                    # number of enc oder layers
            num_heads: int = 8,                     # number of attention heads
            dropout_p: float = 0.3,                 # probability of dropout
            joint_ctc_attention: bool = False,
            num_classes: int = None,                # number of classification
    ):
        super(TransformerEncoder, self).__init__(input_dim=input_dim, extractor=extractor, d_model=d_model,
                                                num_classes=num_classes, dropout_p=dropout_p,
                                                joint_ctc_attention=joint_ctc_attention,
                                                )
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_proj = Linear(self.conv_output_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
            ) for _ in range(num_layers)
        ])

    def forward(self, inputs, input_lengths):
        # pdb.set_trace()
        encoder_log_probs = None
        conv_outputs, output_lengths = self.conv(inputs, input_lengths)

        self_attn_mask = get_attn_pad_mask(conv_outputs, output_lengths, conv_outputs.size(1))
        outputs = self.input_norm(self.input_proj(conv_outputs))
        outputs += self.positional_encoding(outputs.size(1))
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, attn = layer(outputs, self_attn_mask)
        if self.joint_ctc_attention:
            encoder_log_probs = self.fc(outputs.transpose(1, 2)).log_softmax(dim=-1)

        return outputs, output_lengths, encoder_log_probs



class AVTransformerEncoder(BaseEncoder):
    def __init__(
            self,
            input_dim: int,                         # dimension of feature vector
            extractor: str = 'vgg',                 # convolutional extractor
            d_model: int = 512,                     # dimension of model
            d_ff: int = 2048,                       # dimension of feed forward network
            num_layers: int = 6,                    # number of enc oder layers
            num_heads: int = 8,                     # number of attention heads
            dropout_p: float = 0.3,                 # probability of dropout
            joint_ctc_attention: bool = False,
            num_classes: int = None,                # number of classification
    ):
        super(AVTransformerEncoder, self).__init__(input_dim=input_dim, extractor=extractor, d_model=d_model,
                                                num_classes=num_classes, dropout_p=dropout_p,
                                                joint_ctc_attention=joint_ctc_attention,
                                                )
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_proj = Linear(self.conv_output_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
            ) for _ in range(num_layers)
        ])

    def forward(self, inputs, input_lengths):
        # pdb.set_trace()
        encoder_log_probs = None
        conv_outputs, output_lengths = self.conv(inputs, input_lengths)

        self_attn_mask = get_attn_pad_mask(conv_outputs, output_lengths, conv_outputs.size(1))
        outputs = self.input_norm(self.input_proj(conv_outputs))
        outputs += self.positional_encoding(outputs.size(1))
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, attn = layer(outputs, self_attn_mask)

        return outputs, output_lengths, encoder_log_probs




