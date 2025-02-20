import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import torch.nn.functional as F
from models.attention import MultiHeadAttention
from models.decoder import BaseDecoder
from models.modules import Linear
from models.transformer.sublayers import PositionwiseFeedForward
from models.transformer.embeddings import Embedding, PositionalEncoding
from models.transformer.mask import get_attn_pad_mask, get_attn_subsequent_mask
import torch
import math
from torch.nn.modules.utils import _single
import pdb

class TransformerDecoderLayer(nn.Module):
    """
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)
    """

    def __init__(
            self,
            d_model: int = 512,             # dimension of model
            num_heads: int = 8,             # number of attention heads
            d_ff: int = 2048,               # dimension of feed forward network
            dropout_p: float = 0.3,         # probability of dropout
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention_prenorm = nn.LayerNorm(d_model)
        self.encoder_attention_prenorm = nn.LayerNorm(d_model)
        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    def forward(
            self,
            inputs: Tensor,
            encoder_outputs: Tensor,
            self_attn_mask: Optional[Tensor] = None,
            encoder_outputs_mask: Optional[Tensor] = None
    ) :
        # pdb.set_trace()
        residual = inputs
        inputs = self.self_attention_prenorm(inputs)
        outputs, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.encoder_attention_prenorm(outputs)
        outputs, encoder_attn = self.encoder_attention(outputs, encoder_outputs, encoder_outputs, encoder_outputs_mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, self_attn, encoder_attn


class TransformerDecoder(BaseDecoder):


    def __init__(
            self,
            num_classes: int,               # number of classes
            d_model: int = 512,             # dimension of model
            d_ff: int = 512,                # dimension of feed forward network
            num_layers: int = 6,            # number of decoder layers
            num_heads: int = 8,             # number of attention heads
            dropout_p: float = 0.3,         # probability of dropout
            pad_id: int = 0,                # identification of pad token
            sos_id: int = 1,                # identification of start of sentence token
            eos_id: int = 2,                # identification of end of sentence token
            max_length: int = 400,          # max length of decoding
    ) :
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.embedding = Embedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
            ) for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            Linear(d_model, num_classes, bias=False),
        )

    def forward_step(
            self,
            decoder_inputs,
            decoder_input_lengths,
            encoder_outputs,
            encoder_output_lengths,
            positional_encoding_length,
    ) :
        #pdb.set_trace()
        dec_self_attn_pad_mask = get_attn_pad_mask(
            decoder_inputs, decoder_input_lengths, decoder_inputs.size(1)
        )
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_inputs)
        self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        encoder_attn_mask = get_attn_pad_mask(
            encoder_outputs, encoder_output_lengths, decoder_inputs.size(1)
        )

        outputs = self.embedding(decoder_inputs) + self.positional_encoding(positional_encoding_length)
        outputs = self.input_dropout(outputs)
        #pdb.set_trace()
        for layer in self.layers:
            outputs, self_attn, memory_attn = layer(
                inputs=outputs,
                encoder_outputs=encoder_outputs,
                self_attn_mask=self_attn_mask,
                encoder_outputs_mask=encoder_attn_mask,
            )

        return outputs

    def forward(
            self,
            targets: Tensor,
            encoder_outputs: Tensor,
            encoder_output_lengths: Tensor,
            target_lengths: Tensor,
    ) :
        #pdb.set_trace()
        batch_size = encoder_outputs.size(0)

        targets = targets[targets != self.eos_id].view(batch_size, -1)
        target_length = targets.size(1)

        outputs = self.forward_step(
            decoder_inputs=targets,
            decoder_input_lengths=target_lengths,
            encoder_outputs=encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            positional_encoding_length=target_length,
        )
        return self.fc(outputs).log_softmax(dim=-1)

    @torch.no_grad()
    def decode(self, encoder_outputs: Tensor, encoder_output_lengths: Tensor) -> Tensor:

        logits = list()
        batch_size = encoder_outputs.size(0)

        input_var = encoder_outputs.new_zeros(batch_size, self.max_length).long()
        input_var = input_var.fill_(self.pad_id)
        input_var[:, 0] = self.sos_id

        for di in range(1, self.max_length):
            input_lengths = torch.IntTensor(batch_size).fill_(di)
            # pdb.set_trace()
            outputs = self.forward_step(
                decoder_inputs=input_var[:, :di],
                decoder_input_lengths=input_lengths,
                encoder_outputs=encoder_outputs,
                encoder_output_lengths=encoder_output_lengths,
                positional_encoding_length=di,
            )
            # pdb.set_trace()
            step_output = self.fc(outputs).log_softmax(dim=-1)

            logits.append(step_output[:, -1, :])
            input_var[:,di] = logits[-1].topk(1)[1].squeeze(1)

        return torch.stack(logits, dim=1)


