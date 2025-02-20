import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class DecoderInterface(nn.Module):
    def __init__(self):
        super(DecoderInterface, self).__init__()

    def count_parameters(self) :
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float):
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p


class BaseDecoder(DecoderInterface):
    """ ASR Decoder Super Class for KoSpeech model implementation """
    def __init__(self):
        super(BaseDecoder, self).__init__()

    def forward(self, targets, encoder_outputs, **kwargs):
        """
        Forward propagate a `encoder_outputs` for training.
        Args:
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, encoder_outputs, *args):
        """
        Decode encoder_outputs.
        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        raise NotImplementedError


class TransducerDecoder(DecoderInterface):
    """ ASR Transducer Decoder Super Class for KoSpeech model implementation """
    def __init__(self):
        super(TransducerDecoder, self).__init__()

    def forward(self, inputs, input_lengths):
        """
        Forward propage a `inputs` (targets) for training.
        Args:
            inputs (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor):
            * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """
        raise NotImplementedError