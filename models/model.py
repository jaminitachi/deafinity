import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from models.modules import Linear
from models.encoder import BaseEncoder,TransducerEncoder
from models.decoder import BaseDecoder,TransducerDecoder


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def count_parameters(self) :
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) :
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor):
        raise NotImplementedError


class EncoderModel(BaseModel):
    """ Super class of KoSpeech's Encoder only Models """
    def __init__(self):
        super(EncoderModel, self).__init__()
        self.decoder = None

    def set_decoder(self, decoder):
        """ Setter for decoder """
        self.decoder = decoder

    def forward(self, inputs: Tensor, input_lengths: Tensor) :
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, predicted_log_probs: Tensor) :
        return predicted_log_probs.max(-1)[1]

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) :
        predicted_log_probs, _ = self.forward(inputs, input_lengths)
        if self.decoder is not None:
            return self.decoder.decode(predicted_log_probs)
        return self.decode(predicted_log_probs)


class AV_EncoderDecoderModel(BaseModel):
    """ Super class of WOOKSpeech's Audio_visual_Encoder-Decoder Models """
    def __init__(self,  video_encoder: BaseEncoder,
                        audio_encoder: BaseEncoder, 
                        decoder: BaseDecoder,
                        ) :
        super(AV_EncoderDecoderModel, self).__init__()
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.decoder = decoder


    def set_encoder(self, video_encoder,audio_encoder):
        """ Setter for encoder """
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder

    def set_decoder(self, decoder):
        """ Setter for decoder """
        self.decoder = decoder

    def count_parameters(self) :
        """ Count parameters of encoder """
        num_video_encoder_parameters = self.video_encoder.count_parameters()
        num_audio_encoder_parameters = self.audio_encoder.count_parameters()
        num_decoder_parameters = self.decoder.count_parameters()
        return num_video_encoder_parameters + num_audio_encoder_parameters + num_decoder_parameters

    def update_dropout(self, dropout_p) :
        """ Update dropout probability of model """
        self.video_encoder.update_dropout(dropout_p)
        self.audio_encoder.update_dropout(dropout_p)
        self.decoder.update_dropout(dropout_p)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            *args,
    ) :
        raise NotImplementedError

    @torch.no_grad()
    def recognize(self, video_inputs, video_input_lengths, audio_inputs,audio_input_lengths):

        raise NotImplementedError
        
class EncoderDecoderModel(BaseModel):
    """ Super class of KoSpeech's Encoder-Decoder Models """
    def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder) :
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def set_encoder(self, encoder):
        """ Setter for encoder """
        self.encoder = encoder

    def set_decoder(self, decoder):
        """ Setter for decoder """
        self.decoder = decoder

    def count_parameters(self) :
        """ Count parameters of encoder """
        num_encoder_parameters = self.encoder.count_parameters()
        num_decoder_parameters = self.decoder.count_parameters()
        return num_encoder_parameters + num_decoder_parameters

    def update_dropout(self, dropout_p) :
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)
        self.decoder.update_dropout(dropout_p)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            *args,
    ) :
        """
        Forward propagate a `inputs` and `targets` pair for training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
        Returns:
            (Tensor, Tensor, Tensor)
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
            * encoder_output_lengths: The length of encoder outputs. ``(batch)``
            * encoder_log_probs: Log probability of encoder outputs will be passed to CTC Loss.
                If joint_ctc_attention is False, return None.
        """
        raise NotImplementedError

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor):
        """
        Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        encoder_outputs, encoder_output_lengths, _ = self.encoder(inputs, input_lengths)
        return self.decoder.decode(encoder_outputs, encoder_output_lengths)
class TransducerModel(BaseModel):
    def __init__(
            self,
            encoder: TransducerEncoder,
            decoder: TransducerDecoder,
            d_model: int,
            num_classes: int,
    ):
        super(TransducerModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc = Linear(d_model << 1, num_classes, bias=False)

    def set_encoder(self, encoder):
        """ Setter for encoder """
        self.encoder = encoder

    def set_decoder(self, decoder):
        """ Setter for decoder """
        self.decoder = decoder

    def count_parameters(self) :
        """ Count parameters of encoder """
        num_encoder_parameters = self.encoder.count_parameters()
        num_decoder_parameters = self.decoder.count_parameters()
        return num_encoder_parameters + num_decoder_parameters

    def update_dropout(self, dropout_p) :
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)
        self.decoder.update_dropout(dropout_p)

    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor):
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.fc(outputs).log_softmax(dim=-1)

        return outputs

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ) :
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        decoder_outputs, _ = self.decoder(targets, target_lengths)
        return self.joint(encoder_outputs, decoder_outputs)

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) :
        pred_tokens, hidden_state = list(), None
        decoder_input = encoder_output.new_tensor([[self.decoder.sos_id]], dtype=torch.long)

        for t in range(max_length):
            decoder_output, hidden_state = self.decoder(decoder_input, hidden_states=hidden_state)
            step_output = self.joint(encoder_output[t].view(-1), decoder_output.view(-1))
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)

        return torch.LongTensor(pred_tokens)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) :
        outputs = list()

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

        return outputs