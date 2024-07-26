from torch import Tensor
from typing import Tuple
from models.model import AV_EncoderDecoderModel
from models.transformer.decoder import TransformerDecoder
from models.transformer.encoder import AVTransformerEncoder
import torch
import torch.nn as nn
from models.modules import Linear,Transpose
from models.video_frontend import Video_frontend
import pdb

class AV_Transformer_inference(AV_EncoderDecoderModel):

    def __init__(
            self,
            input_vid_dim: int,
            input_dim: int,
            num_classes: int,
            extractor: str,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            encoder_dropout_p: float = 0.2,
            decoder_dropout_p: float = 0.2,
            d_model: int = 512,
            d_ff: int = 2048,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            num_heads: int = 8,
            joint_ctc_attention: bool = False,
            max_length: int = 400,
    ) :
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        video_encoder = AVTransformerEncoder(
            input_dim=input_vid_dim,
            extractor=extractor,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout_p=encoder_dropout_p,
            joint_ctc_attention=joint_ctc_attention,
            num_classes=num_classes,
        )
        audio_encoder = AVTransformerEncoder(
            input_dim=input_dim,
            extractor=extractor,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout_p=encoder_dropout_p,
            joint_ctc_attention=joint_ctc_attention,
            num_classes=num_classes,
        )
        decoder = TransformerDecoder(
            num_classes=num_classes,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout_p=decoder_dropout_p,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            max_length=max_length,
        )
        super(AV_Transformer_inference, self).__init__(video_encoder,audio_encoder, decoder)

        self.num_classes = num_classes
        self.joint_ctc_attention = joint_ctc_attention
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_length = max_length
        
        self.fc1 = nn.Sequential(
                    nn.Linear(d_model*2,d_model*4),
                    Transpose(shape=(1, 2)),
                    nn.BatchNorm1d(d_model*4),
                    Transpose(shape=(1, 2)),
                    nn.ReLU(),
                    Linear(d_model*4,d_model),
                    )
        if self.joint_ctc_attention:
            self.fc = nn.Sequential(
                nn.BatchNorm1d(d_model),
                Transpose(shape=(1, 2)),
                nn.Dropout(encoder_dropout_p),
                Linear(d_model, num_classes, bias=False),
            )
        self.visual_frontend = Video_frontend()
    def forward(
            self,
            video_inputs: Tensor,
            video_input_lengths: Tensor,
            audio_inputs: Tensor,
            audio_input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ):
        #pdb.set_trace()
        video_inputs = self.visual_frontend(video_inputs)
        video_inputs = video_inputs.squeeze(3)
        video_inputs = video_inputs.squeeze(3)
        video_inputs = video_inputs.permute(0,2,1)
        vid_encoder_outputs, vid_output_lengths, _ = self.video_encoder(video_inputs, video_input_lengths)
        aud_encoder_outputs, aud_output_lengths, _ = self.audio_encoder(audio_inputs, audio_input_lengths)
        #B T F --> B F T
        vid_encoder_outputs = vid_encoder_outputs.permute(0,2,1)
        aud_encoder_outputs = aud_encoder_outputs.permute(0,2,1)
        vid_encoder_outputs_upsample = torch.nn.functional.interpolate(
                vid_encoder_outputs,
                aud_encoder_outputs.size(2)
                )
            
        aud_encoder_outputs = aud_encoder_outputs.permute(0,2,1)
        vid_encoder_outputs_upsample = vid_encoder_outputs_upsample.permute(0,2,1)
        fusion_encoder_out = torch.cat((aud_encoder_outputs,vid_encoder_outputs_upsample),2)
        fusion_encoder_out = self.fc1(fusion_encoder_out)

        predicted_log_probs = self.decoder(targets, fusion_encoder_out, aud_output_lengths, target_lengths)
        if self.joint_ctc_attention:
            encoder_log_probs = self.fc(fusion_encoder_out.transpose(1, 2)).log_softmax(dim=-1)
        else:
            encoder_log_probs = None
        return predicted_log_probs, aud_output_lengths, encoder_log_probs

    @torch.no_grad()
    def recognize(
            self,
            video_inputs: Tensor,
            video_input_lengths: Tensor,
            audio_inputs: Tensor,
            audio_input_lengths: Tensor,
    ):
        # pdb.set_trace()
        video_inputs = self.visual_frontend(video_inputs)
        video_inputs = video_inputs.squeeze(3)
        video_inputs = video_inputs.squeeze(3)
        video_inputs = video_inputs.permute(0,2,1)
        vid_encoder_outputs, vid_output_lengths,_ = self.video_encoder(video_inputs, video_input_lengths)
        aud_encoder_outputs, aud_output_lengths,_ = self.audio_encoder(audio_inputs, audio_input_lengths)
        #B T F --> B F T
        vid_encoder_outputs = vid_encoder_outputs.permute(0,2,1)
        aud_encoder_outputs = aud_encoder_outputs.permute(0,2,1)
        vid_encoder_outputs_upsample = torch.nn.functional.interpolate(
                vid_encoder_outputs,
                aud_encoder_outputs.size(2)
                )
            
        aud_encoder_outputs = aud_encoder_outputs.permute(0,2,1)
        vid_encoder_outputs_upsample = vid_encoder_outputs_upsample.permute(0,2,1)
        fusion_encoder_out = torch.cat((aud_encoder_outputs,vid_encoder_outputs_upsample),2)
        fusion_encoder_out = self.fc1(fusion_encoder_out)
        return self.decoder.decode(fusion_encoder_out, aud_output_lengths)