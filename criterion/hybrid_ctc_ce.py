import torch.nn as nn
from typing import Tuple
from torch import Tensor

from criterion.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyLoss


class JointCTCCrossEntropyLoss(nn.Module):

    def __init__(
            self,
            num_classes: int,                     # the number of classfication
            ignore_index: int,                    # indexes that are ignored when calcuating loss
            dim: int = -1,                        # dimension of caculation loss
            reduction='mean',                     # reduction method [sum, mean]
            ctc_weight: float = 0.3,              # weight of ctc loss
            cross_entropy_weight: float = 0.7,    # weight of cross entropy loss
            blank_id: int = None,                 # identification of blank token
            smoothing: float = 0.1,               # ratio of smoothing (confidence = 1.0 - smoothing)
    ) -> None:
        super(JointCTCCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.ignore_index = ignore_index
        self.reduction = reduction.lower()
        self.ctc_weight = ctc_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction=self.reduction, zero_infinity=True)
        if smoothing > 0.0:
            self.cross_entropy_loss = LabelSmoothedCrossEntropyLoss(
                num_classes=num_classes,
                ignore_index=ignore_index,
                smoothing=smoothing,
                reduction=reduction,
                dim=-1,
            )
        else:
            self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=self.ignore_index)

    def forward(
            self,
            encoder_log_probs: Tensor,
            decoder_log_probs: Tensor,
            output_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ctc_loss = self.ctc_loss(encoder_log_probs, targets, output_lengths, target_lengths)
        cross_entropy_loss = self.cross_entropy_loss(decoder_log_probs, targets.contiguous().view(-1))
        loss = cross_entropy_loss * self.cross_entropy_weight + ctc_loss * self.ctc_weight
        return loss, ctc_loss, cross_entropy_loss