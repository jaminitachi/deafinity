import os
import math
import torch
import random
from omegaconf import DictConfig
from torch.utils.data import Dataset
from dataloader.vocabulary import Vocabulary
from dataloader.augment import SpecAugment
from dataloader.feature import MelSpectrogram,MFCC,Spectrogram,FilterBank
from torch import Tensor, FloatTensor
import torchaudio
import numpy as np
import pdb
import csv
import sys
from astropy.modeling import ParameterError
from numpy.lib.stride_tricks import as_strided
import warnings
import librosa

def load_dataset(transcripts_path):
    """
    Provides dictionary of filename and labels
    Args:
        transcripts_path (str): path of transcripts
    Returns: target_dict
        - **target_dict** (dict): dictionary of filename and labels
    """
    video_paths = list()
    audio_paths = list()
    korean_transcripts = list()
    transcripts = list()

    with open(transcripts_path) as f:
        for idx, line in enumerate(f.readlines()):
            # pdb.set_trace()
            video_path, audio_path, korean_transcript, transcript = line.split('\t')
            transcript = transcript.replace('\n', '')
            video_paths.append(video_path)
            audio_paths.append(audio_path)
            korean_transcripts.append(korean_transcript)
            transcripts.append(transcript)

    return video_paths, audio_paths, korean_transcripts, transcripts

def prepare_dataset(config, transcripts_path: str, vocab: Vocabulary, Train=True):

    print("prepare dataset start !!")

    tr_video_paths, tr_audio_paths, tr_korean_transcripts, tr_transcripts = load_dataset(transcripts_path)

    if Train == True:
        trainset = AV_Dataset(
                video_paths=tr_video_paths,
                audio_paths=tr_audio_paths,
                korean_transcripts=tr_korean_transcripts,
                transcripts=tr_transcripts,
                sos_id=vocab.sos_id, 
                eos_id=vocab.eos_id,
                config=config,
                spec_augment = config.audio.spec_augment,
                )
    
    elif Train == False:
        trainset = AV_Dataset(
            video_paths=tr_video_paths,
            audio_paths=tr_audio_paths,
            korean_transcripts=tr_korean_transcripts,
            transcripts=tr_transcripts,
            sos_id=vocab.sos_id, 
            eos_id=vocab.eos_id,
            config=config, 
            spec_augment = False,
            )
            
    return trainset



class AV_Dataset(Dataset):
    
    
    def __init__(
            self,
            video_paths: list,              # list of video paths
            audio_paths: list,              # list of audio paths
            korean_transcripts: list,
            transcripts: list,              # list of transcript paths
            sos_id: int,                    # identification of start of sequence token
            eos_id: int,                    # identification of end of sequence token
            config,             # set of arguments
            spec_augment: bool = False,     # flag indication whether to use spec-augmentation of not
            ):
        super(AV_Dataset, self).__init__()
        
        if config.audio.transform_method.lower() == 'fbank':
            self.transforms = FilterBank(config.audio.sample_rate, 
                                            config.audio.n_mels, 
                                            config.audio.frame_length, 
                                            config.audio.frame_shift,
                                            )


        self.video_paths = list(video_paths)
        self.audio_paths = list(audio_paths)
        self.korean_transcripts = list(korean_transcripts)
        self.transcripts = list(transcripts)
        self.dataset_size = len(self.audio_paths)

        self.sos_id=sos_id
        self.eos_id=eos_id
        self.normalize = config.audio.normalize

        self.VANILLA = 0           # Not apply augmentation
        self.SPEC_AUGMENT = 1      # SpecAugment
        self.augment_methods = [self.VANILLA] * len(self.audio_paths)

        self.spec_augment = SpecAugment(config.audio.freq_mask_para, 
                                    config.audio.time_mask_num, 
                                    config.audio.freq_mask_num,
                                    )

        self._augment(spec_augment)
        self.shuffle()


    def parse_audio(self, audio_path: str, augment_method):
        # pdb.set_trace()
        signal, _ = librosa.load(audio_path,sr = 16000, mono=True)

        feature = self.transforms(signal)
        
        if self.normalize:
            feature -= feature.mean()
            feature /= np.std(feature)

        feature = FloatTensor(feature).transpose(0, 1)

        if augment_method == self.SPEC_AUGMENT:
            feature = self.spec_augment(feature)

        return feature
    
    def parse_video(self, video_path: str):
        video = np.load(video_path)
        video = torch.from_numpy(video).float()

        video -= torch.mean(video)
        video /= torch.std(video)
        video_feature  = video
        # video_feature = video_feature.permute(3,0,1,2) #T H W C --> C T H W
        return video_feature


    def __getitem__(self, index):
        video_feature = self.parse_video(self.video_paths[index])
        audio_feature = self.parse_audio(self.audio_paths[index],self.augment_methods)
        transcript = self.parse_transcript(self.transcripts[index])
        korean_transcript = self.parse_korean_transcripts(self.korean_transcripts[index])
        return video_feature, audio_feature, transcript, korean_transcript

    def parse_transcript(self, transcript):
        tokens = transcript.split(' ')
        transcript = list()

        transcript.append(int(self.sos_id))
        for token in tokens:
            transcript.append(int(token))
        transcript.append(int(self.eos_id))

        return transcript
    
    def parse_korean_transcripts(self, korean_transcript):
        
        tokens = korean_transcript.split(' ')
        korean_transcript = list()

        korean_transcript.append(str(self.sos_id))
        for token in tokens:
            korean_transcript.append(str(token))
        korean_transcript.append(str(self.eos_id))

        return korean_transcript

    def _augment(self, spec_augment):
        """ Spec Augmentation """
        if spec_augment:
            print("Applying Spec Augmentation...")

            for idx in range(self.dataset_size):
                self.augment_methods.append(self.SPEC_AUGMENT)
                self.video_paths.append(self.video_paths[idx])
                self.audio_paths.append(self.audio_paths[idx])
                self.korean_transcripts.append(self.korean_transcripts[idx])
                self.transcripts.append(self.transcripts[idx])

    def shuffle(self):
        """ Shuffle dataset """
        tmp = list(zip(self.video_paths,self.audio_paths,self.korean_transcripts, self.transcripts, self.augment_methods))
        random.shuffle(tmp)
        self.video_paths,self.audio_paths, self.korean_transcripts, self.transcripts, self.augment_methods = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)


def _collate_fn(batch):
    
    """ functions that pad to the maximum sequence length """
    def vid_length_(p):
        return len(p[0])

    def seq_length_(p):
        return len(p[1])

    def target_length_(p):
        return len(p[2])
    
    # sort by sequence length for rnn.pack_padded_sequence()
    # pdb.set_trace()
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    vid_lengths = [len(s[0]) for s in batch]

    seq_lengths = [len(s[1]) for s in batch]
    target_lengths = [len(s[2]) - 1 for s in batch]

    max_vid_sample = max(batch, key=vid_length_)[0]
    max_seq_sample = max(batch, key=seq_length_)[1]
    max_target_sample = max(batch, key=target_length_)[2]

    max_vid_size = max_vid_sample.size(0)
    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)
    
    vid_feat_x = max_vid_sample.size(1)
    vid_feat_y = max_vid_sample.size(2)
    vid_feat_c = max_vid_sample.size(3)
    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    vids = torch.zeros(batch_size, max_vid_size, vid_feat_x,vid_feat_y,vid_feat_c)
    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(0)
    # pdb.set_trace()
    for x in range(batch_size):
        sample = batch[x]
        video_ = sample[0]
        tensor = sample[1]
        target = sample[2]

        vid_length = video_.size(0)
        seq_length = tensor.size(0)
        vids[x,:vid_length,:,:,:] = video_
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    vid_lengths = torch.IntTensor(vid_lengths)
    seq_lengths = torch.IntTensor(seq_lengths)
    #B T W H C -->B C T W H
    # pdb.set_trace()
    vids = vids.permute(0,4,1,2,3)
    return vids, seqs, targets, vid_lengths, seq_lengths, target_lengths



