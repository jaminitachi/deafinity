import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, FloatTensor
from dataloader.vocabulary import KsponSpeechVocabulary,Vocabulary
from metric.metric import CharacterErrorRate
from tqdm import tqdm
from skimage import io
import numpy as np
import torchaudio
import skvideo.io
import glob
import librosa
import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_audio(audio_path: str):
    # pdb.set_trace()
    signal, _ = librosa.load(audio_path,sr = 16000, mono=True)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    feature = FloatTensor(feature).transpose(0, 1)
    return feature
    
def parse_video(video_path: str):
    
    # reader = skvideo.io.FFmpegReader(video_path)
    # video = []
    # for frame in reader.nextFrame(): 
    #     video.append(frame)
    # pdb.set_trace()
    # video = np.array(video)
    video = np.load(video_path)
    video = torch.from_numpy(video).float()

    video -= torch.mean(video)
    video /= torch.std(video)
    video_feature  = video
    video_feature = video_feature.permute(3,0,1,2) #T H W C --> C T H W
    return video_feature

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= '4'
    multi_gpu = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = KsponSpeechVocabulary('dataset/labels.csv')
    test_model = "./temp_model/Docker_model.pt"
    model = torch.load(test_model, map_location=lambda storage, loc: storage).to(device)

    if multi_gpu : 
        model = nn.DataParallel(model)
    model.eval()
    val_metric = CharacterErrorRate(vocab)
    print(model)
    print(count_parameters(model))
    model.eval()
    
    mp4_lists = glob.glob('./temp_data/*.npy')
    mp4_lists = sorted(mp4_lists)

    mp3_lists = glob.glob('./temp_data/*.mp3')
    mp3_lists = sorted(mp3_lists)
    print('Inference start!!!')

    if len(mp3_lists) != len(mp4_lists):
        print("Error!!!!!!!!!!")
        pdb.set_trace()
    for i in range(len(mp4_lists)):
        # pdb.set_trace()
        audio_inputs = parse_audio(mp3_lists[i])
        audio_input_lengths = torch.IntTensor([audio_inputs.shape[0]])
        video_inputs = parse_video(mp4_lists[i])
        video_input_lengths = torch.IntTensor([video_inputs.shape[1]])

        audio_inputs = audio_inputs.unsqueeze(0)
        video_inputs = video_inputs.unsqueeze(0)

        video_inputs = video_inputs.to(device)
        audio_inputs = audio_inputs.to(device)
        video_input_lengths = video_input_lengths.to(device)
        audio_input_lengths = audio_input_lengths.to(device)

        outputs = model.recognize(video_inputs, 
                                video_input_lengths, 
                                audio_inputs,
                                audio_input_lengths,
                                )
        # pdb.set_trace()
        y_hats = outputs.max(-1)[1]
        sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
        print(sentence)
    
    

