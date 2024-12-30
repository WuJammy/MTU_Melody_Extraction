from email.mime import audio
import imp
from os import read
import os
from matplotlib.pyplot import flag, plot
from pydub import AudioSegment
import librosa
import numpy as np
import math
from pyparsing import col
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import medleydb as mdb
from pedalboard import Pedalboard, Reverb, Chorus, Gain
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

p_flag = 0

class Audio_dataset(Dataset):

    def __init__(self, audio_names, config, augmentation,dataset_name):

        if dataset_name == 'medleydb':
            self.audio_segments, self.label_segments = get_all_audio_and_label_segments(
                audio_names=audio_names,
                padding_audio_path=config.PADDING_AUDIO_PATH,
                audio_segment_time=config.SEGMENT.AUDIO_SEGMENT_TIME,
                overlap_time=config.SEGMENT.OVERLAP_TIME,
                label_segment_length=config.SPECTRUM.SHAPE[1],
                augmentation=augmentation)
        elif dataset_name == 'mir1k':
            self.audio_segments, self.label_segments = get_mir1k_audio_and_label_segments(
                audio_names=audio_names,
                padding_audio_path=config.PADDING_AUDIO_PATH,
                audio_folder=config.MIR1K.AUDIO_PATH,
                label_folder=config.MIR1K.LABEL_PATH,
                audio_segment_time=config.SEGMENT.AUDIO_SEGMENT_TIME,
                overlap_time=config.SEGMENT.OVERLAP_TIME,
                label_segment_length=65)
        elif dataset_name == 'orchset':
            self.audio_segments, self.label_segments = get_orchsset_audio_and_label_segments(
                audio_names=audio_names,
                padding_audio_path=config.PADDING_AUDIO_PATH,
                audio_folder=config.Orchset.AUDIO_PATH,
                label_folder=config.Orchset.LABEL_PATH,
                audio_segment_time=config.SEGMENT.AUDIO_SEGMENT_TIME,
                overlap_time=config.SEGMENT.OVERLAP_TIME,
                label_segment_length=130)
            
        self.config = config
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        global p_flag
        pitch_label = self.label_segments[idx]          

        cqt_frenquency = librosa.cqt_frequencies(
            n_bins=self.config.SPECTRUM.N_BINS,
            bins_per_octave=self.config.SPECTRUM.BINS_PER_OCTAVE,
            fmin=librosa.note_to_hz('C2'))
        
        if self.dataset_name == 'medleydb' :
            map_label = seq2map(pitch_label, cqt_frenquency)
        elif self.dataset_name == 'orchset':
            map_label = seq2map(pitch_label, cqt_frenquency)                  
            map_label = cv2.resize(map_label, (224, 224))
            
        elif self.dataset_name == 'mir1k':
            pitch_label = pitchmidi_to_hz(pitch_label) #midi to hz
            map_label = seq2map(pitch_label, cqt_frenquency)
    
             # label resize to 224*224
            map_label = cv2.resize(map_label, (224, 224))

        #map_label做guassian blur
        blur_label = np.zeros_like(map_label)
        
        for i in range(map_label.shape[1]):
            column = map_label[:, i]
            blur_column = cv2.GaussianBlur(column.reshape(-1, 1), (1, 3), sigmaX=0.5, sigmaY=1)
            blur_label[:, i] = blur_column.flatten()

        map_label = blur_label

        


        # map_label = cv2.GaussianBlur(map_label, (1, 5),sigmaX=0.5, sigmaY=1.1)
        # map_label = F.softmax(torch.tensor(map_label/3),dim=0).numpy()
        # map_label = cv2.resize(map_label, (224, 224))

        audio_y, audio_sr = self.audio_segments[idx], 44100
        
        h = [0.5,1, 2, 3, 4, 5]
        audio_hcqt = []
        for i in h:
            audio_hcqt.append(
                librosa.cqt(y=audio_y,
                            sr=audio_sr,
                            fmin=librosa.note_to_hz('C2') * i,
                            n_bins=self.config.SPECTRUM.N_BINS,
                            bins_per_octave=self.config.SPECTRUM.BINS_PER_OCTAVE,
                            hop_length=self.config.SPECTRUM.HOP_LENGTH))
        change_type_audio_hcqt = np.transpose(audio_hcqt, (1, 2, 0))
        power_to_db = librosa.power_to_db((np.abs(change_type_audio_hcqt)))

        sample = {'audio': power_to_db, 'label': map_label}

        return sample


class Audio_test_dataset(Dataset):

    def __init__(self, audio_segments, config):
        self.audio_segments = audio_segments

        self.config = config

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        audio_segment = self.audio_segments[idx]

        h = [0.5, 1, 2, 3, 4, 5]
        audio_hcqt = []
        for i in h:
            audio_hcqt.append(
                librosa.cqt(y=audio_segment,
                            sr=44100,
                            fmin=librosa.note_to_hz('C2') * i,
                            n_bins=self.config.SPECTRUM.N_BINS,
                            bins_per_octave=self.config.SPECTRUM.BINS_PER_OCTAVE,
                            hop_length=self.config.SPECTRUM.HOP_LENGTH))
        change_type_audio_hcqt = np.transpose(audio_hcqt, (1, 2, 0))
        power_to_db = librosa.power_to_db((np.abs(change_type_audio_hcqt)))

        # resize_audio = np.zeros((224, 224, 6))
        # for i in range(resize_audio.shape[2]):
        #     resize_audio[:, :, i] = cv2.resize(power_to_db[:, :, i], (224, 224))

        # return resize_audio
        return power_to_db


class DiceLoss(nn.Module):

    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)  # 修改!!!我們不需要one-hot
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(
            inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def pitchmidi_to_hz(pitches_midi):
    pitches_hz  = librosa.midi_to_hz(pitches_midi)

    for i in range(len(pitches_hz)):
        if pitches_hz[i] < 10 :
            pitches_hz[i] = 0.0

    return pitches_hz

def audio_start_time_to_label_start_index(start_time, label_sampling_time):
    '''
    輸入一時間點及melody sampling rate，輸出含此時間點所包含的第一個melody label的index

    start_time: 時間點 (單位: ms)
    label_sampling_time: melody sampling rate (單位: s/個)
    '''

    return math.ceil(float(start_time / 1000) / label_sampling_time)


def spectrum_to_pitches(spectrum, config):
    '''
    輸入一個spectrum，輸出此spectrum內含的所有pitches的list
    '''
    cqt_frequencies = librosa.cqt_frequencies(n_bins=config.SPECTRUM.N_BINS,
                                              fmin=librosa.note_to_hz('C2'),
                                              bins_per_octave=config.SPECTRUM.BINS_PER_OCTAVE)

    pitches_index = np.argmax(spectrum, axis=0)

    return list(
        map(
            lambda pitch_index: 0.0 if cqt_frequencies[pitch_index] == librosa.note_to_hz('C2') else
            cqt_frequencies[pitch_index], pitches_index))


def pitches_to_notes(pitches):
    '''
    pitch list轉note list。若為unvoiced (pitch為0.0)則note為「_」
    '''
    return list(
        map(lambda pitch: '_'
            if pitch == 0.0 else librosa.hz_to_note(pitch).replace('♯', '#'), pitches))


def mark_timestamps(data, interval):
    return np.column_stack((np.linspace(0.0, (len(data) - 1) * interval,
                                        num=len(data)), np.array(data)))


def seq2map(seq, CenFreq):
    '''
    seq: 一個音檔的label的所有值
    CenFreq: 各種音高代表的頻率的表
    回傳: label的頻率與最近的音高頻率one-hot對應
    '''

    gtmap = np.zeros((len(CenFreq), len(seq)))  # 高為len(CenFreq), 寬為len(seq)
    #gtmap隨機出現0.1~0.6的數字
    # gtmap = np.random.rand(len(CenFreq), len(seq)) * 0.5 + 0.1

    for i in range(len(seq)):
        for j in range(len(CenFreq)):
            if seq[i] < 0.1:  # 在unvoiced的地方標註成C2
                gtmap[0, i] = 1

                break
            elif CenFreq[j] > seq[i]:
                gtmap[j, i] = 1

                break

    return gtmap


def get_medleydb_audio_names():
    audio_names = []

    melody_multitracks = mdb.load_melody_multitracks()

    for mtrack in melody_multitracks:
        audio_names.append(mtrack.track_id)

    # 移除一些壞掉的音檔
    # audio_names.remove('AimeeNorwich_Child')

    # folder_path = pathlib.Path('/home/ktpss97094/medleydb/Audio/')
    # for subdir in folder_path.iterdir():
    #     if subdir.is_dir():
    #         for file in subdir.glob('*.wav'):
    #             all_audio_file_name.append(file.stem)

    # audio_path = pathlib.Path(dataset_config[dataset_name]['audio_folder_path'])
    # for element in list((audio_path.glob("*.wav"))):
    #     all_audio_file_name.append(pathlib.Path(element).stem)

    return audio_names


def get_medleydb_vocal_audio_names():
    audio_names_vocal_path = '/home/wujammy/vocal_train.txt'

    audio_names = []

    with open(audio_names_vocal_path, 'r') as f:
        for line in f:
            audio_names.append(line.strip())

    #check the AimeeNorwich_Child is in the list or not
    # if 'AimeeNorwich_Child' in audio_names:
    #     audio_names.remove('AimeeNorwich_Child')


    return audio_names


def get_medleydb_train_audio_names():
    audio_names_non_vocal_path = '/home/wujammy/melody_extraction_swin/medleydb_train_names.txt'

    audio_names = []

    with open(audio_names_non_vocal_path, 'r') as f:
        for line in f:
            audio_names.append(line.strip())

    return audio_names

def get_medleydb_test_audio_names():
    audio_names_non_vocal_path = '/home/wujammy/melody_extraction_swin/medleydb_test_names.txt'

    audio_names = []

    with open(audio_names_non_vocal_path, 'r') as f:
        for line in f:
            audio_names.append(line.strip())

    return audio_names

def get_medleydb_validation_audio_names():
    audio_names_non_vocal_path = '/home/wujammy/melody_extraction_swin/medleydb_valid_names.txt'

    audio_names = []

    with open(audio_names_non_vocal_path, 'r') as f:
        for line in f:
            audio_names.append(line.strip())

    return audio_names

def get_orchset_name():

    orchset_path = '/home/wujammy/Orchset'
    
    #get all audio names(in dir)
    audio_names = [os.path.splitext(f)[0] for f in os.listdir(orchset_path) if f.endswith('.wav')]
    
    return audio_names 

def get_all_audio_and_label_segments(audio_names, padding_audio_path, audio_segment_time: int,
                                     overlap_time: int, label_segment_length: int,
                                     augmentation: bool):
    '''
    輸入一個所有音檔名的list，輸出兩個list，為音檔segments及melody label的segments

    audio_names: list，存放所有要segment的音檔的名稱
    padding_audio_path: 用來作padding的音檔的路徑
    audio_segment_time: 一個audio segment的時間 (單位: ms)
    overlap_time: audio segments之間有重疊的時間 (單位: ms)
    label_segment_length: 一個label segment含的melody label數
    '''

    all_audio_segments, all_label_segments = [], []

    for audio_name in tqdm(audio_names, desc="Processing MedleyDB files"):
        audio_path = mdb.MultiTrack(audio_name).mix_path
        label_path = mdb.MultiTrack(audio_name).melody2_fpath

        tmp1, tmp2 = segment_audio_and_label(
            audio_path=audio_path,
            padding_audio_path=padding_audio_path,
            label_path=label_path,
            audio_segment_time=audio_segment_time,
            overlap_time=overlap_time,
            label_segment_length=
            label_segment_length,  # = ceil((audio_segment_time / 1000) / melody取樣間隔(label_sampling_time))
            augmentation=augmentation)

        all_audio_segments.extend(tmp1)
        all_label_segments.extend(tmp2)

    return all_audio_segments, all_label_segments


def segment_audio(audio_path, padding_audio_path, segment_time: int, overlap_time: int):
    '''
    將一個音檔切成多個audio segment

    audio_path: 要segment的音檔的路徑
    padding_audio_path: 用來作padding的音檔的路徑
    segment_time: 一個audio segment的時間 (單位: ms)
    overlap_time: audio segments之間有重疊的時間 (單位: ms)
    '''

    audio_segments = []
    input_audio = AudioSegment.from_file(audio_path)
    padding_audio = AudioSegment.from_file(padding_audio_path)

    # 若input音檔為multi-channel，則將channels combine
    if input_audio.channels != 1:
        monos = input_audio.split_to_mono()
        output = monos[0]

        for i in range(1, len(monos)):
            output = output.overlay(monos[i])

        input_audio = output

    # padding audio不能為多channel的音檔
    if padding_audio.channels != 1:
        raise NotImplementedError('padding audio不能為多channel的音檔')

    input_audio_time = len(input_audio)
    start_time = int(0)
    stop_time = segment_time

    while start_time < input_audio_time:
        # 切割wav檔
        audio_chunk = input_audio[start_time:stop_time]

        if len(audio_chunk) != segment_time:  # 最後一個segment可能會不滿segment_time時間，所以要做padding
            pad_time = segment_time - len(audio_chunk)

            merge_part = padding_audio[0:pad_time]

            # 調整聲音大小
            db = audio_chunk.dBFS - merge_part.dBFS
            merge_part += db

            # 合併音檔
            audio_chunk += merge_part

        # AudioSegment轉librosa格式
        samples = audio_chunk.get_array_of_samples()
        arr = np.array(samples).astype(np.float32) / 32768
        arr = librosa.core.resample(y=arr,
                                    orig_sr=audio_chunk.frame_rate,
                                    target_sr=44100,
                                    res_type='kaiser_best')

        audio_segments.append(arr)

        start_time = stop_time - overlap_time
        stop_time = start_time + segment_time

    return audio_segments


def segment_audio_and_label(audio_path, padding_audio_path, label_path, audio_segment_time: int,
                            overlap_time: int, label_segment_length: int, augmentation: bool):
    '''
    將一個音檔及他的melody label切成多個audio segment及對應的label

    audio_path: 要segment的音檔的路徑
    padding_audio_path: 用來作padding的音檔的路徑
    label_path: 要segment的音檔的melody label的路徑 (label檔的格式為column 0為timestamp (單位: s)、column 1為melody frequency)
    audio_segment_time: 一個audio segment的時間 (單位: ms)
    overlap_time: audio segments之間有重疊的時間 (單位: ms)
    label_segment_length: 一個label segment含的melody label數
    augmentation: 是否要做augmentation
    '''

    audio_segments, label_segments = [], []
    input_audio = AudioSegment.from_file(audio_path)
    padding_audio = AudioSegment.from_file(padding_audio_path)
    input_label = np.genfromtxt(
        label_path, delimiter=',',
        dtype=float).T  # .T: 作transpose，作transpose之後input_label[0]為time、input_label[1]為frequency

    # 若input音檔為multi-channel，則將channels combine
    if input_audio.channels != 1:
        monos = input_audio.split_to_mono()
        output = monos[0]

        for i in range(1, len(monos)):
            output = output.overlay(monos[i])

        input_audio = output

    # padding audio不能為多channel的音檔
    if padding_audio.channels != 1:
        raise NotImplementedError('padding audio不能為多channel的音檔')

    input_audio_time = len(input_audio)
    start_time = int(0)
    stop_time = audio_segment_time

    while start_time < input_audio_time:
        # 切割音檔
        audio_chunk = input_audio[start_time:stop_time]

        # 切割label檔
        label_sampling_time = input_label[0][1] - input_label[0][0]
        start_index = audio_start_time_to_label_start_index(start_time, label_sampling_time)
        label_chunk = input_label[1][
            start_index:start_index +
            label_segment_length]  # FIXME: medleydb不會有這問題，但如果label數比實際音檔長度少的話可以做padding 0.0

        if len(audio_chunk) != audio_segment_time:  # 最後一個segment可能會不滿segment_time時間，所以要做padding
            pad_time = audio_segment_time - len(audio_chunk)

            merge_part = padding_audio[0:pad_time]

            # 調整聲音大小
            db = audio_chunk.dBFS - merge_part.dBFS
            merge_part += db

            # padding audio_chunk
            audio_chunk += merge_part

            # padding label_chunk
            label_chunk = np.append(label_chunk, np.zeros(label_segment_length - len(label_chunk)))

        # AudioSegment轉librosa格式
        samples = audio_chunk.get_array_of_samples()
        arr = np.array(samples).astype(np.float32) / 32768
        arr = librosa.core.resample(y=arr,
                                    orig_sr=audio_chunk.frame_rate,
                                    target_sr=44100,
                                    res_type='kaiser_best')

        audio_segments.append(arr)
        label_segments.append(label_chunk)

        if augmentation:
            board = Pedalboard([
                Reverb(wet_level=float(torch.rand(1))),
                Chorus(mix=float(torch.rand(1))),
                Gain(gain_db=10 * float(torch.rand(1))),
            ])

            arr_augmentation = board(arr, 44100)


            audio_segments.append(arr_augmentation)
            label_segments.append(label_chunk)

        start_time = stop_time - overlap_time
        stop_time = start_time + audio_segment_time



    return audio_segments, label_segments

# mir1k dataset
def get_mir1k_audio_names(mik1k_folder_path):
    mir1k_path = mik1k_folder_path
    
    audio_names = [file_name.stem for file_name in Path(mir1k_path).iterdir()]

    return audio_names

def mir1k_segment_audio_and_label(audio_path, padding_audio_path, label_path, audio_segment_time: int,
                            overlap_time: int, label_segment_length: int):
    
    audio_segments, label_segments = [], []
    input_audio = AudioSegment.from_file(audio_path) 
    padding_audio = AudioSegment.from_file(padding_audio_path)
    input_label = np.genfromtxt(
        label_path, delimiter=',',
        dtype=float)
    
      # 若input音檔為multi-channel，則將channels combine
    if input_audio.channels != 1:
        monos = input_audio.split_to_mono()
        output = monos[0]

        for i in range(1, len(monos)):
            output = output.overlay(monos[i])

        input_audio = output
    
    input_audio_time = len(input_audio)
    start_time = int(0)
    stop_time = audio_segment_time
    
    while start_time < input_audio_time:
        # 切割音檔
        audio_chunk = input_audio[start_time:stop_time]

        # 切割label檔
        label_sampling_time = 0.02
        start_index = audio_start_time_to_label_start_index(start_time, label_sampling_time)
        label_chunk = input_label[start_index:start_index + label_segment_length]

    

        if len(audio_chunk) != audio_segment_time:  # 最後一個segment可能會不滿segment_time時間，所以要做padding
            pad_time = audio_segment_time - len(audio_chunk)

            merge_part = padding_audio[0:pad_time]

            # 調整聲音大小
            db = audio_chunk.dBFS - merge_part.dBFS
            merge_part += db

            # padding audio_chunk
            audio_chunk += merge_part

            # padding label_chunk
            label_chunk = np.append(label_chunk, np.zeros(label_segment_length - len(label_chunk)))


        # AudioSegment轉librosa格式
        samples = audio_chunk.get_array_of_samples()
        arr = np.array(samples).astype(np.float32) / 32768
        arr = librosa.core.resample(y=arr,
                                    orig_sr=audio_chunk.frame_rate,
                                    target_sr=44100,
                                    res_type='kaiser_best')
    
        audio_segments.append(arr)
        label_segments.append(label_chunk)

        start_time = stop_time - overlap_time
        stop_time = start_time + audio_segment_time

    return audio_segments, label_segments

def get_mir1k_audio_and_label_segments(audio_names, padding_audio_path, audio_folder, label_folder, audio_segment_time: int,
                            overlap_time: int, label_segment_length: int):
 
    audio_segments, label_segments = [], []

    for audio_name in tqdm(audio_names,desc="Processing MIR1K files"):
        audio_path = audio_folder + '/' + audio_name + '.wav'
        label_path = label_folder + '/' + audio_name + '.pv'

        tmp1, tmp2 = mir1k_segment_audio_and_label(
            audio_path=audio_path,
            padding_audio_path=padding_audio_path,
            label_path=label_path,
            audio_segment_time=audio_segment_time,
            overlap_time=overlap_time,
            label_segment_length=label_segment_length,)

        audio_segments.extend(tmp1)
        label_segments.extend(tmp2)

    return audio_segments, label_segments

def orchsset_segment_audio_and_label(audio_path, padding_audio_path, label_path, audio_segment_time: int,
                            overlap_time: int, label_segment_length: int):
    
    audio_segments, label_segments = [], []
    input_audio = AudioSegment.from_file(audio_path) 
    padding_audio = AudioSegment.from_file(padding_audio_path)
    input_label = np.loadtxt(label_path,dtype=float)
    
      # 若input音檔為multi-channel，則將channels combine
    if input_audio.channels != 1:
        monos = input_audio.split_to_mono()
        output = monos[0]

        for i in range(1, len(monos)):
            output = output.overlay(monos[i])

        input_audio = output
    
    input_audio_time = len(input_audio)
    start_time = int(0)
    stop_time = audio_segment_time
    
    while start_time < input_audio_time:
        # 切割音檔
        audio_chunk = input_audio[start_time:stop_time]

        # 切割label檔
        label_sampling_time = 0.01
        start_index = audio_start_time_to_label_start_index(start_time, label_sampling_time)
        label_chunk = input_label[start_index:start_index + label_segment_length,1]

        if len(audio_chunk) != audio_segment_time or label_chunk.shape[0] != label_segment_length:  # 最後一個segment可能會不滿segment_time時間，所以要做padding
            pad_time = audio_segment_time - len(audio_chunk)

            merge_part = padding_audio[0:pad_time]

            # 調整聲音大小
            db = audio_chunk.dBFS - merge_part.dBFS
            merge_part += db

            # padding audio_chunk
            audio_chunk += merge_part

            # padding label_chunk
            label_chunk = np.append(label_chunk, np.zeros(label_segment_length - len(label_chunk)))

        # AudioSegment轉librosa格式
        samples = audio_chunk.get_array_of_samples()
        arr = np.array(samples).astype(np.float32) / 32768
        arr = librosa.core.resample(y=arr,
                                    orig_sr=audio_chunk.frame_rate,
                                    target_sr=44100,
                                    res_type='kaiser_best')
        audio_segments.append(arr)
        label_segments.append(label_chunk)

        start_time = stop_time - overlap_time
        stop_time = start_time + audio_segment_time

    return audio_segments, label_segments


def get_orchsset_audio_and_label_segments(audio_names, padding_audio_path, audio_folder, label_folder, audio_segment_time: int,
                            overlap_time: int, label_segment_length: int):
 
    audio_segments, label_segments = [], []

    for audio_name in tqdm(audio_names,desc="Processing Orchset files"):
        audio_path = audio_folder + '/' + audio_name + '.wav'
        label_path = label_folder + '/' + audio_name + '.mel'

        tmp1, tmp2 = orchsset_segment_audio_and_label(
            audio_path=audio_path,
            padding_audio_path=padding_audio_path,
            label_path=label_path,
            audio_segment_time=audio_segment_time,
            overlap_time=overlap_time,
            label_segment_length=label_segment_length,)

        audio_segments.extend(tmp1)
        label_segments.extend(tmp2)

    return audio_segments, label_segments