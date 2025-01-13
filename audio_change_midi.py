import librosa
from midiutil.MidiFile import MIDIFile
import argparse
from mamba_transformer_unet.predict_model import melody_extraction_mamba_transformer_unet_model
from mamba_transformer_unet.config import get_test_config as get_mamba_transformer_unet_test_config
import numpy as np
from pydub import AudioSegment
from utils import spectrum_to_pitches

def change_midifile(hz_array, output_path):    

    # hz to note(librosa) if hz = 0 then midi = -1
    for i in range(len(hz_array)):
        if hz_array[i] == 0:
                 hz_array[i] = 0
        else:
            hz_array[i] = int(librosa.hz_to_midi(hz_array[i]))

    #find same note(duration)
    notes = []
    note_start_time = 0
    now = 0
    note_start_index = 0

    while True:
        # 當前音符頻率
        same_note = hz_array[now]
        # 尋找所有與當前音符相同的頻率並累加到 now
        for i in range(now, len(hz_array)):
            if hz_array[i] == same_note:
                now = i + 1
            else:
                break

        # 計算音符持續時間
        duration = (now - note_start_index) * 0.0058

        # 記錄音符和其對應的時間段
        notes.append([same_note, note_start_time, note_start_time + duration])

        # 更新音符開始時間
        note_start_index = now
        note_start_time += duration

        # 如果處理到最後，結束
        if now >= len(hz_array):
            break

    # Create the MIDIFile Object
    MyMIDI = MIDIFile(1)
    track = 0
    time = 0
    MyMIDI.addTrackName(track, time, "Sample Track")
    MyMIDI.addTempo(track, time, 60)
    # Add a note. addNote expects the following information:
    channel = 0
    pitch = 0
    time = 0
    volume = 127
    # Now add the note.
    for i in range(len(notes)):
        if notes[i][0] == 0:
            continue
        pitch = int(notes[i][0])
        onset = notes[i][1]
        duration = notes[i][2] - notes[i][1]
        MyMIDI.addNote(track, channel, pitch, onset, duration, volume)

    #change instrument
    MyMIDI.addProgramChange(track, channel, time, 71)

    # And write it to disk.
    binfile = open(output_path, 'wb')
    MyMIDI.writeFile(binfile)
    binfile.close()
    
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--output_midi_file_dir', type=str, required=True)
parser.add_argument('--audio_path', type=str, required=True)

args = parser.parse_args()

config = get_mamba_transformer_unet_test_config(args)
model = melody_extraction_mamba_transformer_unet_model(config)

model_path = args.model_path
output_midi_file_dir = args.output_midi_file_dir
audio_path = args.audio_path

audio_file_name = audio_path.split('/')[-1].split('.')[0]

spectrums = model.predict(audio_path)

start_time = int(0)
stop_time = config.SEGMENT.AUDIO_SEGMENT_TIME
i = 0
input_audio_time = len(AudioSegment.from_file(audio_path))
total_audio_len_number = int(((input_audio_time/1000)/0.0058))

merge_spectrum = []

for i in range(spectrums.shape[0]):
    if i == 0:
        merge_spectrum = spectrums[i]
        
    else:
        
        merge_spectrum = np.concatenate((merge_spectrum, spectrums[i, :, 52:]), axis=1)


    # merge_spectrum = np.concatenate((merge_spectrum, spectrums[i]), axis=1) 

print(merge_spectrum.shape)

prediction = np.array(spectrum_to_pitches(merge_spectrum, config))


change_midifile(prediction, f'/home/wujammy/melody_extraction_swin/{audio_file_name}.mid')    



