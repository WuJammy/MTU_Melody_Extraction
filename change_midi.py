import librosa
from midiutil.MidiFile import MIDIFile

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