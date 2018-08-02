
# coding: utf-8

# In[1]:


####Ryan Blow


# In[2]:


from microphone import record_audio
import numpy as np
listen_time=5
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion 
from scipy.ndimage.morphology import iterate_structure
import pickle
from collections import Counter


# In[3]:


notes={}
notes["C_0"]=16.351
notes["C#/Db_0"]=17.324
notes["D_0"]=18.354
notes["D#/Eb_0"]=19.445
notes["E_0"]=20.601
notes["F_0"]=21.827
notes["F#/Gb_0"]=23.124
notes["G_0"]=24.499
notes["G#/Ab_0"]=25.956
notes["A_0"]=27.5
notes["A#/Bb_0"]=29.135
notes["B_0"]=30.868
notes["C_1"]=32.703
notes["C#/Db_1"]=34.648
notes["D_1"]=36.708
notes["D#/Eb_1"]=38.891
notes["E_1"]=41.203
notes["F_1"]=43.654
notes["F#/Gb_1"]=46.249
notes["G_1"]=48.999
notes["G#/Ab_1"]=51.913
notes["A_1"]=55
notes["A#/Bb_1"]=58.27
notes["B_1"]=61.735
notes["C_2"]=65.406
notes["C#/Db_2"]=69.296
notes["D_2"]=73.416
notes["D##/Eb_2"]=77.782
notes["E_2"]=82.407
notes["F_2"]=87.307
notes["F#/Gb_2"]=92.499
notes["G_2"]=97.999
notes["G#/Ab_2"]=103.826
notes["A_2"]=110
notes["A#/Bb_2"]=116.541
notes["B_2"]=123.471
notes["C_3"]=130.813
notes["C#/Db_3"]=138.591
notes["D_3"]=146.832
notes["D#/Eb_3"]=155.563
notes["E_3"]=164.814
notes["F_3"]=174.614
notes["F#/Gb_3"]=184.997
notes["G_3"]=195.998
notes["G#/Ab_3"]=207.652
notes["A_3"]=220
notes["A#/Bb_3"]=233.082
notes["B_3"]=246.942
notes["C_4"]=261.626
notes["C#/Db_4"]=277.183
notes["D_4"]=293.665
notes["D#/Eb_4"]=311.127
notes["E_4"]=329.628
notes["F_4"]=349.228
notes["F#/Gb_4"]=369.994
notes["G_4"]=391.995
notes["G#/Ab_4"]=415.305
notes["A_4"]=440
notes["A#/Bb_4"]=466.164
notes["B_4"]=493.883
notes["C_5"]=523.251
notes["C#/Db_5"]=554.365
notes["D_5"]=587.33
notes["D#/Eb_5"]=622.254
notes["E_5"]=659.255
notes["F_5"]=698.456
notes["F#/Gb_5"]=739.989
notes["G_5"]=783.991
notes["G#/Ab_5"]=830.609
notes["A_5"]=880
notes["A#/Bb_5"]=932.328
notes["B_5"]=987.767
notes["C_6"]=1046.502
notes["C#/Db_6"]=1108.731
notes["D_6"]=1174.659
notes["D#/Eb_6"]=1244.508
notes["E_6"]=1318.51
notes["F_6"]=1396.913
notes["F#/Gb_6"]=1479.978
notes["G_6"]=1567.982
notes["G#/Ab_6"]=1661.219
notes["A_6"]=1760
notes["A#/Bb_6"]=1864.655
notes["B_6"]=1975.533
notes["C_7"]=2093.005
notes["C#/Db_7"]=2217.461
notes["D_7"]=2349.318
notes["D#/Eb_7"]=2489.016
notes["E_7"]=2637.021
notes["F_7"]=2793.826
notes["F#/Gb_7"]=2959.955
notes["G_7"]=3135.964
notes["G#/Ab_7"]=3322.438
notes["A_7"]=3520
notes["A#/Bb_7"]=3729.31
notes["B_7"]=3951.066
notes["C_8"]=4186.009
notes["C#/Db_8"]=4434.922
notes["D_8"]=4698.636
notes["D#/Eb_8"]=4978.032
notes["E_8"]=5274.042
notes["F_8"]=5587.652
notes["F#/Gb_8"]=5919.91
notes["G_8"]=6271.928
notes["G#/Ab_8"]=6644.876
notes["A_8"]=7040
notes["A#/Bb_8"]=7458.62
notes["B_8"]=7902.132
notes["C_9"]=8372.018
notes["C#/Db_9"]=8869.844
notes["D_9"]=9397.272
notes["D#/Eb_9"]=9956.064
notes["E_9"]=10548.084
notes["F_9"]=11175.304
notes["F#/Gb_9"]=11839.82
notes["G_9"]=12543.856
notes["G#/Ab_9"]=13289.752
notes["A_9"]=14080
notes["A#/Bb_9"]=14917.24
notes["B_9"]=15804.264
with open("notes.pkl", mode="wb") as opened_file:
        pickle.dump(notes, opened_file)


# In[4]:


def recognize_notes():
    frames, sample_rate = record_audio(listen_time)
    audio_data = np.hstack([np.frombuffer(i, np.int16) for i in frames])
    time = np.arange(len(audio_data)) * sample_rate # corresponding time (sec) for each sample
    #fig, ax = plt.subplots()
    sampling_rate=44100
    fig, ax = plt.subplots()

    S, freqs, times, im = ax.specgram(audio_data, NFFT=4096, Fs=sampling_rate,
                                  window=mlab.window_hanning,
                                  noverlap=4096 // 2)
    ax.set_ylim(0,2500)
    sum_S=S.sum(axis=1)
    freq=[]
    print(sum_S.shape)
    for i in range(len(sum_S)):
        if sum_S[i]>200000:
            print(sum_S[i])
            print(i)
            freq_bins=freqs[i]
            freq.append(freq_bins)
    with open("notes.pkl", mode="rb") as opened_file:
        notes_dic = pickle.load(opened_file)
    notes_reversed = dict(map(reversed, notes_dic.items()))
    notes_list_values=[i for i in notes_dic.values()]
    notes_keys=[i for i in notes_dic.keys()]
    frequencies_array_pickle=np.array(notes_list_values)
    notes=[]
    for i in range(len(freq)):
        if freq[i]<130.8 or freq[i]>2093:
            i=i
        else:
            difference_array=frequencies_array_pickle-freq[i]
            lowest_freq=np.min(np.abs(difference_array))
            if lowest_freq not in difference_array:
                lowest_freq=lowest_freq*-1
            note_freq=np.around(lowest_freq+freq[i], decimals=3)
            real_note=notes_reversed[note_freq]
            for i in range(len(notes_keys)):
                if notes_keys[i] is real_note:
                    notes.append(notes_keys[i])
    counter=Counter(notes)
    guess=counter.most_common()
    return guess

