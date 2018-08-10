

from microphone import record_audio
import numpy as np
from flask import Flask
from flask_ask import session, Ask, statement, question, request
app= Flask(__name__)
ask = Ask(app, '/')
@app.route('/')
def homepage():
    return "MusicNotes"
@ask.launch
def start_skill():
    welcome_message= 'Hello, please sing a note for 5 seconds. Are you ready?'
    return question(welcome_message)

listen_time=5
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion 
from scipy.ndimage.morphology import iterate_structure
import pickle
from collections import Counter
@ask.intent("YesIntent")
def recognize_notes():
    frames, sample_rate = record_audio(listen_time)
    audio_data = np.hstack([np.frombuffer(i, np.int16) for i in frames])
    sampling_rate=44100
    fig, ax = plt.subplots()

    S, freqs, times, im = ax.specgram(audio_data, NFFT=4096, Fs=sampling_rate,
                                  window=mlab.window_hanning,
                                  noverlap=4096 // 2)
    ax.set_ylim(0,2500)
    sum_S=S.sum(axis=1)
    freq=[]
    for i in range(len(sum_S)):
        if sum_S[i]>70000:
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
            hello=1
        else:
            difference_array=frequencies_array_pickle-freq[i]
            lowest_freq=np.min(np.abs(difference_array))
            if lowest_freq not in difference_array:
                lowest_freq=lowest_freq*-1
            note_freq=np.around(lowest_freq+freq[i], decimals=3)
            real_note=notes_reversed[note_freq]
            for j in range(len(notes_keys)):
                if notes_keys[j] is real_note:
                    notes.append(notes_keys[j])
    counter=Counter(notes)
    guess=counter.most_common(3)
    guesses_notes=[]
    for values in range(len(guess)):
        guesses_notes.append(guess[values][0])
    guesses=" and ".join(guesses_notes)
    guesses="you played " + guesses
    if len(guesses_notes) == 0:
        guesses="We couldn't tell what note you played"
    return statement(guesses)
@ask.intent('AMAZON.CancelIntent')
@ask.intent('AMAZON.StopIntent')
@ask.intent('AMAZON.NoIntent')
def stop_alexa():
    quit = "Alright, goodbye."
    return statement(quit)

if __name__ == '__main__':
    app.run(debug=True)
