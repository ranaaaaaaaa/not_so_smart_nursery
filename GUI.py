from tkinter import *
import numpy as np
# import random
from time import *

import pyaudio

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from serialpython import *
from trigger_alert import *
from video_player import play

####################################
#Model's Training Libraries
import pandas as pd
import pickle
#__rolling
from collections import deque
#Audio Processing Libraries (Sarah)
import os
import librosa
import noisereduce as nr
import warnings
warnings.filterwarnings("ignore")
#load outside to avoid being loaded multiple times
with open("already_trained_model.pkl", "rb") as file:
    saved_model = pickle.load(file)
#The label map
label_map = {0:"hungry" , 1:"discomfort", 2:"tired"}
#Essential Functions in the secondary file
def extract_cry_features(signal, sr=16000):
    cleaned_signal = nr.reduce_noise(y=signal, sr=sr)
    intervals = librosa.effects.split(cleaned_signal, top_db=20)
    if len(intervals) > 0:
        VAD_signal = np.concatenate([cleaned_signal[start:end] for start, end in intervals])
    else:
        VAD_signal = cleaned_signal
    if len(VAD_signal) == 0:
        return None
    MFCCs = librosa.feature.mfcc(y=VAD_signal, sr=sr, n_mfcc=13)
    MFCC_mean = np.mean(MFCCs, axis=1)
    MFCC_std = np.std(MFCCs, axis=1)
    f0 = librosa.yin(VAD_signal, fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C7'), frame_length=2048)
    f0_clean = f0[~np.isnan(f0)] if f0 is not None else []
    f0_mean = np.mean(f0_clean) if len(f0_clean) > 0 else 0.0
    f0_std = np.std(f0_clean) if len(f0_clean) > 0 else 0.0
    rms = librosa.feature.rms(y=VAD_signal)
    intensity_mean = np.mean(rms)
    intensity_std = np.std(rms)
    duration = librosa.get_duration(y=VAD_signal, sr=sr)
    feature_vector = np.hstack([
        MFCC_mean,
        MFCC_std,
        f0_mean,
        f0_std,
        intensity_mean,
        intensity_std,
        duration])
    return feature_vector
# Getting Live Records <<<@ Rana GUI
# Runs on every 64ms chunk → servo responds instantly
def update_servo(audio_array):
    live_signal = audio_array.astype(np.float32) / 32768.0
    if np.max(np.abs(live_signal)) < 0.15:
        # print("Silence detected (No sound)")
        cry('C0')
    else:
        cry('C1')

# Runs on the 6-second rolling window → ML classification
def get_live_features(window_samples, sr=16000):
    live_signal = window_samples.astype(np.float32) / 32768.0
    if np.max(np.abs(live_signal)) < 0.15:
        return None
    return extract_cry_features(live_signal, sr=sr)

#Live Recording Classifying>>> @rana GUI
def classify_live_record(x_live):
    Prediction= saved_model.predict(x_live)
    return str(label_map[Prediction.item()])
####################################

gas_value = 0
baby_state= "fine"
video_window= None
alert_window= None

MAX_AMPLITUDE = 4000

CHUNK_SIZE = 1024
SAMPLE_RATE = 16000
BAR_COUNT = 80
#__rolling
audio_buffer = deque(maxlen=96000)   # 6 seconds at 16 kHz

def create_waveform(window):
    ############ graph ############
    wave_frame = Frame(
        window,
        width=700,
        height=300,
        bg="white"
    )
    wave_frame.pack(
        side=TOP,
        padx=3,
        pady=3
    )

    levels= [0] * BAR_COUNT

    figure = Figure(
        figsize= (6,2)
    ) # It does not display anything by itself yet. It is just the container for the graph.
    axis = figure.add_subplot(1, 1, 1) # (rows, columns, position) # axes object

    x_values = np.arange(BAR_COUNT) # This creates the horizontal positions for the bars.
    # matplotlib.axes.Axes.bar
    # matplotlib.axes.Axes.set_ylim
    # matplotlib.axes.Axes.set_xlim
    bars = axis.bar(
        x_values,
        levels,
        width=0.7
    ) # zip(bars, levels) pairs each bar with its corresponding height

    axis.set_ylim(0, MAX_AMPLITUDE) # set vertical range of graph
    axis.set_xlim(-1, BAR_COUNT) # -1 to 80 prevents the edge bars from touching the graph boundaries
    axis.axis("off") #  hides the graph decorations

    figure.tight_layout() # graph fits better in its available area.

    # FigureCanvasTkAgg
    # links figure to GUI
    canvas = FigureCanvasTkAgg(
        figure,
        master= wave_frame
    )

    canvas.get_tk_widget().pack()
    # Tkinter window -> wave_frame -> Matplotlib canvas -> figure -> axis -> bars

    ############ audio_input ############
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format= pyaudio.paInt16,
        channels= 1,
        rate= SAMPLE_RATE,
        input= True,
        frames_per_buffer= CHUNK_SIZE # 23 milliseconds of audio
    )

    def read_stream():
        nonlocal levels
        global baby_state

        data = stream.read(
            CHUNK_SIZE,
            exception_on_overflow= False
        )

        audio_array = np.frombuffer(
            data,
            dtype= np.int16 # tells NumPy how to interpret the raw bytes that PyAudio returned.
        )

        ####################################
        # bytes has no attribute 'flatten'
        # VAD on 64ms chunk → servo responds instantly
        update_servo(audio_array)
        # split so that the servo update won't be delayed 6 seconds
        # Feed samples into 6-second rolling buffer
        audio_buffer.extend(audio_array.tolist())

        # ML classification once we have a full 6-second window
        if len(audio_buffer) == 96000: # 16000*6
            window_samples = np.array(audio_buffer, dtype=np.int16)
            live_feature_vector = get_live_features(window_samples)
            if live_feature_vector is not None:
                X_live = live_feature_vector.reshape(1, -1)  # this line was added by Dhuha
                # print("Live Feature Vector Ready! Shape:", X_live.shape)
                # calling pretrained classifying model
                baby_state = classify_live_record(X_live)
                # slide the window forward by 1 second (16000 samples)
            else:
                baby_state = "fine"
            for i in range(16000):
                 audio_buffer.popleft()
        ####################################

        amplitude = float(
            np.mean(np.abs(audio_array))
        )
        levels.pop(0)
        levels.append(amplitude)

        for bar, height in zip(bars, levels):
            bar.set_height(height)

        canvas.draw_idle() # update without redrawing the whole canvas

        # stream.read() already waits for about 23 ms of audio
        window.after(1, read_stream)

    def close_waveform():
        stream.stop_stream() # Stops recording from the microphone. (cleanup)
        stream.close()
        audio.terminate()

    read_stream()

    return close_waveform
############ suggestion ############
def update_suggestion(window, SUGGESTIONS_label):
    global baby_state
    if baby_state == "hungry":
        SUGGESTIONS_label.config(text=f"SUGGESTION: Consider feeding your baby.", fg="green")
    if baby_state == "tired":
        SUGGESTIONS_label.config(text=f"SUGGESTION: URGENT! Your baby requires your care right now.", fg="red")
    if baby_state == "discomfort":
        SUGGESTIONS_label.config(text=f"SUGGESTION: Consider checking diaper, clothing, position.", fg="green")
    if baby_state == "fine":
        SUGGESTIONS_label.config(text=f"SUGGESTION: NO suggestions.", fg="green")

    window.after(500, update_suggestion, window, SUGGESTIONS_label)
############ baby_state ############
def update_state(window, babystate_label):
    global baby_state
    global video_window
    global gas_value
    babystate_label.config(text=f"YOUR BABY IS {baby_state}")
    if baby_state == "hungry" and gas_value <= 500:
        if video_window is None:
            video_window= play(window)
    else:
        if video_window is not None:
            video_window.destroy()
            video_window = None
    if baby_state == "tired":
        ser.write("B1\r".encode())
    else: ser.write("B0\r".encode())
    if baby_state == "discomfort" or baby_state == "tired" or baby_state == "fine":
        if video_window is not None:
            video_window.destroy()
            video_window = None

    window.after(500, update_state, window, babystate_label)

############ serial ############
def update_gui(window, temp_label, gas_label, awake_label, fan_label, light_label):
    global gas_value
    global alert_window
    global video_window
    if ser.in_waiting > 0:  # only read if there's actually new data waiting
        raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
        # serial device sends bytes -> ser.readline() -> decode bytes into text -> remove newline characters
        if raw_line:
            data = parse_line(raw_line)
            if "Temp" in data:
                temp_label.config(text=f"Temp: {data['Temp']} °C")
            if "Gas" in data:
                gas_label.config(text=f"Gas: {data['Gas']}")
                gas_value = float(data['Gas'])
            if "Awake" in data:
                awake_text = "Yes" if data['Awake'] == "1" else "No"
                awake_label.config(text=f"Awake: {awake_text}")
            if "Fan" in data:
                fan_text = "ON" if data['Fan'] == "1" else "OFF"
                fan_label.config(text=f"Fan: {fan_text}")
            if "Light" in data:
                light_text = "ON" if data['Light'] == "1" else "OFF"
                light_label.config(text=f"Light: {light_text}")

    if gas_value > 500 and alert_window is None:
        alert_window = trigger_alert(window)
        if video_window is not None:
            video_window.destroy()
            video_window = None
    elif gas_value <= 500 and alert_window is not None:
        alert_window.destroy()
        alert_window = None

    window.after(500, update_gui, window, temp_label, gas_label, awake_label, fan_label, light_label)


def main():
    window = Tk()
    window.title("Smart Nursery")
    window.config(bg="#d4f8ff", cursor="star")
    window_icon = PhotoImage(file="3ayoota.png")
    window.iconphoto(True, window_icon)
    ############ clock ############
    time_label = Label(window,
                       font=("ALGERIAN", 15, 'bold'),
                       fg="#78d4ff",
                       bg="white"
                       )
    time_label.pack(
        pady= 2
    )
    def update_time():
        time_string = strftime("%H:%M:%S %p")
        time_label.config(text=time_string)

        window.after(1000, update_time)
    update_time()
    ############ voice-memo waveform ############
    wave_label = Label(window,
                       text="LIVE AUDIO PROCESSING",
                       font=("ALGERIAN", 15, 'bold'),
                       fg='#b17ebf',
                       bg='white',
                       )
    wave_label.pack(
        side=TOP,
        pady= 2
    )
    close_waveform = create_waveform(window) # it's a local fn inside the "create_waveform" so I need to keep a ref to it

    def close_application():
        close_waveform()
        ser.close()
        window.destroy()

    window.protocol(
        "WM_DELETE_WINDOW",
        close_application
    )

    ############ SECTION1 ############
    SECTION1_label = Label(window,
                       text="ENVIRONMENT",
                       font=("ALGERIAN", 15, 'bold'),
                       fg='#b17ebf',
                        bg='white',
                       )
    SECTION1_label.pack(
        side= TOP,
    )
    ############ temp/gas/awake/fan/light ############
    temp_label = Label(window,
                       text="Temp: -- °C",
                       font=("ALGERIAN", 15, 'bold'),
                       fg='#78d4ff',
                       bg='white',
                       )
    temp_label.pack(side=TOP, pady=2)

    gas_label = Label(window,
                      text="Gas: --",
                      font=("ALGERIAN", 15, 'bold'),
                       fg='#78d4ff',
                       bg='white',
                      )
    gas_label.pack(side=TOP, pady=2)

    awake_label = Label(window,
                        text="Awake: --",
                        font=("ALGERIAN", 15, 'bold'),
                        fg='#78d4ff',
                        bg='white',
                        )
    awake_label.pack(side=TOP, pady=2)

    fan_label = Label(window,
                      text="☢FAN☢: --",
                      font=("ALGERIAN", 15, 'bold'),
                      fg='#78d4ff',
                      bg='white',
                      )
    fan_label.pack(side=TOP, pady=2)

    light_label = Label(window,
                        text="Light: --",
                        font=("ALGERIAN", 15, 'bold'),
                        fg='#78d4ff',
                        bg='white',
                        )
    light_label.pack(side=TOP, pady=2)

    update_gui(window, temp_label, gas_label, awake_label, fan_label, light_label)
    ############ SECTION2 ############
    SECTION2_label = Label(window,
                           text="Baby STATE",
                           font=("ALGERIAN", 15, 'bold'),
                           fg='#b17ebf',
                           bg='white',
                           )
    SECTION2_label.pack(
        side=TOP,
        pady= 2
    )
    ############ baby_state ############
    # LIST= ('HUNGRY','UNCOMFORTABLE','TIRED','FINE')
    # baby_state= random.choice(LIST)
    # if baby_state == 'HUNGRY' and gas_value <= 500:
    #    play()
    # connect with serial

    babystate_label = Label(window,
                            text=f"YOUR BABY IS --",
                            font=("ALGERIAN", 15, 'bold'),
                            fg='#78d4ff',
                            bg='white',
                            )
    babystate_label.pack(
        side=TOP,
        pady= 2
    )
    update_state(window, babystate_label)
    ############ suggestions ############
    # Laptop to Microcontroller
    # Cry detected / Cry ended
    # Laptop to Microcontroller
    # Servo start / Servo stop
    # Laptop to Microcontroller
    # Buzzer on (tired cry alert)
    SUGGESTIONS_label = Label(window,
                              text=f"SUGGESTION: --",
                              font=("ALGERIAN", 15, 'bold'),
                              fg='green',
                              bg='white',
                              )
    SUGGESTIONS_label.pack(
        side=TOP,
        pady=2
    )
    update_suggestion(window, SUGGESTIONS_label)
    def dismiss_cry():
        babystate_label.config(text="YOUR BABY IS FINE.")
    dismiss_button = Button(window,
                            text="DISMISS",
                            command= dismiss_cry,
                            state=ACTIVE,
                            )
    dismiss_button.pack(
        side=TOP,
        pady=2
    )
    ############ MAINLOOP ############
    window.mainloop()

main()
