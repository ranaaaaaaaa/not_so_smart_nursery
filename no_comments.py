from tkinter import *
import numpy as np
import random
from time import *

import pyaudio

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from video_player import play
from serialpython import *

MAX_AMPLITUDE = 4000
FIG_SIZE = (6, 2)

CHUNK_SIZE = 1024
SAMPLE_RATE = 44100
BAR_COUNT = 80

def create_waveform(window):
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

    levels = np.zeros(BAR_COUNT)

    figure = Figure(figsize=FIG_SIZE)
    axis = figure.add_subplot(111)

    x_values = np.arange(BAR_COUNT)

    bars = axis.bar(
        x_values,
        levels,
        width=0.7
    )

    axis.set_ylim(0, MAX_AMPLITUDE)
    axis.set_xlim(-1, BAR_COUNT)
    axis.axis("off")

    figure.tight_layout()

    canvas = FigureCanvasTkAgg(
        figure,
        master=wave_frame
    )

    canvas.get_tk_widget().pack()

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    def check_audio():
        nonlocal levels

        data = stream.read(
            CHUNK_SIZE,
            exception_on_overflow=False
        )

        audio_array = np.frombuffer(
            data,
            dtype=np.int16
        )

        amplitude = float(
            np.mean(np.abs(audio_array))
        )

        levels = np.roll(levels, -1)
        levels[-1] = amplitude

        for bar, height in zip(bars, levels):
            bar.set_height(height)

        canvas.draw_idle()

        # stream.read() already waits for about 23 ms of audio
        window.after(1, check_audio)

    def close_waveform():
        stream.stop_stream()
        stream.close()
        audio.terminate()

    check_audio()

    return close_waveform


def update_gui(window, temp_label, gas_label, awake_label, fan_label, light_label):
    if ser.in_waiting > 0:  # only read if there's actually new data waiting
        raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
        if raw_line:
            data = parse_line(raw_line)

            if "Temp" in data:
                temp_label.config(text=f"Temp: {data['Temp']} C")
            if "Gas" in data:
                gas_label.config(text=f"Gas: {data['Gas']}")
            if "Awake" in data:
                awake_text = "Yes" if data['Awake'] == "1" else "No"
                awake_label.config(text=f"Awake: {awake_text}")
            if "Fan" in data:
                fan_text = "ON" if data['Fan'] == "1" else "OFF"
                fan_label.config(text=f"Fan: {fan_text}")
            if "Light" in data:
                light_text = "ON" if data['Light'] == "1" else "OFF"
                light_label.config(text=f"Light: {light_text}")

    window.after(23, update_gui, window, temp_label, gas_label, awake_label, fan_label, light_label)

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
    def update():
        time_string = strftime("%H:%M:%S %p")
        time_label.config(text=time_string)

        window.after(1000, update)
    update()
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
    close_waveform = create_waveform(window)

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
    ############ baby_state(randomized for now) ############
    LIST= ('HUNGRY','UNCOMFORTABLE','TIRED','FINE')
    baby_state= random.choice(LIST)
    # needs fixing
    # if baby_state == 'UNCOMFORTABLE' and data['Gas'] <= 500:
    #     play()

    babystate_label = Label(window,
                            text=f"YOUR BABY IS {baby_state}.",
                            font=("ALGERIAN", 15, 'bold'),
                            fg='#78d4ff',
                            bg='white',
                            )
    babystate_label.pack(
        side=TOP,
        pady= 2
    )
    ############ MAINLOOP ############
    window.mainloop()

main()