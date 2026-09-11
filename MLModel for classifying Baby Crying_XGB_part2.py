#Model's Training Libraries
import numpy as np
import pandas as pd
import pickle

#Audio Procesing Libraries (Sarah)
import librosa
import noisereduce as nr
import warnings
warnings.filterwarnings("ignore")
#The label map
label_map = {0:"hungry" , 1:"discomfort", 2:"tired"}
#Essential Functions in the secondary file
def extract_cry_features(signal, sr=16000):
  # 1. Zero/Silence Check
  if signal is None or len(signal) == 0 or np.max(np.abs(signal)) < 0.01:
    return None

  cleaned_signal = nr.reduce_noise(y=signal, sr=sr)
  intervals = librosa.effects.split(cleaned_signal, top_db=20)
  if len(intervals) > 0:
    VAD_signal = np.concatenate(
        [cleaned_signal[start:end] for start, end in intervals]
    )
  else:
    VAD_signal = cleaned_signal
  if len(VAD_signal) == 0 or np.max(np.abs(VAD_signal)) < 0.01:
    return None

  # 1. Base Features (MFCCs + Delta)
  MFCCs = librosa.feature.mfcc(y=VAD_signal, sr=sr, n_mfcc=13)
  MFCC_mean = np.mean(MFCCs, axis=1)
  MFCC_std = np.std(MFCCs, axis=1)

  if MFCCs.shape[1] >= 3:
    MFCC_delta = librosa.feature.delta(MFCCs, width=3)
  else:
    MFCC_delta = np.zeros_like(MFCCs)
  MFCC_delta_mean = np.mean(MFCC_delta, axis=1)

  # 2. Additional Features (Spectral Centroid & Zero Crossing Rate)
  cent = librosa.feature.spectral_centroid(y=VAD_signal, sr=sr)
  cent_mean = np.mean(cent)

  zcr = librosa.feature.zero_crossing_rate(y=VAD_signal)
  zcr_mean = np.mean(zcr)

  # 3. Fundamental Frequency (F0), RMS & Duration
  f0 = librosa.yin(
      VAD_signal,
      fmin=librosa.note_to_hz("C3"),
      fmax=librosa.note_to_hz("C7"),
      frame_length=2048,
  )
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
      MFCC_delta_mean,
      cent_mean,
      zcr_mean,
      f0_mean,
      f0_std,
      intensity_mean,
      intensity_std,
      duration,
  ])
  return feature_vector
# Getting Live Records <<<@ Rana GUI
def get_live_features(audio, sr=16000):
    live_signal = audio.flatten()
    if np.max(np.abs(live_signal)) < 0.01:
        print("Silence detected (No sound)")
        return None
    return extract_cry_features(live_signal, sr=sr)

#Live Recording Classifying>>> @rana GUI
def classify_live_record(x_live):
    #load pickeled model
    with open("already_trained_model.pkl", "rb") as file:
        saved_model = pickle.load(file)
    Prediction=saved_model.predict(x_live) 
    return str(label_map[Prediction.item()])  


#calling Function
#record audio .. Put the recorded stream instead of that
audio_signal, sr = librosa.load(
    r"C:\Users\hamoa\Downloads\baby-crying-01.wav", sr=16000
)

live_feature_vector = get_live_features(audio_signal)

if live_feature_vector is not None:
    X_live = live_feature_vector.reshape(1, -1) #this line was added by Dhuha
    print("Live Feature Vector Ready! Shape:", X_live.shape)
#calling pretrained classifying model
print(classify_live_record(X_live))


stream = audio.open(
        format= pyaudio.paInt16,
        channels= 1,
        rate= SAMPLE_RATE,
        input= True,
        frames_per_buffer= CHUNK_SIZE # 23 milliseconds of audio
    )

    def read_stream():
        nonlocal levels

        data = stream.read(
            CHUNK_SIZE,
            exception_on_overflow= False
        )
        # =========================
        # stream.read() already waits for about 23 ms of audio
        window.after(1, read_stream)