#Model's Training Libraries
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV,SelectFromModel
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix, classification_report,multilabel_confusion_matrix,roc_auc_score,make_scorer,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_diabetes
#Audio Procesing Libraries 
import glob
import os
import shutil
from sklearn.model_selection import train_test_split
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
import sounddevice as sd
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import butter, lfilter
#*****************************************************************************************
#Stage 1 : Audio Preprocessing (By : Sarah Mustafa 7017E)
#*****************************************************************************************
# --- 10 Audio Augmentation Functions ---

# 1. Noise Injection
def add_white_noise(signal, noise_factor):
  noise = np.random.normal(0, signal.std(), signal.size)
  return signal + noise * noise_factor

# 2. Time Stretch
def time_stretch(signal, stretch_rate):
  return librosa.effects.time_stretch(y=signal, rate=stretch_rate)

# 3. Pitch Shift
def pitch_scale(signal, sr, n_steps):
  return librosa.effects.pitch_shift(
      y=signal, sr=sr, n_steps=n_steps, bins_per_octave=12
  )

# 4. Gain / Volume Control
def change_gain(signal, gain_factor):
  return signal * gain_factor

# 5. Time Shift / Rolling
def time_shift(signal, shift_ratio):
  shift_amt = int(len(signal) * shift_ratio)
  return np.roll(signal, shift_amt)

# 6. Dynamic Range Compression
def compress_dynamic_range(signal, threshold):
  return np.clip(signal, -threshold, threshold)

# 7. Low Pass Filter
def low_pass_filter(signal, sr, cutoff):
  nyquist = 0.5 * sr
  normal_cutoff = cutoff / nyquist
  b, a = butter(5, normal_cutoff, btype='low', analog=False)
  return lfilter(b, a, signal)

# 8. High Pass Filter
def high_pass_filter(signal, sr, cutoff):
  nyquist = 0.5 * sr
  normal_cutoff = cutoff / nyquist
  b, a = butter(5, normal_cutoff, btype='high', analog=False)
  return lfilter(b, a, signal)

# 9. Time Masking
def apply_time_mask(signal, mask_ratio):
  masked_sig = signal.copy()
  mask_len = int(len(masked_sig) * mask_ratio)
  start = np.random.randint(0, max(1, len(masked_sig) - mask_len))
  masked_sig[start:start+mask_len] = 0
  return masked_sig

# 10. Frequency Masking
def apply_freq_mask(signal, sr, cutoff_low, cutoff_high):
  nyquist = 0.5 * sr
  low = cutoff_low / nyquist
  high = cutoff_high / nyquist
  b, a = butter(3, [low, high], btype='bandstop', analog=False)
  return lfilter(b, a, signal)


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

def plot_two_signals(signal, augmented_signal, sr, figure_name):
  fig, ax = plt.subplots(nrows=2, num=figure_name)
  librosa.display.waveshow(signal, sr=sr, ax=ax[0])
  ax[0].set(title="OG Signal")
  librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1])
  ax[1].set(title="AG Signal")
  plt.tight_layout()
  plt.show(block=False)
#---------------------------------------------------------------------------------------------  
# Loading and Splitting Raw Data set
RawData = r"D:\dhuha\Mind\donateacry_corpus"
classes_labels = ["hungry", "discomfort", "tired"]
label_map = {"hungry": 0, "discomfort": 1, "tired": 2}
X_train, Y_train = [], []
X_test, Y_test = [], []

# Extreme distant values for 10 augmentation techniques
noise_factors = [0.01, 0.25]
stretch_rates = [0.65, 1.45]
pitch_steps = [-5, 5]
gain_factors = [0.4, 2.2]
shift_ratios = [-0.3, 0.3]
compress_thresholds = [0.15, 0.85]
lowpass_cutoffs = [1200, 4500]
highpass_cutoffs = [200, 1500]
time_mask_ratios = [0.05, 0.25]
freq_mask_bands = [(500, 1500), (2000, 4000)]

for cls in classes_labels:
  label = label_map[cls]
  files = glob.glob(f"{RawData}/{cls}/*.wav")
  train_files, test_files = train_test_split(
      files, test_size=0.20, random_state=40
  )
  os.makedirs(f"dataset_split_2/test/{cls}", exist_ok=True)
  os.makedirs(f"dataset_split_2/train/{cls}", exist_ok=True)

# Feature Extraction of audio data set
  for f in test_files:
    path_test = f"dataset_split_2/test/{cls}/{os.path.basename(f)}"
    shutil.copy(f, path_test)
    sig, sr_in = librosa.load(path_test, sr=16000, mono=True)
    feat = extract_cry_features(sig, sr=sr_in)
    if feat is not None:
      X_test.append(feat)
      Y_test.append(label)

  for f in train_files:
    path_train = f"dataset_split_2/train/{cls}/{os.path.basename(f)}"
    shutil.copy(f, path_train)
    sig, sr_in = librosa.load(path_train, sr=16000, mono=True)
    feat = extract_cry_features(sig, sr=sr_in)
    if feat is not None:
      X_train.append(feat)
      Y_train.append(label)

# Data Augmentation for minority classes
  if cls == "hungry":
    continue
  else:
    os.makedirs(f"dataset_split_2/train_augmented/{cls}", exist_ok=True)
    for f in train_files:
      signal, sr = librosa.load(f, sr=16000)
      base_name = os.path.splitext(os.path.basename(f))[0]

      # 1. Noise Injection
      for idx, val in enumerate(noise_factors):
        aug = add_white_noise(signal, val)
        sf.write(f"dataset_split_2/train_augmented/{cls}/{base_name}_noise_{idx}.wav", aug, sr)
        feat = extract_cry_features(aug, sr)
        if feat is not None: X_train.append(feat); Y_train.append(label)

      # 2. Time Stretch
      for idx, val in enumerate(stretch_rates):
        aug = time_stretch(signal, val)
        sf.write(f"dataset_split_2/train_augmented/{cls}/{base_name}_stretch_{idx}.wav", aug, sr)
        feat = extract_cry_features(aug, sr)
        if feat is not None: X_train.append(feat); Y_train.append(label)

      # 3. Pitch Scale
      for idx, val in enumerate(pitch_steps):
        aug = pitch_scale(signal, sr, val)
        sf.write(f"dataset_split_2/train_augmented/{cls}/{base_name}_pitch_{idx}.wav", aug, sr)
        feat = extract_cry_features(aug, sr)
        if feat is not None: X_train.append(feat); Y_train.append(label)

      # 4. Gain Change
      for idx, val in enumerate(gain_factors):
        aug = change_gain(signal, val)
        sf.write(f"dataset_split_2/train_augmented/{cls}/{base_name}_gain_{idx}.wav", aug, sr)
        feat = extract_cry_features(aug, sr)
        if feat is not None: X_train.append(feat); Y_train.append(label)

      # 5. Time Shift
      for idx, val in enumerate(shift_ratios):
        aug = time_shift(signal, val)
        sf.write(f"dataset_split_2/train_augmented/{cls}/{base_name}_shift_{idx}.wav", aug, sr)
        feat = extract_cry_features(aug, sr)
        if feat is not None: X_train.append(feat); Y_train.append(label)

      # 6. Dynamic Range Compression
      for idx, val in enumerate(compress_thresholds):
        aug = compress_dynamic_range(signal, val)
        sf.write(f"dataset_split_2/train_augmented/{cls}/{base_name}_compress_{idx}.wav", aug, sr)
        feat = extract_cry_features(aug, sr)
        if feat is not None: X_train.append(feat); Y_train.append(label)

      # 7. Low Pass Filter
      for idx, val in enumerate(lowpass_cutoffs):
        aug = low_pass_filter(signal, sr, val)
        sf.write(f"dataset_split_2/train_augmented/{cls}/{base_name}_lowpass_{idx}.wav", aug, sr)
        feat = extract_cry_features(aug, sr)
        if feat is not None: X_train.append(feat); Y_train.append(label)

      # 8. High Pass Filter
      for idx, val in enumerate(highpass_cutoffs):
        aug = high_pass_filter(signal, sr, val)
        sf.write(f"dataset_split_2/train_augmented/{cls}/{base_name}_highpass_{idx}.wav", aug, sr)
        feat = extract_cry_features(aug, sr)
        if feat is not None: X_train.append(feat); Y_train.append(label)

      # 9. Time Masking
      for idx, val in enumerate(time_mask_ratios):
        aug = apply_time_mask(signal, val)
        sf.write(f"dataset_split_2/train_augmented/{cls}/{base_name}_timemask_{idx}.wav", aug, sr)
        feat = extract_cry_features(aug, sr)
        if feat is not None: X_train.append(feat); Y_train.append(label)

      # 10. Frequency Masking
      for idx, (flow, fhigh) in enumerate(freq_mask_bands):
        aug = apply_freq_mask(signal, sr, flow, fhigh)
        sf.write(f"dataset_split_2/train_augmented/{cls}/{base_name}_freqmask_{idx}.wav", aug, sr)
        feat = extract_cry_features(aug, sr)
        if feat is not None: X_train.append(feat); Y_train.append(label)

# saving processed data into extracted features folder
os.makedirs("extracted_features_2", exist_ok=True)
np.save("extracted_features_2/X_train.npy", np.array(X_train))
np.save("extracted_features_2/Y_train.npy", np.array(Y_train))
np.save("extracted_features_2/X_test.npy", np.array(X_test))
np.save("extracted_features_2/Y_test.npy", np.array(Y_test))

# printing training and testing counts after augmentation
for cls in classes_labels:
  train_count = len(glob.glob(f"dataset_split_2/train/{cls}/*.wav"))
  test_count = len(glob.glob(f"dataset_split_2/test/{cls}/*.wav"))
  print(f"{cls} Train: {train_count} Test: {test_count}")

# Samples for the 3 Plots comparison
augmented_signal_N = add_white_noise(signal, noise_factors[1])
augmented_signal_T = time_stretch(signal, stretch_rates[1])
augmented_signal_P = pitch_scale(signal, sr, pitch_steps[1])

# Playing audio sample & Plotting comparison
print("Playing Original Audio")
sd.play(signal, sr)
sd.wait()
print("Playing Augmented (Noise) Audio")
sd.play(augmented_signal_N, sr)
sd.wait()

plot_two_signals(signal, augmented_signal_N, sr, "Noise Added")
plot_two_signals(signal, augmented_signal_T, sr, "Time Stretched")
plot_two_signals(signal, augmented_signal_P, sr, "Pitch Scaling")
plt.show()
#******************************************************************************************************
#Stage 2: Training and Testing the Model (By : Dhuha Ali 7020E)
#******************************************************************************************************
# ------Importing processed data code files-------------
X_train_raw = np.load("extracted_features_2/X_train.npy")
Y_train_raw = np.load("extracted_features_2/Y_train.npy")
X_test = np.load("extracted_features_2/X_test.npy")
Y_test = np.load("extracted_features_2/Y_test.npy")

label_map = {0: "Hungry", 1: "Discomfort", 2: "Tired"}

# Train / Validation Split (15% validation) ((By : Sarah Mustafa 7017E))
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_raw, Y_train_raw, test_size=0.15, stratify=Y_train_raw, random_state=42
)
#-----------------------------------------------------------------------------------------------

#Training Block {{{{{XGBoost Model}}}}}
#---------------

#Applying Randomized search cross validation using Grid Search to find best parameters
"""multi_class_auc = make_scorer(roc_auc_score, multi_class='ovr', response_method='predict_proba')

#Creating the pipeline <performs auto search for the best parameter for the model classifier>
pipeline = Pipeline([
    ('feature_selection', SelectFromModel(
        RandomForestClassifier(random_state=42, n_estimators=100), 
        threshold='mean'
    )),
    ('classifier', XGBClassifier(random_state=42, eval_metric='mlogloss',class_weight='balanced'))
])

#creating the parameters grid for XGBoost
parameter_grid = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__learning_rate': [0.01,0.02,0.03,0.04, 0.05, 0.1, 0.2],
    'classifier__max_depth': [2,3,4,5,6,7],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0],
    'classifier__reg_alpha': [0.0, 0.1, 1.0],
    'classifier__reg_lambda': [1.0, 5.0, 10.0]
}
#Scorere F1
macro_f1_scorer = make_scorer(f1_score, average='macro')

#applying Random Search strategy for the best parameters
grid_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_grid,
    n_iter=50,  
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), 
    #scoring=multi_class_auc,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    random_state=42
)
#fit
from sklearn.utils.class_weight import compute_sample_weight
#class_weights = compute_sample_weight(class_weight='balanced', y=Y_train)
#grid_search.fit(X_train, Y_train,classifier__sample_weight=class_weights)
grid_search.fit(X_train, Y_train)
#grid_search.fit(X_train_resampled, Y_train_resampled)
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

#saving best scores to CSV file
results_dict = {
    'Model': ['XGBoost'],
    'Best_CV_Score': [grid_search.best_score_],
    **{k: [v] for k, v in grid_search.best_params_.items()} 
}
results_df = pd.DataFrame(results_dict)
file_name = 'model_tuning_results.csv'
if os.path.exists(file_name):
    results_df.to_csv(file_name, mode='a', header=False, index=False)
else:
    results_df.to_csv(file_name, index=False)"""
#---------------------------------------------------------------------------------------------------    
#Applying Best Parameters
#------------------------
#Feature Selection
selector = SelectFromModel(
    RandomForestClassifier(random_state=42, n_estimators=100), 
    threshold='mean'
)
X_train_selected = selector.fit_transform(X_train, Y_train)
X_test_selected = selector.transform(X_test)

#Fitting the model
XGBModel = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=5,
    random_state=42,
    eval_metric='mlogloss',
    class_weight='balanced'
)
XGBModel.fit(X_train_selected, Y_train)

#Saving already trained model as a pickle file in the hard disk
#Saving already trained model as a picle file in the hard disk
with open('already_trained_model.pkl', 'wb') as file:
    pickle.dump(XGBModel, file)
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#Testing Block
#-------------
#load pickeled model
with open('already_trained_model.pkl', 'rb') as file:
    saved_model = pickle.load(file)
Y_predict=saved_model.predict(X_test_selected)
#Y_predict=grid_search.best_estimator_.predict(X_test)
classes_encoded=[0,1,2]
classes_labels = ["hungry", "discomfort", "tired"]
#Confusion Matrix
cm=confusion_matrix(Y_test,Y_predict,labels=classes_encoded)
#plot confusion matrix
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes_labels)
disp.plot(cmap=plt.cm.Reds)
plt.title('Confusion Matrix',fontsize=15)
plt.xlabel('Prediction',fontsize=13)
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().figure.subplots_adjust(bottom=0.2)

plt.show()
#Multilabel confusion matrix
mcm = multilabel_confusion_matrix(Y_test, Y_predict)
print("Multilabel confusion matrix is like :" "\n", np.array([['TN','FP'],
                                                        ['FN','TP']]) )
for class_label, multi_label_matrix in zip(classes_labels,mcm):
    print(f"{class_label} Confusion Matrix :\n {multi_label_matrix} ")
#Diagnosing Overfitting/ Underfitting
#training_accuracy_score=accuracy_score(Y_train,grid_search.best_estimator_.predict(X_train))
#training_accuracy_score=accuracy_score(Y_train_resampled,grid_search.best_estimator_.predict(X_train_resampled))
training_accuracy_score=accuracy_score(Y_train,saved_model.predict(X_train_selected))
testing_accuracy_score=accuracy_score(Y_test,Y_predict)
accuracy_gap=abs(training_accuracy_score-testing_accuracy_score)
macro_f1=f1_score(Y_test, Y_predict, average='macro')
weighted_f1=f1_score(Y_test, Y_predict, average='weighted')

print(f"Training Accuracy Score : {round(training_accuracy_score*100,1)}%")
print(f"Testing Accuracy Score : {round(testing_accuracy_score*100,1)}%")
print(f"Accuracy Gap : {round(accuracy_gap,2)}")
print(f"Macro Score for testing (Indication of model's overall performance)  :{round(macro_f1,2)}")
print(f"Weighted Score for testing (Idication of model's performance) : {round(weighted_f1,2)}\n")

if (training_accuracy_score < 0.60 ):
    print("Underfitting Model: both train and test accuracy < 0.60")
elif (training_accuracy_score - testing_accuracy_score) > 0.29:
    print("Overfitting Model: train accuracy much higher than test accuracy (gap > 0.10)")
elif (training_accuracy_score >= 0.60 and accuracy_gap <= 0.29):
    print("Good Fit: balanced performance and small gap")
else:
    print("Mixed/Unclear Fit: doesn't cleanly match under/over/good-fit criteria — inspect scores manually")
if macro_f1 < 0.5:
    print("Poor Performance (Biased model due to class imbalance): Macro F1 < 0.5")     
          