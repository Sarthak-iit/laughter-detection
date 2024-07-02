import tkinter as tk
from tkinter import filedialog
import os
import wave
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from librosa import display
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import pickle
import copy
import math
import moviepy
    
model_path = 'laughter.keras'
    
def get_mel_spec(file):
    block_duration = 0.5
    window = 'hann'
    audio_path = file
    audio,sr = librosa.load(audio_path,sr=44100)
    buffer = int(block_duration * sr)
    samples_total = len(audio)
    samples_wrote = 0
    j= 0
    feature_list =[]
    while samples_wrote < samples_total:
        if buffer > (samples_total - samples_wrote):
            break
        y_block = audio[samples_wrote : (samples_wrote + buffer)]
        #check if the buffer is not exceeding total samples 
        mels = librosa.feature.melspectrogram(y=y_block,
                                                sr=sr,
                                                n_fft=512,
                                                hop_length=512,
                                                center=False,
                                                window=window)
        mels_db = librosa.power_to_db(mels, ref=1.0)
        feature_list.append(mels_db.reshape((128, int(round(21.4*block_duration*4)), 1)))
        samples_wrote += int(buffer)
        j+=1
    feature_list = np.array(feature_list)
    return feature_list

def clip_audio(y_corrected,audio_path):
    video_path = audio_path.replace('.mp3','.mp4')
    folder_name = video_path.replace('.mp4','')
    only_folder_name = folder_name.replace('test/','')
    folders = os.listdir('test/')
    if only_folder_name not in folders:
        os.makedirs(f'{folder_name}/laugh')
        os.makedirs(f'{folder_name}/notlaugh')
    
    for i in y_corrected:
        clip = moviepy.editor.VideoFileClip(video_path).subclip(k*0.5, (k+1)*0.5)
        # clips.append(clip)
        # final_clip = concatenate_videoclips(clips,method="compose")
        if i:
            output_file = video_path.replace('.mp4',f'/laugh/{k}.mp4')
        else:
            output_file = video_path.replace('.mp4',f'/notlaugh/{k}.mp4')
        clip.write_videofile(output_file, codec='libx264',audio_codec="aac" ,audio=True)
        k+=1

def run_model(file_path):
    model = tf.keras.models.load_model(model_path)
    feature_list = get_mel_spec(file_path)
    y_predicted = np.argmax(model.predict(x=feature_list), axis=1)
    return y_predicted

def get_laugh_duration(predicted_list):
    predicted_list = list(predicted_list)
    laugh = predicted_list.count(1)
    not_laugh = predicted_list.count(0)
    return laugh*0.5,(not_laugh+laugh)*0.5

def get_prediction(file_path,create_clips):
    y_predicted = run_model(file_path)
    laugh_dur,total_dur = get_laugh_duration(y_predicted)
    return laugh_dur,2,total_dur

# Function to process the audio file

# Function to handle file selection
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav")])
    
    if file_path:
        audio_file_entry.delete(0, tk.END)
        audio_file_entry.insert(0, file_path)

# Function to handle audio processing
def process_audio_file():
    file_path = audio_file_entry.get()
    create_clips = create_clips_var.get()
    if file_path:
        result_label.config(text=f"Processing...")
        laugh_dur,laughter_instances,total_dur = get_prediction(file_path,create_clips)
        # Update the UI with the processed information
        result_label.config(text=f"Laugh: {laugh_dur} Seconds \n Number of Laughter Instances: {laughter_instances} \n Total: {total_dur} Seconds")
    

# Create the main window
root = tk.Tk()
root.title("Audio File Info")

# Set fixed window size
root.geometry("400x250")

# Create a label and entry for selecting the audio file
audio_file_label = tk.Label(root, text="Select Audio File:")
audio_file_label.pack()
audio_file_entry = tk.Entry(root, width=40)
audio_file_entry.pack()
select_button = tk.Button(root, text="Browse", command=select_file)
select_button.pack()

# Create a checkbox for "Create Clips"
create_clips_var = tk.BooleanVar()
create_clips_checkbox = tk.Checkbutton(root, text="Create Clips", variable=create_clips_var)
create_clips_checkbox.pack()

# Create a "Process" button to trigger audio processing
process_button = tk.Button(root, text="Process Audio", command=process_audio_file)
process_button.pack()

# Create a label to display the result
result_label = tk.Label(root, text="")
result_label.pack()

# Run the main event loop
root.mainloop()
