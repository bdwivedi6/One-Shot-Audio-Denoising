import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
import random
import os
import shutil


fuss_csv_path =  "Audio-Denoising\\FSD50K\\fuss_fsd_data\\fuss_pairs.csv"
df_filetags = pd.read_csv(fuss_csv_path)

mixed_samples_path ="Audio-Denoising\\FSD50K\\fuss_fsd_data\\mixed_noise_samples\\"
audio_dir_path = "Audio-Denoising\\FSD50K\\fuss_fsd_data\\all_samples\\"

# for index, row in df_filetags.iterrows():
#
#     signalFilename = row['SignalFilename']
#     noiseFilename = row['NoiseFilename']
#     # print(f"signalFilename {signalFilename} and noiseFilename {noiseFilename}")
#     signal = AudioSegment.from_file(audio_dir_path+signalFilename)
#     noise = AudioSegment.from_file(audio_dir_path+noiseFilename)
#     noiseVolume = np.random.uniform(low=-4, high=4)
#     mix = signal.overlay(noise,gain_during_overlay=noiseVolume)
#     mix.export(mixed_samples_path + signalFilename, format='wav')

train_df = df_filetags[df_filetags['Split']=="train"]
val_df = df_filetags[df_filetags['Split']=="val"]
test_df = df_filetags[df_filetags['Split']=="test"]

df_list = [train_df,val_df,test_df]

for df in df_list:
    split_type = df.iloc[0,4]
    clean_sample = df['SignalFilename'].tolist()

    with open(split_type+'_dataset_temp.txt', 'w+') as f:
        for sample in clean_sample:
            f.write(mixed_samples_path+sample+' '+audio_dir_path+sample+'\n')
