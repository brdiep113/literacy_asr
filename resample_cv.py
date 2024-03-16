import re
import os
import json
import tarfile
import torch
import torchaudio
import webdataset as wds
import numpy as np
from tqdm import tqdm
import subprocess
import librosa
import argparse
import pandas as pd
import soundfile as sf

resampler = torchaudio.transforms.Resample(
    48000,
    16000,
    lowpass_filter_width=64,
    rolloff=0.9475937167399596,
    resampling_method="sinc_interp_kaiser",
    dtype=torch.float32,
    beta=14.769656459379492,
)

resampler.to('cpu')


def resample_audio(input_path,output_path):
    
    #subprocess.call(['ffmpeg', '-i', input_path, '-ar', '16000', '-loglevel', 'quiet','-b', '16','-y',output_path])
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'quiet', '-nostats', '-i',input_path, '-acodec', 'pcm_s16le', '-f', 'wav', '-ar', '16000',output_path]
    subprocess.call(command)
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--code',type=str, default='en')
    parser.add_argument("--set",type=str, default='train')
    parser.add_argument("--transcript_path",type=str,default='sw_output.tsv')
    parser.add_argument("--outdir",type=str,default="~/scratch/processed_common_voice/")
    parser.add_argument("--dataset_path",type=str,default='~/scratch/common_voice/audio/')
    args = parser.parse_args()
    
    
    #path = args.transcript_path
    #files = os.listdir(path)
    
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    dataset_path = args.dataset_path
    
    file = args.code

    code = file
    set = args.set

    print("Processing %s"%file)
    
    data = pd.read_csv(args.transcript_path,sep='\t')

    transcripts = {k:v for k,v in zip(data['path'].tolist(),data['sentence'].tolist())}


    audio_path = os.path.join(dataset_path, code, set)
    audio_files = [i for i in os.listdir(audio_path) if i.endswith('.tar')]

    for audio_file in audio_files:

        print("Processing %s"%audio_file)

        outfile = os.path.join(outdir, file+"--"+audio_file)
        if not os.path.exists(outfile):
            sink = wds.TarWriter(outfile)

            tmp_folder = os.path.join(audio_path,audio_file.replace('.tar',''))

            print("Tmp folder at: %s"%tmp_folder)
            if not os.path.exists(tmp_folder):
                os.mkdir(tmp_folder)

            #file_list = set([os.path.join(audio_file.replace('.tar',''),k+'.mp3') for k in transcripts.keys()])

            #with tarfile.open(os.path.join(audio_path,audio_file), 'r') as tar:
            #    for file_name in file_list:
            #        try:
            #            tar.extract(file_name, audio_path)
            #        except KeyError:
            #            continue
            subprocess.call(['tar','-xvf',os.path.join(audio_path,audio_file),'-C',audio_path])

            clips = [i for i in os.listdir(tmp_folder) if 'mp3' in i]

            if len(clips)>0:
                for clip in tqdm(clips):
                    idx = clip.replace('.mp3','')
                    if idx in transcripts.keys():
                        #y,sr = torchaudio.load(os.path.join(tmp_folder,clip),format="mp3")
                        resample_audio(os.path.join(tmp_folder,clip),os.path.join(tmp_folder,clip.replace('.mp3','.wav')))
                        #y,sr = librosa.load(os.path.join(tmp_folder,clip),sr=48000)
                        #assert sr == 48000
                        #y_hat = resampler(torch.tensor(y))
                        y,sr = sf.read(os.path.join(tmp_folder,clip.replace('.mp3','.wav')))
                        example = {
                                    "__key__": idx,
                                    "pth": torch.tensor(y).float(),
                                    "txt": transcripts[idx]
                                      }
                        sink.write(example)

            subprocess.call(['rm','-rf',tmp_folder])