import os
import csv
import argparse
from yt_dlp import YoutubeDL
from pydub import AudioSegment
from tqdm import tqdm

def download_audio(youtube_id, start_time, output_directory, duration=10):
    try:
        os.makedirs(output_directory, exist_ok=True)
        
        output_file = f'{output_directory}/{youtube_id}.wav'
        if os.path.exists(output_file):
            print(f"File already exists, skipping: {output_file}")
            return  # Skip downloading and processing if the file already exists


        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'outtmpl': f'./tmp/{youtube_id}.%(ext)s', 
            'quiet': True
        }
        url = f'https://www.youtube.com/watch?v={youtube_id}'

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            original_file = ydl.prepare_filename(info)

            # Load the downloaded audio from tmp
            audio = AudioSegment.from_file(original_file)

            # Resample the audio to 44.1 kHz
            audio = audio.set_frame_rate(44100)

            # Extract the 10-second clip
            start_ms = start_time * 1000
            end_ms = start_ms + (duration * 1000)
            audio_clip = audio[start_ms:end_ms]

            # Ensure the output directory exists for storing the final clipped .wav file
            os.makedirs(output_directory, exist_ok=True)

            # Save the clipped file as a .wav file in the output directory
            output_file = f'{output_directory}/{youtube_id}.wav'
            audio_clip.export(output_file, format='wav')  # Save clipped file as .wav

            # Clean up the original file in tmp
            if os.path.exists(original_file):
                os.remove(original_file)
    except Exception as e:
        print(f"Failed to download or process {youtube_id}: {e}")

def process_dataset(csv_file, output_directory) :
    with open(csv_file, newline='') as file :
        tmp_directory = "./tmp"
        os.makedirs(tmp_directory, exist_ok=True)
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in tqdm(reader, desc=f"Processing {os.path.basename(csv_file)}", unit="clip"):
            audiocaps_id, youtube_id, start_time, caption = row
            download_audio(youtube_id, int(start_time), output_directory)

        

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./data/AudioCaps/audio")
    parser.add_argument("--annotation", type=str, default="./data/AudioCaps/annotation")
    args = parser.parse_args()
    

    for split in ["train", "val", "test"]:
        csv_file = os.path.join(args.annotation, f"{split}.csv")
        output_directory = os.path.join(args.output_path, split)
        process_dataset(csv_file, output_directory)