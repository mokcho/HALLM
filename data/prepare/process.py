import os
import pandas as pd
import argparse

def process_entailment_files(audio_dir, entailment_dir):
    csv_files = [f for f in os.listdir(entailment_dir) if os.path.isfile(os.path.join(entailment_dir, f))]
    for item in csv_files:
        process_data(audio_dir, os.path.join(entailment_dir,item))

def process_data(data_dir, input_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Extract columns into lists
    audio_paths = df['Audio file'].tolist()
    entailment = df['Entailment'].tolist()
    neutral = df['Neutral'].tolist()
    contradiction = df['Contradiction'].tolist()

    processed_audio_paths = []
    processed_premise = []
    labels = []

    # Process the data
    for i in range(len(audio_paths)):
        processed_premise.append(entailment[i])
        processed_premise.append(neutral[i])
        processed_premise.append(contradiction[i])

        labels.append(0)  # Entailment
        labels.append(1)  # Neutral
        labels.append(2)  # Contradiction

        audio_path = os.path.join(data_dir, audio_paths[i])
        processed_audio_paths.append(audio_path)
        processed_audio_paths.append(audio_path)
        processed_audio_paths.append(audio_path)

    # Create the flattened DataFrame
    flatten_df = pd.DataFrame({
        'Audio file': processed_audio_paths,
        'Hypothesis': processed_premise,
        'Label': labels
    })

    # Save the flattened DataFrame to a CSV file
    output_csv = "flattened_" + os.path.basename(input_csv)
    flatten_df.to_csv(output_csv, index=False)
    print(f"Data processed and saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Process audio data for training.")
    parser.add_argument('--data', type=str, default='all', help= " All, Clotho, AudioCaps ")
    parser.add_argument('--data_dir', type=str, default='./data', help="Directory containing data files.")
    args = parser.parse_args()

    if args.data.lower() in ['all', 'clotho']:
        clotho_audio_dir = os.path.join(args.data_dir, "Clotho/audio")
        clotho_entailment_dir = os.path.join(args.data_dir, "Clotho/entailment")
        process_entailment_files(clotho_audio_dir, clotho_entailment_dir)

    if args.data.lower() in ['all', 'audiocaps']:
        audiocaps_audio_dir = os.path.join(args.data_dir, "AudioCaps/audio")
        audiocaps_entailment_dir = os.path.join(args.data_dir, "AudioCaps/entailment")
        process_entailment_files(audiocaps_audio_dir, audiocaps_entailment_dir)

    else :
        print("Please give 'all', 'clotho', or 'audiocaps' for --data")
        return


if __name__ == "__main__":
    main()
