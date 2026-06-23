import os
import pickle
import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter
import librosa

# --- 1. Define the Core Functions ---
def get_spectrogram(audio_data, fs, nperseg=1024, noverlap=512):
    f, t, Sxx = signal.spectrogram(audio_data, fs, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db

def get_constellation(Sxx_db, neighborhood_size=20, threshold_percentile=85):
    local_max = maximum_filter(Sxx_db, size=neighborhood_size) == Sxx_db
    threshold = np.percentile(Sxx_db, threshold_percentile)
    is_loud_enough = Sxx_db > threshold
    peaks = local_max & is_loud_enough
    peak_freq_idx, peak_time_idx = np.where(peaks)
    sort_idx = np.argsort(peak_time_idx)
    return peak_time_idx[sort_idx], peak_freq_idx[sort_idx]

def generate_hashes(peak_time_idx, peak_freq_idx, fan_value=15):
    hashes = []
    num_peaks = len(peak_time_idx)
    for i in range(num_peaks):
        t1, f1 = peak_time_idx[i], peak_freq_idx[i]
        for j in range(1, fan_value):
            if i + j < num_peaks:
                t2, f2 = peak_time_idx[i + j], peak_freq_idx[i + j]
                time_delta = t2 - t1
                if 0 < time_delta <= 200: 
                    hashes.append(((f1, f2, time_delta), t1))
    return hashes

# --- 2. Build and Save the Database ---
def create_pkl_database():
    database_folder = "database_songs"
    output_file = "song_database.pkl"
    database = {} 
    
    print(f"Scanning folder: '{database_folder}'...")
    
    if not os.path.exists(database_folder):
        print(f"Error: The folder '{database_folder}' does not exist.")
        return

    # Loop through every mp3 file in the folder
    for filename in os.listdir(database_folder):
        if filename.endswith(".mp3"):
            print(f"Processing: {filename}...")
            filepath = os.path.join(database_folder, filename)
            
            # Load the audio
            audio_data, fs = librosa.load(filepath, sr=None, mono=True)
            song_name = os.path.splitext(filename)[0] 
            
            # Generate the fingerprints
            f, t, Sxx_db = get_spectrogram(audio_data, fs)
            peak_time_idx, peak_freq_idx = get_constellation(Sxx_db)
            hashes = generate_hashes(peak_time_idx, peak_freq_idx)
            
            # Store in the dictionary
            for hash_val, t1 in hashes:
                if hash_val not in database:
                    database[hash_val] = []
                database[hash_val].append((song_name, t1))
                
    # Save the dictionary to a .pkl file
    print("Saving database to disk...")
    with open(output_file, 'wb') as file:
        pickle.dump(database, file)
        
    print(f"Success! {output_file} has been created.")

# Run the function
create_pkl_database()