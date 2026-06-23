import streamlit as st
import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

# ==========================================
# CORE ALGORITHM: AUDIO FINGERPRINTING
# ==========================================

# Use a lower sample rate to dramatically speed up processing
TARGET_SR = 11025 

def get_spectrogram(audio_data, fs, nperseg=1024, noverlap=512):
    """Computes the spectrogram of the audio signal."""
    f, t, Sxx = signal.spectrogram(audio_data, fs, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db

def get_constellation(Sxx_db, neighborhood_size=20, threshold_percentile=85):
    """Finds local maxima (peaks) in the spectrogram to form a constellation."""
    local_max = maximum_filter(Sxx_db, size=neighborhood_size) == Sxx_db
    threshold = np.percentile(Sxx_db, threshold_percentile)
    is_loud_enough = Sxx_db > threshold
    
    peaks = local_max & is_loud_enough
    peak_freq_idx, peak_time_idx = np.where(peaks)
    
    sort_idx = np.argsort(peak_time_idx)
    return peak_time_idx[sort_idx], peak_freq_idx[sort_idx]

def generate_hashes(peak_time_idx, peak_freq_idx, fan_value=15):
    """Pairs nearby peaks to create robust (f1, f2, dt) hashes."""
    hashes = []
    num_peaks = len(peak_time_idx)
    
    for i in range(num_peaks):
        t1 = peak_time_idx[i]
        f1 = peak_freq_idx[i]
        
        for j in range(1, fan_value):
            if i + j < num_peaks:
                t2 = peak_time_idx[i + j]
                f2 = peak_freq_idx[i + j]
                
                time_delta = t2 - t1
                if 0 < time_delta <= 200: 
                    hash_tuple = (f1, f2, time_delta)
                    hashes.append((hash_tuple, t1))
    return hashes

def build_database(database_folder):
    """Indexes MP3 songs and builds BOTH the hash database and peak database for visualization."""
    hash_db = {} 
    peak_db = {} 
    
    if not os.path.exists(database_folder):
        return hash_db, peak_db

    files = [f for f in os.listdir(database_folder) if f.endswith(".mp3")]
    
    # Optional: progress bar if run from Streamlit interface directly
    if 'st' in globals():
        progress = st.progress(0)
        status = st.empty()
    
    for idx, filename in enumerate(files):
        if 'st' in globals():
            status.text(f"Indexing {idx+1}/{len(files)}: {filename}")
            progress.progress((idx + 1) / len(files))
            
        filepath = os.path.join(database_folder, filename)
        audio_data, fs = librosa.load(filepath, sr=TARGET_SR, mono=True)
        song_name = os.path.splitext(filename)[0]
        
        f, t, Sxx_db = get_spectrogram(audio_data, fs)
        peak_time_idx, peak_freq_idx = get_constellation(Sxx_db)
        
        # Store real time (s) and frequency (Hz) for the visualizer
        peak_db[song_name] = (t[peak_time_idx], f[peak_freq_idx])
        
        hashes = generate_hashes(peak_time_idx, peak_freq_idx)
        
        for hash_val, t1 in hashes:
            if hash_val not in hash_db:
                hash_db[hash_val] = []
            hash_db[hash_val].append((song_name, t1))
            
    if 'st' in globals():
        status.empty()
        progress.empty()
                
    return hash_db, peak_db

def match_query(query_data, fs, hash_db):
    """Matches a query audio clip against the database using offset histograms."""
    f, t, Sxx_db = get_spectrogram(query_data, fs)
    peak_time_idx, peak_freq_idx = get_constellation(Sxx_db)
    query_hashes = generate_hashes(peak_time_idx, peak_freq_idx)
    
    matches_per_song = {} 
    
    for hash_val, t_query in query_hashes:
        if hash_val in hash_db:
            for song_name, t_db in hash_db[hash_val]:
                offset = t_db - t_query
                if song_name not in matches_per_song:
                    matches_per_song[song_name] = []
                matches_per_song[song_name].append(offset)
                
    scores = {}
    best_song = None
    best_score = 0
    best_offsets = []
    
    for song_name, offsets in matches_per_song.items():
        if len(offsets) == 0:
            continue
        counts = pd.Series(offsets).value_counts()
        max_count = counts.max()
        scores[song_name] = max_count
        
        if max_count > best_score:
            best_score = max_count
            best_song = song_name
            best_offsets = offsets
            
    return best_song, best_score, best_offsets, (f, t, Sxx_db), (peak_time_idx, peak_freq_idx)

# ==========================================
# STREAMLIT USER INTERFACE 
# ==========================================

st.set_page_config(page_title="Zapptain America", layout="wide")
st.title("🎵 Zapptain America: Sonic Signatures")

DB_FILE = "song_database.pkl"
AUDIO_DIR = "database_songs" 

@st.cache_resource
def load_or_build_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'rb') as f:
            return pickle.load(f)
    else:
        st.info(f"Database not found. Building database from {AUDIO_DIR}... This will run much faster now.")
        db_tuple = build_database(AUDIO_DIR)
        with open(DB_FILE, 'wb') as f:
            pickle.dump(db_tuple, f)
        return db_tuple

# Unpack both the search index and the visualization data
hash_db, peak_db = load_or_build_db()

st.sidebar.header("Mode Selection")
mode = st.sidebar.radio("Choose operation mode:", ["Single-Clip Mode", "Batch Mode"])

if mode == "Single-Clip Mode":
    st.header("Single-Clip Analysis")
    uploaded_file = st.file_uploader("Upload a query audio clip (.mp3)", type=["mp3"])
    
    if uploaded_file is not None:
        # Downsample incoming query clips to match the database
        audio_data, fs = librosa.load(uploaded_file, sr=TARGET_SR, mono=True)
        st.audio(uploaded_file, format='audio/mp3')
        
        with st.spinner("Analyzing..."):
            best_song, score, offsets, spec_data, peaks_data = match_query(audio_data, fs, hash_db)
            
        if best_song:
            st.success(f"### 🎯 Matched Song: **{best_song}** (Confidence Score: {score})")
            
            # Unpack visualization data
            f, t, Sxx_db = spec_data
            p_time, p_freq = peaks_data
            
            # Set global plot style for dark theme
            plt.style.use('dark_background')
            
            # ---------------------------------------------------------
            # GRAPH 1 & 2: STEP 1 - FEATURE EXTRACTION
            # ---------------------------------------------------------
            st.markdown("### STEP 1 • FEATURE EXTRACTION")
            st.markdown("#### From spectrogram to constellation")
            st.write(f"The clip was converted into a time-frequency map (left). From that rich image, the **{len(p_time)} most prominent peaks** were kept (right).")
            
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Left: Spectrogram
            ax1.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma')
            ax1.scatter(t[p_time], f[p_freq], c='cyan', s=10, marker='o', alpha=0.8)
            ax1.set_ylabel('frequency (Hz)')
            ax1.set_xlabel('time (s)')
            
            # Right: Constellation Map
            ax2.scatter(t[p_time], f[p_freq], c='cyan', s=15, marker='o', alpha=0.8)
            ax2.set_ylabel('frequency (Hz)')
            ax2.set_xlabel('time (s)')
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(ax1.get_ylim())
            ax2.grid(True, alpha=0.2)
            
            st.pyplot(fig1)

            # ---------------------------------------------------------
            # GRAPH 3: STEP 2 - DATABASE SEARCH
            # ---------------------------------------------------------
            st.markdown("### STEP 2 • DATABASE SEARCH")
            st.markdown("#### Where in the song?")
            st.write(f"The fingerprint hashes were looked up against the indexed track. Below is the full fingerprint of **{best_song}**. The highlighted window is exactly where the query clip sits inside the full song.")
            
            fig2, ax3 = plt.subplots(figsize=(14, 5))
            
            if best_song in peak_db:
                # Plot the full database song's constellation
                db_t, db_f = peak_db[best_song]
                ax3.scatter(db_t, db_f, c='#1f77b4', s=10, marker='o', alpha=0.6)
                
                # Calculate window alignment
                dt = t[1] - t[0] # Time step per frame
                most_common_offset_frames = pd.Series(offsets).mode()[0]
                alignment_time_sec = most_common_offset_frames * dt
                query_duration_sec = t[-1]
                
                # Highlight the matched window
                ax3.axvspan(alignment_time_sec, alignment_time_sec + query_duration_sec, 
                            color='orange', alpha=0.2)
                ax3.axvline(alignment_time_sec, color='orange', linestyle='--', alpha=0.8)
                ax3.axvline(alignment_time_sec + query_duration_sec, color='orange', linestyle='--', alpha=0.8)
                
            ax3.set_ylabel('frequency (Hz)')
            ax3.set_xlabel('time (s)')
            ax3.grid(True, alpha=0.2)
            st.pyplot(fig2)
            
            # Reset style so it doesn't affect other components if added later
            plt.style.use('default')

        else:
            st.error("No match found in the database. Ensure the song is indexed or try a clearer clip.")

elif mode == "Batch Mode":
    st.header("Batch File Processing")
    st.write("Upload multiple query files to generate a `results.csv`.")
    
    uploaded_files = st.file_uploader("Upload query clips (.mp3)", type=["mp3"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Batch"):
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                # Downsample query clips to match the database
                audio_data, fs = librosa.load(file, sr=TARGET_SR, mono=True)
                best_song, _, _, _, _ = match_query(audio_data, fs, hash_db)
                
                pred = best_song if best_song else "Unknown"
                
                results.append({
                    "filename": file.name,
                    "prediction": pred
                })
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success("Batch processing complete!")
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download results.csv",
                data=csv,
                file_name='results.csv',
                mime='text/csv',
            )