# MIDI PROCESSING FUNCTIONS

import os
import shutil
import librosa
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from matplotlib.animation import FuncAnimation
import mido.midifiles.meta as meta
import madmom



def get_all_files(root_dir, file_type):
    """
    Find all MIDI files under the given root directory recursively.

    Args:
        root_dir (str): The root directory to start the search.

    Returns:
        List[str]: A list of absolute paths to the MIDI files found.
    """

    midi_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file extension is .mid
            if filename.lower().endswith((file_type)):
                # Add the absolute path of the MIDI file to the list
                midi_files.append(os.path.abspath(os.path.join(dirpath, filename)))
    
    return midi_files




def get_valid_midi_files_with_melody_and_duration(midi_paths):
    """
    Filters a list of MIDI files and returns only those that have a track containing the word "melody"
    in their name, have a single track without tempo changes, and are at least 60 seconds long.

    Args:
        midi_files (list of str): A list of paths to MIDI files.

    Returns:
        list of str: A list of paths to valid MIDI files that contain a track with the word "melody" in their name,
        have a single track without tempo changes, and are at least 30 seconds long.
    """
    midi_with_melody_duration = []
    melody_word = "melody"

    for midi_file in tqdm(midi_paths):
        try:
            with warnings.catch_warnings(record=True) as w:
                midi_data = pretty_midi.PrettyMIDI(midi_file)
                if any(issubclass(warn.category, RuntimeWarning) and
                       "Tempo, Key or Time signature change events found on non-zero tracks" in str(warn.message) 
                       for warn in w):
                    # Skip files with the warning
                    continue

            # Check for time signature changes
            time_signature_change = midi_data.time_signature_changes
            if len(time_signature_change) > 1:  
                # print(f"Skipping file {midi_file} - contains time signature changes")
                continue
            else:
                # Get list of all tracks
                all_tracks = midi_data.instruments

                # Check number of tracks
                if len(all_tracks) != 1:
                    for track in all_tracks:
                        if melody_word in track.name.lower():
                            duration = midi_data.get_end_time()
                            if duration >= 60.0:
                                midi_with_melody_duration.append(midi_file)
                            else:
                                # print(f"Skipping file {midi_file} - duration less than 30 seconds")
                                continue
                        else:
                            # print(f"Skipping file {midi_file} - no melody track labeled")
                            continue
                else:
                    # print(f"Skipping file {midi_file} - single track")
                    continue
        except (IOError, ValueError, EOFError, KeyError, meta.KeySignatureError):
            # Skips MIDI files that can't be opened or have a KeySignatureError
            # print(f"Skipping file {midi_file} - could not be opened or parsed")
            continue
        except:
            # Catch-all exception handler
            # print(f"Skipping file {midi_file} - an error occurred while processing file {midi_file}")
            continue

    return midi_with_melody_duration



def separate_melody_harmony(midi_paths, melody_word="melody"):
    """
    Filters a list of MIDI files and separates the tracks containing the word "melody" in their name
    from the other tracks. Returns two lists of paths, one for the melody tracks and one for the other tracks.

    Args:
        midi_files (list of str): A list of paths to MIDI files.
        melody_word (str): The word that should be contained in the name of the melody track.

    Returns:
        (list of str, list of str): A tuple of two lists of paths, one for the melody tracks and one for the other tracks.
    """
    melody_paths = []
    other_paths = []

    for midi_file in midi_paths:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)

            # Get list of all tracks
            all_tracks = midi_data.instruments

            # Check number of tracks
            if len(all_tracks) > 1:
                for track in all_tracks:
                    if melody_word in track.name.lower():
                        melody_paths.append(midi_file)
                        break
                else:
                    other_paths.append(midi_file)
            else:
                other_paths.append(midi_file)
        except:
            continue

    return melody_paths, other_paths




def copy_files(file_list, dst_dir):
    """
    Copies a list of files from a source directory to a destination directory.

    Args:
        file_list (list): A list of file paths to copy.
        dst_dir (str): The path to the destination directory.

    Returns:
        list: A list of the copied file paths.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    copied_files = []
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        dst_path = os.path.join(dst_dir, file_name)
        try:
            shutil.copy(file_path, dst_path)
            copied_files.append(dst_path)
        except shutil.SameFileError:
            pass
        except Exception as e:
            print(f"Error copying file {file_name}: {str(e)}")

    return copied_files



# source: inspred by https://stackoverflow.com/questions/44844581
def remove_drums(midi_paths, dst_dir):
    """
    Remove all drum instruments from a list of MIDI files and write the modified files to disk.

    Args:
        midi_paths (list): A list of paths to the input MIDI files.

    Returns:
        list: A list of paths to the output MIDI files.
    """
    output_paths = []
    for midi_path in tqdm(midi_paths):
        # Load the MIDI file
        midi_file = pretty_midi.PrettyMIDI(midi_path)

        # Find the indices of all drum instruments in the file
        drum_instruments_index = [i for i, inst in enumerate(midi_file.instruments) if inst.is_drum]

        # Delete all drum instruments from the file
        for i in sorted(drum_instruments_index, reverse=True):
            del midi_file.instruments[i]

        # Get the basename of the input file
        base_name = os.path.basename(midi_path)

        # Split the filename at '.mid' and append '_drumless'
        file_name, _ = os.path.splitext(base_name)
        output_name = file_name + '_drumless.mid'

        # Create the output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), dst_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write the modified MIDI file to disk
        output_path = os.path.join(output_dir, output_name)
        midi_file.write(output_path)
        output_paths.append(output_path)
    return output_paths




def mid_to_wav(midi_paths, dst_src, soundfont_path='./alex_gm.sf2'):
    """
    Convert a list of MIDI files to WAV format using the fluidsynth command line tool.

    Args:
        midi_paths (list): A list of MIDI file paths.
        soundfont_path (str): The path to the soundfont file.

    Returns:
        None
    """
    output_dir = os.path.join(os.getcwd(), dst_src)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wav_paths = []
    for midi_path in tqdm(midi_paths):
        # Get the basename of the MIDI file
        base_name = os.path.basename(midi_path)

        # Create the output file name by changing the extension to .wav
        output_name = os.path.splitext(base_name)[0] + '.wav'

        # Create the output file path
        output_path = os.path.join(output_dir, output_name)

        # Run the fluidsynth command to convert the MIDI file to WAV format
        command = f'fluidsynth -ni "{soundfont_path}" "{midi_path}" -F "{output_path}" -r 44100'
        os.system(command)

        # Add the output file path to the list of converted audio file paths
        wav_paths.append(output_path)

    return wav_paths



def audio_to_cqt(audio_paths, dst_src, figsize=(16,9), hop_length=4096,
                 n_bins=84, bins_per_octave=12, window='hamming'):
    """
    Compute CQT for each audio file in the list and save the CQT arrays to a desired folder by adding '_cqt.npy'
    at the end of file name, and save the CQT plots to the same folder by adding '_cqt.png' at the end of file name.

    Args:
        audio_paths (list): A list of paths to the input audio files.

    Returns:
        list: A list of tuples containing the paths to the saved CQT arrays and plots.
    """
    cqt_paths = []
    for audio_path in tqdm(audio_paths):
        # Load the audio file
        y, sr = librosa.load(audio_path)

        # Compute the CQT
        C = np.abs(librosa.cqt(y,
                               sr              = sr,
                               hop_length      = hop_length,
                               n_bins          = n_bins,
                               bins_per_octave = bins_per_octave,
                               window          = window,
                               fmin            = librosa.note_to_hz('C1')))

        # Normalize the CQT between 0 and 1
        C = librosa.power_to_db(C**2, ref=np.max)
        C = (C - C.min()) / (C.max() - C.min())

        # Create the output file names by appending '_cqt.npy' and '_cqt.png' to the input file name
        output_name = os.path.basename(audio_path)
        output_root, _ = os.path.splitext(output_name)
        output_npy = output_root + '_cqt.npy'
        output_png = output_root + '_cqt.png'
        output_npy_path = os.path.join(dst_src, output_npy)
        output_png_path = os.path.join(dst_src, output_png)

        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_npy_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the CQT array to disk
        np.save(output_npy_path, C)

        # Save the CQT plot to disk
        fig, ax = plt.subplots(figsize=figsize)
        librosa.display.specshow(C, sr=sr, ax=ax)
        fig.savefig(output_png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        cqt_paths.append((output_npy_path, output_png_path))

    return cqt_paths



def get_notes_by_measure(midi_files):
    """
    Returns a numpy array of notes played at each measure of multiple MIDI files and their velocities.

    Args:
        midi_files (list): A list of file paths to the MIDI files.

    Returns:
        np.ndarray: A 3D numpy array with shape (num_files, num_measures, num_notes_per_measure, 2).
                    The last dimension contains the pitch and velocity of each note.
    """
    notes_by_measure_all_files = []
    for midi_file in tqdm(midi_files):
        # Load the MIDI file using pretty_midi
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        # Get the downbeats
        downbeats = midi_data.get_downbeats()

        # Initialize a list to store the notes played at each measure and their velocities
        notes_by_measure = [[] for _ in range(len(downbeats))]

        # Iterate over all the notes in the MIDI file
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                # Calculate the measure in which the note was played
                for i, downbeat in enumerate(downbeats):
                    if note.start < downbeat:
                        measure = i - 1
                        break

                # Add the note and its velocity to the list for the corresponding measure
                notes_by_measure[measure].append([note.pitch, note.velocity])

        notes_by_measure_all_files.append(notes_by_measure)

    return notes_by_measure_all_files



# Courtesy of Magdalena Fuentes
def window_audio(audio, sample_rate, audio_seg_size, segments_overlap):
    """
    Segment audio into windows with a specified size and overlap. Padding is added only to the
    last window.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal to be segmented.
    sample_rate : int
        The sampling rate of the audio signal.
    audio_seg_size : float
        The duration of each window in seconds.
    segments_overlap : float
        The duration of the overlap between consecutive windows in seconds.

    Returns
    -------
    audio_windows : list of np.ndarray
        A list of windows of the audio signal.

    Example
    -------
    >>> import librosa
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> audio_windows = window_audio(y, sr, audio_seg_size=1, segments_overlap=0.5)
    """
    # Calculate the window size in samples
    window_size = int(audio_seg_size * sample_rate)

    # Calculate the overlap size in samples
    overlap_size = int(segments_overlap * sample_rate)

    audio_windows = []

    for i in range(0, len(audio), window_size-overlap_size):
        start = i
        end = min(i+window_size, len(audio))
        window = audio[start:end]
        if len(window) < window_size:
            # Padding the last window with zeros if it extends beyond the audio length
            padding = window_size - len(window)
            window = np.pad(window, (0, padding), mode='constant')

        # Add the window to the list of audio windows
        audio_windows.append(window)

    return audio_windows



def window_cqt_normalized(audio_file, hop_length=512, n_bins=84, bins_per_octave=12, window = 'hann', fmin=librosa.note_to_hz('C1')):
    """
    Compute CQT slices for a given audio file.

    Parameters
    ----------
    audio_file : str
        Path to the audio file.
    hop_length : int
        Hop length for the CQT computation.
    n_bins : int
        Number of frequency bins for the CQT computation.
    bins_per_octave : int
        Number of bins per octave for the CQT computation.
    fmin : float
        Minimum frequency for the CQT computation.

    Returns
    -------
    cqt_slices : np.ndarray
        Array of CQT slices for the audio file.
    """

    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Segment the audio into windows
    audio_windows = window_audio(y, sr, audio_seg_size=1, segments_overlap=0.5)

    # Compute CQT slices for each window
    cqt_window_slices = []
    for window in audio_windows:
        C = np.abs(librosa.cqt(window,
                                sr=sr,
                                hop_length=hop_length,
                                n_bins=n_bins,
                                bins_per_octave=bins_per_octave,
                                window='hann',
                                fmin=fmin))

        # Normalize the CQT between 0 and 1
        C = librosa.power_to_db(C**2, ref=np.max)
        C = (C - C.min()) / (C.max() - C.min())
        cqt_window_slices.append(C)

    # Stack the CQT slices into an array
    cqt_slices = np.array(cqt_window_slices)

    return cqt_slices



def assign_chords_to_cqt(cqt, chord_intervals, chord_labels, cqt_seg_size, segments_overlap):

    # Divide the CQT into equal segments with overlap
    cqt_segments = window_cqt(cqt, 44100, cqt_seg_size, segments_overlap)

    # Calculate the number of frames in each segment
    segment_size_frames = cqt_seg_size * cqt.shape[1] // 512

    # Assign chord label to each frame in the CQT segments
    chord_labels_cqt = []
    for segment in cqt_segments:
        # Calculate start and end frame indices for the segment
        start_frame = cqt.tolist().index(segment[0].tolist())
        end_frame = start_frame + segment_size_frames

        # Find the chord interval that overlaps with the segment
        chord_index = -1
        for i, interval in enumerate(chord_intervals):
            if start_frame >= interval[0] and end_frame <= interval[1]:
                chord_index = i
                break

        # Assign chord label to each frame within the segment
        if chord_index == -1:
            # If no chord interval overlaps with the segment, assign "N" for no chord
            chord_labels_cqt += ["N"] * segment_size_frames
        else:
            chord_label = chord_labels[chord_index]
            chord_labels_cqt += [chord_label] * segment_size_frames

    return chord_labels_cqt




# similar to window_audio
def segment_cqt_with_chords_fixed_windows(cqt, chord_labels, sample_rate, audio_seg_size, segments_overlap):
    """
    Segment the CQT spectrogram into windows of fixed duration and apply the correct chord label to each window.

    Parameters
    ----------
    cqt : np.ndarray
        The CQT spectrogram to be segmented.
    chord_labels : np.ndarray
        The chord labels, with each row representing the start and end times of a chord label.
    sample_rate : int
        The sampling rate of the audio signal.
    audio_seg_size : float
        The duration of each window in seconds.
    segments_overlap : float
        The duration of the overlap between consecutive windows in seconds.

    Returns
    -------
    cqt_windows : list of np.ndarray
        A list of windows of the CQT spectrogram.
    chord_labels_windows : list of np.ndarray
        A list of chord labels, with each row representing the start and end times of a chord label.

    Example
    -------
    >>> import librosa
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> cqt = np.abs(librosa.cqt(y, sr))
    >>> chord_labels = np.array([[ 0. ,  3.3],
                                 [ 3.3,  4.2],
                                 [ 4.2,  5.7],
                                 [ 5.7,  8.5],
                                 [ 8.5, 10.4]])
    >>> cqt_windows, chord_labels_windows = segment_cqt_with_chords_fixed_windows(cqt, chord_labels, sr, 1, 0.5)
    """
    # Calculate the window size and overlap size in samples
    window_size = int(audio_seg_size * sample_rate)
    overlap_size = int(segments_overlap * sample_rate)

    # Initialize lists to store the CQT windows and chord label windows
    cqt_windows = []
    chord_labels_windows = []

    # Calculate the number of windows
    num_windows = int(np.ceil(cqt.shape[1] / (window_size - overlap_size)))

    # Iterate through the CQT windows, extracting the corresponding chord label
    for i in range(num_windows):
        start_idx = i * (window_size - overlap_size)
        end_idx = min(start_idx + window_size, cqt.shape[1])
        cqt_window = cqt[:, start_idx:end_idx]

        # Find the chord label that overlaps with the current CQT window
        start_time = start_idx / sample_rate
        end_time = end_idx / sample_rate
        label_idx = np.where((chord_labels[:, 0] < end_time) & (chord_labels[:, 1] > start_time))[0]

        # If no chord label overlaps with the current CQT window, use the previous label
        if len(label_idx) == 0:
            label_idx = prev_label_idx

        # Save the current label index for the next iteration
        prev_label_idx = label_idx

        # Get the chord label for the current CQT window
        label_start = max(start_time, chord_labels[label_idx[0], 0])
        label_end = min(end_time, chord_labels[label_idx[0], 1])
        label_window = np.array([[label_start, label_end]])

        # Add the CQT window and label windows to their respective lists
        cqt_windows.append(cqt_window)
        chord_labels_windows.append(label_window)

    return cqt_windows, chord_labels_windows




def trim_audio_with_cqt(file_path, start_time=None, duration=30):
    """
    Loads an audio file from the specified file path and trims it to the desired duration. Computes the Constant-Q 
    Transform (CQT) of the trimmed audio.

    Parameters
    ----------
    file_path : str
        The file path of the audio file.
    start_time : float, optional
        The start time (in seconds) of the trimmed audio. If not specified, defaults to the middle of the audio file.
    duration : float, optional
        The duration (in seconds) of the trimmed audio. Defaults to 30 seconds.

    Returns
    -------
    tuple
        A tuple containing the file path of the audio file, the trimmed audio signal, the sample rate of the audio 
        file, and the CQT of the trimmed audio.

    Raises
    ------
    ValueError
        If the start time is earlier than the start of the audio file or the duration exceeds the end time of the audio 
        file.
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Compute the start and end times for the trimmed audio
    if start_time is None:
        middle_time = len(y) / 2 / sr
        start_time = middle_time - duration / 2
    end_time = start_time + duration

    # Check if start time is valid
    if start_time < 0:
        raise ValueError("Start time is earlier than the start of the audio file.")

    # Check if duration exceeds the end time of the audio file
    if end_time > librosa.get_duration(y=y, sr=sr):
        raise ValueError("Duration exceeds the end time of the audio file.")

    # Compute the sample indices corresponding to the start and end times
    start_idx = int(start_time * sr)
    end_idx = int(end_time * sr)

    # Trim the audio signal to the desired duration
    audio_trimmed = y[start_idx:end_idx]

    # Compute the CQT
    cqt = librosa.cqt(y, sr=sr, hop_length=4096, n_bins=84, bins_per_octave=12, window='hamming', fmin=librosa.note_to_hz('C1'))

    # Compute the time values
    frame_times = librosa.frames_to_time(np.arange(cqt.shape[1]), sr=sr, hop_length=4096)

    # Find the indices of the CQT frames corresponding to the start and end times
    start_idx_cqt = np.searchsorted(frame_times, start_time)
    end_idx_cqt = np.searchsorted(frame_times, end_time)

    # Trim the CQT array to the desired duration
    cqt_trimmed = cqt[:, start_idx_cqt:end_idx_cqt]

    return file_path, audio_trimmed, sr, cqt_trimmed



def trim_audio_with_cqt_normalized(file_path, start_time=None, duration=30, hop_length=512,
                                   n_bins=84, bins_per_octave=12, window='hann'):
    """
    Loads an audio file from the specified file path and trims it to the desired duration. Computes the Constant-Q 
    Transform (CQT) of the trimmed audio.

    Parameters
    ----------
    file_path : str
        The file path of the audio file.
    start_time : float, optional
        The start time (in seconds) of the trimmed audio. If not specified, defaults to the middle of the audio file.
    duration : float, optional
        The duration (in seconds) of the trimmed audio. Defaults to 30 seconds.
    hop_length : int, optional
        The hop length (in samples) of the CQT. Defaults to 512.
    n_bins : int, optional
        The number of frequency bins in the CQT. Defaults to 84.
    bins_per_octave : int, optional
        The number of bins per octave in the CQT. Defaults to 12.
    window : str, optional
        The window function to use for the CQT. Defaults to 'hann'.

    Returns
    -------
    tuple
        A tuple containing the file path of the audio file, the trimmed audio signal, the sample rate of the audio 
        file, and the CQT of the trimmed audio.

    Raises
    ------
    ValueError
        If the start time is earlier than the start of the audio file or the duration exceeds the end time of the audio 
        file.
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Compute the start and end times for the trimmed audio
    if start_time is None:
        middle_time = len(y) / 2 / sr
        start_time = middle_time - duration / 2
    end_time = start_time + duration

    # Check if start time is valid
    if start_time < 0:
        raise ValueError("Start time is earlier than the start of the audio file.")

    # Check if duration exceeds the end time of the audio file
    if end_time > librosa.get_duration(y=y, sr=sr):
        raise ValueError("Duration exceeds the end time of the audio file.")

    # Compute the sample indices corresponding to the start and end times
    start_idx = int(start_time * sr)
    end_idx = int(end_time * sr)

    # Trim the audio signal to the desired duration
    audio_trimmed = y[start_idx:end_idx]

    # Compute the CQT
    C = np.abs(librosa.cqt(y=audio_trimmed, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave, window=window, fmin=librosa.note_to_hz('C1')))

    # Normalize the CQT between 0 and 1
    C = librosa.power_to_db(C**2, ref=np.max)
    C = (C - C.min()) / (C.max() - C.min())

    return file_path, audio_trimmed, sr, C



def estimate_chords(audio_path):
    """Compute chords using the Deep Chroma Chord Recogntion model, implemented in madmom.
    
    Parameters
    ----------
    audio_path : str
        Path to input audio file

    Returns
    -------
    chord_intervals : np.ndarray, shape=(n, 2)
        Chord intervals [start_time, end_time] in seconds
    chord_labels : list, shape=(n,)
        List of chord labels, e.g. ['A:maj', 'G:min', ...]

    """
    # initializing and calling dcp
    dcp = madmom.audio.chroma.DeepChromaProcessor()
    decode = madmom.features.chords.DeepChromaChordRecognitionProcessor()
    chroma = dcp(audio_path)
    decoded_chroma = decode(chroma)

    # assigning chord intervals and labels
    chord_intervals = []
    chord_labels = []
    for i in decoded_chroma:
        chord_intervals.append([i[0], i[1]])
        chord_labels.append(i[2])
    chord_intervals = np.array(chord_intervals)
    return chord_intervals, chord_labels



def chord_to_number(chord):
    """
    Convert a chord name to a number between 0 and 23, with A:maj represented by 0.
    
    Parameters
    ----------
    chord : str
        The chord name to convert, e.g. 'C:maj', 'G:min', etc.
    
    Returns
    -------
    int
        The corresponding number between 0 and 23 if chord is recognized,
        otherwise -1.
    """
    # Define a mapping of chord names to numbers
    chord_map = {'A:maj' : 0,  'A:min' : 12,
                 'A#:maj': 1,  'A#:min': 13,
                 'B:maj' : 2,  'B:min' : 14,
                 'C:maj' : 3,  'C:min' : 15,
                 'C#:maj': 4,  'C#:min': 16,
                 'D:maj' : 5,  'D:min' : 17,
                 'D#:maj': 6,  'D#:min': 18,
                 'E:maj' : 7,  'E:min' : 19,
                 'F:maj' : 8,  'F:min' : 20,
                 'F#:maj': 9,  'F#:min': 21,
                 'G:maj' : 10, 'G:min' : 22,
                 'G#:maj': 11, 'G#:min': 23,
                 'N': 24}
    return chord_map.get(chord, -1)



def number_to_chord(num):
    """
    Convert a number between 0 and 23 to a chord name, with A:maj represented by 0.
    
    Parameters
    ----------
    num : int
        The number to convert, between 0 and 23.
    
    Returns
    -------
    str
        The corresponding chord name if num is valid, otherwise 'N'.
    """
    # Define a mapping of numbers to chord names
    chord_map = {0 : 'A:maj',  12: 'A:min',
                 1 : 'A#:maj', 13: 'A#:min',
                 2 : 'B:maj',  14: 'B:min',
                 3 : 'C:maj',  15: 'C:min',
                 4 : 'C#:maj', 16: 'C#:min',
                 5 : 'D:maj',  17: 'D:min',
                 6 : 'D#:maj', 18: 'D#:min',
                 7 : 'E:maj',  19: 'E:min',
                 8 : 'F:maj',  20: 'F:min',
                 9 : 'F#:maj', 21: 'F#:min',
                 10: 'G:maj',  22: 'G:min',
                 11: 'G#:maj', 23: 'G#:min'}
    return chord_map.get(num, 'N')



def animate_cqt_with_vline(cqt, sr, start_time=0, end_time=None):
    """
    Plot the Constant-Q Transform (CQT) of an audio signal using librosa,
    and animate a vertical line moving from left to right to indicate
    the current playback position of the audio signal.

    Parameters:
    -----------
    cqt : np.ndarray
        Constant-Q Transform (CQT) of the audio signal.
    sr : int
        Sampling rate of the audio signal.
    start_time : float, optional
        Start time in seconds of the audio signal to plot (default is 0).
    end_time : float, optional
        End time in seconds of the audio signal to plot (default is None).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    ax : matplotlib.axes.Axes
        Axes object containing the plot.
    line : matplotlib.lines.Line2D
        Line object representing the moving vertical line.
    """

    # If end_time is not specified, set it to the end of the audio signal
    if end_time is None:
        end_time = librosa.get_duration(cqt.shape[1], sr)

    # Compute the time values corresponding to each column of the CQT
    times = librosa.times_like(cqt, sr=sr)

    # Compute the start and end indices of the CQT columns to plot
    start_col = np.argmax(times >= start_time)
    end_col = np.argmax(times >= end_time)

    # Plot the CQT
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(cqt[:, start_col:end_col]), sr=sr, x_axis='time', ax=ax)

    # Create a vertical line representing the current playback position
    line = ax.axvline(times[start_col], color='r')

    # Calculate the time it takes to move one step of the CQT
    step_time = (times[end_col] - times[start_col]) / (end_col - start_col)

    def update_line(num):
        """
        Update function for the animation.
        Moves the vertical line to the next position.
        """
        nonlocal start_col
        nonlocal line

        # Move the vertical line to the next position
        line.set_xdata(times[start_col])
        start_col += 1

    # Create the animation
    ani = FuncAnimation(fig, update_line, frames=end_col - start_col, interval=step_time * 1000)

    return fig, ax, line




def plot_chromagram(chroma_data, tempo, time_signature):
    """
    Plot a chromagram heatmap given chromagram data and the duration of the audio in beats.

    Parameters:
    chroma_data (numpy.ndarray): Chromagram data, as returned by `get_chroma()` method of a midi file object.
    duration_seconds (int): Duration of the audio in beats.

    Returns:
    None
    """
    # Calculate duration of each beat in seconds
    beat_duration = 60 / tempo

    # Calculate duration of each bar in seconds
    bar_duration = beat_duration * time_signature[0]

    # Create time axis in seconds
    time_axis = np.arange(chroma_data.shape[1]) * beat_duration
    bar_ticks = np.arange(0, time_axis[-1], bar_duration)
    bar_tick_labels = np.arange(1, len(bar_ticks) + 1)

    # Normalize data to be between 0 and 1
    chroma_data = chroma_data / np.max(chroma_data)

    # Create heatmap plot with flipped axes
    plt.imshow(chroma_data, aspect='auto', origin='lower', cmap='gray_r', extent=[0, time_axis[-1], 0, chroma_data.shape[0]])

    # Set y-axis tick labels
    plt.yticks(np.arange(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
    
    # Set axis labels and title
    plt.xlabel('Time (beats)')
    plt.ylabel('Pitch Class')
    plt.title('Chromagram')

    # Show the plot
    plt.show()



def midi_to_note_name(midi_num):
    """
    Converts a MIDI note number to a string representation of the note name.

    Args:
        midi_num (int): MIDI note number

    Returns:
        str: String representation of the note name.

    Raises:
        ValueError: If midi_num is less than 0 or greater than 127.
    """
    if midi_num < 0 or midi_num > 127:
        raise ValueError('MIDI value must be between 0 and 127, inclusive.')

    # calculate octave number
    octave = midi_num // 12 - 1

    # calculate note name
    note_num = midi_num % 12
    if note_num == 0:
        note_name = 'C'
    elif note_num == 1:
        note_name = 'C#'
    elif note_num == 2:
        note_name = 'D'
    elif note_num == 3:
        note_name = 'D#'
    elif note_num == 4:
        note_name = 'E'
    elif note_num == 5:
        note_name = 'F'
    elif note_num == 6:
        note_name = 'F#'
    elif note_num == 7:
        note_name = 'G'
    elif note_num == 8:
        note_name = 'G#'
    elif note_num == 9:
        note_name = 'A'
    elif note_num == 10:
        note_name = 'A#'
    else:
        note_name = 'B'

    # combine note name and octave number
    # note_name += '-' + str(octave)

    return note_name



def note_name_to_midi(note_name):
    """
    Converts a string representation of a note name to the corresponding MIDI note number.

    Args:
        note_name (str): String representation of a note name in the format "note-octave", e.g. "C-4".

    Returns:
        int: MIDI note number corresponding to the input note name.

    Raises:
        ValueError: If note_name is not in the correct format, or if the note name or octave number is invalid.
    """
    try:
        # split note name and octave number
        note, octave = note_name.split('-')
    except ValueError:
        raise ValueError('note_name must be in the format "note-octave", e.g. "C-4".')

    # lookup table for note names
    note_table = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}

    # check if note name is valid
    if note not in note_table:
        raise ValueError(f'Invalid note name: {note}')

    # check if octave number is valid
    try:
        octave = int(octave)
    except ValueError:
        raise ValueError(f'Invalid octave number: {octave}')
    if octave < 0 or octave > 9:
        raise ValueError(f'Octave number must be between 0 and 9: {octave}')

    # calculate MIDI note number
    midi_num = note_table[note] + (octave + 1) * 12

    # check if MIDI note number is valid
    if midi_num < 0 or midi_num > 127:
        raise ValueError(f'MIDI value must be between 0 and 127 inclusive: {midi_num}')

    return midi_num



def midi_to_hz(midi_num):
    """
    Converts a MIDI note value to its corresponding frequency in Hz.

    Args:
        midi_num (int): MIDI note number

    Returns:
        float: Frequency in Hz corresponding to the input MIDI note number.

    Raises:
        ValueError: If midi_num is less than 0 or greater than 127.
    """
    if midi_num < 0 or midi_num > 127:
        raise ValueError('MIDI value must be between 0 and 127, inclusive.')

    return 2 ** ((midi_num - 69) / 12) * 440



def hz_to_midi(freq):
    """
    Converts a frequency in Hz to the corresponding MIDI note number.

    Args:
        freq (float): Frequency in Hz.

    Returns:
        int: MIDI note number corresponding to the input frequency.
    """
    midi_num = 12 * (np.log2(freq) - np.log2(440)) + 69
    return int(round(midi_num))



def hz_to_midi(freq, base_freq=440.0):
    """
    Converts a frequency or list of frequencies in Hz to MIDI note numbers.

    Args:
        freq (float or list): Frequency or list of frequencies in Hz
        base_freq (float, optional): Base frequency for A4 (default 440.0)

    Returns:
        int or list: MIDI note number(s) corresponding to the input frequency(ies).
    """
    if isinstance(freq, list):
        return [int(round(12 * np.log2(f / base_freq) + 69)) for f in freq]
    else:
        return int(round(12 * np.log2(freq / base_freq) + 69))



def midi_to_hz(midi_num, base_freq=440.0):
    """
    Converts a MIDI note number or list of MIDI note numbers to frequencies in Hz.

    Args:
        midi_num (int or list): MIDI note number(s)
        base_freq (float, optional): Base frequency for A4 (default 440.0)

    Returns:
        float or list: Frequency(ies) in Hz corresponding to the input MIDI note number(s).
    """
    if isinstance(midi_num, list):
        return [base_freq * 2 ** ((m - 69) / 12) for m in midi_num]
    else:
        return base_freq * 2 ** ((midi_num - 69) / 12)