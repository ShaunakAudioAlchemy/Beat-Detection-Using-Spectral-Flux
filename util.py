import mirdata
import librosa
import mir_eval
import numpy as np
import matplotlib.pyplot as plt
import IPython.display

def load_data(dataset_name, data_home, dataset_version='mini'):
    """
    Load a specific version of a dataset using the mirdata library.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load, e.g., "gtzan_genre".
    
    dataset_version : str
        Version of the dataset to load. To load the "mini" version, specify "mini". Default is "1.0".
    
    data_home : str
        Path to where the dataset is located. 

    Returns
    -------
    dataset : mirdata.Dataset
        The initialized mirdata Dataset object corresponding to the specified dataset and version.

    Notes
    -----
    The function is optimized for GTZAN-genre dataset but can be potentially used for other datasets supported by mirdata.
    """

    dataset = mirdata.initialize(dataset_name = 'gtzan_genre', data_home = data_home, version = dataset_version)
    
    return dataset



def estimate_beats_spectral_flux(audio_path, hop_length=512):
    """
    Compute beat positions using the spectral flux onset novelty function, followed by computing a tempogram and PLP.
    
    Parameters
    ----------
    audio_path : str
        Path to input audio file
    hop_length : int, optional
        Hop length, by default 512

    Returns
    -------
    beat_times : 1-d np.array
        Array of time stamps of the estimated beats in seconds.
    activation : 1-d np.array
        Array with the activation (or novelty function) values.
    """

    y, sr = librosa.load(audio_path, sr = None)
    activation = librosa.onset.onset_strength(y=y, sr=sr, hop_length = hop_length) #calculating spectral flux
    plp = librosa.beat.plp(y=y, sr=sr, hop_length = hop_length, onset_envelope = activation) #calculating plp
    beat_plp = librosa.util.localmax(plp) # obtaining a boolean array identifying max and non max values
    beat_plp = np.where(beat_plp)[0] # retaining indices (in frames) of only true values
    beat_times = librosa.frames_to_time(beat_plp, sr=sr, hop_length=hop_length)

    return beat_times, activation
    


def evaluate_estimated_beats(data, estimated_beats):
    """
    Evaluates the estimated beats for all tracks in the given data.

    Parameters:
        data (dict): Dictionary of tracks with track_id as the key and track information as the value.
        estimated_beats (dict): Dictionary with track_id as the key and array of estimated beat times as the value.

    Returns:
        dict: A dictionary with track_id as the key and evaluation score as the value.
    """
    diction = {}
    for track_id in data:
        diction[track_id] = mir_eval.beat.f_measure(data[track_id].beats.times, estimated_beats[track_id])
    
    return diction



def split_by_genre(scores_dictionary, tracks_dictionary):
    """Split scores by genre.

    Parameters
    ----------
    scores_dictionary : dict
        Dictionary of scores keyed by track_id
    tracks_dictionary: dict
        Dictionary of mirdata tracks keyed by track_id

    Returns
    -------
    genre_scores : dict
        Dictionary with genre as keys and a
        dictionary of scores keyed by track_id as values

    """
    genre_scores = {}
    genres = []  # Change to a list

    for track_id in tracks_dictionary:
        if tracks_dictionary[track_id].genre not in genres:
            genres.append(tracks_dictionary[track_id].genre)
    
    for genre in genres:
        genre_tracks = {}
        for track_id in scores_dictionary:
            if genre == tracks_dictionary[track_id].genre:
                genre_tracks[track_id] = scores_dictionary[track_id]
        genre_scores[genre] = genre_tracks

    return genre_scores



def get_tempo_vs_performance(scores_dictionary, tracks_dictionary):
    """Get score values as a function of tempo.

    Parameters
    ----------
    scores_dictionary : dict
        Dictionary of scores keyed by track_id
    tracks_dictionary: dict
        Dictionary of mirdata tracks keyed by track_id        

    Returns
    -------
    tempo : np.array
        Array of tempo values with the same number of elements as scores_dictionary
    scores : np.array
        Array of scores with the same number of elements as scores_dictionary
    """
    #creating empty arrays for tempo and scores
    tempo = []  
    scores = []  

    for track_id in scores_dictionary:
        
        tempo_value = tracks_dictionary[track_id].tempo
        score_value = scores_dictionary[track_id]

    
        tempo.append(tempo_value)
        scores.append(score_value)

    # to convert lists to arrays
    tempo = np.array(tempo)
    scores = np.array(scores)

    return tempo, scores


def compute_time_sf(novelty_sf, hop_size=512, sr=22050):
    """
    Compute the time axis for spectral flux novelty.

    Parameters
    ----------
    novelty_sf : np.array
        The spectral flux novelty curve.
    hop_size : int, optional
        Hop size used to compute the spectral flux. Default is 512.
    sr : int, optional
        Sampling rate of the signal. Default is 22050.

    Returns
    -------
    time_sf : np.array
        Time axis corresponding to the spectral flux novelty curve.
    """
    number_frames = len(novelty_sf)
    time_sf = np.arange(number_frames) * hop_size / sr #time values in seconds for each frame of the curve

    return time_sf


def compute_time_ml(novelty_ml, sr_ml=100):
    """
    Compute the time axis for machine learning activation function novelty.

    Parameters
    ----------
    novelty_ml : np.array
        The machine learning activation function novelty curve.
    sr_ml : int, optional
        Rate at which the novelty is computed (usually at 100Hz for madmom). Default is 100.

    Returns
    -------
    time_ml : np.array
        Time axis corresponding to the machine learning novelty curve.
    """
    time_ml = np.arange(len(novelty_ml)) / sr_ml  # Compute time axis

    return time_ml
    

def sonify_track_data(track_id, estimated_beats, tracks_dictionary):
    """
    Sonify the estimated beats for a given track ID.

    The purpose of this function is to sonify or "audify" the beats estimated 
    using different methods for a given track in the GTZAN dataset. This function 
    generates three audio clips for each track:
    1. The original audio with superimposed click sounds at the estimated beats.
    2. The original audio with superimposed click sounds at the reference beats.

    Steps:
    - Load the audio data for the provided track ID.
    - Select the estimated beats from the corresponding dictionary.
    - Generate click tracks (sonifications) for the estimated and reference beats. Hint: use mir_eval.sonify
    - Superimpose or add these click tracks to the original audio. Hint: make sure lengths of audio signals match.
    - Display the resulting audio clips for playback using IPython.display.

    Parameters
    ----------
    track_id : str
        A string representing the unique identifier for a track in the GTZAN 
        dataset.
    estimated_beats: dict
        Dictionary of estimated beats per track keyed by track_id
    tracks_dictionary: dict
        Dictionary of mirdata tracks keyed by track_id

    Returns
    -------
    None: 
        While the function doesn't return any values, it displays the audio clips 
        for playback in the environment (e.g., Jupyter notebook).
    """
    pass
