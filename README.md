# Beat-Detection-Using-Spectral-Flux
This code is a collection of functions designed to perform beat detection using spectral flux, specifically tailored for the GTZAN genre dataset (mini version). The GTZAN dataset is a widely used music classification dataset that contains audio tracks from various genres.

Here's a breakdown of the functions and their purpose:

****load_data**:** This function initializes and loads the GTZAN genre dataset using the mirdata library. It allows you to specify the dataset version and the path to where the dataset is located.

**estimate_beats_spectral_flux:** This function takes an audio file path as input and calculates beat positions using the spectral flux onset novelty function. It also computes a tempogram and PLP (Perceptual Linear Predictive) features. The beat times and activation values are returned.

**evaluate_estimated_beats:** This function evaluates the estimated beat positions for all tracks in the dataset. It computes F-measure scores to assess how well the estimated beats match the ground truth beats provided in the dataset.

**split_by_genre:** This function organizes the evaluation scores by genre. It creates a dictionary where each genre is a key, and the associated value is a dictionary containing evaluation scores for tracks in that genre.

**get_tempo_vs_performance:** This function extracts tempo values and their corresponding evaluation scores from the dataset. This can be useful for analyzing the relationship between tempo and beat detection performance.

compute_time_sf: A utility function to compute the time axis corresponding to the spectral flux novelty curve, given the novelty curve, hop size, and sampling rate.

compute_time_ml: A utility function to compute the time axis corresponding to the machine learning activation function novelty curve, given the novelty curve and its sampling rate.

sonify_track_data: This function is designed to sonify (audify) the estimated beats for a specific track in the GTZAN dataset. It generates audio clips with click sounds at both the estimated and reference beats, allowing you to audibly compare the two.
