# English-speaker-accent-recognition-using-Transfer-Learning


## Introduction
This example demonstrates how to classify different English accents within audio waves by utilizing feature extraction techniques.

Instead of starting from scratch, we leverage transfer learning, tapping into advanced deep learning models, specifically using a pre-trained model (Yamnet) as feature extractors.

## Our Process:
Utilize a pre-trained TF Hub model (Yamnet) within the tf.data pipeline, transforming audio files into feature vectors.
Train a dense model using these extracted feature vectors.
Apply the trained model to classify accents in new audio files.
Notes:
TensorFlow IO is required for audio file resampling to 16 kHz, as demanded by the Yamnet model.
The test section uses ffmpeg to convert mp3 files to wav.
## Yamnet Model
Yamnet is an audio event classifier trained on the AudioSet dataset, predicting audio events from the AudioSet ontology. It's accessible on TensorFlow Hub.

Yamnet expects a 1-D tensor of audio samples at a 16 kHz sample rate. It produces a 3-tuple:

Scores (N, 521): Representing scores for 521 classes.
Embeddings (N, 1024).
Log-mel spectrogram of the entire audio frame.
We'll utilize the embeddings, the extracted audio features, as our dense model's input.

For more detailed information about Yamnet, please refer to its TensorFlow Hub page.

## Dataset
The dataset used comprises high-quality UK and Ireland English Dialect speech data, totaling 17,877 audio wav files.

This dataset encompasses over 31 hours of recordings from 120 volunteers identifying as native speakers of Southern England, Midlands, Northern England, Wales, Scotland, and Ireland.

For additional information, refer to the provided link or the following paper: "Open-source Multi-speaker Corpora of the English Accents in the British Isles."

## TensorFlow Dataset
To build a tf.data.Dataset, a dataframe_to_dataset function is created:

Creates a dataset using filenames and labels.
Retrieves Yamnet embeddings by calling filepath_to_embeddings.
Applies caching, reshuffling, and sets batch size.
filepath_to_embeddings:

Loads audio files.
Resamples audio to 16 kHz.
Generates scores and embeddings from Yamnet model.
As Yamnet produces multiple samples for each audio file, this function duplicates the label for generated samples with score=0 (speech), while setting 'other' for non-speech segments, indicating non-attribution to specific accents.

The load_16k_audio_file function is adapted from the tutorial "Transfer learning with YAMNet for environmental sound classification."

