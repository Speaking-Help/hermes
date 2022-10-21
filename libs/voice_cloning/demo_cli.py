import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder


if __name__ == '__main__':

    encoder_weights = Path("saved_models/default/encoder.pt")
    vocoder_weights = Path("saved_models/default/vocoder.pt")
    syn_dir = Path("saved_models/default/synthesizer.pt")
    encoder.load_model(encoder_weights)
    synthesizer = Synthesizer(syn_dir)
    vocoder.load_model(vocoder_weights)
    
    print("Interactive generation loop")
    num_generated = 0
    while True:
        try:
            # Get the reference audio filepath
            message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                      "wav, m4a, flac, ...):\n"
            in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))

            ## Computing the embedding
            # First, we load the wav using the function that the speaker encoder provides. This is
            # important: there is preprocessing that must be applied.

            # The following two methods are equivalent:
            # - Directly load from the filepath:
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            # - If the wav is already loaded:
            original_wav, sampling_rate = librosa.load(str(in_fpath))
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            print("Loaded file succesfully")

            # Then we derive the embedding. There are many functions and parameters that the
            # speaker encoder interfaces. These are mostly for in-depth research. You will typically
            # only use this function (with its default parameters):
            embed = encoder.embed_utterance(preprocessed_wav)
            print("Created the embedding")


            ## Generating the spectrogram
            text = input("Write a sentence (+-20 words) to be synthesized:\n")

            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            embeds = [embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")


            ## Generating the waveform
            print("Synthesizing the waveform:")

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(spec)


            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

            # Trim excess silences to compensate for gaps in spectrograms (issue #53)
            generated_wav = encoder.preprocess_wav(generated_wav)

            # Save it on the disk
            filename = "demo_output_%02d.wav" % num_generated
            print(generated_wav.dtype)
            sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            num_generated += 1
            print("\nSaved output as %s\n\n" % filename)


        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
