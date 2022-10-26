import sys, os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.append("../../libs/voice_cloning")

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

class Cloner:

    # Locate the pretrained models
    encoder_weights = Path("../../libs/voice_cloning/saved_models/default/encoder.pt")
    vocoder_weights = Path("../../libs/voice_cloning/saved_models/default/vocoder.pt")
    syn_dir = Path("../../libs/voice_cloning/saved_models/default/synthesizer.pt")

    # Load the models
    encoder.load_model(encoder_weights)
    synthesizer = Synthesizer(syn_dir)
    vocoder.load_model(vocoder_weights)

    def synthesize(self, text, in_filename, out_filename):
        """
        Synthesizes speech by cloning `in_filename`, using the text provided by `text`, outputting to `out_filename`.
        """

        # Get the path of the audio to be cloned
        in_fpath = Path(in_filename)

        # Load the audio from the path
        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        original_wav, sampling_rate = librosa.load(str(in_fpath))

        # Process the wav
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        embed = encoder.embed_utterance(preprocessed_wav)

        # Create mel spectogram
        specs = self.synthesizer.synthesize_spectrograms([text], [embed])
        generated_wav = vocoder.infer_waveform(specs[0])

        # Synthesize audio
        generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")
        generated_wav = encoder.preprocess_wav(generated_wav)

        # Export audio
        sf.write(out_filename, generated_wav.astype(np.float32), self.synthesizer.sample_rate)
