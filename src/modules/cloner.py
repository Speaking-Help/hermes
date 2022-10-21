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

    encoder_weights = Path("../../libs/voice_cloning/saved_models/default/encoder.pt")
    vocoder_weights = Path("../../libs/voice_cloning/saved_models/default/vocoder.pt")
    syn_dir = Path("../../libs/voice_cloning/saved_models/default/synthesizer.pt")
    
    encoder.load_model(encoder_weights)
    synthesizer = Synthesizer(syn_dir)
    vocoder.load_model(vocoder_weights)

    def synthesize(self, text, in_filename, out_filename):

        in_fpath = Path(in_filename)

        reprocessed_wav = encoder.preprocess_wav(in_fpath)
        original_wav, sampling_rate = librosa.load(str(in_fpath))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        embed = encoder.embed_utterance(preprocessed_wav)

        specs = self.synthesizer.synthesize_spectrograms([text], [embed])
        generated_wav = vocoder.infer_waveform(specs[0])
        generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")

        sf.write(out_filename, generated_wav.astype(np.float32), self.synthesizer.sample_rate)

cl = Cloner()
cl.synthesize("Hello. Yesterday I went to the store. Today I'll go shopping.", "persian.m4a", "persian_cloned.wav")
