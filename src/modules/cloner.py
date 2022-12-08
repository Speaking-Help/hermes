import sys, os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment


from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

sys.path.append('../src/modules')


class Cloner:

    # Locate Pytorch models
    ensure_default_models(Path("saved_models"))
    print(os.path.abspath("saved_models"))
    encoderModel = Path("saved_models/default/encoder.pt")
    synthesizerModel = Path("saved_models/default/synthesizer.pt")
    vocodermodel = Path("saved_models/default/vocoder.pt")
        

    #Load models
    encoder.load_model(encoderModel)
    synthesizer = Synthesizer(synthesizerModel)
    vocoder.load_model(vocodermodel)


    #Prepare embedings 
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    embed = np.random.rand(speaker_embedding_size)
    embed /= np.linalg.norm(embed)
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["test 1", "test 2"]
    mels = synthesizer.synthesize_spectrograms(texts, embeds)
    mel = np.concatenate(mels, axis=1)
    no_action = lambda *args: None
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
    num_generated = 0



    def synthesize(self, text, language):
        """
        Synthesizes speech by cloning `in_filename`, using the text provided by `text`, outputting to `out_filename`.
        """

        # Prepare global variables
        #global synthesizer

        texts = [text]
        embeds = [embed]

        # Synthesize the spectogram from embeddings and the text
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]


        # Synthesize the wave form
        generated_wav = vocoder.infer_waveform(spec)

        ## Post-generation
        # Sounddevice cuts audio one second earlier, so we must pad it.
        generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")

        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        generated_wav = encoder.preprocess_wav(generated_wav)

        # Play the audio (non-blocking)
        import sounddevice as sd
        try:
            sd.play(generated_wav, self.synthesizer.sample_rate)
        except sd.PortAudioError as e:
            print("\nCaught exception: %s" % repr(e))
            print("Continuing without audio playback.\n")
        except:
            raise

        # TODO return compatible audio file for React Native consumption

        # Save on the disk
        filename = "demo_output_%02d.wav" % self.num_generated
        sf.write(filename, generated_wav.astype(np.float32), self.synthesizer.sample_rate)
        self.num_generated += 1



    def train(self, file):
        """
        Trains our embedding, using the audio file `file` to create a voice clone.
        """

        # Prepare global varibles
        global synthesizer
        global embed
        global num_generated
        audioFile = file

        # Prepare path to passed in audio file
        audioSegment = AudioSegment.from_file(audioFile)
        audioSegment.export(out_f="out.wav",format="wav")
        in_fpath = Path(("out.wav").replace("\"", "").replace("\'", ""))


        ## The following 2 preprocesssing methods are equivalent.
        
        ## Method 1
        # First, preprocess the audio file using the encoder's helper function.
        # This is important for further steps.
        preprocessed_wav = encoder.preprocess_wav(in_fpath)

        ## Method 2
        # Use the librosa package to load the audio file and determine sampling rate
        original_wav, sampling_rate = librosa.load(str(in_fpath))


        print("HERE")
        # Proprocess the wave file
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)

        # Derive the embedding. There are many functions and parameters that the
        # speaker encoder interfaces.
        embed = encoder.embed_utterance(preprocessed_wav)
        print("HERE")
