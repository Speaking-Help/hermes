import os
from os.path import exists
# import required libraries s
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
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
from os import path
import sys




embed = None
num_generated = None
synthesizer = None

"""
ARCHIVED FLASK APPLICATION 

NOT FOR USE BUT FOR REFERENCE


def create_app(test_config=None):
    # create and configure the app
    application = Flask(__name__, instance_relative_config=True)
     
    application.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(application.instance_path, 'flaskr.sqlite'),
    )


    if test_config is None:
        # load the instance config, if it exists, when not testing
        application.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        application.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(application.instance_path)
    except OSError:
        pass
    

    @application.post('/grammarlyify')
    def grammarlyify():

        data = request.get_json()
        text = data.get('value')
        happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
        args = TTSettings(num_beams=5, min_length=1)
        result = happy_tt.generate_text(text, args=args)
        return str(result.text)



    @application.post('/upload')
    def login_post():
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)

        filename = uploaded_file.filename

        track = AudioSegment.from_file(filename,  format= 'm4a')
        file_handle = track.export('newFILE.wav', format='wav')

        text = "Default"
        filename = 'newFILE.wav'

        r = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            text = r.recognize_google(audio_data)

        return text

    @application.post('/toAudio')
    def TTS():
        data = request.get_json()
        
        text = data.get('value')

        language = data.get('language').lower()

        # DEFAULT TTS
        # tts = gTTS(text, lang=language)
        # tts.save("temp.mp3")
        # os.system("afplay " + "temp.mp3")

        # If seed is specified, reset torch seed and force synthesizer reload

        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        embeds = [embed]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        global synthesizer
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        # If seed is specified, reset torch seed and reload vocoder

        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        generated_wav = vocoder.infer_waveform(spec)


        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        generated_wav = encoder.preprocess_wav(generated_wav)

        # Play the audio (non-blocking)
        
        import sounddevice as sd
        try:
            sd.play(generated_wav, synthesizer.sample_rate)
        except sd.PortAudioError as e:
            print("\nCaught exception: %s" % repr(e))
            print("Continuing without audio playback.\n")
        except:
            raise

        # Save it on the disk
        global num_generated
        filename = "demo_output_%02d.wav" % num_generated
        sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
        num_generated += 1
        return data


    @application.post('/train')
    def train():
        audioFile = request.files['file']
        ensure_default_models(Path("saved_models"))
        encoder.load_model(Path("saved_models/default/encoder.pt"))
        global synthesizer
        synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
        vocoder.load_model(Path("saved_models/default/vocoder.pt"))
        encoder.embed_utterance(np.zeros(encoder.sampling_rate))
        global embed
        embed = np.random.rand(speaker_embedding_size)
        embed /= np.linalg.norm(embed)
        embeds = [embed, np.zeros(speaker_embedding_size)]
        texts = ["test 1", "test 2"]
        mels = synthesizer.synthesize_spectrograms(texts, embeds)
        mel = np.concatenate(mels, axis=1)
        no_action = lambda *args: None
        vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
        global num_generated
        num_generated = 0
        aSeg =AudioSegment.from_file(audioFile)
        aSeg.export(out_f="out.wav",format="wav")
        message = "out.wav"
        in_fpath = Path(("out.wav").replace("\"", "").replace("\'", ""))
        ## Computing the embedding
        # First, we load the wav using the function that the speaker encoder provides. This is
        # important: there is preprocessing that must be applied.
        # The following two methods are equivalent:
        # - Directly load from the filepath:
        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        # - If the wav is already loaded:
        original_wav, sampling_rate = librosa.load(str(in_fpath))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        # Then we derive the embedding. There are many functions and parameters that the
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        embed = encoder.embed_utterance(preprocessed_wav)

        return None

    return application
     
"""