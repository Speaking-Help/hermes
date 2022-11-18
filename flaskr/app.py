import os
from pydub import AudioSegment
from os.path import exists
# import required libraries s
import speech_recognition as sr
from happytransformer import HappyTextToText, TTSettings
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
from pydub import AudioSegment

import sys

from flask import Flask
from flask import request


embed = None
num_generated = None
synthesizer = None

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

    # a simple page that says hello
    @application.route('/hello')
    def hello():
        return 'HEYO'
    
    @application.route('/jojo')
    def jojo():
        return 'jojo'
    

    @application.post('/grammarlyify')
    def grammarlyify():
        #text = ???

        data = request.get_json()

        text = data.get('value')

        happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
        args = TTSettings(num_beams=5, min_length=1)


        result = happy_tt.generate_text(text, args=args)
        print("HOHOHO" + str(result.text))
        return str(result.text)

    @application.post('/upload')
    def login_post():
            
        print("JO")
        
        #print(request.files['file'])
        #print(exists(request.data))
        print(request.files['file'])

        print(exists(request.files['file'].filename))
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
        
        print(exists(request.files['file'].filename))


        filename = uploaded_file.filename

        track = AudioSegment.from_file(filename,  format= 'm4a')
        file_handle = track.export('newFILE.wav', format='wav')


        print("Audio recorded...")

        #---OPTIONAL- play the iput file
        #wave_obj = sa.WaveObject.from_wave_file(filename)
        #play_obj = wave_obj.play()
        #play_obj.wait_done()  # Wait until sound has finished playing


        text = "Default"
        filename = 'newFILE.wav'

        r = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            text = r.recognize_google(audio_data)
            print("You said: \"" + text + "\"")

        
        #wav_audio = AudioSegment.from_file(request.data[7:], format="caf")
        #wav_audio.export("audio.mp3", format="mp3")

        return text
   
   
    # @app.route('/login', methods=['GET', 'POST'])
    # def login():
    #     if flask.request.method == 'GET':
    #         return '''
    #             <form action='login' method='POST'>
    #                 <input type='text' name='email' id='email' placeholder='email'/>
    #                 <input type='password' name='password' id='password' placeholder='password'/>
    #                 <input type='submit' name='submit'/>
    #             </form>
    #             '''

    #     email = flask.request.form['email']
    #     if email in users and flask.request.form['password'] == users[email]['password']:
    #         user = User()
    #         user.id = email
    #         flask_login.login_user(user)
    #         return flask.redirect(flask.url_for('protected'))

    #     return 'Bad login'


    # @app.route('/protected')
    # @flask_login.login_required
    # def protected():
    #     return 'Logged in as: ' + flask_login.current_user.id

    @application.post('/toAudio')
    def TTS():
        print
        data = request.get_json()
        
        text = data.get('value')

        print(text)
        language = data.get('language').lower()
        print("LANGUAGE IS " + str(language))
        tts = gTTS(text, lang=language)
        tts.save("temp.mp3")



        os.system("afplay " + "temp.mp3")

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
            sd.stop()
            #sd.play(generated_wav, synthesizer.sample_rate)
        except sd.PortAudioError as e:
            print("\nCaught exception: %s" % repr(e))
            print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
        except:
            raise

        # Save it on the disk
        global num_generated
        filename = "demo_output_%02d.wav" % num_generated
        print(generated_wav.dtype)
        sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
        num_generated += 1
        print("\nSaved output as %s\n\n" % filename)





        return data

    @application.post('/train')
    def train():



#         print("JO")
        
#         #print(request.files['file'])
#         #print(exists(request.data))
        
#         print(request.files['file'])

#         print(exists(request.files['file'].filename))
#         uploaded_file = request.files['file']
#         if uploaded_file.filename != '':
#             uploaded_file.save(uploaded_file.filename)
        
#         print(exists(request.files['file'].filename))


#         filename = uploaded_file.filename

#         track = AudioSegment.from_file(filename,  format= 'm4a')
#         file_handle = track.export('newFILE.wav', format='wav')


#         print("Audio recorded...")

#         #---OPTIONAL- play the iput file
#         #wave_obj = sa.WaveObject.from_wave_file(filename)
#         #play_obj = wave_obj.play()
#         #play_obj.wait_done()  # Wait until sound has finished playing


#         text = "Default"
#         filename = 'newFILE.wav'
# ############################################






#         print("HERE")
#         import argparse
#         if torch.cuda.is_available():
#             device_id = torch.cuda.current_device()
#             gpu_properties = torch.cuda.get_device_properties(device_id)
#             ## Print some environment information (for debugging purposes)
#             print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
#                 "%.1fGb total memory.\n" %
#                 (torch.cuda.device_count(),
#                 device_id,
#                 gpu_properties.name,
#                 gpu_properties.major,
#                 gpu_properties.minor,
#                 gpu_properties.total_memory / 1e9))
#         else:
#             print("Using CPU for inference.\n")

#         ## Load the models one by one.
        
#         print("HERO")
        
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
        print("MADE IT HERE")
        
        message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                    "wav, m4a, flac, ...):\n"
        
        in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))
        #in_fpath = Path(filename)
        

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
        return None


    @application.post('/trip')
    def trip():
            ## Generating the spectrogram
        text = input("Write a sentence (+-20 words) to be synthesized:\n")

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
            sd.stop()
            sd.play(generated_wav, synthesizer.sample_rate)
        except sd.PortAudioError as e:
            print("\nCaught exception: %s" % repr(e))
            print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
        except:
            raise

        # Save it on the disk
        global num_generated
        filename = "demo_output_%02d.wav" % num_generated
        print(generated_wav.dtype)
        sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
        num_generated += 1
        print("\nSaved output as %s\n\n" % filename)
                

    @application.post('/z')
    def login_post_2():
            
        
        #print(request.files['file'])
        #print(exists(request.data))
        print(request.files['file'])

        print(exists(request.files['file'].filename))
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
        
        print(exists(request.files['file'].filename))


        filename = uploaded_file.filename

        track = AudioSegment.from_file(filename,  format= 'm4a')
        file_handle = track.export('newFILE.wav', format='wav')


        print("Audio recorded...")

        #---OPTIONAL- play the iput file
        #wave_obj = sa.WaveObject.from_wave_file(filename)
        #play_obj = wave_obj.play()
        #play_obj.wait_done()  # Wait until sound has finished playing


        text = "Default"
        filename = 'newFILE.wav'

        r = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            text = model.transcribe(audio_data, language='english')
            print("You said: \"" + text["text"] + "\"")

        
        #wav_audio = AudioSegment.from_file(request.data[7:], format="caf")
        #wav_audio.export("audio.mp3", format="mp3")

        return text["text"]


    return application