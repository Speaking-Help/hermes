import os
from pydub import AudioSegment
from os.path import exists
# import required libraries s
import speech_recognition as sr
from happytransformer import HappyTextToText, TTSettings
from gtts import gTTS
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

from flask import Flask
from flask import request
from cloner import Cloner
from transcriber import Transcriber
from grammar_corrector import Grammar_Corrector

embed = None
num_generated = None
synthesizer = None

cloner = Cloner()
grammarCorrect = Grammar_Corrector()
transcriber = Transcriber()


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
        """
        Recieves text with grammatical errors
        Fixes grammatical errors in text
        Returns fixed text and changed indices
        """

        # Recieve text values
        text = request.get_json().get('value')

        # Retrieve fixed text versions
        fixedText = grammarCorrect.correct_sentence(text)

        ## deltas = grammarCorrect.find_deltas(text, fixedText)
        # TODO return deltas as well

        return fixedText



    @application.post('/upload')
    def login_post():
        """
        Recieves audio file
        Transcribes text from audio file
        Returns transcribed text
        """

        # Recieve audio file and prepare for processing
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
        filename = uploaded_file.filename
        path = Path(filename)

        # Return transcribed audio
        text = transcriber.transcribe_from_audio_whisper(path)
        return text

    @application.post('/toAudio')
    def TTS():
        """
        Recieves grammatically correct text
        Completes text to speech operation using clone voice
        Returns audio file of voice clone speaking text
        """

        # Recieve text and language values
        data = request.get_json()
        text = data.get('value')
        language = data.get('language').lower()

        # Synthesize cloned voice using text
        cloner.synthesize(text, language)


    @application.post('/train')
    def train():
        """
        Recieves audio file of user speaking
        Trains voice cloner to create user voice clone
        """

        # Retrieve audio file
        audioFile = request.files['file']

        # Train cloner on audio file
        cloner.train(audioFile)

        return None

    return application
