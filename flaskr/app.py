# import required libraries s
import os
import sys
from os import path

#Using encoder, synthesizer, utils, vocoder
sys.path.append('../src/modules')

from flask import Flask
from flask import request
import cloner
import transcriber
# import grammar_corrector


# grammarCorrect = grammar_corrector.Grammar_Corrector()
transcriber = transcriber.Transcriber()
cloner = cloner.Cloner()

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
        print("grammarly CALLED")
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
        print("UPLOAD CALLED")
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
        print("TTS CALLED")
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
        return "Completed"


    @application.post('/train')
    def train():
        print("TRAINING CALLED")
        """
        Recieves audio file of user speaking
        Trains voice cloner to create user voice clone
        """

        # Retrieve audio file
        audioFile = request.files['file']

        # Train cloner on audio file
        cloner.train(audioFile)

        return "Completed"

    return application
