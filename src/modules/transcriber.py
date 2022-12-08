import speech_recognition as sr
import whisper

from scipy.io import wavfile
import os
import noisereduce as nr
from pathlib import Path
from pydub import AudioSegment

class Transcriber:

  # Create the recognizers
  _r = sr.Recognizer()
  _model = whisper.load_model("base.en")

  # Transcribes using 
  def transcribe_from_mic(self):
    """
    Transcribes audio coming in from the devices microphone.

    Ideally, this method should only be used for local debugging.

    Returns the transcribed audio as a string.
    """
    with sr.Microphone() as source:
        print("Say Something")
        audio = self._r.listen(source)
    try:
        return transcribe_from_audio(audio)
    except:
        print("Could not recognize")

  def transcribe_from_audio(self, input_audio):

    if input_audio.filename != '':
            input_audio.save(input_audio.filename)

    filename = input_audio.filename

    track = AudioSegment.from_file(filename,  format= 'm4a')
    file_handle = track.export('newFILE.wav', format='wav')

    text = "Default"
    filename = 'newFILE.wav'

    """
    Transcribes audio given a WAV/AIFF/FLAC audio file path ``input_audio``. 

    Returns the transcribed audio as a string.
    """
    print("TYPE IS " + str(type(input_audio)))

    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)

    return text

  def transcribe_from_audio_whisper(self, input_audio):
    """
    Transcribes audio given a WAV/AIFF/FLAC audio file path ``input_audio``. 
    Uses OpenAI's Whisper 'base.en' model.

    Returns the transcribed audio as a string.
    """

    # rate, data = wavfile.read('sample.wav')
    # reduced_noise = nr.reduce_noise(y=data, sr=rate)
    # wavfile.write("denoised.wav", rate, reduced_noise)
    if input_audio.filename != '':
      input_audio.save(input_audio.filename)

    filename = input_audio.filename

    track = AudioSegment.from_file(filename,  format= 'm4a')
    file_handle = track.export('newFILE.wav', format='wav')

    text = "Default"
    pa = Path('newFILE.wav')


    result = self._model.transcribe(audio='newFile.wav', language='english')
    print("RESULT IS " + str(result["text"]))
    print(result)
    print("\n\n\n\n\n")
    # os.remove("sample.wav")

    return result["text"]