import speech_recognition as sr
import whisper

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
    """
    Transcribes audio given a WAV/AIFF/FLAC audio file path ``input_audio``. 

    Returns the transcribed audio as a string.
    """
    with sr.AudioFile(input_audio) as source:
        audio = self._r.record(source)
    try:
        transcribed = self._r.recognize_google(audio, language="en-US")
        return transcribed
    except:
        print("Could not recognize")

  def transcribe_from_audio_whisper(self, input_audio):
    """
    Transcribes audio given a WAV/AIFF/FLAC audio file path ``input_audio``. 
    Uses OpenAI's Whisper 'base.en' model.

    Returns the transcribed audio as a string.
    """
    result = self._model.transcribe(input_audio, language='english')
    return result["text"]
