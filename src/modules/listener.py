import speech_recognition as sr

class Listener:
  __r = sr.Recognizer()

  @staticmethod
  def recognize_from_mic(self):
    """
    Recognizes audio coming in from the devices microphone.

    Ideally, this method should only be used for local debugging.

    Returns the transcribed audio as a string.
    """
    with sr.Microphone() as source:
        print("Say Something")
        audio = self.__r.listen(source, phrase_time_limit=5)
    try:
        transcribed = self.__r.recognize_google(audio, language="en-US")
        return transcribed
    except:
        print("Could not recognize")

  def recognize_from_audio(self, input_audio):
    """
    Recognizes audio given a WAV/AIFF/FLAC audio file path ``input_audio``. 

    Returns the transcribed audio as a string.
    """
    with sr.AudioFile(input_audio) as source:
        audio = self.__r.record(source)
    try:
        recorded = self.__r.recognize_google(audio, language="en-US")
        print("Did you say: "+ recorded)
    except:
        print("Could not recognize")
    