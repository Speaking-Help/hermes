from ast import main
import speech_recognition as sr

r = sr.Recognizer()

def recognize_from_mic():
  with sr.Microphone() as source:
      print("Say Something")
      audio = r.listen(source)
  try:
      recd = r.recognize_sphinx(audio, language="en-US")
      print("Did you say: "+ recd)
  except:
      print("Could not recognize")

def recognize_from_audio(input_audio):
  with sr.AudioFile(input_audio) as source:
      audio = r.record(source)
  try:
      recd = r.recognize_sphinx(audio, language="en-US")
      print("Did you say: "+ recd)
  except:
      print("Could not recognize")

recognize_from_mic();
  