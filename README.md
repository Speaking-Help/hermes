# Hermes
Back-end responsible for grammar correcting a dictated phrase and outputting it in a synthesized audio format.

# Usage
````
git clone ----
cd hermes
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
pip3 install -U git+https://github.com/PrithivirajDamodaran/Gramformer.git
pip3 install git+https://github.com/openai/whisper.git
python -m spacy download en
// NOTE THAT THE FOLLOWING STEP TO ADD FFMPEG, A REQUIRED COMPONENT,
// REQUIRES HOMEBREW ON MACOS (https://brew.sh).
// ON WINDOWS AND LINUX INSTALL FFMPEG HOWEVER APPROPRIATE.
brew install ffmpeg
cd flaskr
flask run
````
 
# Structure

hermes\
&emsp;flaskr\
&emsp;&emsp;app.py\
&emsp;src/modules\
&emsp;&emsp;encoder\
&emsp;&emsp;saved_models\
&emsp;&emsp;synthesizer\
&emsp;&emsp;utils\
&emsp;&emsp;vocoder\
&emsp;&emsp;archived.py\
&emsp;&emsp;cloner.py\
&emsp;&emsp;grammar_corrector.py\
&emsp;&emsp;transcriber.py

# API Calls

- Post ("/train") with audio file (short snippet, around 4-8 seconds of clear and constant speaking) | trains an instance of the voice cloner.
- Post ("/upload") with audio file of a phrase to transcribe | returns text transcribed by OpenAI's Whisper 
- Post ("/grammarlyify") with incorrect text snippet | returns fixed text snippet
- Post ("/toAudio") with correct text | recieve text in the sound of their voice 
- Post request to OpenAI's API for chatbot.

# User Flow in Demo App

- Immediately upon signing into app, user makes an API call to '/train' to train the voice cloner. 
- Now, user can go to chat screen and interface with UI. They record a phrase they want to fix, and the app will send a request to '/upload' to recieve a transcription of what they said; then, we get the fixed version of that text using '/grammarlyify' and we display that text on screen.
- The user now has the option to click on the text display to send a request to '/toAudio' and hear their voice say the fixed text back to them.
- From here, the user can either re-record more snippets to fix, or they can retrain the UI.


notes from walkthrough:

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
weasel 0.3.4 requires srsly<3.0.0,>=2.4.3, but you have srsly 1.0.6 which is incompatible.
pydantic-core 2.14.6 requires typing-extensions!=4.7.0,>=4.6.0, but you have typing-extensions 4.4.0 which is incompatible.
confection 0.1.4 requires srsly<3.0.0,>=2.4.0, but you have srsly 1.0.6 which is incompatible.
