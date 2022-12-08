# Hermes
Back-end responsible for grammar correcting a dictated phrase and outputting it in a synthesized audio format.


# Structure

hermes
    flaskr
        app.py
    src/modules
        encoder
        saved_models
        synthesizer
        utils
        vocoder
        archived.py
        cloner.py
        grammar_corrector.py
        transcriber.py

# API Calls

- Post ("/train") with audio file (short snippet, around 4-8 seconds of clear and constant speaking) | trains an instance of the voice cloner.
- Post ("/upload") with audio file of a phrase to transcribe | returns text transcribed by OpenAI's Whisper 
- Post ("/grammarlyify") with incorrect text snippet | returns fixed text snippet
- Post ("/toAudio") with correct text | recieve text in the sound of their voice 
- Post request to OpenAI's API for chatbot.

# User Flow in Demo App

Immediately upon signing into app, user makes an API call to '/train' to train the voice cloner. 
Now, user can go to chat screen and interface with UI. They record a phrase they want to fix, and the app will send a request to '/upload' to recieve a transcription of what they said; then, we get the fixed version of that text using '/grammarlyify' and we display that text on screen.
The user now has the option to click on the text display to send a request to '/toAudio' and hear their voice say the fixed text back to them.
From here, theh user can either re-record more snippets to fix, or they can retrain the UI.