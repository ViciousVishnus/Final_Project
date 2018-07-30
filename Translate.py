
# coding: utf-8

# In[32]:
from flask import Flask
app= Flask(__name__)
from flask_ask import session, Ask, statement, question
@app.route('/')
def homepage():
    return "Translate"
@ask.launch
def start_skill():
    welcome_message= 'Hello, would you like me to translate for you?'
    return welcome_message
####Ryan Blow
####Utilizes the googletrans api
@ask.intent("YesIntent")
def translator_function():
    language_message = "What language would you like me to translate to?"
    language = question(language_message)
    message_message = "What message would you like me to translate?"
    message=question(message_message)
    from googletrans import Translator, LANGUAGES
    translator = Translator()
    final_language=language
    audio_string = message
    """
    Takes in audio and translates it into a new language
    
    Parameters:
    
        audio: The string which you wish to translate
        
        final_language: The language which you wish to translate to 
        
    Returns: 
    
        Translated_string: Gives back the translated string for alexa to speak back
    """
    #figures out what the origin language is 
    language_initial= translator.detect(audio_string)
    #figure out the output language
    language_codes = dict(map(reversed, LANGUAGES.items()))
    final_language=final_language.lower()
    translated_language=language_codes[final_language]
    #translate the thing
    final_language=translator.translate(audio_string, dest=translated_language, src= language_initial.lang)
    return statement(final_language.text)
@ask.intent('AMAZON.CancelIntent')
@ask.intent('AMAZON.StopIntent')
@ask.intent('AMAZON.NoIntent')
def stop_alexa():
    quit = "Alright, goodbye."
    return statement(quit)

if __name__ == '__main__':
    app.run(debug=True)