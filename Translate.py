
# coding: utf-8

# In[32]:
from flask import Flask
from flask_ask import session, Ask, statement, question, request
app= Flask(__name__)
ask = Ask(app, '/')

@app.route('/')
def homepage():
    return "Translate"

@ask.launch
def start_skill():
    welcome_message= 'Hello, would you like me to translate for you?'
    return question(welcome_message)
####Ryan Blow
####Utilizes the googletrans api

@ask.intent("YesIntent")
def language():
    language_message = "What language would you like me to translate to?"
    return question(language_message)

@ask.intent("LanguageIntent")
def language(Language):
    session.attributes['language']=Language
    message_message = "What message would you like me to translate? Please say translate then your sentence"
    return question(message_message)

@ask.intent("MessageIntent")
def message_alexa(message):
    session.attributes['message']=message
    translated_message=translator_function()
    return statement(translated_message)

def translator_function():
    from googletrans import Translator, LANGUAGES
    translator = Translator()
    final_language=session.attributes['language']
    audio_string = session.attributes['message']
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
    final_translate=translator.translate(audio_string, dest=translated_language, src= language_initial.lang)
    #back_language=translator.translate(final_translate.text)
    if final_translate.dest=='it':
        output = final_translate.text
    elif final_translate.dest=='ja':
        output = final_translate.text
    elif final_translate.dest=='de':
        output = final_translate.text
    elif final_translate.dest=='fr':
        output = final_translate.text
    elif final_translate.dest=='es':
        output = final_translate.text
    else:
        translation=[]
        audio_string_list = audio_string.split()
        for i in range(len(audio_string_list)):
            final_translate = translator.translate(audio_string_list[i], dest=translated_language, src=language_initial.lang)
            if final_translate.pronunciation is None:
                translation.append(final_translate.text)
            else:
                translation.append(final_translate.pronunciation)
        output = " ".join(translation)
    return output
@ask.intent('AMAZON.CancelIntent')
@ask.intent('AMAZON.StopIntent')
@ask.intent('AMAZON.NoIntent')
def stop_alexa():
    quit = "Alright, goodbye."
    return statement(quit)

if __name__ == '__main__':
    app.run(debug=True)