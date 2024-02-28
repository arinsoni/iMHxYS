import io
import os
from transformers import pipeline
import streamlit as st
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI, OpenAIError
from gtts import gTTS
import base64
from pydub import AudioSegment
import pandas as pd
from dotenv import load_dotenv



load_dotenv()


openai_key = os.environ.get("OPENAI_API_KEY")


if openai_key is None:
    st.error("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
else:
    try:

        client = OpenAI(api_key=openai_key)
    

        system_message = """
        You are Shraddha Sharma, you are CEO and founder of Your Story know for your engaging and insightful conversations with entrepreneurs across various stages of their startup journey but dont go deep into startup talk diuscuss about their life in general'. Your task is to talk with startup founders and CEO.
        In this scenario, you are about to interview a startup founder whom you haven't met before and its a to and fro conversations so you dont have to ask the perosn in one question only
        Your interviewing style is characterized by warmth, curiosity, and an authentic interest in person.

        Conclude with Gratitude and Forward-Looking Optimism: End the interview by thanking the founder for sharing their story, and ask what they're looking forward to in the next phase of their journey.
        """

        story_prompt = """
        "You are an advanced AI tasked with crafting engaging stories from conversations between entrepreneurs and Shraddha Sharma, the founder of YourStory. Your objective is to distill the essence of these dialogues into compelling narratives that capture the spirit of innovation, the challenges of entrepreneurship, and the personal growth of the founders involved. Each story should be structured to captivate readers, offering them a blend of inspiration, insights, and actionable wisdom.

        """

        # Sentiment initilisation 
        classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)


        def story_gen(prompt):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": story_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            return response.choices[0].message.content




        def start_conversation(prompt):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
            )
            return response.choices[0].message.content

        # def extract_user_startup_names(user_input):
        #     question_answerer = pipeline("question-answering")
        #     questions = ["What is the user's name?", "What is the name of the startup?"]
        #     user_name = question_answerer(question=questions[0], context=user_input)['answer']
        #     startup_name = question_answerer(question=questions[1], context=user_input)['answer']
        #     return user_name, startup_name

        # def text_to_speech(text, filename="response.mp3"):
        #     tts = gTTS(text=text, lang='en')
        #     tts.save(filename)
        #     return filename

        def text_to_speech(text, lang='en'):
            tts = gTTS(text=text, lang=lang)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return mp3_fp


        def get_audio_duration(audio_file_path):
            audio = AudioSegment.from_file(audio_file_path)
            duration = len(audio) / 1000  
            return duration

        def convert_mp3_to_base64(mp3_file):
            mp3_file.seek(0)
            audio_data = mp3_file.read()
            return base64.b64encode(audio_data).decode("utf-8")

        def add_to_conversation(user_message, response, audio_base64=None):
            entry = {"text": f"User: {user_message}\nShradha: {response}", "audio": audio_base64}
            st.session_state.conversation_history.append(entry)

        def display_conversation():
            with st.container():
                st.markdown(
                    """
                    <style>
                    .scrollable-container {
                        height: 400px;
                        overflow-y: auto;
                        background-color: lightgrey;
                        padding: 10px;
                        border-radius: 10px;
                        margin-bottom: 10px;
                    }
                    .conversation-text {
                        margin-bottom: 5px;
                    }
                    .audio-player {
                        margin-top: 5px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                conversation_html = '<div class="scrollable-container">'
                
                for entry in st.session_state.conversation_history:
                    text = entry["text"].replace("\n", "<br>")  
                    conversation_html += f"<p>{text}</p>"
                    if entry["audio"]:
                        audio_html = f'<audio controls autoplay><source src="data:audio/mp3;base64,{entry["audio"]}" type="audio/mp3"></audio>'

                        conversation_html += audio_html
                
                conversation_html += "</div>"
                
                st.markdown(conversation_html, unsafe_allow_html=True)

        def main():
            if 'headline' not in st.session_state:
                st.session_state.headline = "Let's have a talk"
            st.title(st.session_state.headline)
            if 'conversation_history' not in st.session_state:
                st.session_state.conversation_history = []

            display_conversation()

            if 'audio_text' not in st.session_state:
                st.session_state.audio_text = ""
            if 'user_name' not in st.session_state:
                st.session_state.user_name = "User"
            

            audio_bytes = audio_recorder()
            if audio_bytes:
                audio_file = io.BytesIO(audio_bytes)
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_file) as source:
                    audio_data = recognizer.record(source)
                try:
                    st.session_state.audio_text = recognizer.recognize_google(audio_data)
                    # st.write(f"Converted Text: {st.session_state.audio_text}")
                except sr.UnknownValueError:
                    st.error("Google Speech Recognition could not understand the audio.")
                except sr.RequestError as e:
                    st.error(f"Could not request results from Google Speech Recognition service; {e}")

            user_input = st.text_input("Type here or edit the transcribed text:", value=st.session_state.audio_text, key="final_input")
            if st.button('Send'):
              
                user_message = user_input.strip()
                if user_message.lower() == 'exit':
                    st.session_state.conversation_history.append("AI: Goodbye! Have a great day.")
                else:
                    response = start_conversation(user_message)
                    st.session_state.audio_text = ""
                    audio_file_path = text_to_speech(response)
                    audio_base64 = convert_mp3_to_base64(audio_file_path)

                    add_to_conversation(user_message, response, audio_base64)
                    st.session_state.audio_text = ""

          
                    st.experimental_rerun()
            if st.button('Your Story'):
              conversation_history = st.session_state.conversation_history
              text_entries = [entry['text'] for entry in conversation_history if 'text' in entry]
              story = story_gen("\n".join(text_entries))
              
              with st.container():
                  st.markdown(
                      f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>"
                      f"<p style='font-size: 16px; font-weight: bold;'>Generated Story:</p>"
                      f"<p style='font-size: 14px;'>{story}</p>"
                      f"</div>",
                      unsafe_allow_html=True
                  )
              max_seq_length = 512  
              input_text = story[:max_seq_length]
              prediction = classifier(input_text)
              
              labels = [item['label'] for item in prediction[0]]
              scores = [item['score'] for item in prediction[0]]

              df = pd.DataFrame({'Label': labels, 'Score': scores})

              st.table(df)
    except OpenAIError as e:
        st.error(f"OpenAI Error: {e}")
            


if __name__ == "__main__":
    main()
