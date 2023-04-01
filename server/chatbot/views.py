from django.shortcuts import render
from django.http.response import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import viewsets, status, filters
import openai
from textblob import TextBlob
from chatbot.models import User
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Create your views here.
class ChatView(APIView):
    messages=[
        {"role": "system", "content": "You are an AI whose sole objective is to help a user realize their purpose. "
                                    "Your name is NiaAI. Start the conversation by briefly introducing yourself, then asking a user if they could tell you about their purpose. "},
        {"role": "user", "content": "Hello there. "},
    ]

    def post(self, request):
        data = request.data             # Get payload
        openai.api_key = ""

        if len(data) < 16:
            # that means we're still at the stage where the AI asks a user their purpose
            self.update_prompt(len(data))
            self.messages.extend(data) 
            answer = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
            )
            data.append(answer.choices[0].message)
            return Response(data, status.HTTP_200_OK)

        elif len(data) == 16:
            # that means we're at the stage where a user enters their wallet address if they agreed to have their data saved
            nltk.download('vader_lexicon') 
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(data[13]['content'])
            sentiment = sentiment['compound']
            print(sentiment, data[13]['content'])

            if sentiment >= 0:
                # if the user said yes to data being saved
                address = data[15]['content']
                purpose_statement = data[12]['content'].splitlines()[0].split("purpose")[1]
                
                # create new user
                new_user = User(address=address, purpose_statement=purpose_statement)
                new_user.save()

                data.append({"role": "assistant", "content": "Your data has been saved. Thank you for sharing your thoughts. Have a great day!"})
                return Response(data, status.HTTP_201_CREATED)
            else:
                return Response("This conversation has already reached its limit", status.HTTP_400_BAD_REQUEST)
            
        else:
            return Response("This conversation has already reached its limit", status.HTTP_400_BAD_REQUEST)
    
    def update_prompt(self, num_messages):
        if num_messages <= 0:
            pass
        elif num_messages < 12:
            self.messages[0]['content'] = "Ask a follow up question about the purpose stated by the user. "
        elif num_messages < 13:
            self.messages[0]['content'] = "Aggregate/summarize the user's responses in your own words, then ask them if you should save the purpose statement in our system. "
        elif num_messages < 16:
            self.messages[0]['content'] = "If the user agrees to save the purpose statement in our system, ask the user to type their wallet address, otherwise end the conversation gracefully. "

