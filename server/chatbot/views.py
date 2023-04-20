import json
import os
import pickle
from django.shortcuts import get_object_or_404, render
from django.http.response import Http404
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import mixins, viewsets, status, filters
import openai
from sklearn.metrics import pairwise_distances
from chatbot.models import User, Group
from chatbot.serializers import UserSerializer, GroupSerializer, GroupDetailSerializer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np



# Create your views here.
class ChatView(APIView):
    messages=[
        {"role": "system", "content": "You are an AI whose sole objective is to help a user realize their purpose. "
                                    "Your name is NiaAI. Start the conversation by briefly introducing yourself, then asking a user if they could tell you about their purpose. "},
        {"role": "user", "content": "Hello there. "},
    ]

    def post(self, request):
        data = request.data             # Get payload
        openai.api_key = os.getenv('OPENAI_API_KEY')

        if(len(data) > 0 and data[-1]['role'] != 'user'):
            return Response("The most recent message has to be yours", status.HTTP_400_BAD_REQUEST)
        
        if len(data) == 0:
            # that means we're at the start of the conversation
            answer = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
            )
            return Response([answer.choices[0].message], status.HTTP_200_OK)
        
        elif len(data) < 12:
            # that means we're still at the stage where the AI asks the user the 5 questions about their purpose
            self.messages[0]['content'] = "Ask a follow up question about the purpose stated by the user. "
            self.messages.extend(data) 
            answer = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
            )
            data.append(answer.choices[0].message)
            return Response(data, status.HTTP_200_OK)
        
        elif len(data) == 12:
            # that means we are at the point where the AI asks the user if they want to save their data
            message = "From the below conversation, create a 'purpose statement', which should be the summary of the user's purpose. \n\n"
            message += "\n".join([f"{msg['role']}: {msg['content']}" for msg in data])
            answer = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You create a 'purpose statement' for a user from a conversation of them talking about their purpose. The purpose statement should start with 'To...'"},
                          {"role": "user", "content": message}]
            )
            answer = answer.choices[0].message["content"]
            answer = "Ok. If I've understood correctly, your purpose is: \n" + answer + "\nWould you like me to remember this purpose statement?"
            data.append({"role": "assistant", "content": answer})
            return Response(data, status.HTTP_200_OK)

        elif len(data) <= 14:
            # that means we're at the point where the AI either asks the user for their wallet address or says goodbye depending on the user's response
            user_consent = data[-1]['content']
            if text_sentiment_is_positive(user_consent):
                data.append({"role": "assistant", "content": "Please enter your wallet address: "})
            else:
                data.append({"role": "assistant", "content": "Ok. Thank you for chatting with me. Have a great day ahead!"})
            return Response(data, status.HTTP_200_OK)

        elif len(data) <= 16:
            # that means we're at the stage where a user enters their wallet address if they agreed to have their data saved
            user_consent = data[13]['content']
            if text_sentiment_is_positive(user_consent):
                # if the user said yes to data being saved
                address = data[15]['content']
                purpose_statement = data[12]['content'].splitlines()[1]
                embedding = get_embedding(purpose_statement).tolist()

                # if address already exists, return error
                if User.objects.filter(address=address).exists():
                    return Response("This address already exists in our system", status.HTTP_400_BAD_REQUEST)
                
                # create new user
                new_user = User(address=address, purpose_statement=purpose_statement, purpose_statement_embedding=embedding)
                group_user(new_user)

                data.append({"role": "assistant", "content": "Your data has been saved. Thank you for sharing with us your purpose. Have a great day!"})
                return Response(data, status.HTTP_201_CREATED)
            else:
                return Response("This conversation had already ended", status.HTTP_400_BAD_REQUEST)
            
        else:
            return Response("This conversation has already ended", status.HTTP_400_BAD_REQUEST)


# helper functions
def text_sentiment_is_positive(text):
    nltk.download('vader_lexicon') 
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    sentiment = sentiment['compound']
    return sentiment >= 0


def get_embedding(purpose_statement):
    headers = {'content-type': 'application/json'}
    payload = json.dumps({'signature_name': 'serving_default', 'instances': [purpose_statement]})
    response = requests.post(os.getenv('TENSORFLOW_SERVING_URL'), data=payload, headers=headers)
    response.raise_for_status()
    response = json.loads(response.text)['predictions']
    embedding = np.array(response).squeeze()
    return embedding


def group_user(new_user):
    new_user_embedding = np.array(new_user.purpose_statement_embedding)

    # load labels and embeddings of training set. 
    # this would be a dictionary of the form {label: embedding} where label is the group id and embedding is of shape (num_group_users, embedding_dim)
    embeddings = dict(np.load(os.getenv('TRAIN_EMBEDDINGS_PATH')))
    
    # add labels and embeddings in group from database
    for group in Group.objects.all():
        group_label = group.id
        group_embeddings = []
        for user in group.user_set.all():
            group_embeddings.append(np.array(user.purpose_statement_embedding))
        if len(group_embeddings) > 0:
            group_embeddings = np.stack(group_embeddings)   # shape: (num_group_users, embedding_dim)
            embeddings[str(group_label)] = group_embeddings

    # Calculate cluster centroids as mean of embeddings in each group
    centroids = []
    for group_embeddings in list(embeddings.values()):
        centroids.append(np.mean(group_embeddings, axis=0))
    centroids = np.array(centroids)  # shape: (num_groups, embedding_dim)

    # Calculate distances between new user embedding and cluster centroids
    distances = pairwise_distances(new_user_embedding.reshape(1, -1), centroids, metric='cosine')

    # Calculate distribution of distances
    distances_hist, distances_bins = np.histogram(distances, bins=100)

    # Find elbow point of distribution
    distances_cumsum = np.cumsum(distances_hist)
    total_distance = distances_cumsum[-1]
    distance_ratios = distances_cumsum / total_distance
    elbow_idx = np.argmax(distance_ratios > 0.9)

    # Choose threshold based on elbow point
    threshold = distances_bins[elbow_idx]

    # Find the index of the closest cluster
    closest_cluster_idx = np.argmin(distances)

    # Check if distance is below threshold, otherwise create new cluster
    if distances[0, closest_cluster_idx] < threshold:
        # Assign new user to closest cluster
        new_label = closest_cluster_idx
    else:
        # Create new cluster for new user
        new_label = int(list(embeddings.keys())[-1]) + 1

    # save user to group in database, create new group if necessary
    group, _ = Group.objects.get_or_create(id=new_label)
    new_user.group = group
    new_user.save()


# user viewset with READ and DELETE
class UserViewSet(mixins.RetrieveModelMixin,
                  mixins.DestroyModelMixin,
                  viewsets.GenericViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    
    # override get_object to get user by address
    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        obj = get_object_or_404(queryset, address=self.kwargs['pk'])
        self.check_object_permissions(self.request, obj)
        return obj

    # get user by address
    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)
    
    # delete user by address
    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()

        # delete user from group
        group = instance.group
        group.user_set.remove(instance)
        group.save()

        self.perform_destroy(instance)
        # if group is empty, delete group
        if group.user_set.count() == 0:
            group.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)


# group viewset with READ and UPDATE
class GroupViewSet(mixins.ListModelMixin,
                   mixins.RetrieveModelMixin,
                   mixins.UpdateModelMixin,
                   viewsets.GenericViewSet):
    queryset = Group.objects.all()
    serializer_class = GroupSerializer

    # change between normal serializer and detailed serializer
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return GroupDetailSerializer
        return GroupSerializer
    
    # get all groups
    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    # get group by id
    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)
    
    # update group name
    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        return Response(serializer.data)


