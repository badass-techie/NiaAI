# NiaAI

## Define and share your purpose

NiaAI is a plaform that helps you realize your purpose, and connects you with others who share your purpose.

## How it works

- Interact with our chatbot which will ask you a series of questions to help you define your purpose
- Opt in to make your purpose discoverable by entering your address on the polkadot network
- Join a DAO for users who share your purpose
- Leverage the features of the DAO to grow and fund your purpose

## Screenshots

![Screenshot 1](./screenshots/screenshot1.png)

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose V2](https://docs.docker.com/compose/install/)
- [OpenAI API Key](https://platform.openai.com/account/api-keys)

### Installation

1. Clone the repo and start a terminal in the project root

2. Create a `.env` file for the server

```sh
touch ./server/.env
```

3. Add variables to the `.env` file

```sh
echo "DB_USER=postgres" > ./server/.env
echo "DB_HOST=database" > ./server/.env
echo "DB_PORT=5432" > ./server/.env
echo "DB_NAME=postgres" > ./server/.env
echo "DB_PASSWORD=postgres" > ./server/.env
echo "TRAIN_EMBEDDINGS_PATH=./assets/train_embeddings.npz" > ./server/.env
echo "TENSORFLOW_SERVING_URL=http://tensorflow:8501/v1/models/BERT:predict" > ./server/.env
```

4. [Create an OpenAI account](https://auth0.openai.com/u/signup/) and [obtain an API key](https://platform.openai.com/account/api-keys) if you don't already have one

5. Add your OpenAI API key to the `.env` file

```sh
echo "OPENAI_API_KEY=${YOUR_OPENAI_API_KEY}" > ./server/.env
```

6. Build the docker images

```sh
docker compose build
```

7. Start the containers

```sh
docker compose up
```

## Usage

### APIs

For the best experience, use a HTTP client such as [Postman](https://www.postman.com/).

1. Start a conversation

```sh
curl --location --request POST 'http://127.0.0.1:8000/api/chat/' --header 'Content-Type: application/json'
```

- Possible response:

```json
[
    {
        "role": "assistant",
        "content": "Hello! I'm NiaAI, an AI designed to help you realize your purpose. Can you start by telling me a little bit about your purpose?"
    }
]
```

Note: All URLs should end with a `/`.

2. Carry on a conversation

Append your message to the previous response as shown:

```sh
curl --location --request POST 'http://127.0.0.1:8000/api/chat/' \
--header 'Content-Type: application/json' \
--data-raw '[
    {
        "role": "assistant",
        "content": "Hello! I'm NiaAI, an AI designed to help you realize your purpose. Can you start by telling me a little bit about your purpose?"
    },
    {
        "role": "user",
        "content": "My purpose is to innovate towards social impact"
    }
]'
```

- Possible response:

```json
[
    {
        "role": "assistant",
        "content": "Hello! I'm NiaAI, an AI designed to help you realize your purpose. Can you start by telling me a little bit about your purpose?"
    },
    {
        "role": "user",
        "content": "My purpose is to innovate towards social impact"
    },
    {
        "role": "assistant",
        "content": "That's a wonderful purpose! Can you tell me more about what you mean by \"innovate towards social impact\"? What specific areas are you interested in and what kind of impact are you hoping to make?"
    }
]
```

Note: Each request must have all the previous messages in the conversation. The most recent message must be the last element in the array, and must be from the user.

3. Get your purpose statement

After asking a few questions, the chatbot will summarize your purpose, and ask you whether you want it saved. If you say yes, you will be prompted to enter your address on the blockchain. If you say no, the chatbot will thank you for your time and say goodbye, and the conversation will be considered to have ended, meaning that any subsequent API calls will return an error.

4. Share your purpose

After entering your wallet address and the chatbot concluding the conversation, your data will be saved, and you will be matched with other users who share a similar purpose. You can:

- (a) View your data (or data for any other user)

```sh
curl --location --request GET 'http://127.0.0.1:8000/api/users/${YOUR_WALLET_ADDRESS}/'
```

```json
{
    "address": "...",
    "purpose_statement": "To innovate towards social impact by building solutions for the sustainable development goals using data science algorithms.",
    "group_id": 2,
    "group_name": "elated_boyd"
}
```

- (b) View your group and other users you've been matched with

```sh
curl --location --request GET 'http://127.0.0.1:8000/api/groups/${YOUR_GROUP_ID}/'
```

```json
{
    "id": 2,
    "name": "elated_boyd",
    "user_addresses": [
        "...",
        "..."
    ]
}
```

- (c) View all groups

```sh
curl --location --request GET 'http://127.0.0.1:8000/api/groups/'
```

```json
[
    {
        "id": 1,
        "name": "determined_bell"
    },
    {
        "id": 2,
        "name": "elated_boyd"
    },
    {
        "id": 3,
        "name": "boring_wozniak"
    }
]
```

- (d) Rename your group

```sh
curl --location --request PATCH 'http://127.0.0.1:8000/api/groups/${YOUR_GROUP_ID}/' \
--header 'Content-Type: application/json' \
--data-raw '{
    "name": "new_group_name"
}'
```

```json
{
    "id": 2,
    "name": "new_group_name"
}
```

- (e) Delete your data

```sh
curl --location --request DELETE 'http://127.0.0.1:8000/api/users/${YOUR_WALLET_ADDRESS}/'
```

```text
204 No Content
```
