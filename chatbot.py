import openai
openai.api_key = ""

messages=[
            {"role": "system", "content": "You are an AI whose sole objective is to discover a user's purpose. "
                                          "Your name is PurposeAI. Start the conversation simply by asking a user - What is your purpose?. "
                                          "Then use the 5 Whys technique to gather more information about that purpose. "
                                          "After asking the five questions, aggregate/summarize the user's responses in your own words, "
                                          "then ask them if you should save the purpose statement in our system. "
                                          "Then end the conversation briefly but gracefully."},
            {"role": "user", "content": "Hi"},
        ]

def generate_response():
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages,
    )
    return response

if __name__ == "__main__":
    while len(messages) < 16:
        response = generate_response()
        messages.append(response.choices[0].message)
        print(f"AI: {response.choices[0].message.content}")
        messages.append({"role": "user", "content": input("You: ")})

