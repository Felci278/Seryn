import anthropic
import os

# Set your API key
os.environ["ANTHROPIC_API_KEY"] = "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"

client = anthropic.Anthropic()

def chat_with_claude(message, model="claude-3-haiku-20240307"):
    """Simple function to chat with Claude using the Anthropic SDK directly"""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    response = chat_with_claude("Hello, how are you?")
    if response:
        print(response)