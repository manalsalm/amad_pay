import requests
import json

class Ather:

    def process_query(self, query):

        payload = {
            "model": "llama2",
            "prompt": query,
            "stream": False,
            "options": {
            'temperature': 0.7,     # Creativity level (0.0 to 1.0)
            'top_p': 0.9,          # Nucleus sampling
            'top_k': 40,           # Top-k sampling
            'repeat_penalty': 1.1,  # Penalty for repetition
            'num_ctx': 2048,       # Context window size
            }
        }

        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response_json = response.json()
        
        ai_response = response_json["response"] 
        return ai_response

if __name__ == "__main__":
    ather = Ather()

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        result = ather.process_query(user_input)
        print(f"Ather: {result}")
