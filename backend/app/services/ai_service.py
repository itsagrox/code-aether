import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face API key (for optional authentication if needed)
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Load the model and tokenizer locally
def load_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=huggingface_api_key)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        use_auth_token=huggingface_api_key
    )

    return model, tokenizer, device

# Initialize the model and tokenizer
model, tokenizer, device = load_model()

# Function to refactor code using the loaded model
def refactor_code_locally(code: str, language: str,model_type:str):
    prompt = f"""
        You are an AI that strictly outputs refactored code following best practices. 
        Do not add any explanations, messages, or comments. 

        Refactor the following {language} code to follow best practices. 
        Only return the refactored code inside triple backticks:

        ```{language}
        {code}"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=300,
            temperature=0.7,  # Controls randomness (lower is more deterministic)
            top_p=0.9,        # Controls diversity (higher is more diverse)
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to wrap the refactor process with retry logic (if needed)
def call_huggingface_api(code: str, language: str, model_type: str, retries=10, delay=10):
    # Directly use the local model to get refactored code
    try:
        refactored_code = refactor_code_locally(code, language, model_type)
        return refactored_code
    except Exception as e:
        return f"Error during refactoring: {str(e)}"
