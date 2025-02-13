import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Choose a smaller model
model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Force the use of CPU
device = "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_api_key)

# Load model (no quantization)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map="cpu",
    token=huggingface_api_key
)

# Function to refactor code
def refactor_code_locally(code: str, language: str, model_type: str):
    prompt = f"""### Task:
Refactor the following {language} code to follow best practices.
Strictly return only the refactored code. **Do NOT add any explanations, comments, or additional functionality.**
Maintain the original functionality and structure as much as possible.

### Input Code:
{code}

### Refactored {language} Code:
"""

    # Convert input text into tokenized form
    inputs = tokenizer(prompt, return_tensors="pt",return_attention_mask=True).to(device)

    # Generate output while ignoring input tokens
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=500,  # Limit output length
            temperature=0.1,
            top_p=0.7,
            do_sample=True,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )

    # Extract only new tokens (ignore input tokens)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]  # Remove input tokens from output

    # Decode response (keep raw)
    raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\n===== DEBUG: RAW OUTPUT =====\n", raw_response)  # Debugging output

    # Extract only the refactored code part (if needed)
    if "### Refactored" in raw_response:
        raw_response = raw_response.split("### Refactored")[1].strip()

    return raw_response


# Function to call the refactoring process
def call_huggingface_api(code: str, language: str, model_type: str):
    try:
        return refactor_code_locally(code, language, model_type)
    except Exception as e:
        return f"Error: {str(e)}"
