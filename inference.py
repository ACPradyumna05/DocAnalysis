import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL_PATH = "/home/prady/dparse/models/mistral-7b"   
ADAPTER_PATH = "/home/prady/dparse/project/mistral_lora_out/adapter"


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    use_fast=False
)


print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)


print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable policy assistant. "
    "You answer questions clearly, concisely, and with reasoning. "
    "If additional context is needed, you ask the user politely."
)

def chat():
    print("\n🔥 Custom Chatbot Ready (Mistral-7B + LoRA)\nType 'exit' to quit.\n")

    while True:
        user_msg = input("🧑 User: ").strip()
        if user_msg.lower() in ["exit", "quit"]:
            break

        prompt = (
            f"### Instruction:\n{SYSTEM_PROMPT}\n\n"
            f"### Input:\n{user_msg}\n\n"
            f"### Response:\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()

        print(f"🤖 Model: {response}\n")

if __name__ == "__main__":
    chat()
