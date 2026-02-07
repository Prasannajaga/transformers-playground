import torch 
from transformers import GPT2Tokenizer
from customTransformers import DecodeTransformer
from config import FineTuneConfig
import sys 

config = FineTuneConfig()

tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2",
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token
SPECIAL_TOKENS = ["<|story|>", "<|user|>", "<|assistant|>"]
tokenizer.add_tokens(SPECIAL_TOKENS)
tokenizer.pad_token = tokenizer.eos_token 

model = DecodeTransformer(
    num_layers=config.n_layer,
    n_emb=config.n_embd,
    n_head=config.n_head,
    block_size=config.block_size,
    vocab_size=len(tokenizer),
)

# Load fine-tuned checkpoint
checkpoint_path =  sys.argv[1] if len(sys.argv) > 1 else "./finetuned/model_sft_final.pt"
print(f"Loading fine-tuned checkpoint from {checkpoint_path}...")
model.resize_token_embeddings(len(tokenizer))  
checkpoint = torch.load(checkpoint_path, map_location=config.device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)

# # Handle both raw state_dict and checkpoint dict formats
# if isinstance(checkpoint, dict) and "model" in checkpoint:
#     model.load_state_dict(checkpoint["model"])
#     print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
#     if "train_loss" in checkpoint:
#         print(f"Training loss: {checkpoint['train_loss']:.4f}")
#     if "val_loss" in checkpoint:
#         print(f"Validation loss: {checkpoint['val_loss']:.4f}")
# else:
#     model.load_state_dict(checkpoint)
#     print("Loaded checkpoint (raw state_dict)")

model.to(config.device)
model.eval()

print(f"Model loaded on {config.device} and set to eval mode\n")

memory = {}

def extract_entity(question: str):
    stopwords = {"who", "what", "is", "are", "doing"}
    tokens = question.lower().replace("?", "").split()

    for t in tokens:
        if t.isalpha() and t not in stopwords:
            return t
    return None


def extract_assistant_text(decoded: str) -> str:
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1]

    if tokenizer.eos_token in decoded:
        decoded = decoded.split(tokenizer.eos_token)[0]

    return decoded.strip()



@torch.no_grad()
def generate_story(model, tokenizer, entity, device):
    prompt = (
        "<|user|>\n"
        f"tell me a short story about {entity}\n"
        "<|assistant|>\n"
    )

    print("Updated prompt" , prompt)

    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    out = model.generate(
        ids,
        max_new_tokens=64,
        temperature=0.3,
        top_k=20,
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    story = extract_assistant_text(decoded)

    print("response" , story)

    return story


@torch.no_grad()
def answer_question(model, tokenizer, question, device):
    prompt = ( 
        # "<|story|> \n"+
        #  story + "\n"
        "<|user|>\n" + question + "\n"
        "<|assistant|>\n"
    )

    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    out = model.generate(
        ids,
        max_new_tokens=100,
        temperature=0.3,
        top_k=40,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=False) 
    answer = extract_assistant_text(decoded)

    return answer
 
def chat(model, tokenizer, question, device):
    # entity = extract_entity(question) 
    # if entity is None:
    #     return "I don't understand the question."

    # if entity not in memory:
    #     memory[entity] = generate_story(model, tokenizer, entity, device)
    # else:
    #     print("memory Found" , entity, memory[entity])

    return answer_question(
        model,
        tokenizer,
        # memory[entity],
        question,
        device
    )


def main():
    """
    Interactive chat loop for the fine-tuned QA model.
    """
    print("=" * 60)
    print("Fine-Tuned QA Chatbot")
    print("=" * 60)
    print("\nThis chatbot generates stories about entities and answers")
    print("questions about them. Ask questions like:")
    print("  - 'What is Alice doing?'")
    print("  - 'Who is the dragon?'")
    print("\nType 'quit' or 'exit' to stop.\n")
    print("=" * 60)
    
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break
            
            answer = chat(model, tokenizer, question, config.device)
            print(f"\nAssistant: {answer}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
