import time
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    # Initialize vLLM engine
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=256, # Limit context to save memory if needed
        gpu_memory_utilization=0.8,
        enforce_eager=True  # Disables CUDA Graph capture to save memory
    )

    prompts = [
        "<|im_start|>user\nWrite a quick sort in Python.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain async/await in JavaScript.<|im_end|>\n<|im_start|>assistant\n"
    ]

    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    # Generate output with continuous batching
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    for output in outputs:
        print(f"Prompt: {output.prompt!r}, \nGenerated: {output.outputs[0].text}\n")

    time_taken = end_time - start_time
    total_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_second = total_generated_tokens / time_taken

    print("\n--- Inference Stats ---")
    print(f"Generated {total_generated_tokens} tokens in {time_taken:.2f} seconds")
    print(f"Speed: {tokens_per_second:.2f} tokens/sec")