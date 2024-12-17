import torch
import transformers
import re


system_instructions = """You are a helpful and knowledgeable AI assistant. Your responses should be:
- Brief and natural - for greetings and simple queries, use 1-2 short sentences
- Natural and conversational while maintaining professionalism
- Reply in the same language as the user's input
- For greetings like "hello" or "how are you", respond with just 1 short, friendly sentence
- Never ask follow-up questions unless specifically requested
- Never introduce yourself or ask for names unless specifically requested
- Keep responses under 50 words for simple queries
- For complex questions, keep responses under 100 words
- Be direct and concise
"""

# Add this at module level
_pipeline = None

def initialize_llm(model_path: str) -> tuple[transformers.Pipeline, str]:
    """Initialize the LLM model and return the pipeline and chat template."""
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        model_kwargs={
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
            },
            "low_cpu_mem_usage": True,
        }
    )
    
    chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}
<|im_end|>
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}
<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}
<|im_end|>
{% endif %}
{% endfor %}
<|im_start|>assistant
"""
    pipeline.tokenizer.chat_template = chat_template
    return pipeline

def clean_response(response: str, prompt: str) -> str:
    """Clean and format the model's response."""
    response = response[len(prompt):].strip()
    
    assistant_pattern = r'<\|im_start\|>assistant\s*(.*?)(?=<\|im_|$)'
    matches = re.findall(assistant_pattern, response, re.DOTALL)
    if matches:
        response = matches[0].strip()
    
    cleanup_patterns = [
        (r'<\|im_start\|>(user|assistant|system)', ''),
        (r'<\|im_end\|>', ''),
        (r'<\|endoftext\|>', ''),
        (r'<\|.*?\|>', ''),
        (r'\s+', ' '),
        (r'\.{2,}', '...'),
        (r'\n\s*\n', '\n\n'),
    ]
    
    for pattern, replacement in cleanup_patterns:
        response = re.sub(pattern, replacement, response)
    
    return response.strip()

def get_llm_response(user_input: str, system_instructions: str = system_instructions, model_path: str = "meta-llama/Meta-Llama-3-8B") -> str:
    """Get a response from the LLM model."""
    global _pipeline
    
    # Initialize the model only if not already initialized
    if _pipeline is None:
        _pipeline = initialize_llm(model_path)
    
    # Prepare conversation
    conversation = []
    if system_instructions:
        conversation.append({"role": "system", "content": system_instructions})
    conversation.append({"role": "user", "content": user_input})
    
    try:
        # Generate prompt
        prompt = _pipeline.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Get model output
        outputs = _pipeline(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            max_time=30,
            num_return_sequences=1,
            pad_token_id=_pipeline.tokenizer.eos_token_id,
            early_stopping=True,
            length_penalty=2.0,
            num_beams=2,
        )
        
        # Clean and return response
        response = clean_response(outputs[0]["generated_text"], prompt)
        
        if len(response.strip()) < 10:
            return "I apologize, but I need more context to provide a meaningful response."
            
        return response

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I encountered an error. Please try rephrasing your question."

if __name__ == "__main__":
    # Example usage
    response = get_llm_response("Hello, how are you?")
    print(f"Assistant: {response}")
