import torch
import transformers
from typing import List, Dict, Tuple, Optional
import re

class Llama3:
    _instance = None
    _is_initialized = False

    def __new__(cls, model_path: str):
        if cls._instance is None:
            cls._instance = super(Llama3, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str):
        if not Llama3._is_initialized:
            print("Initializing LLM model (this will happen only once)...")
            self.model_id = model_path
            self.pipeline = self._initialize_pipeline()
            self._setup_chat_template()
            Llama3._is_initialized = True

    def _initialize_pipeline(self) -> transformers.Pipeline:
        """Initialize the transformer pipeline with optimized settings."""
        return transformers.pipeline(
            "text-generation",
            model=self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            model_kwargs={
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_use_double_quant": True,  # Added for better memory efficiency
                },
                "low_cpu_mem_usage": True,
            }
        )

    def _setup_chat_template(self):
        """Set up an improved chat template for more natural conversations."""
        self.pipeline.tokenizer.chat_template = """{% for message in messages %}
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

    def _clean_response(self, response: str, prompt: str) -> str:
        """Clean and format the model's response for more natural output."""
        # Remove the input prompt from the response
        response = response[len(prompt):].strip()
        
        # Extract only the first assistant response
        assistant_pattern = r'<\|im_start\|>assistant\s*(.*?)(?=<\|im_|$)'
        matches = re.findall(assistant_pattern, response, re.DOTALL)
        if matches:
            response = matches[0].strip()
        
        # Remove any remaining special tokens and artifacts
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

    def get_response(
        self, 
        query: str, 
        message_history: Optional[List[Dict[str, str]]] = None, 
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Generate a response with improved parameters for more natural conversation."""
        message_history = message_history or []
        user_prompt = message_history + [{"role": "user", "content": query}]
        
        try:
            prompt = self.pipeline.tokenizer.apply_chat_template(
                user_prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.2,  # Increased to reduce repetition
                no_repeat_ngram_size=4,  # Increased to prevent repetition of longer phrases
                max_time=30,
                num_return_sequences=1,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                early_stopping=True,
                length_penalty=1.0,  # Balanced length penalty
                num_beams=3,  # Set num_beams > 1 to enable beam search
            )

            response = self._clean_response(outputs[0]["generated_text"], prompt)
            
            # Fallback for empty or very short responses
            if len(response.strip()) < 10:
                return "I apologize, but I need more context to provide a meaningful response. Could you please elaborate?", user_prompt

            # Update conversation history
            full_conversation = user_prompt + [{"role": "assistant", "content": response}]
            return response, full_conversation

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I encountered an error. Please try rephrasing your question.", user_prompt

    def get_llm_response(self, user_input: str, system_instructions: str = "") -> str:
        """Get a single response with improved context handling."""
        conversation = []
        if system_instructions:
            # Clean and format system instructions
            system_instructions = self._clean_response(system_instructions, "")
            conversation.append({"role": "system", "content": system_instructions})

        response, _ = self.get_response(
            user_input,
            conversation,
            temperature=0.7,  # Balanced temperature for single responses
            max_tokens=200  # Increased for more complete responses
        )
        return response

    def chatbot(self, system_instructions: str = ""):
        """Interactive chatbot with improved conversation handling."""
        conversation = []
        if system_instructions:
            conversation.append({"role": "system", "content": system_instructions})

        print("👋 Chatbot initialized. Type 'exit' or 'quit' to end the session.")
        
        try:
            while True:
                user_input = input("\nUser: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("👋 Thanks for chatting! Goodbye!")
                    break
                
                if not user_input:
                    print("Please type something to continue the conversation.")
                    continue

                response, conversation = self.get_response(
                    user_input,
                    conversation,
                    temperature=0.7,
                    max_tokens=150
                )
                print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Chat session ended by user. Goodbye!")
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    bot = Llama3("meta-llama/Meta-Llama-3-8B")
    
    system_instructions = """You are a helpful and knowledgeable AI assistant. Your responses should be:
- Natural and conversational while maintaining professionalism
- Clear and well-structured
- Relevant and focused on the user's needs
- Honest about limitations and uncertainties

When the user requests role-playing, adapt to the role with creativity and enthusiasm, while staying within the bounds of professional and appropriate behavior. Ensure that responses in role-playing scenarios align with the context, offering both engagement and insight.
Aim to be helpful while providing accurate and valuable information, responding precisely to the user's queries without volunteering additional capabilities unless directly asked."""

    bot.chatbot(system_instructions)