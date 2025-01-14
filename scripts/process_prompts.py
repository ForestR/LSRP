import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI  # Updated import for newer OpenAI client
from tqdm import tqdm
from dotenv import load_dotenv
from enum import Enum
import re

# Add this at the beginning of your script, before any OpenAI operations
load_dotenv()

class APIProvider(Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"

def sanitize_json(raw_text):
    """Remove markdown formatting and sanitize JSON content."""
    raw_text = re.sub(r"```(?:json)?", "", raw_text)
    return raw_text.strip()

class PromptProcessor:
    def __init__(self, 
                 api_key: str = None, 
                 provider: str = "openai",
                 model: str = "gpt-4", 
                 temperature: float = 0.2,
                 debug: bool = False):
        """
        Initialize the PromptProcessor.
        
        Args:
            api_key: API key. If None, will try to get from environment variable
            provider: API provider ("openai" or "deepseek")
            model: Model to use
            temperature: Temperature parameter for generation (0.0 to 1.0)
            debug: Whether to print debug information
        """
        self.provider = APIProvider(provider.lower())
        self.debug = debug
        
        # Set up API key based on provider
        if api_key is None:
            env_key = "OPENAI_API_KEY" if self.provider == APIProvider.OPENAI else "DEEPSEEK_API_KEY"
            api_key = os.getenv(env_key)
            if not api_key:
                raise ValueError(f"{env_key} must be provided or set in environment variable")
        
        # Initialize appropriate client
        if self.provider == APIProvider.OPENAI:
            self.client = OpenAI(api_key=api_key)
        else:  # DEEPSEEK
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            
        self.model = model
        self.temperature = temperature
        
    def process_single_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process a single prompt using the selected API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical risk assessment assistant. Respond with valid JSON, ensuring no repeated keys or structural errors."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            response_text = sanitize_json(response.choices[0].message.content.strip())
            
            if self.debug:
                print("\nRaw API Response:")
                print(response_text)
                print("\n---")
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as je:
                if self.debug:
                    print(f"JSON parsing error: {str(je)}")
                    print(f"Failed to parse response: {response_text}")
                return None
                
        except Exception as e:
            if self.debug:
                print(f"API error with {self.provider.value}: {str(e)}")
            return None
            
    def process_prompts_file(self, 
                           input_file: str, 
                           output_file: str,
                           delay: float = 1.0,
                           test_mode: bool = False) -> Dict[str, Any]:
        """
        Process prompts from a file and save results.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            delay: Delay between API calls in seconds
            test_mode: If True, process only first prompt
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        prompts = content.split('-'*50)
        prompts = [p.strip() for p in prompts if p.strip()]
        
        if self.debug:
            print(f"Number of prompts found: {len(prompts)}")
        
        if test_mode:
            prompts = prompts[:1]
            if self.debug:
                print("Test mode: Processing only first prompt")
        
        results = {}
        
        for prompt in tqdm(prompts, desc="Processing prompts"):
            parts = prompt.split('|||', 1)
            if len(parts) != 2:
                if self.debug:
                    print(f"Warning: Skipping malformed prompt (missing delimiter)")
                continue
                
            hospital_number, prompt_text = parts
            hospital_number = hospital_number.strip()
            prompt_text = prompt_text.strip()
            
            if self.debug:
                print(f"\nProcessing hospital number: {hospital_number}")
            
            result = self.process_single_prompt(prompt_text)
            if result:
                results[hospital_number] = result
                # Save after each successful processing
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            time.sleep(delay)
            
        return results

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get configuration from environment
    provider = os.getenv("API_PROVIDER", "openai")
    model = "gpt-4" if provider == "openai" else "deepseek-chat"
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
    
    processor = PromptProcessor(
        provider=provider,
        model=model,
        debug=debug_mode
    )
    
    results = processor.process_prompts_file(
        input_file="data/generated_prompts.txt",
        output_file="data/processed/risk_assessments.json",
        delay=1.0,
        test_mode=test_mode
    )
    
    if debug_mode:
        print(f"\nProcessing complete:")
        print(f"- Provider: {provider}")
        print(f"- Prompts processed: {len(results)}")
        print(f"- Output saved to: data/processed/risk_assessments.json")

if __name__ == "__main__":
    main() 
    