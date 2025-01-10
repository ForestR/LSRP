import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import openai
from tqdm import tqdm
from dotenv import load_dotenv

# Add this at the beginning of your script, before any OpenAI operations
load_dotenv()

class PromptProcessor:
    def __init__(self, api_key: str = None, model: str = "gpt-4", temperature: float = 0.2):
        """
        Initialize the PromptProcessor.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable.
            model: OpenAI model to use
            temperature: Temperature parameter for generation (0.0 to 1.0)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        openai.api_key = self.api_key
        self.model = model
        self.temperature = temperature
        
    def process_single_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Process a single prompt using the OpenAI API.
        
        Args:
            prompt: The prompt text to process
            
        Returns:
            Parsed JSON response
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content.strip()
            
            # Parse the JSON response
            return json.loads(response_text)
            
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")
            return None
            
    def process_prompts_file(self, 
                           input_file: str, 
                           output_file: str,
                           delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Process all prompts from a file and save results.
        
        Args:
            input_file: Path to file containing prompts
            output_file: Path to save results
            delay: Delay between API calls in seconds
            
        Returns:
            List of processed results
        """
        # Read prompts from file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split prompts (assuming they're separated by ---------------------------------------------------)
        prompts = content.split('---------------------------------------------------')
        # Each prompt should now be a tuple of (hospital_number, prompt_text)
        prompts = [p.strip().split('|||') for p in prompts if p.strip()]
        
        results = {}  # Changed from list to dict to store hospital number as key
        
        # Process each prompt with progress bar
        for hospital_number, prompt_text in tqdm(prompts, desc="Processing prompts"):
            result = self.process_single_prompt(prompt_text)
            if result:
                results[hospital_number.strip()] = result  # Use hospital number as key
            
            # Add delay to avoid rate limiting
            time.sleep(delay)
            
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        return results

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = PromptProcessor()
    
    # Process prompts
    results = processor.process_prompts_file(
        input_file="data/generated_prompts.txt",
        output_file="data/processed/risk_assessments.json",
        delay=1.0  # Adjust delay as needed based on your API rate limits
    )
    
    print(f"Successfully processed {len(results)} prompts")

if __name__ == "__main__":
    main() 
    