import json
import os
from pathlib import Path
from typing import Dict, Any, Union
from openai import OpenAI
from dotenv import load_dotenv
import time
from tqdm import tqdm
from enum import Enum

load_dotenv()

class APIProvider(Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"

class Translator:
    def __init__(self, 
                 api_key: str = None,
                 provider: str = "openai",
                 model: str = "gpt-4",
                 temperature: float = 0.1,
                 debug: bool = False):
        """
        Initialize the Translator.
        
        Args:
            api_key: API key. If None, will try to get from environment variable
            provider: API provider ("openai" or "deepseek")
            model: Model to use for translation
            temperature: Temperature parameter for generation (0.0 to 1.0)
            debug: Whether to print debug information
        """
        self.debug = debug
        self.provider = APIProvider(provider.lower())
        
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
        
        # Load or create translation map
        self.map_file = Path("data/translation_map.json")
        self.translation_map = self._load_translation_map()
        
    def _load_translation_map(self) -> Dict[str, str]:
        """Load existing translation map or create new one."""
        if self.map_file.exists():
            with open(self.map_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_translation_map(self):
        """Save translation map to file."""
        with open(self.map_file, 'w', encoding='utf-8') as f:
            json.dump(self.translation_map, f, ensure_ascii=False, indent=2)
            
    def translate_text(self, text: str) -> str:
        """Translate single text using OpenAI API."""
        if not text or not isinstance(text, str):
            return text
            
        # Check if translation exists in map
        if text in self.translation_map:
            return self.translation_map[text]
            
        try:
            if self.debug:
                print(f"\nTranslating: {text[:100]}...")  # Show first 100 chars
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate the following English text to Chinese. Keep any technical terms accurate and professional."},
                    {"role": "user", "content": text}
                ],
                temperature=self.temperature,
                timeout=30  # Add timeout
            )
            translation = response.choices[0].message.content.strip()
            
            # Save to translation map
            self.translation_map[text] = translation
            self._save_translation_map()
            
            if self.debug:
                print(f"Translated: {text[:50]} -> {translation[:50]}")
                
            return translation
            
        except Exception as e:
            if self.debug:
                print(f"Translation error for text '{text[:100]}...': {str(e)}")
            return text
            
    def translate_value(self, value: Any) -> Any:
        """Recursively translate values in data structure."""
        try:
            if isinstance(value, str):
                return self.translate_text(value)
            elif isinstance(value, list):
                return [self.translate_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: self.translate_value(v) for k, v in value.items()}  # Don't translate keys
            return value
        except Exception as e:
            if self.debug:
                print(f"Error in translate_value: {str(e)}")
            return value
        
    def translate_json_file(self, 
                           input_file: str, 
                           output_file: str,
                           delay: float = 1.0) -> None:
        """Translate JSON file content."""
        # Read input JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Translate content
        translated_data = {}
        
        # Create progress bar
        pbar = tqdm(data.items(), desc="Translating reports")
        for key, value in pbar:
            try:
                if self.debug:
                    pbar.write(f"\nProcessing entry: {key}")
                    
                translated_data[key] = self.translate_value(value)
                
                # Update progress bar description
                pbar.set_description(f"Processed {len(translated_data)}/{len(data)} reports")
                
                # Save progress after each successful translation
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(translated_data, f, ensure_ascii=False, indent=2)
                    
                time.sleep(delay)  # Avoid rate limiting
                
            except Exception as e:
                pbar.write(f"Error processing entry {key}: {str(e)}")
                continue
                
        if self.debug:
            print(f"\nTranslation completed:")
            print(f"- Input file: {input_file}")
            print(f"- Output file: {output_file}")
            print(f"- Translation map saved to: {self.map_file}")

def main():
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("data/translated")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get configuration from environment
        provider = os.getenv("TRANSLATE_MODEL", "openai")
        model = "gpt-4" if provider == "openai" else "deepseek-chat"
        debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        temperature = float(os.getenv("TEMPERATURE", "0.1"))
        delay = float(os.getenv("API_DELAY", "1.0"))
        
        # Initialize translator
        translator = Translator(
            provider=provider,
            model=model,
            temperature=temperature,
            debug=debug_mode
        )
        
        # Translate risk assessments
        translator.translate_json_file(
            input_file="data/processed/risk_assessments.json",
            output_file="data/translated/risk_assessments_zh.json",
            delay=delay
        )
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
