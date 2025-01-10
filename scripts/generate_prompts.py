import pandas as pd
import json
from pathlib import Path
from typing import Dict, List

class PromptGenerator:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.translation_map: Dict[str, str] = {}
        self.df: pd.DataFrame = None
        
    def load_translation_map(self) -> None:
        """Load translation map from JSON file."""
        json_path = self.data_dir / 'translation_map.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            self.translation_map = json.load(f)
            
    def load_data(self, filename: str) -> None:
        """Load and preprocess the CSV data."""
        csv_path = self.data_dir / filename
        self.df = pd.read_csv(csv_path, encoding='gbk', dtype=str)
        
        # Translate column names using translation map
        self.df = self.df.rename(columns=self.translation_map)
        
    def generate_prompt_from_row(self, row: pd.Series) -> str:
        """Generate a ChatGPT prompt based on a row of data."""
        prompt = (
            f"{row['Hospital Number']}|||"
            f"Patient Data:\n"
            f"Diagnosis Year: {row['Diagnosis Year']}, "
            f"Age: {row['Age']}, "
            f"Gender: {'Male' if row['Gender (Male 1, Female 2)'] == 1 else 'Female'}, "
            f"Smoking Status: {'Yes' if row['Smoking (Yes 1, No 2)'] == 1 else 'No'}, "
            f"Nodule Size: {row['Nodule Size']} mm, "
            f"Tumor Component: {row['Tumor Component']}, "
            f"Location: {row['Location']}, "
            f"WBC: {row['WBC']}, "
            f"Neutrophils: {row['Neutrophils']}, "
            f"Platelets: {row['Platelets']}, "
            f"Lymphocytes: {row['Lymphocytes']}, "
            f"Monocytes: {row['Monocytes']}, "
            f"Albumin: {row['Albumin']}, "
            f"Surgery Time: {row['Surgery Time']}, "
            f"Intraoperative Bleeding: {row['Intraoperative Bleeding']}, "
            f"Lymph Node Sampling Count: {row['Lymph Node Sampling Count']}, "
            f"Postoperative Drainage: {row['Postoperative Drainage']}, "
            f"Postoperative Hospital Stay: {row['Postoperative Hospital Stay']}, "
            f"Catheter Days: {row['Catheter Days']}, "
            f"Postoperative Pathology: {'Adenocarcinoma' if row['Postoperative Pathology (Adenocarcinoma 1, Other 2)'] == 1 else 'Other'}, "
            f"Stage: {row['Stage']}, "
            f"Air Leak: {'Yes' if row['Air Leak (No 0, Yes 1)'] == 1 else 'No'}.\n\n"
            f"Based on the patient data above, generate a risk assessment in the following JSON format:\n\n"
            "{\n"
            '  "risk_score": <integer between 0-100>,\n'
            '  "key_indices": {\n'
            '    "age": <value>,\n'
            '    "nodule_size_mm": <value>,\n'
            '    "albumin_g_l": <value>,\n'
            '    "wbc_x10_9_l": <value>,\n'
            '    "neutrophils_x10_9_l": <value>,\n'
            '    "postoperative_drainage_ml": <value>,\n'
            '    "surgery_time_minutes": <value>,\n'
            '    "air_leak": <"Yes" or "No">\n'
            '  },\n'
            '  "possible_adverse_reactions": [\n'
            '    <list of specific potential complications based on patient profile>\n'
            '  ],\n'
            '  "recommendations": {\n'
            '    "monitoring": [<list of specific monitoring recommendations>],\n'
            '    "nutritional_support": <specific nutritional recommendation>,\n'
            '    "early_mobilization": <specific mobilization strategy>,\n'
            '    "follow_up": <specific follow-up plan>\n'
            '  }\n'
            '}\n\n'
            'Ensure all numerical values are appropriate for the units specified and provide detailed, medically relevant recommendations.'
        )
        return prompt
    
    def generate_prompts(self) -> List[str]:
        """Generate prompts for all rows in the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data first.")
            
        prompts = []
        for _, row in self.df.iterrows():
            try:
                prompt = self.generate_prompt_from_row(row)
                prompts.append(prompt)
            except KeyError as e:
                print(f"Missing column in row: {e}")
        return prompts
    
    def save_prompts(self, prompts: List[str], output_filename: str) -> None:
        """Save generated prompts to a text file."""
        output_path = self.data_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, prompt in enumerate(prompts, 1):
                f.write(f"Prompt {i}:\n")
                f.write(prompt)
                f.write("\n" + "-"*50 + "\n")
        print(f"Prompts generated and saved to {output_path}")

def main():
    # Setup paths
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = PromptGenerator(data_dir)
    
    # Load data
    generator.load_translation_map()
    generator.load_data('tiny_dataset.csv')
    
    # Generate and save prompts
    prompts = generator.generate_prompts()
    generator.save_prompts(prompts, 'generated_prompts.txt')

if __name__ == "__main__":
    main()
