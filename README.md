# Lung Surgery Risk Prediction with AI

This repository hosts the code, data, and documentation for research on developing AI-driven risk prediction models for lung surgery prognosis. The project aims to leverage structured and time-series patient data to predict risks associated with lung surgeries, including complications and mortality.

## Features
- **Structured Data Analysis**: Extract, preprocess, and analyze patient demographic, laboratory, and surgical data.
- **Time-Series Analysis**: Incorporate and analyze physiological time-series data for dynamic monitoring.
- **Risk Prediction Models**: Develop AI models that generate risk scores with detailed explanations.
- **Data Pipeline**: Automatically generate structured prompts for large language models (LLMs) to refine outputs with expert validation.
- **Visualization Tools**: Plot and analyze trends for key physiological metrics.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data](#data)
5. [Contributing](#contributing)
6. [License](#license)

## Project Overview
Lung surgeries, including cancer treatment and transplantation, carry significant risks. Accurate prediction of postoperative outcomes is crucial for patient management. This research project uses AI technologies such as:
- **Large Language Models (LLMs)** for unstructured data processing and interpretability.
- **XGBoost and similar models** for structured data risk prediction.
- **Physiological time-series modeling** for dynamic analysis.

The generated outputs include:
- A risk score (0-100) with two decimal precision.
- Key physiological indices influencing the risk score.
- Potential adverse reactions associated with the profile.

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/ForestR/LSRP.git
    ```
2. Navigate to the project directory:
    ```bash
    cd LSRP
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Prepare the Data**: Place your CSV file in the `data/` directory.
2. **Generate Prompts**:
    Run the script to generate ChatGPT prompts:
    ```bash
    python scripts/generate_prompts.py --input data/patient_data.csv --output output/prompts.txt
    ```
3. **Train Models**: Use the scripts in the `models/` directory to train risk prediction models.
4. **Visualize Results**: Run visualization scripts in the `scripts/` directory to analyze trends.

## Data
### Input
- Structured data (e.g., age, gender, lab results).
- Time-series data (e.g., heart rate, respiratory rate).

### Output
- Risk score (0-100) with detailed explanation.
- Key physiological indices and adverse reactions.

### Sample Data
Refer to the `data/sample_patient_data.csv` file for an example dataset format.

## Contributing
We welcome contributions to this research project! To contribute:
1. Fork the repository.
2. Create a feature branch:
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes:
    ```bash
    git commit -m "Description of changes"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-name
    ```
5. Submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
We acknowledge the contributions of domain experts and the open-source AI community for their support in developing this project.
