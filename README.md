
---
title: Flutter Code Generator
emoji: 💻
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.18.1"
app_file: app.py
pinned: false
---

# Flutter Code Generator - Hugging Face Space

This project leverages Transformers and Hugging Face's Spaces to generate Flutter code based on user prompts. It is designed to simplify Flutter development by providing intelligent suggestions for UI and functionality implementations.

## Features
- **Customizable Flutter Code Generation**: Enter prompts like "Create a responsive login screen" to get Dart code snippets.
- **Streamlit Web App**: Interactive UI for generating Flutter code with adjustable parameters.
- **Fine-tuned Model**: Trained on multiple datasets for Flutter-specific code generation.

## Installation
To run the app locally, follow these steps:

### Clone the Repository
```bash
git clone https://github.com/cod-e-codes/flutter-code-generator.git
cd flutter-code-generator
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run app.py
```

## Parameters
Adjust the following settings via the sidebar for code generation:
- **Temperature**: Controls randomness (higher = more random outputs).
- **Top-p**: Cumulative probability for nucleus sampling.
- **Max Length**: Maximum tokens in output.
- **Repetition Penalty**: Penalizes repetitive text.
- **Top-k**: Limits the sampling pool.

## Deploying to Hugging Face Spaces
1. Create a Hugging Face Space and select the "Streamlit" template.
2. Upload the code files from this repository.
3. Configure the environment by adding the required packages.
4. Deploy and access the app via your Space's URL.

## Model Training
The model was fine-tuned using:
- Datasets from Hugging Face such as `wraps/codegen-flutter-v1`, `limcheekin/flutter-website-3.7`, and `deepklarity/top-flutter-packages`.
- A checkpoint from Salesforce's CodeGen model (`codegen-350M-mono`).

## License
This project is open-source and available under the [MIT License](LICENSE).

---

Built with ❤️ by [Cod-e-Codes](https://github.com/cod-e-codes)
