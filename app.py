import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model and tokenizer from the checkpoint
model_name = "./flutter_codegen_model/checkpoint-1500"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to clean up repetitive lines in code
def clean_code_response(response):
    lines = response.splitlines()
    unique_lines = []
    for line in lines:
        if line.strip() not in unique_lines:  # Avoid duplicates
            unique_lines.append(line.strip())
    return "\n".join(unique_lines)

# Function to generate Flutter code
def generate_flutter_code(prompt, temperature, top_p, max_length, num_return_sequences, repetition_penalty, top_k):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    code = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return [clean_code_response(c) for c in code]

# App Title
st.title("Flutter Code Generator")

# Default parameter values
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_LENGTH = 512
DEFAULT_NUM_RETURN_SEQUENCES = 1
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_TOP_K = 50

# Sidebar for settings
st.sidebar.title("Generation Settings")

temperature = st.sidebar.slider(
    "Temperature (randomness)",
    0.1, 1.0, DEFAULT_TEMPERATURE, step=0.1,
)

top_p = st.sidebar.slider(
    "Top-p (cumulative probability)",
    0.1, 1.0, DEFAULT_TOP_P, step=0.1,
)

max_length = st.sidebar.slider(
    "Max Output Length (tokens)",
    128, 1024, DEFAULT_MAX_LENGTH, step=64,
)

num_return_sequences = st.sidebar.slider(
    "Number of Outputs",
    1, 5, DEFAULT_NUM_RETURN_SEQUENCES,
)

repetition_penalty = st.sidebar.slider(
    "Repetition Penalty",
    1.0, 2.0, DEFAULT_REPETITION_PENALTY, step=0.1,
)

top_k = st.sidebar.slider(
    "Top-k (limit sampling pool)",
    0, 100, DEFAULT_TOP_K,
)

# Reset to defaults button
if st.sidebar.button("Reset to Defaults"):
    st.session_state.update(
        {
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "max_length": DEFAULT_MAX_LENGTH,
            "num_return_sequences": DEFAULT_NUM_RETURN_SEQUENCES,
            "repetition_penalty": DEFAULT_REPETITION_PENALTY,
            "top_k": DEFAULT_TOP_K,
        }
    )

# Input Section
user_input = st.text_area(
    "Enter your prompt (e.g., 'Create a responsive login screen'):",
    max_chars=200,
)

# Output Section
if st.button("Generate Code"):
    if user_input.strip():
        prompt = f"{user_input.strip()}"
        generated_code = generate_flutter_code(
            prompt, temperature, top_p, max_length, num_return_sequences, repetition_penalty, top_k
        )
        for i, code in enumerate(generated_code, start=1):
            st.subheader(f"Output {i}")
            st.code(code, language="dart")
    else:
        st.error("Please enter a prompt before clicking 'Generate Code'.")
