# Import libraries
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import pandas as pd

from huggingface_hub import login
from datasets import load_dataset
from unsloth import FastLanguageModel

# Set Paths and Hyperparameters
# Base path
base_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', '..', '..'))

# Source and target language
source_language = "English" #"Early Modern Bohemian German"
target_language = "Early Modern Bohemian German"

# Translation direction
translation_direction = "DE_to_EN" if source_language == "Early Modern Bohemian German" else "EN_to_DE"

# Model parameters
unsloth_model_name = 'unsloth/Phi-3.5-mini-instruct-bnb-4bit'
company_name = 'microsoft'

model_name = unsloth_model_name.split('/')[1]
max_new_tokens = 2000       # Maximum number of model output
max_seq_length = 128000     # Maximum of input tokens
dtype = None                # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True            # Use 4bit quantization to reduce memory usage. Can be False.

# Number of icl examples
shots = 8
formatted_shots = f"{shots:03}" 

# Icl examples path
icl_examples_path = os.path.join(
    base_path, 
    'data', 
    'icl_examples',
    translation_direction, 
    f'{formatted_shots}_example_prompt.txt'
)

# Icl prompts path
icl_prompts_path = os.path.join(
    base_path, 
    'data', 
    'icl_prompts',
    company_name,
    model_name,
    translation_direction, 
    f'{formatted_shots}_prompt_check.txt'
)

# Save inference dataset
save_path = os.path.join(
    base_path, 
    'results', 
    f'{company_name}',
    f'{model_name}', 
    'icl', 
    translation_direction, 
    f'{translation_direction}_{formatted_shots}_example_prompt.json'
)

# Print paths
print(f'Company name: {company_name}')
print(f'Model name: {model_name}')
print(f'Base path: {base_path}')
print(f'Translation direction: {translation_direction}')
print(f'Number of icl examples: {shots}')
print(f'ICL examples path: {icl_examples_path}')
print(f'ICL prompts path: {icl_prompts_path}')
print(f'Save ICL inference dataset path: {save_path}')

# Hugging face login
hub_token = "hf_..."
login(hub_token, add_to_git_credential=True)

# Load test dataset
dataset = load_dataset('niclasgriesshaber/EarlyModernGerman_to_EN')
test_dataset = dataset['test']
print(f'Loaded test dataset: \n {test_dataset}')

# Load icl examples as string
with open(icl_examples_path, 'r') as file:
    icl_examples = file.read()

# Prompt template
prompt_template = """<|system|>

You are a helpful assistant tasked with translating from {} to {}. NEVER provide an introduction to the translation (e.g. 'Here is the translation:', 'Translate to', 'Hier ist die Übersetzung:', etc.), explanations or clarifications.
NEVER provide a note after your translation. In the following, there are some examples how you should translate in the translation task.<|end|>
<|user|>
{}
### Translation Task. Only translate the following text. Nothing else!

{}:
{}

Translate to {} and match the structure of the source text. Output only this translation and nothing else.<|end|>
<|assistant|>{}"""

# Apply prompt template to test dataset
def formatting_prompts_func(examples, source_language, target_language, icl_examples):
    
    source_texts = examples[source_language]
    texts = []

    for source_text in source_texts:
        # Format the prompt with dynamic source and target languages
        text = prompt_template.format(
            source_language,  # Dynamic source language
            target_language,  # Dynamic target language
            icl_examples,     # Example shots (prompt text before asking for translation)
            source_language,  # Target language for translation request
            source_text,      # Actual source text to translate
            target_language,  # Target language for translation output (no actual output included)
            ""                # Placeholder for the output, left empty for inference
        ) 
        texts.append(text)

    return {"text": texts}

# Apply prompt template to all test samples
test_dataset = test_dataset.map(
    lambda examples: formatting_prompts_func(examples, source_language, target_language, icl_examples),
    batched=True
)

# Output a text file to check prompt
with open(icl_prompts_path, "w") as f:
    f.write(test_dataset['text'][1])

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=unsloth_model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Set model to inference mode
FastLanguageModel.for_inference(model)

# Save inferences in new dataset
def process_dataset(test_dataset, model, tokenizer, max_new_tokens, formatted_shots, translation_direction):

    # Extract target language
    if translation_direction == "DE_to_EN":
        target_language = "English"
    else:
        target_language = "Early Modern Bohemian German"
    
    # Convert Hugging Face dataset to Pandas DataFrame
    df = pd.DataFrame(test_dataset)
    
    total_time = 0  # Initialize the total time accumulator

    # Loop through each row in the dataframe
    for i, row in df.iterrows():
        start_time = time.time()  # Start timer for the current inference

        try:
            print(f"Processing test point {i + 1} of {len(df)}")

            # Get the text for the current row
            inputs = tokenizer([row['text']], return_tensors="pt").to("cuda")

            # Generate the model outputs
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, repetition_penalty=1.1, temperature=0.01)

            # Decode the outputs, converting from token IDs back to text
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            # Since decoded_output has only one entry, extract the single output
            output = decoded_outputs[0]

            # Define the exact substring to search for
            search_string = f"<|assistant|>"

            # Find the index of the search_string in the output
            start_idx = output.find(search_string)

            # Extract everything after the search_string if it's found
            if start_idx != -1:
                extracted_text = output[start_idx + len(search_string):]
            else:
                extracted_text = 'NA'

            # Manually remove the special token '<|eot_id|>' from the beginning and the end, if present
            #extracted_text = extracted_text.replace("<|start_header_id|>assistant<|end_header_id|>\n", "").strip()
            extracted_text = extracted_text.replace("<|end|><|endoftext|>", "").strip()

            # Print the output and extracted text
            #print('_________________________________________________')
            #print(output)
            print('Sheilagh Ogilvie:')
            print('_________________________________________________')
            print(test_dataset[i][target_language])
            print('_________________________________________________')
            #print('LLM-generated Response:')
            print('_________________________________________________')
            print(extracted_text)
            print('_________________________________________________')
            print('_________________________________________________')
            print('_________________________________________________')
            print('_________________________________________________')

            # Add extracted_text directly to the dataframe at index i
            df.at[i, f'{translation_direction}_{formatted_shots}_example_prompt'] = extracted_text

        except Exception as e:
            print(f"Error occurred at test point {i + 1}: {str(e)}")
        
        end_time = time.time()  # End timer for the current inference
        elapsed_time = end_time - start_time  # Calculate elapsed time
        total_time += elapsed_time  # Accumulate total time

        print(f"Time for test point {i + 1}: {elapsed_time:.2f} seconds, Total time: {total_time:.2f} seconds")

    return df

# Call the function
processed_dataset = process_dataset(test_dataset, model, tokenizer, max_new_tokens, formatted_shots, translation_direction)

# Save the dataset as a JSON file
processed_dataset.to_json(save_path, orient='records', lines=True, force_ascii=False)
print(f"Dataset saved successfully to {save_path}")