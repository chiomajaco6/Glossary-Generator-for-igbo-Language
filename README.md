
# Glossary-Generator-for-igbo-Language
This project builds a machine learning model for generating Igbo word explanations using transformer-based NLP architectures (T5).  It supports training with PyTorch or TensorFlow and is tailored to empower natural language understanding in low-resource African languages.


# Team Members

Odirichukwu, Jacinta Chioma, PhD.
Senior Lecturer.
Department of Computer Science,
School of Information and Communication Technology,
Federal University of Technology Owerri, Imo State Nigeria.
+2348037394691
jacinta.odirichukwu@futo.edu.ng

Reginald Nnadozie Nnamdi (PhD)
Department of Philosophy
Veritas University Abuja,
FCT, Nigeria,
reginaldnnamdi5@gmail.com

Simon Peter Chimaobi Odirichukwu
Department of Health,
Primary Health Development Agency, Owerri,
Imo State, Nigeria
sodirichukwu@gmail.com

# Introduction
Over 30 million people worldwide speak Igbo language, and the Igbo language is predominantly spoken by Igbo people in South-Eastern Nigeria (Odirichukwu and Nnamdi, 2023)

# Problem Statement
Unlike other low-resource African languages, Igbo language faces the same challenge of inadequate or in most cases complete lack of resources dataset and methods which supports the research and development of even basic NLP tools for well over 30 million users.

# Objectives

1. To Loads a dataset of Igbo words, their meanings, and related tokens.
2. Cleans the dataset to ensure all fields are usable.
3. Prepares a list of dictionary samples in this format:

  {
    "input_text": "translate Igbo to explanation: abalị uchichi anyasị itiri",
    "target_text": "Ọchịchịrị. Ehihie laa, chi ejiela."
  }

# Project Highlights

Igbo-to-Igbo text generation using contextual tokens
Cleaned and structured dataset for low-resource language modeling
Fine-tuned HuggingFace `t5-small` transformer model
Supports both PyTorch (Trainer) and TensorFlow (Keras) backends
Optimized input formatting: Igbo word + context ➝ explanation
Includes preprocessing, training, and inference scripts
 Ideal for language researchers, data scientists, and lexicon developers
 
 Project Structure


 Dataset

Each row in the training data contains:
- `okwu`: Igbo headword
- `cleaned_tokens`: list of semantically related tokens (e.g. synonyms, dialect forms)
- `mpüta`: Igbo explanation or gloss

The training script formats input as:  
**translate Igbo to explanation: okwu cleaned_tokens ➝ mpüta**

🧪 Model Training

PyTorch Training:
- Fine-tuned using HuggingFace `Trainer`
- `T5ForConditionalGeneration` + `T5Tokenizer`
- 5 epochs, max_length 64, batch size 8

TensorFlow Training:
- Uses `TFT5ForConditionalGeneration` and `model.fit()`
- Tokenized input/output with padding and truncation

📥 Inference Example

def generate_igbo_gloss(word, tokens):

    input_text = f"translate Igbo to explanation: {word} {' '.join(tokens)}"

    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    output = model.generate(input_ids, max_length=64)

    return tokenizer.decode(output[0], skip_special_tokens=True)

🌍 Use Cases

- Digital Igbo Dictionary Projects
- Lexicography and Linguistics Research
- Language Learning Apps
- Machine Translation Gloss Generation

💼 Portfolio Impact

This project demonstrates ability to:
- Solve NLP problems in underrepresented languages
- Handle noisy linguistic data (token lists, glosses)
- Fine-tune modern transformers using clean supervision
- Adapt ML tools for educational, research, or production use
- Publish code with scalable and documented workflow

# Methodology
# Igbo Token Cleaning Script for Lexical Modeling
🔍 Project Component: Dataset Preprocessing

This script is part of the Igbo-to-Igbo Contextual Translation project, focused on low-resource NLP tasks for the Igbo language. It processes a tokenized Igbo dictionary dataset stored in Excel format by:

Converting stringified tokens to Python lists

Cleaning non-alphabetic and non-Igbo words

Retaining valid Igbo tokens with tonal marks

Generating a cleaned dataset for language modeling

📝 Python Script: clean_igbo_tokens.py

# Sample output
| tokenized\_definition            | cleaned\_tokens                 |
| -------------------------------- | ------------------------------- |
| \['Abalị', 'uchichi', 'anyasị']  | \['abalị', 'uchichi', 'anyasị'] |
| \['Ọchịchịrị.', 'Ehihie', 'laa'] | \['ọchịchịrị', 'ehihie', 'laa'] |

![alt text](image-2.png)

![alt text](image-3.png)

# Cleaning the Igbo Dataset to remove Noise from tokenized_definitiion colulmn such as non-word characters like:

# '~', ':', '1', '2', '.', ',', etc. Removal of punctuation, numbers, mixed tokens, empty/whitespace only tokens

This script cleans the tokenized_definition column in the Igbo language dataset to remove noisy, non-linguistic tokens such as:

* Punctuation: ~, :, ., ,

* Numbers: 1, 2, 10, etc.

* Mixed or malformed tokens

* Empty or whitespace-only entries

The cleaned version is saved in a new column: cleaned_tokens.

| Step                | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| 1. 📥 Load Dataset  | Reads an Excel file containing tokenized Igbo word definitions   |
| 2. 🔍 Define Filter | A regex keeps only valid Igbo tokens with letters and tone marks |
| 3. 🧹 Clean Tokens  | Removes digits, punctuation, and non-Igbo patterns               |
| 4. ➕ Add Column     | Appends a `cleaned_tokens` column with the filtered result       |

🧪 Example
# Before cleaning:
['Nwoke', '1', ':', 'bụ', 'onye', '~', 'ọma.']

# After cleaning:
['nwoke', 'bụ', 'onye', 'ọma']

📦 Dependencies
# pip install pandas openpyxl

🧠 Ideal For
* Low-resource NLP preprocessing

* Igbo language modeling

* Lexicon and dictionary cleaning tasks

* T5 / BERT fine-tuning input preparation



## PyTorch Implementation with HuggingFace Trainer



## TensorFlow Implementation (Keras-style)


# Results

![alt text](image.png)

# Reference
J. C. Odirichukwu and R. N. , Nnamdi, (2023) “Web-based igbo thesaurus
with real-time retrieval.