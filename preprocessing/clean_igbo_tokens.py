import pandas as pd
import ast
import re

# Step 1: Load your Excel file
file_path = "C:/Users/hp/Desktop/MLs/IgboCorpora/dataokwumputaDataset.xlsx"
df = pd.read_excel(file_path)

# Step 2: Define a cleaning function
def clean_tokens(token_list):
    cleaned = []
    for token in token_list:
        # Lowercase and keep only alphabetic Igbo-looking tokens (allow accented & tonal characters)
        if re.match(r"^[a-zA-Zịọụñṅáéíóúʼ]+$", token):
            cleaned.append(token.lower())
    return cleaned

# Step 3: Convert 'tokenized_definition' from string to actual list and clean
df['cleaned_tokens'] = df['tokenized_definition'].dropna().apply(ast.literal_eval).apply(clean_tokens)

# Step 4: Preview
print(df[['tokenized_definition', 'cleaned_tokens']].head())