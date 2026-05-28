import json
import pandas as pd

# =====================================================
# === USER SETTINGS ==================================
# =====================================================
# Set the directory containing your split CSV files and the desired output
# directory for the generated JSONL files. Each split folder is expected to
# contain `train {N}.csv` and `test {N}.csv`.
INPUT_DIR = "/path/to/splits"           # contains split_0{N}/train {N}.csv and split_0{N}/test {N}.csv
OUTPUT_DIR = "/path/to/output"          # where the generated .jsonl files will be written
N = 4                                   # split index to process

def save_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')

def generate_json_templates(df):
    # List to store JSON objects
    output_json = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize the JSON template
        json_template = {
            "messages": [
                {"role": "system", "content": "You are a materials scientist specializing in low Earth orbit atomic oxygen interactions with polymers. Given a structured input describing a polymer sample, including the polymer name, the SMILES string, a description of the coating, the NASA mission from which the sample's data were obtained, the orientation of the sample during space exposure, the mission time (years of direct space exposure while attached to the ISS), the solar exposure (equivalent sun hours) and the atomic oxygen fluence (atoms/cm^2), predict the base-10 logarithm of the atomic oxygen erosion yield in (Angstroms^3/atom). Rules: 1) Use only the provided input fields. 2) Keep reasoning internal; do not explain your steps. 3) Do not include text, labels, or units. 4) Output only the final numeric value with 3 significant figures. 5) Negative values are allowed. 6) If prediction is not possible, output exactly null."},
                {"role": "user", "content": "What is the base-10 logarithm of the atomic oxygen erosion yield of the polymer {} represented by SMILES {}, with {} coating, flown on the {} mission oriented in the {} direction for a mission time of {} years, subjected to a solar exposure of {} equivalent sun hours and an atomic oxygen fluence of {} atom/cm^2?"},
                {"role": "assistant", "content": '{}'}
            ]
        }

        # Populate the JSON template with polymer and monomer information from the CSV
        layers = row['layers']
        mission=row['mission']
        orientation=row['orientation']
        thickness = row['thickness (mm)']
        polymer_name =  row['polymer name']
        polymer_smiles = row['smiles']
        coating = row['coating name']
        if pd.isna(coating) or str(coating).strip() == "":
            coating = "no"
        solar = round(row['ram_solar_exposure (esh)'])
        erosion_yield = round(float(row['log(e_y)']), 3)
        fluence = f"{float(row['ram_ao_fluence (atoms/cm2)']):.3e}"
        mission_time = row['mission_time (yr)']
        #monomer2_smiles = row['reactant2']

        # Update the JSON template with the current polymer and monomer
        json_template['messages'][1]['content'] = json_template['messages'][1]['content'].format(polymer_name,polymer_smiles,coating,mission,orientation,mission_time,solar,fluence)
        json_template['messages'][2]['content'] = json_template['messages'][2]['content'].format(erosion_yield)


        # Conditionally add Monomer2 only if it's not Na

        # Append the updated JSON template to the output list
        output_json.append(json_template)

    return output_json

# Load the data from your CSV file into a DataFrame
df = pd.read_csv(f'{INPUT_DIR}/split_0{N}/test {N}.csv')

# Generate JSON templates
output_json = generate_json_templates(df)

# Save the data in JSONL format
save_jsonl(output_json, f'{OUTPUT_DIR}/v3AM-AF_RGlog_all_test_aBo_split{N}.jsonl')

df = pd.read_csv(f'{INPUT_DIR}/split_0{N}/train {N}.csv')

# Generate JSON templates
output_json = generate_json_templates(df)

# Save the data in JSONL format
save_jsonl(output_json, f'{OUTPUT_DIR}/v3AM-AF_RGlog_all_train_aBo_split{N}.jsonl')
