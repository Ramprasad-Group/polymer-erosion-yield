import os
import csv
import json
from datetime import datetime, timezone

import openai
from openai import OpenAI
import pandas as pd

# =====================================================
# === USER SETTINGS ==================================
# =====================================================
# Set your OpenAI API key here (or load it from an environment variable).
openai.api_key = "YOUR_OPENAI_API_KEY"
api_key = "YOUR_OPENAI_API_KEY"

# Set the directory containing your split JSONL files and the desired output
# directory for the generated CSV result files.
INPUT_DIR = "/path/to/splits"           # contains split_0{N}/v3AM-AF_RGlog_all_{train,test}_aBo_split{N}.jsonl
OUTPUT_DIR = "/path/to/output"          # where the result CSVs will be written

# Job/split selection
JOB_ID = ""                             # fine-tuning job ID to fetch / diagnose
T = 0.2                                 # sampling temperature for inference
N = 5                                   # split index to process


def fetch_results(job_id):
    client = openai.OpenAI(api_key=api_key)

    # Retrieve the fine-tuning job
    job = client.fine_tuning.jobs.retrieve(job_id)
    result_files = job.result_files  # list of file IDs (strings)

    if result_files:
        result_id = result_files[0]

        # Get file content using the new SDK method
        response = client.files.with_raw_response.retrieve_content(result_id)
        output_file_name = f"{job_id}_result.csv"
        with open(output_file_name, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded result file: {output_file_name}")

        # Save the fine-tuned model ID
        fine_tuned_model = job.fine_tuned_model
        if fine_tuned_model:
            with open(f"{job_id}_fine_tuned_model.txt", "w") as f:
                f.write(fine_tuned_model)
            print(f"✅ Fine-tuned model ID saved: {fine_tuned_model}")
        else:
            print("⚠️ Fine-tuned model not available yet.")

        return fine_tuned_model
    else:
        print("⚠️ No result files found.")
        return None


def ts_to_str(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def diagnose_finetune(job_id: str):
    client = OpenAI(api_key=api_key)

    # 1) Retrieve job
    job = client.fine_tuning.jobs.retrieve(job_id)
    print("=== Job Summary ===")
    print(f"job_id:           {job.id}")
    print(f"status:           {job.status}")
    print(f"base_model:       {job.model}")
    print(f"created_at:       {ts_to_str(job.created_at)}")
    if getattr(job, "finished_at", None):
        print(f"finished_at:      {ts_to_str(job.finished_at)}")
    if getattr(job, "fine_tuned_model", None):
        print(f"fine_tuned_model: {job.fine_tuned_model}")
    if getattr(job, "error", None):
        print(f"error:            {job.error}")

    # 2) Recent events
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=50)
    ev_list = events.data if hasattr(events, "data") else []

    print("\n=== Last Events ===")
    if not ev_list:
        print("(no events found)")
    else:
        for ev in ev_list[:20]:
            created = ts_to_str(ev.created_at)
            lvl = getattr(ev, "level", "")
            msg = getattr(ev, "message", "")
            print(f"[{created}] {lvl.upper():7s} {msg}")

    return job, ev_list


def process_jsonl_file(jsonl_file, output_csv):

    data = []


    with open(jsonl_file, 'r') as file:
        lines = file.readlines()


        for line in lines:

            obj = json.loads(line)


            user_message = obj['messages'][1]['content']


            response = client.chat.completions.create(
                model=fine_tuned_model,
                messages=[
                    {"role": "system", "content": "You are a materials scientist specializing in low Earth orbit (LEO) atomic oxygen interactions with polymers. Given a structured input describing a polymer sample, including the number of stacked thin film layers, the thickness of each layer, the polymer name, the SMILES string, a description of the coating, the mission time (years spent attached to the ISS in the ram facing direction), the ram facing solar exposure (equivalent sun hours) and the ram atomic oxygen fluence (atoms/cm^2), predict the base-10 logarithm of the atomic oxygen erosion yield in (Angstroms^3/atom). Rules: 1) Use only the provided input fields. 2) Keep reasoning internal; do not explain your steps. 3) Do not include text, labels, or units. 4) Output only the final numeric value with 3 significant figures. 5) Negative values are allowed. 6) If prediction is not possible, output exactly null."},
                    {"role": "user", "content": user_message}
                ],
                temperature = 0.2

            )


            ground_truth = obj['messages'][2]['content']

            # Append data to the list
            data.append({
                'Question': user_message,
                'Answer': response.choices[0].message.content,
                'Assistant': ground_truth
            })


    df = pd.DataFrame(data)


    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL, escapechar="\\")


# =====================================================
# === MAIN ============================================
# =====================================================

# Fetch the fine-tuned model from a completed job
fine_tuned_model = fetch_results(JOB_ID)
print(fine_tuned_model)

# Optional: diagnose a (possibly still-running) fine-tuning job
job, events = diagnose_finetune(JOB_ID)

# Run inference on the train/test JSONL files with the fine-tuned model
client = openai
process_jsonl_file(f"{INPUT_DIR}/split_0{N}/v3AM-AF_RGlog_all_train_aBo_split{N}.jsonl", f"{OUTPUT_DIR}/AM-AF_RGlog_all_ABo_train_results_split{N}.csv")
process_jsonl_file(f"{INPUT_DIR}/split_0{N}/v3AM-AF_RGlog_all_test_aBo_split{N}.jsonl", f"{OUTPUT_DIR}/AM-AF_RGlog_all_ABo_test_results_split{N}.csv")
process_jsonl_file(f"{INPUT_DIR}/split_0{N}/v3AM-AF_RGlog_all_train_aBo_split{N}.jsonl", f"{OUTPUT_DIR}/AM-AF_RGlog_all_ABo_train_TR2results_split{N}.csv")
process_jsonl_file(f"{INPUT_DIR}/split_0{N}/v3AM-AF_RGlog_all_test_aBo_split{N}.jsonl", f"{OUTPUT_DIR}/AM-AF_RGlog_all_ABo_test_TR2results_split{N}.csv")
