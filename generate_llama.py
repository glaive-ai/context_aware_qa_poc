from transformers import AutoTokenizer,AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
from utils import GPT_4O_MINI_SYSTEM_PROMPT,JUDGE_SYSTEM_PROMPT,get_client,infer
import json
from concurrent.futures import ThreadPoolExecutor
import threading

openai_client = get_client()
local_client = get_client(local=True)

datasets = ["Google_NQ_Dataset_Chunked.csv","MSMarco_Dataset_Chunked.csv","MuSiQue_Dataset_Chunked.csv","google_nq_500_irrelevant.csv","ms_marco_500_irrelevant.csv","musique_500_irrelevant.csv"]

# Create a thread-safe list for results
thread_safe_results = []
results_lock = threading.Lock()

def process_row(row_data):
    index, row = row_data
    input = json.loads(row["input"])
    question = input["question"]
    documents = input["documents"]
    true_documents = []
    try:
        true_answer = json.loads(row["expected"])["true_answer"]
        true_documents = json.loads(row["expected"])["true_documents"]
    except:
        true_answer = row["expected"]
    
    if "irrelevant" in dataset:
        true_answer = "I don't know / Answer not in context"
    
    docs_string = ""
    for doc in documents:
        docs_string += f"Document {doc['id']}:\n {doc['text']}\n\n"

    messages = [
        {"role":"user","content":f"Question: {question}\n\nContext:\n{docs_string}"}
    ]
    output = infer(local_client,messages,"glaiveai/dropbox_qa_v2")
    judge_messages = [
        {"role":"system","content":JUDGE_SYSTEM_PROMPT.format(question=question,generated_answer=output,ground_truth=true_answer)}
    ]
    judge_response = infer(openai_client,judge_messages,"openai/chatgpt-4o-latest")
    correct = judge_response.lower() == "y"
    with results_lock:
        thread_safe_results.append({
            "question": question,
            "generated_answer": output,
            "ground_truth": true_answer,
            "judge_evaluation": correct,
            "true_documents": true_documents
        })

eval_generations = []
for dataset in datasets:
    print(f"Evaluating {dataset}")
    thread_safe_results = []  # Reset for each dataset
    df = pd.read_csv(dataset)
    
    # Process rows in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(
            executor.map(process_row, df.iterrows()),
            total=len(df),
            desc=f"Processing {dataset}"
        ))
    
    eval_generations.append({dataset: thread_safe_results})

with open("eval_generations_llama.json","w") as f:
    json.dump(eval_generations,f,indent=4)