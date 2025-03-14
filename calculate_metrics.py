import json
import pandas as pd
import numpy as np
import re

def load_eval_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_source_metrics(generated_answer, true_documents):
    """
    Calculate source precision, recall, and F1 score using the Citation field
    and true_documents from the results
    """
    generated_sources = set()
    # Extract source citations from the Citation field at the end of the answer
    try:
        pattern = r'Citation: \[(.*?)\]'
        match = re.search(pattern, generated_answer)

        if match:
            # Split the matched string by commas and strip whitespace
            ids = [id.strip() for id in match.group(1).split(',')]
            generated_sources = set(ids)
    except:
        generated_sources = set()
    
    # Convert true_documents to set
    true_sources = set(true_documents)
    
    # Calculate metrics
    if len(generated_sources) == 0:
        precision = 0
    else:
        precision = len(generated_sources.intersection(true_sources)) / len(generated_sources) * 100
        
    if len(true_sources) == 0:
        recall = 0
    else:
        recall = len(generated_sources.intersection(true_sources)) / len(true_sources) * 100
    
    # Calculate F1 score
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1

def is_rejection_response(answer):
    """Helper function to check if the answer is a rejection response"""
    rejection_phrases = [
        "i don't know",
        "i cannot find information",
        "i don't have enough context",
        "i don't have enough information"
        "i don't",
        "i cannot",
        "no relevant information",
        "provided context",
        "context you provided",
    ]
    answer_lower = answer.lower()
    # Check for exact phrases
    if any(phrase in answer_lower for phrase in rejection_phrases):
        return True
    # Check if answer begins with common rejection starters
    rejection_starters = ["i don't", "i cannot","i can't","sorry,"]
    if any(answer_lower.strip().startswith(starter) for starter in rejection_starters):
        return True
    return False

def calculate_metrics(eval_data):
    metrics = {}
    
    dataset_pairs = {
        'Google_NQ': {
            'main': 'Google_NQ_Dataset_Chunked.csv',
            'irrelevant': 'google_nq_500_irrelevant.csv'
        },
        'MSMarco': {
            'main': 'MSMarco_Dataset_Chunked.csv',
            'irrelevant': 'ms_marco_500_irrelevant.csv'
        },
        'MuSiQue': {
            'main': 'MuSiQue_Dataset_Chunked.csv',
            'irrelevant': 'musique_500_irrelevant.csv'
        }
    }

    for dataset_type, config in dataset_pairs.items():
        metrics[dataset_type] = {}
        main_results = None
        irrelevant_results = None
        
        # Find the results for both datasets
        for dataset_entry in eval_data:
            if config['main'] in dataset_entry:
                main_results = dataset_entry[config['main']]
            if config['irrelevant'] in dataset_entry:
                irrelevant_results = dataset_entry[config['irrelevant']]

        if main_results and irrelevant_results:
            # Calculate metrics on main dataset only
            correct_answers = sum(1 for r in main_results if r["judge_evaluation"])
            total_answerable = sum(1 for r in main_results if r["true_documents"])
            false_rejections = sum(1 for r in main_results 
                                 if is_rejection_response(r["generated_answer"])
                                 and r["true_documents"])
            rejection_rate = (false_rejections / total_answerable) * 100 if total_answerable > 0 else 0
            
            # Calculate hallucination rate on irrelevant dataset
            hallucinations = 0
            for r in irrelevant_results:
                answer = r["generated_answer"]
                is_rejection = is_rejection_response(answer)
                if not is_rejection:
                    hallucinations += 1

            hallucination_rate = (hallucinations / len(irrelevant_results)) * 100

            # Source evaluation metrics (main dataset only)
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            valid_examples = 0

            for result in main_results:
                # Skip examples where the model rejected to answer
                if is_rejection_response(result["generated_answer"]):
                    continue
                    
                # Use true_documents field from the result
                true_documents = result["true_documents"]
                
                # Calculate source metrics for this example
                precision, recall, f1 = calculate_source_metrics(
                    result["generated_answer"], 
                    true_documents
                )
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                valid_examples += 1

            # Average the source metrics
            avg_precision = total_precision / valid_examples if valid_examples > 0 else 0
            avg_recall = total_recall / valid_examples if valid_examples > 0 else 0
            avg_f1 = total_f1 / valid_examples if valid_examples > 0 else 0

            metrics[dataset_type].update({
                # Main dataset metrics
                "correctness": (correct_answers / len(main_results)) * 100,
                "rejection_rate": rejection_rate,
                "source_metrics": {
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "f1": avg_f1
                },
                # Irrelevant dataset metrics
                "hallucination_rate": hallucination_rate
            })

    return metrics

def print_metrics_report(metrics,model_name):
    print(f"\n=== Evaluation Metrics Report for {model_name} ===\n")
    
    for dataset_type, dataset_metrics in metrics.items():
        print(f"\n{dataset_type} Dataset:")
        print("-" * 50)
        
        print("Metrics on Main Dataset:")
        print(f"LLM-Judge Correctness: {dataset_metrics['correctness']:.1f}%")
        print(f"Rejection Rate: {dataset_metrics['rejection_rate']:.1f}%")
        
        print("\nSource Evaluation Metrics (Main Dataset):")
        source_metrics = dataset_metrics['source_metrics']
        print(f"Source Precision: {source_metrics['precision']:.1f}%")
        print(f"Source Recall: {source_metrics['recall']:.1f}%")
        print(f"Source F1 Score: {source_metrics['f1']:.1f}%")
        
        print("\nMetrics on Irrelevant Dataset:")
        print(f"Hallucination Rate: {dataset_metrics['hallucination_rate']:.1f}%")

def save_metrics_json(metrics, output_file):
    """Save metrics to a JSON file"""
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)

def main():
    # Process 4o mini results
    eval_results_4o_mini = load_eval_results("eval_generations_4o_mini.json")
    metrics_4o_mini = calculate_metrics(eval_results_4o_mini)
    print_metrics_report(metrics_4o_mini, "4o_mini")
    save_metrics_json(metrics_4o_mini, "evaluation_metrics_4o_mini.json")

    # Process llama results 
    eval_results_llama = load_eval_results("eval_generations_llama.json")
    metrics_llama = calculate_metrics(eval_results_llama)
    print_metrics_report(metrics_llama, "llama")
    save_metrics_json(metrics_llama, "evaluation_metrics_llama.json")
if __name__ == "__main__":
    main()