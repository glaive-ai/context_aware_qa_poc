# Dropbox Evaluation Pipeline

This repository contains scripts to evaluate question-answering models on various datasets.

## Prerequisites

- GPU machine required
- API keys are pre-configured in the code

## Running the Evaluation

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Start the VLLM server:

```bash
vllm serve glaiveai/dropbox_qa_v2 --tensor-parallel 1 --host 0.0.0.0 --port 8000
```

3. Generate model responses:
```bash
python generate_gpt4o_mini.py
python generate_llama.py
```

4. Calculate metrics:
```bash
python calculate_metrics.py
```

The generate commands will save model responses to JSON files, which will then be processed by the metrics calculation script.

## Results

### GPT-4O Mini Model

| Dataset    | Metric                  | Main Dataset | Irrelevant Dataset |
|------------|------------------------|--------------|-------------------|
| Google NQ  | LLM-Judge Correctness  | 60.0%       | -                |
|            | Rejection Rate         | 28.2%       | -                |
|            | Source Precision       | 78.6%       | -                |
|            | Source Recall          | 78.6%       | -                |
|            | Source F1 Score        | 78.6%       | -                |
|            | Hallucination Rate     | -           | 0.2%             |
| MSMarco    | LLM-Judge Correctness  | 82.5%       | -                |
|            | Rejection Rate         | 0.0%        | -                |
|            | Source Precision       | 19.0%       | -                |
|            | Source Recall          | 67.5%       | -                |
|            | Source F1 Score        | 28.6%       | -                |
|            | Hallucination Rate     | -           | 91.4%            |
| MuSiQue    | LLM-Judge Correctness  | 35.0%       | -                |
|            | Rejection Rate         | 50.0%       | -                |
|            | Source Precision       | 67.5%       | -                |
|            | Source Recall          | 56.2%       | -                |
|            | Source F1 Score        | 59.5%       | -                |
|            | Hallucination Rate     | -           | 9.2%             |

### Llama Model

| Dataset    | Metric                  | Main Dataset | Irrelevant Dataset |
|------------|------------------------|--------------|-------------------|
| Google NQ  | LLM-Judge Correctness  | 62.5%       | -                |
|            | Rejection Rate         | 51.3%       | -                |
|            | Source Precision       | 92.5%       | -                |
|            | Source Recall          | 95.0%       | -                |
|            | Source F1 Score        | 93.3%       | -                |
|            | Hallucination Rate     | -           | 34.9%            |
| MSMarco    | LLM-Judge Correctness  | 85.0%       | -                |
|            | Rejection Rate         | 7.5%        | -                |
|            | Source Precision       | 17.4%       | -                |
|            | Source Recall          | 62.2%       | -                |
|            | Source F1 Score        | 26.1%       | -                |
|            | Hallucination Rate     | -           | 95.0%            |
| MuSiQue    | LLM-Judge Correctness  | 60.0%       | -                |
|            | Rejection Rate         | 32.5%       | -                |
|            | Source Precision       | 68.5%       | -                |
|            | Source Recall          | 54.0%       | -                |
|            | Source F1 Score        | 58.4%       | -                |
|            | Hallucination Rate     | -           | 35.4%            |

