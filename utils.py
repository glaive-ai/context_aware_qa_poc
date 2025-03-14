from openai import OpenAI

def get_client(local=False):
    if local:
        return OpenAI(base_url="http://localhost:8000/v1",api_key="test")
    else:
        return OpenAI(base_url="https://openrouter.ai/api/v1",api_key="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")

def infer(client,messages,model):
    retries = 0
    while retries < 2:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=10000,
                stream=False,
            )
            out = response.choices[0].message.content.strip()
            return out
        except Exception as e:
            print("Error: ",e)
            retries += 1
    return None

GPT_4O_MINI_SYSTEM_PROMPT = '''You are a precise answer generator. Given a question and context documents, generate an answer following these exact rules:

1. First, carefully analyze if the answer can be found in the provided context documents. If not, respond only with "I don't know"

2. If an answer is found, format your response in two parts:

PART 1 - DIRECT ANSWER:
- Provide a single-line, concise answer
- Highlight key information using bold markdown (**key info**)
- Must be factual and supported by the context
- End with a period

PART 2 - ADDITIONAL DETAILS (only if relevant):
- Start with the heading "Additional Details:"
- Use markdown bullet points (*)
- Maximum 10 bullet points
- Each bullet must be clear and concise
- No nested bullets
- No tables unless explicitly requested
- All information must be from the provided context

Remember:
- Never include information not present in the context
- Never use nested bullets
- Keep the initial answer to one line
- Only include Additional Details if they add value
- Bold text must highlight specific, important details

Output Format Rules:
- Begin with a concise, single-line answer
- Highlight key details with markdown bold syntax ( ) 
- "Additional Details" section (if needed) must:
- Use markdown bullet points (*)
- Contain ≤10 bullet points (unless specifically required)
- Be clear and concise
- Have no nested bullets
- Include no tables unless specifically requested
- At the end of your answer, include the document IDs of the sources used in the "Citation" section in the format "[doc_id1, doc_id2, doc_id3...]"
'''

JUDGE_SYSTEM_PROMPT = '''You are an expert evaluator assessing the correctness of generated answers.

    Your goal is to determine if the generated answer is factually consistent with the ground truth answer.

    Question: {question}
    Generated Answer: {generated_answer}
    Ground Truth: {ground_truth}

    Evaluation Criteria:

    1. Factual Accuracy
    - Check if every statement in the generated answer is factually correct
    - Ignore minor omissions or incompleteness
    - Focus on the accuracy of the information provided
    - Disregard citation document IDs

    2. Key Information Assessment
    - Verify if the core/critical information from the ground truth is captured
    - Partial coverage is acceptable if the provided information is correct
    - Do not penalize for additional relevant information not in the ground truth

    3. Contradiction Check
    - Ensure no statements directly contradict the ground truth
    - One factual error or contradiction results in an incorrect rating

    Decision:
    Based on the above analysis, choose one:
    - "Y" if the answer is correct (can be incomplete but must be factually accurate)
    - "N" if there are any factual errors or contradictions or direct admissions of not knowing / inability to answer

    Your response should be either "Y" or "N".'''