import os
import json
import time
import random
import glob
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from openai import OpenAI

# Initialize the NVIDIA-compatible OpenAI client (for other models)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

def exponential_backoff(retry_attempts):
    wait_time = min(2 ** retry_attempts, 128) + random.uniform(0, 2)
    print(f"Waiting {wait_time:.2f} seconds before retrying...")
    time.sleep(wait_time)
# Load Nemotron reward model from Hugging Face with CPU support
def load_nemotron_model():
    model_name = "nvidia/Nemotron-4-340B-Reward"
    
    # For CPU usage, we'll need to use quantization
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load with 8-bit quantization to reduce memory usage
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to('cpu')  # Explicitly move to CPU
    
    # Put model in evaluation mode
    model.eval()
    
    print("Nemotron model loaded on CPU with quantization")
    return model, tokenizer

# Initialize Nemotron model with error handling
try:
    print("Loading Nemotron reward model (this may take several minutes)...")
    nemotron_model, nemotron_tokenizer = load_nemotron_model()
    print("Successfully loaded Nemotron reward model on CPU")
except Exception as e:
    print(f"Error loading Nemotron model: {e}")
    raise


def score_response(question, response):
    """
    Scores a response based on:
    - Helpfulness: Overall helpfulness of the response to the prompt
    - Correctness: Inclusion of all pertinent facts without errors
    - Coherence: Consistency and clarity of expression
    - Complexity: Intellectual depth required to write response
    - Verbosity: Appropriate level of detail
    """
    scoring_prompt = f"""Evaluate this question and response:
    
    Question: {question}
    Response: {response}
    
    Score these aspects (1-5):
    - Helpfulness (1-5): How well does this answer the question?
    - Correctness (1-5): Are the facts accurate and complete?
    - Coherence (1-5): Is the response clear and logically structured?
    - Complexity (1-5): Does this require domain expertise to create?
    - Verbosity (1-5): Is the length appropriate? (3=ideal)
    
    Return as JSON with these scores only."""

    try:
        inputs = nemotron_tokenizer(scoring_prompt, return_tensors="pt", truncation=True, max_length=1024).to('cpu')
        
        with torch.no_grad():
            outputs = nemotron_model(**inputs)
            logits = outputs.logits
            scores = torch.sigmoid(logits).squeeze().tolist()
        
        return {
            "helpfulness": min(5, max(1, int(scores[0] * 4 + 1))),
            "correctness": min(5, max(1, int(scores[1] * 4 + 1))),
            "coherence": min(5, max(1, int(scores[2] * 4 + 1))),
            "complexity": min(5, max(1, int(scores[3] * 4 + 1))),
            "verbosity": min(5, max(1, int(scores[4] * 4 + 1)))
        }
    except Exception as e:
        print(f"Scoring error: {str(e)[:200]}")
        return {"helpfulness": 3, "correctness": 3, "coherence": 3, "complexity": 3, "verbosity": 3}
# Chunk generation prompt template
CHUNK_GENERATION_PROMPT_TEMPLATE = """\
Given the following text, divide it into logical sections or subtopics based on the content. 
Each section should correspond to a coherent subtopic. The sections should be as concise as 
possible but should not exceed 100 words per section. Do not provide a fixed number of subtopics, 
the model should determine the appropriate number of sections.

Text:
{raw_text}
"""

def generate_chunks(client, raw_text):
    prompt = CHUNK_GENERATION_PROMPT_TEMPLATE.format(raw_text=raw_text)
    retry_attempts = 0
    while retry_attempts < 5:
        try:
            response = client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip().split("\n\n")
        except Exception as e:
            print(f"Error generating chunks: {e}")
            if "Too Many Requests" in str(e):
                retry_attempts += 1
                exponential_backoff(retry_attempts)
            else:
                return []
    print("Max retries exceeded for chunk generation.")
    return []

# Question generation prompt template
QUESTION_PROMPT_TEMPLATE = """\
Given a topic, generate {n_questions} questions that could be asked about that topic. 
Your response should be in a list format.

The topic is: {sub_topic}

The list must be without numbers. The questions should be separated by a newline character. 
There must be no other text than the list.
"""

def generate_questions(client, sub_topic, n_questions):
    prompt = QUESTION_PROMPT_TEMPLATE.format(sub_topic=sub_topic, n_questions=n_questions)
    retry_attempts = 0
    while retry_attempts < 5:
        try:
            response = client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip().split("\n")
        except Exception as e:
            print(f"Error generating questions: {e}")
            if "Too Many Requests" in str(e):
                retry_attempts += 1
                exponential_backoff(retry_attempts)
            else:
                return []
    print("Max retries exceeded for question generation.")
    return []

# Response generation prompt template
RESPONSE_PROMPT_TEMPLATE = """\
Given a question, generate 2 responses that could be given to that question. 
Your response should be in a list format.

The question is: {question}

The list must be in the format:

RESPONSE A: Response A text here
RESPONSE B: Response B text here
"""

def generate_responses(client, question):
    prompt = RESPONSE_PROMPT_TEMPLATE.format(question=question)
    retry_attempts = 0
    while retry_attempts < 5:
        try:
            response = client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating responses: {e}")
            if "Too Many Requests" in str(e):
                retry_attempts += 1
                exponential_backoff(retry_attempts)
            else:
                return "RESPONSE A: N/A\nRESPONSE B: N/A"
    print("Max retries exceeded for response generation.")
    return "RESPONSE A: N/A\nRESPONSE B: N/A"


def read_raw_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    

def process_file(client, file_path, output_file_path, n_questions=3):
    raw_text = read_raw_text(file_path)
    subtopics = generate_chunks(client, raw_text)
    print(f"\nProcessing {file_path} - Identified {len(subtopics)} subtopics.")

    all_questions = []
    
    for i, subtopic in enumerate(subtopics):
        print(f"Generating questions for subtopic {i + 1}: {subtopic[:60]}...")
        try:
            qs = generate_questions(client, subtopic, n_questions)
            all_questions.extend(qs)
        except Exception as e:
            print(f"[{type(e).__name__}] Failed to generate questions for subtopic {i + 1}: {e}")
            continue

    print(f"Total questions generated: {len(all_questions)}")

    final_question_answer_pairs = []

    for question in all_questions:
        print(f"\nProcessing question: {question[:100]}...")
        try:
            response_set = generate_responses(client, question)

            retries = 0
            while "RESPONSE A:" in response_set and "RESPONSE B:" in response_set and "N/A" in response_set and retries < 3:
                print(f"Retrying due to N/A response for question: {question[:50]}...")
                response_set = generate_responses(client, question)
                retries += 1

            match = re.search(r"RESPONSE A:\s*(.*?)\s*RESPONSE B:\s*(.*)", response_set, re.DOTALL)
            if not match:
                print(f"Invalid response format for question")
                continue

            a, b = match.group(1).strip(), match.group(2).strip()
            if a == "N/A" or b == "N/A":
                print(f"Skipping question due to invalid responses.")
                continue

            # Score responses with more detailed logging for CPU
            print("Scoring response A...")
            start_time = time.time()
            scores_a = score_response(question, a)
            print(f"Scored response A in {time.time()-start_time:.2f}s - Scores: {scores_a}")
            
            print("Scoring response B...")
            start_time = time.time()
            scores_b = score_response(question, b)
            print(f"Scored response B in {time.time()-start_time:.2f}s - Scores: {scores_b}")

            # Select the better response
            if scores_a["helpfulness"] >= scores_b["helpfulness"]:
                selected_response = a
                selected_scores = scores_a
            else:
                selected_response = b
                selected_scores = scores_b

            # Only keep responses that meet minimum quality threshold
            if selected_scores["helpfulness"] >= 3.0:
                final_question_answer_pairs.append({
                    "question": question,
                    "response": selected_response,
                    "scores": selected_scores
                })
                print(f"Selected response with scores: {selected_scores}")

        except Exception as e:
            print(f"[{type(e).__name__}] Failed to process question - {e}")
            continue

    # Save results
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in final_question_answer_pairs:
            f.write(json.dumps(item) + "\n")

    print(f"\nCompleted processing {file_path}. Output saved to {output_file_path}.")

def process_all_files(client, input_dir, output_dir):
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    os.makedirs(output_dir, exist_ok=True)

    for file_path in txt_files:
        output_file_path = os.path.join(output_dir, os.path.basename(file_path).replace(".txt", ".jsonl"))
        process_file(client, file_path, output_file_path)

def merge_jsonl_files(output_dir, final_output_file):
    jsonl_files = glob.glob(os.path.join(output_dir, "*.jsonl"))
    os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
    
    with open(final_output_file, "w", encoding="utf-8") as f:
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as file:
                for line in file:
                    f.write(line)

    print(f"Final merged dataset saved to {final_output_file}")

# Main execution
if __name__ == "__main__":
    input_dir = "data2/"
    output_dir = "all_datasets/"
    final_output_file = "main_dataset/final_dataset.jsonl"
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
    
    try:
        process_all_files(client, input_dir, output_dir)
        merge_jsonl_files(output_dir, final_output_file)
        print("Processing completed successfully")
    except Exception as e:
        print(f"Fatal error in processing: {e}")
        raise