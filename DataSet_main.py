import os
import json
import time
import random
import glob
import re
import torch
from openai import OpenAI
import requests

# Initialize the NVIDIA-compatible OpenAI client (for other models)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)


def extract_reward_value(response: str) -> float:
    try:
        match = re.search(r"reward:\s*(-?\d+(?:\.\d+)?)", response)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Reward value not found in response.")
    except Exception as e:
        print(f"[ERROR] Could not extract reward: {e}")
        return None



def exponential_backoff(retry_attempts):
    wait_time = min(2 ** retry_attempts, 128) + random.uniform(0, 2)
    print(f"Waiting {wait_time:.2f} seconds before retrying...")
    time.sleep(wait_time)


def call_nvidia_reward_model(prompt: str):
    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-reward",
            messages=[{"role": "user", "content": prompt}],
        )
        # Extracting the response
        response_content = completion.choices[0].message.content

        # Extract reward
        reward_score = extract_reward_value(response_content)

        return response_content, reward_score
    except Exception as e:
        print(f"Error calling model: {e}")
        return None, None


# Scoring function to compare two responses and select the best one
def score_responses(question, response_a, response_b):
    try:
        # Get the scores for both responses
        _, score_a = call_nvidia_reward_model(f"{question} {response_a}")
        _, score_b = call_nvidia_reward_model(f"{question} {response_b}")

        # If both scores are available, compare them
        if score_a is not None and score_b is not None:
            # Compare the two scores
            if score_a >= score_b:
                return response_a, score_a
            else:
                return response_b, score_b
        else:
            print("Error: One of the responses is missing.")
            return None, None
    except Exception as e:
        print(f"Error scoring responses: {e}")
        return None, None


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

# Response generation prompt template
RESPONSE_PROMPT_TEMPLATE = """\
Given a question, generate 2 responses that could be given to that question. 
Your response should be in a list format.

The question is: {question}

The list must be in the format:

RESPONSE A: Response A text here
RESPONSE B: Response B text here
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

            # Log structured response (safe and informative)
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                return content.strip() if content else "RESPONSE A: N/A\nRESPONSE B: N/A"
            else:
                print("Warning: No choices found in the response.")
                return "RESPONSE A: N/A\nRESPONSE B: N/A"
        
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

# Process file and generate best response
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
        
        # Skip empty questions
        if not question.strip():  # Check if the question is empty or contains only whitespace
            print("Skipping empty question.")
            continue

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

            # Score responses and select the best
            selected_response, selected_score = score_responses(question, a, b)

            if selected_response:
                final_question_answer_pairs.append({
                    "question": question,
                    "response": selected_response,
                    "score": selected_score
                })
                print(f"Selected response with score: {selected_score}")

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
    all_lines = []
    for file in jsonl_files:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            all_lines.extend(lines)

    with open(final_output_file, "w", encoding="utf-8") as f_out:
        for line in all_lines:
            f_out.write(line)

    print(f"Merged {len(jsonl_files)} files into {final_output_file}. Total entries: {len(all_lines)}")


if __name__ == "__main__":
    INPUT_DIR = "data/"  
    OUTPUT_DIR = "all_datasets/"
    FINAL_OUTPUT_FILE = "main_dataset/final_output_dataset.jsonl"

    process_all_files(client, INPUT_DIR, OUTPUT_DIR)
    merge_jsonl_files(OUTPUT_DIR, FINAL_OUTPUT_FILE)
