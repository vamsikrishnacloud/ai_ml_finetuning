import os
import json
import time
import random
from openai import OpenAI
import glob

# Initialize the NVIDIA-compatible OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

# Exponential backoff utility function
def exponential_backoff(retry_attempts):
    # Exponential backoff with some randomness to avoid thundering herd
    wait_time = min(2 ** retry_attempts, 64) + random.uniform(0, 1)
    print(f"Waiting {wait_time:.2f} seconds before retrying...")
    time.sleep(wait_time)

CHUNK_GENERATION_PROMPT_TEMPLATE = """\
Given the following text, divide it into logical sections or subtopics based on the content. Each section should correspond to a coherent subtopic. The sections should be as concise as possible but should not exceed 100 words per section. Do not provide a fixed number of subtopics, the model should determine the appropriate number of sections.

Text:
{raw_text}
"""

# Function to process and generate chunks
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
            return response.choices[0].message.content.strip().split("\n\n")  # Split into list of subtopics
        except Exception as e:
            print(f"Error generating chunks: {e}")
            if "Too Many Requests" in str(e):
                retry_attempts += 1
                exponential_backoff(retry_attempts)
            else:
                return []

    print("Max retries exceeded for chunk generation.")
    return []

# 2. Questions Generation
QUESTION_PROMPT_TEMPLATE = """\
Given a topic, generate {n_questions} questions that could be asked about that topic. Your response should be in a list format.

The topic is: {sub_topic}

The list must be without numbers. The questions should be separated by a newline character. There must be no other text than the list.
"""

# Function to generate questions for a subtopic
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

RESPONSE_PROMPT_TEMPLATE = """\
Given a question, generate 2 responses that could be given to that question. Your response should be in a list format.

The question is: {question}

The list must be in the format:

RESPONSE A: Response A text here
RESPONSE B: Response B text here
"""

# Function to generate responses for a question
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

# Function to read raw text from a file
def read_raw_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to process each file and generate Q&A
def process_file(client, file_path, output_file_path, n_questions=3):
    raw_text = read_raw_text(file_path)

    # step 1: chunking
    subtopics = generate_chunks(client, raw_text)
    print(f"\nProcessing {file_path} - Subtopics: {len(subtopics)}")

    # step 2: generate questions for each subtopic
    questions = generate_questions(client, subtopics, n_questions)
    print(f"Generated {len(questions)} questions for {file_path}")

    question_response_pair_list = []

    # generate responses for each question
    for question in questions:
        print(f"Processing question: {question}")
        response_set = generate_responses(client, question)

        # Retry if response is empty or contains placeholders
        while "RESPONSE A:" in response_set and "RESPONSE B:" in response_set and ("N/A" in response_set):
            print(f"Retrying due to N/A response for question: {question}")
            response_set = generate_responses(client, question)

        if "RESPONSE A:" in response_set and "RESPONSE B:" in response_set:
            a = response_set.split("RESPONSE B:")[0].replace("RESPONSE A:", "").strip()
            b = response_set.split("RESPONSE B:")[-1].strip()
        else:
            a = response_set.strip()
            b = "N/A"
        
        # log skipped responses
        if b == "N/A":
            print(f"Skipping response for question: {question} (Response B is N/A)")

        # store valid question-response pairs
        if a != "N/A" and b != "N/A":
            question_response_pair_list.append({
                "question": question,
                "responses": {
                    "response_a": {"response": a},
                    "response_b": {"response": b},
                }
            })
        else:
            print(f"Skipping question: {question} due to invalid responses.")

    # Save the result to a JSONL file
    with open(output_file_path, "w") as f:
        for item in question_response_pair_list:
            f.write(json.dumps(item))
            f.write("\n")

    print(f"Processed {file_path} and saved to {output_file_path}")

# Main flow to process all files in the directory
def process_all_files(client, input_dir, output_dir):
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    for file_path in txt_files:
        output_file_path = os.path.join(output_dir, os.path.basename(file_path).replace(".txt", ".jsonl"))
        process_file(client, file_path, output_file_path)

# function to merge JSONL files into a single file
def merge_jsonl_files(output_dir, final_output_file):
    jsonl_files = glob.glob(os.path.join(output_dir, "*.jsonl"))
    with open(final_output_file, "w") as f:
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r") as file:
                for line in file:
                    f.write(line)

    print(f"Final merged dataset saved to {final_output_file}")

# main execution
input_dir = "data/"
output_dir = "all_datasets/"
final_output_file = "main_dataset/final_dataset.jsonl"

# process all files in the directory
process_all_files(client, input_dir, output_dir)

# merge all the JSONL files into a final dataset
merge_jsonl_files(output_dir, final_output_file)

print("Processing complete and final dataset saved.")


# QUESTION_PROMPT_TEMPLATE = """\
# Given a topic, generate {n_questions} questions that could be asked about that topic. If the text mentions contact information (phone, email, etc.), be sure to include at least one question asking about it.

# The topic is: {sub_topic}

# The list must be without numbers. The questions should be separated by a newline character. There must be no other text than the list.
# """
