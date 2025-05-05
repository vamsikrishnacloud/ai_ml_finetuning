# import os
# import json
# import time
# import random
# from openai import OpenAI
# import glob
# import re

# # Initialize the NVIDIA-compatible OpenAI client
# client = OpenAI(
#     base_url="https://integrate.api.nvidia.com/v1",
#     api_key=os.environ["NVIDIA_API_KEY"]
# )

# # Exponential backoff utility function
# def exponential_backoff(retry_attempts):
#     # Exponential backoff with some randomness to avoid thundering herd
#     wait_time = min(2 ** retry_attempts, 128) + random.uniform(0, 2)
#     print(f"Waiting {wait_time:.2f} seconds before retrying...")
#     time.sleep(wait_time)

# # Adjusted chunk generation prompt
# CHUNK_GENERATION_PROMPT_TEMPLATE = """\
# Given the following text, divide it into logical sections or subtopics based on the content. Each section should correspond to a coherent subtopic. The sections should be as concise as possible but should not exceed 100 words per section. Do not provide a fixed number of subtopics, the model should determine the appropriate number of sections.

# Text:
# {raw_text}
# """

# # Function to process and generate chunks
# def generate_chunks(client, raw_text):
#     prompt = CHUNK_GENERATION_PROMPT_TEMPLATE.format(raw_text=raw_text)
#     retry_attempts = 0
#     while retry_attempts < 5:
#         try:
#             response = client.chat.completions.create(
#                 model="meta/llama-3.1-405b-instruct",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.2,
#                 top_p=0.7,
#                 max_tokens=1024,
#             )
#             return response.choices[0].message.content.strip().split("\n\n")  # Split into list of subtopics
#         except Exception as e:
#             print(f"Error generating chunks: {e}")
#             if "Too Many Requests" in str(e):
#                 retry_attempts += 1
#                 exponential_backoff(retry_attempts)
#             else:
#                 return []

#     print("Max retries exceeded for chunk generation.")
#     return []

# # 2. Questions Generation
# QUESTION_PROMPT_TEMPLATE = """\
# Given a topic, generate {n_questions} questions that could be asked about that topic. Your response should be in a list format.

# The topic is: {sub_topic}

# The list must be without numbers. The questions should be separated by a newline character. There must be no other text than the list.
# """

# # Function to generate questions for a subtopic
# def generate_questions(client, sub_topic, n_questions):
#     prompt = QUESTION_PROMPT_TEMPLATE.format(sub_topic=sub_topic, n_questions=n_questions)
#     retry_attempts = 0
#     while retry_attempts < 5:
#         try:
#             response = client.chat.completions.create(
#                 model="meta/llama-3.1-405b-instruct",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.2,
#                 top_p=0.7,
#                 max_tokens=1024,
#             )
#             return response.choices[0].message.content.strip().split("\n")
#         except Exception as e:
#             print(f"Error generating questions: {e}")
#             if "Too Many Requests" in str(e):
#                 retry_attempts += 1
#                 exponential_backoff(retry_attempts)
#             else:
#                 return []

#     print("Max retries exceeded for question generation.")
#     return []

# RESPONSE_PROMPT_TEMPLATE = """\
# Given a question, generate 2 responses that could be given to that question. Your response should be in a list format.

# The question is: {question}

# The list must be in the format:

# RESPONSE A: Response A text here
# RESPONSE B: Response B text here
# """

# # Function to generate responses for a question
# def generate_responses(client, question):
#     prompt = RESPONSE_PROMPT_TEMPLATE.format(question=question)
#     retry_attempts = 0
#     while retry_attempts < 5:
#         try:
#             response = client.chat.completions.create(
#                 model="meta/llama-3.1-405b-instruct",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.2,
#                 top_p=0.7,
#                 max_tokens=1024,
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"Error generating responses: {e}")
#             if "Too Many Requests" in str(e):
#                 retry_attempts += 1
#                 exponential_backoff(retry_attempts)
#             else:
#                 return "RESPONSE A: N/A\nRESPONSE B: N/A"

#     print("Max retries exceeded for response generation.")
#     return "RESPONSE A: N/A\nRESPONSE B: N/A"

# # Function to read raw text from a file
# def read_raw_text(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return file.read()

# def process_file(client, file_path, output_file_path, n_questions=3):
#     raw_text = read_raw_text(file_path)

#     # Step 1: Chunk into subtopics
#     subtopics = generate_chunks(client, raw_text)
#     print(f"\nProcessing {file_path} - Identified {len(subtopics)} subtopics.")

#     all_questions = []
    
#     # Step 2: Generate questions per subtopic
#     for i, subtopic in enumerate(subtopics):
#         print(f"Generating questions for subtopic {i + 1}: {subtopic[:60]}...")
#         try:
#             qs = generate_questions(client, subtopic, n_questions)
#             all_questions.extend(qs)
#         except Exception as e:
#             print(f"[{type(e).__name__}] Failed to generate questions for subtopic {i + 1}: {e}")
#             continue

#     print(f"Total questions generated: {len(all_questions)}")

#     question_response_pair_list = []

#     # Step 3: Generate responses for each question
#     for question in all_questions:
#         print(f"\nGenerating responses for question: {question}")
#         try:
#             response_set = generate_responses(client, question)

#             # Retry if responses are missing or invalid
#             retries = 0
#             while "RESPONSE A:" in response_set and "RESPONSE B:" in response_set and "N/A" in response_set and retries < 3:
#                 print(f"Retrying due to N/A response for question: {question}")
#                 response_set = generate_responses(client, question)
#                 retries += 1

#             # Extract responses using regex for robustness
#             match = re.search(r"RESPONSE A:\s*(.*?)\s*RESPONSE B:\s*(.*)", response_set, re.DOTALL)
#             if match:
#                 a, b = match.group(1).strip(), match.group(2).strip()
#             else:
#                 print(f"Invalid response format for question: {question}")
#                 continue

#             if a != "N/A" and b != "N/A":
#                 question_response_pair_list.append({
#                     "question": question,
#                     "responses": {
#                         "response_a": {"response": a},
#                         "response_b": {"response": b},
#                     }
#                 })
#             else:
#                 print(f"Skipping question: {question} due to invalid responses.")
#         except Exception as e:
#             print(f"[{type(e).__name__}] Failed to generate responses for question: {question} - {e}")
#             continue

#     # Step 4: Save results
#     with open(output_file_path, "w", encoding="utf-8") as f:
#         for item in question_response_pair_list:
#             f.write(json.dumps(item) + "\n")

#     print(f"\nCompleted processing {file_path}. Output saved to {output_file_path}.")


# # Main flow to process all files in the directory
# def process_all_files(client, input_dir, output_dir):
#     txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

#     for file_path in txt_files:
#         output_file_path = os.path.join(output_dir, os.path.basename(file_path).replace(".txt", ".jsonl"))
#         process_file(client, file_path, output_file_path)

# # function to merge JSONL files into a single file
# def merge_jsonl_files(output_dir, final_output_file):
#     jsonl_files = glob.glob(os.path.join(output_dir, "*.jsonl"))
#     with open(final_output_file, "w") as f:
#         for jsonl_file in jsonl_files:
#             with open(jsonl_file, "r") as file:
#                 for line in file:
#                     f.write(line)

#     print(f"Final merged dataset saved to {final_output_file}")

# # main execution
# input_dir = "data2/"
# output_dir = "all_datasets/"
# final_output_file = "main_dataset/final_dataset.jsonl"

# # process all files in the directory
# process_all_files(client, input_dir, output_dir)

# # merge all the JSONL files into a final dataset
# merge_jsonl_files(output_dir, final_output_file)

# print("Processing complete and final dataset saved.")

import os
import json
import time
import random
import glob
import re
from openai import OpenAI

# Initialize the NVIDIA-compatible OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

# Exponential backoff utility function
def exponential_backoff(retry_attempts):
    wait_time = min(2 ** retry_attempts, 128) + random.uniform(0, 2)
    print(f"Waiting {wait_time:.2f} seconds before retrying...")
    time.sleep(wait_time)

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

# Response scoring function using Nemotron
def score_response(client, question, response):
    prompt = f"""\
Evaluate the following question and response pair based on these criteria:
1. Helpfulness (1-5): How well does the response answer the question?
2. Relevance (1-5): How relevant is the response to the question?
3. Accuracy (1-5): How accurate is the information in the response?

Return your evaluation as a JSON object with these three scores.

Question: {question}
Response: {response}
"""
    retry_attempts = 0
    while retry_attempts < 5:
        try:
            completion = client.chat.completions.create(
                model="nvidia/nemotron-4-340b-reward",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            try:
                scores = json.loads(completion.choices[0].message.content)
                return {
                    "helpfulness": float(scores.get("Helpfulness", 3)),
                    "relevance": float(scores.get("Relevance", 3)),
                    "accuracy": float(scores.get("Accuracy", 3))
                }
            except (json.JSONDecodeError, ValueError, AttributeError):
                return {"helpfulness": 3, "relevance": 3, "accuracy": 3}
                
        except Exception as e:
            print(f"Error scoring response: {e}")
            retry_attempts += 1
            time.sleep(2 ** retry_attempts)
    
    return {"helpfulness": 3, "relevance": 3, "accuracy": 3}

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
        print(f"\nGenerating responses for question: {question}")
        try:
            response_set = generate_responses(client, question)

            retries = 0
            while "RESPONSE A:" in response_set and "RESPONSE B:" in response_set and "N/A" in response_set and retries < 3:
                print(f"Retrying due to N/A response for question: {question}")
                response_set = generate_responses(client, question)
                retries += 1

            match = re.search(r"RESPONSE A:\s*(.*?)\s*RESPONSE B:\s*(.*)", response_set, re.DOTALL)
            if not match:
                print(f"Invalid response format for question: {question}")
                continue

            a, b = match.group(1).strip(), match.group(2).strip()
            if a == "N/A" or b == "N/A":
                print(f"Skipping question: {question} due to invalid responses.")
                continue

            # Score both responses
            scores_a = score_response(client, question, a)
            time.sleep(1.2)
            scores_b = score_response(client, question, b)
            time.sleep(1.2)

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

        except Exception as e:
            print(f"[{type(e).__name__}] Failed to process question: {question} - {e}")
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

    process_all_files(client, input_dir, output_dir)
    merge_jsonl_files(output_dir, final_output_file)
    print("Processing complete and final dataset saved.")