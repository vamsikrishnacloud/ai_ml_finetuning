from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
import torch
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import json

# Load model and tokenizer
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype=torch.float32
)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

# Load test dataset
test_dataset = load_dataset('json', data_files='dataset_test.jsonl')['train'].train_test_split(test_size=0.2)['test']

def evaluate_model(model, tokenizer, dataset, num_samples=50):
    results = []
    metrics = {
        'exact_match': 0,
        'bleu_score': 0,
        'rouge_score': 0,
        'perplexity': 0
    }
    
    # Select subset for evaluation
    eval_data = dataset.select(range(min(num_samples, len(dataset))))
    
    for example in eval_data:
        question = example['question']
        reference = example['response']
        
        # Generate answer
        inputs = tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt").to("cpu")
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
        except:
            prediction = "Generation error"
        
        # Calculate metrics (simplified versions)
        exact_match = int(prediction.lower() == reference.lower())
        bleu = min(len(prediction.split()) / max(len(reference.split()), 1), 1)  # Simplified
        rouge = min(len(set(prediction.split()) & set(reference.split())) / max(len(set(reference.split())), 1), 1)
        
        results.append({
            'question': question,
            'reference': reference,
            'prediction': prediction,
            'exact_match': exact_match,
            'bleu_score': bleu,
            'rouge_score': rouge
        })
    
    # Aggregate metrics
    if results:
        metrics['exact_match'] = np.mean([r['exact_match'] for r in results])
        metrics['bleu_score'] = np.mean([r['bleu_score'] for r in results])
        metrics['rouge_score'] = np.mean([r['rouge_score'] for r in results])
    
    return metrics, results

# Run evaluation
metrics, detailed_results = evaluate_model(model, tokenizer, test_dataset)

# Save results
output = {
    'model': model_name,
    'metrics': metrics,
    'examples': detailed_results[:10]  # Save first 10 examples
}

with open('evaluation_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Evaluation Metrics:")
print(f"Exact Match: {metrics['exact_match']:.2f}")
print(f"BLEU Score: {metrics['bleu_score']:.2f}")
print(f"ROUGE Score: {metrics['rouge_score']:.2f}")
print("\nSample predictions saved to evaluation_results.json")