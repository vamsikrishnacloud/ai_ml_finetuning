from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)


model = AutoModelForCausalLM.from_pretrained("./results_lora")
tokenizer = AutoTokenizer.from_pretrained("./results_lora")
tokenizer.pad_token = tokenizer.eos_token

# set up the generator pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 3. Generate an answer for a custom question
custom_question = "what are key benefits if cloudcomputing in cloudtern solutions?"
prompt = f"### Question:\n{custom_question}\n\n### Answer:\n"

output = generator(prompt, max_length=256, do_sample=True, top_p=0.9, temperature=0.7)

# Print the generated answer
# print(output)
print("\nGenerated Answer:\n", output[0]["generated_text"])
