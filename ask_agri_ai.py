from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "models/agri_t5_qa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

while True:
    question = input("Ask AgriLLaVA Mini: ")
    if question.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    print("🤖:", tokenizer.decode(outputs[0], skip_special_tokens=True))
