from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import pipeline

dataset = load_dataset("json", data_files="data.json")


# Load model and tokenizer
model_name = "deepseek-r1:1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    load_in_8bit=True,  # Reduce memory usage
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Layers to adapt
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Verify trainable params
training_args = TrainingArguments(
    output_dir="./deepseek_finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()

model.save_pretrained("./deepseek_finetuned")
tokenizer.save_pretrained("./deepseek_finetuned")


finetuned_model = AutoModelForCausalLM.from_pretrained("./deepseek_finetuned")
finetuned_pipeline = pipeline("text-generation", model=finetuned_model, tokenizer=tokenizer)

response = finetuned_pipeline("Translate to French: Hello, how are you?")
print(response)