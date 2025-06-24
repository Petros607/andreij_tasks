# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
# from peft import LoraConfig, get_peft_model
# from datasets import load_dataset

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
import torch
from datasets import Dataset

data = pd.read_csv('support_responses.txt', sep='\t', header=None, names=['question', 'answer'])
data['text'] = data['question'] + ' ' + data['answer']
dataset = Dataset.from_pandas(data[['text']])

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def tokenize_function(examples):
    encodings = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    encodings['labels'] = encodings['input_ids'].copy()
    return encodings

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./lora-gpt2",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

trainer.save_model("./lora-gpt2")
tokenizer.save_pretrained("./lora-gpt2")

new_questions = [
    "Как я могу сбросить пароль?",
    "Что делать, если я не могу войти в свою учетную запись?",
    "Как обновить информацию о платеже?",
    "Где я могу найти инструкции по установке?",
    "Как связаться с технической поддержкой?",
    "Что делать, если приложение не запускается?",
    "Как проверить статус моего заказа?",
    "Как изменить настройки конфиденциальности?",
    "Как удалить свою учетную запись?",
    "Как получить доступ к обучающим материалам?"
]

for question in new_questions:
    input_text = question + " "
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=150, num_return_sequences=1)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Вопрос: {question}\nОтвет: {answer}\n")
