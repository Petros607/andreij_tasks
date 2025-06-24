# -*- coding: utf-8 -*-
"""lora_gpt2_fine_tuning.py
Fine-tuning GPT-2 using LoRA on support QA pairs.
"""

# pip install transformers peft datasets torch scikit-learn

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from peft import get_peft_model, LoraConfig

MODEL_NAME = "gpt2"
OUTPUT_DIR = "./lora-gpt2"
SEED = 42
MAX_LENGTH = 512

def load_dataset_from_file(filepath: str):
    """
    Load dataset from a TSV file and prepare the text field.
    """
    df = pd.read_csv(filepath, sep='\t', header=None, names=['question', 'answer'])
    df['text'] = df['question'] + ' ' + df['answer']
    return Dataset.from_pandas(df[['text']])

def prepare_model_and_tokenizer(model_name: str):
    """
    Load GPT-2 model and tokenizer, apply LoRA.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize dataset for causal language modeling.
    """
    def tokenize_function(examples):
        encodings = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH
        )
        encodings['labels'] = encodings['input_ids'].copy()
        return encodings

    return dataset.map(tokenize_function, batched=True)

def fine_tune(model, tokenizer, tokenized_dataset, output_dir):
    """
    Fine-tune the model using the Trainer API.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
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
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

def generate_answers(model, tokenizer, questions):
    """
    Generate answers for a list of questions using the fine-tuned model.
    """
    for question in questions:
        input_text = question + " "
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        output = model.generate(
            input_ids,
            max_length=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Вопрос: {question}\nОтвет: {answer}\n")

if __name__ == "__main__":
    torch.manual_seed(SEED)

    dataset = load_dataset_from_file('support_responses.txt')
    model, tokenizer = prepare_model_and_tokenizer(MODEL_NAME)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    fine_tune(model, tokenizer, tokenized_dataset, OUTPUT_DIR)

    test_questions = [
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

    generate_answers(model, tokenizer, test_questions)
