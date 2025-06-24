# -*- coding: utf-8 -*-
"""lora_gpt2_fine_tuning.py
Fine-tuning GPT-2 using LoRA on support QA pairs.
"""

# pip install transformers peft datasets torch scikit-learn

import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.pipelines import pipeline
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, classification_report
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


MODEL_NAME = "gpt2"
OUTPUT_DIR = "./lora-gpt2"
SEED = 42
MAX_LENGTH = 512

def load_dataset_from_file(filepath: str):
    """
    Load dataset and apply formatting for question-answer.
    """
    df = pd.read_csv(filepath, sep='\t', header=None, names=['question', 'answer'])
    df['text'] = "Вопрос: " + df['question'] + "\nОтвет: " + df['answer']
    return Dataset.from_pandas(df[['text']])

def prepare_model_and_tokenizer(model_name: str):
    """
    Load GPT-2 model and tokenizer, apply LoRA.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
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
        num_train_epochs=7,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

def evaluate_model(peft_model_dir, test_questions, true_answers, max_length=100):
    """
    Load PEFT model and evaluate it on test questions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = PeftConfig.from_pretrained(peft_model_dir)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).to(device)
    model = PeftModel.from_pretrained(base_model, peft_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(peft_model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu_scores = []
    rouge_scores = []

    print("\n=== Evaluation ===")
    for i, (question, true_answer) in enumerate(zip(test_questions, true_answers)):
        prompt = f"Вопрос: {question}\nОтвет:"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        if "Ответ:" in generated:
            generated_answer = generated.split("Ответ:")[-1].strip()
        else:
            generated_answer = generated.strip()

        print(f"\nQ: {question}")
        print(f"Real A: {true_answer}")
        print(f"Pred A: {generated_answer}")

        bleu = sentence_bleu([true_answer.split()], generated_answer.split())
        bleu_scores.append(bleu)

        rouge_result = scorer.score(true_answer, generated_answer)
        rouge_scores.append(rouge_result['rougeL'].fmeasure)

    avg_bleu = np.mean(bleu_scores)
    avg_rouge = np.mean(rouge_scores)

    print(f"\nAverage BLEU: {avg_bleu:.4f}")
    print(f"Average ROUGE-L F1: {avg_rouge:.4f}")

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

    true_answers = [
        "Чтобы сбросить пароль, перейдите на страницу восстановления пароля.",
        "Если вы не можете войти, проверьте правильность введенных данных.",
        "Чтобы обновить информацию о платеже, войдите в свой аккаунт и перейдите в раздел 'Платежи'.",
        "Инструкции по установке можно найти в разделе 'Поддержка' на нашем сайте.",
        "Вы можете связаться с технической поддержкой по телефону или через чат на сайте.",
        "Если приложение не запускается, попробуйте переустановить его.",
        "Статус вашего заказа можно проверить в разделе 'Мои заказы' в аккаунте.",
        "Настройки конфиденциальности можно изменить в разделе 'Настройки' вашего аккаунта.",
        "Чтобы удалить учетную запись, перейдите в настройки и выберите 'Удалить учетную запись'.",
        "Обучающие материалы доступны в разделе 'Обучение' на нашем сайте."
    ]

    evaluate_model(OUTPUT_DIR, test_questions, true_answers)
