#
from huggingface_hub import notebook_login, create_repo
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
import evaluate
import numpy as np

notebook_login()
#
##books = load_dataset("opus_books", "en-fr")
dataset = load_dataset("json", data_files={"train": "/content/drive/MyDrive/Colab Notebooks/dataset-involved_services-raw_logs-10000-with_labels.json"})

#
dataset = dataset["train"].train_test_split(test_size=0.1)
#
# PREPROCESS
#
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
source_lang = "raw_logs"
target_lang = "label"
prefix = "translate raw_logs to label:"


def preprocess_function(examples):

    inputs = [prefix + example for example in examples[source_lang]]
    targets = [example for example in examples[target_lang]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs
#
tokenized_dataset = dataset.map(preprocess_function, batched=True)
#
print(dataset["train"][0])
print(tokenized_dataset["train"][0])


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
#
#EVALUATE
#
metric = evaluate.load("sacrebleu")
#
def postprocess_text(preds, labels):

    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
#
#TRAINING
#
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#
training_args = Seq2SeqTrainingArguments(
    output_dir="MODELLO",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
input("Premi invio per addestrare")
trainer.train()
#
trainer.push_to_hub()
#