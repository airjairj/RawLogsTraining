#
from huggingface_hub import notebook_login, create_repo
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
import evaluate
import numpy as np
import Levenshtein


notebook_login()
#
##books = load_dataset("opus_books", "en-fr")
dataset = load_dataset("json", data_files={"train": "PERCORSO DATASET"})

#region cambio lettere servizi
mappa_sostituzione = {'webservice1': 'A', 'redisservice1': 'B', 'mobservice1': 'C', 'logservice1': 'D', 'dbservice1': 'E', 'redisservice2': 'F' , 'logservice2': 'G', 'mobservice2': 'H', 'dbservice2': 'I', 'webservice2': 'J'}

def trova_servizi_unici(a):
    risultato = ""
    prossima_lettera = 'A'
    # Estrai la stringa dal JSON
    stringa = a["label"]
    elementi = stringa.split('--')

    for elemento in elementi:
        risultato += mappa_sostituzione[elemento] + '--'

    # Rimuovi l'ultimo ' ' extra
    risultato = risultato[:-2]
    #print("PRIMA:",a["label"])
    a["label"] = risultato
    #print("DOPO:",a["label"])

    return a
#endregion

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
tokenized_dataset = dataset.map(trova_servizi_unici)
tokenized_dataset = tokenized_dataset.map(preprocess_function, batched=True)

tokenized_dataset = tokenized_dataset.remove_columns("label")
#
print(mappa_sostituzione)


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
#
#EVALUATE
#

def compute_edit_distance(eval_preds):
    preds, labels = eval_preds

    # Calculate the edit distance for each pair of predictions and labels
    edit_distances = [Levenshtein.distance(pred, label) for pred, label in zip(preds, labels)]

    # Calculate the average edit distance
    avg_edit_distance = sum(edit_distances) / len(edit_distances)

    # Return the average edit distance as the evaluation metric
    result = {"edit_distance": avg_edit_distance}
    return result
#
#TRAINING
#
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#
training_args = Seq2SeqTrainingArguments(
    output_dir="PERCORSO MODELLO",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=18,
    per_device_eval_batch_size=18,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=18,
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
    compute_metrics=compute_edit_distance,
)
input("Premi invio per addestrare")
trainer.train()
#
trainer.push_to_hub()
#