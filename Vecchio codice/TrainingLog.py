from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from transformers import pipeline
import numpy as np
import evaluate
#GPU
print(torch.cuda.is_available())
print(torch.version.cuda)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#GPU FINE

dataset = load_dataset("json", data_files={"train": "C:\\Users\\GAMING EDGE\\Desktop\\LAUREA\\Datasets\\V4\\dataset-involved_services-raw_logs-10000-with_labels.json"})

# Importa il tokenizer da usare (BERT) e crea una funzione di tokenizzazione per il dataset.
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

labels = []

def is_string_in_array(string, string_array):
    return string in string_array

def trova_labels_unici(dataset):
    labels = []
    for val in dataset:

        # Estrai la stringa dal JSON
        data_string = dataset[val]["label"]

        # Rimuovi le stringhe duplicate mantenendo l'ordine dell'apparizione
        labels = list(dict.fromkeys(data_string))
            
    return labels

def tokenize_function(examples):
    return tokenizer(examples["raw_logs"], padding="max_length", truncation=True)

# Applica la funzione di creazione dell'input durante la mappatura
tokenized_datasets = dataset.map(tokenize_function, batched=False)

#region labels UNICI

temp_list = trova_labels_unici(dataset)

# dictionary that maps integer to its string value 
label_dict = {}

# list to store integer labels 
int_labels = []

for i in range(len(temp_list)):
    label_dict[i] = temp_list[i]
    int_labels.append(i)

def cambio_str2int(a):
    val = a["label"]
    for k in int_labels:
        if(val == temp_list[k]):
            #Trovato uguale, il suo numero è k
            #print("PRIMA#",k ,": ", a["label"])
            a["label"] = k
            #print("DOPO#",k ,": ", a["label"])

    return a
            

updated_dataset = tokenized_datasets.map(cambio_str2int)
print(updated_dataset["train"][9999]["label"]) ##dovrebbe essere 69

#endregion

# Dataset più piccoli per ridurre il tempo necessario (se si vuole), 375 alla fine
small_eval_dataset = updated_dataset["train"].shuffle(seed=42).select(range(1000))

# Prendi il modello
n_labels = len(trova_labels_unici(dataset))
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=n_labels)
model = model.to(device)

#region GPU
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

#endregion

# Imposta gli argomenti per il training
training_args = TrainingArguments(output_dir=" C:\\Users\\GAMING EDGE\\Desktop\\LAUREA\\Modelli\\ModelloLog",evaluation_strategy="epoch")
# Carica la metrica di valutazione
metric = evaluate.load("accuracy")

input("Premi invio per continuare:")

# calcola le metriche di valutazione in base ai logits prodotti dal modello e alle etichette 
# di riferimento del dataset di valutazione, utilizzando l'oggetto metric specificato.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Crea l'oggetto Trainer per l'addestramento
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= updated_dataset["train"],
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

input("Premi invio per addestrare:")

# Avvia l'addestramento
trainer.train()

# Salva il modello addestrato
model.save_pretrained("C:\\Users\\GAMING EDGE\\Desktop\\LAUREA\\Modelli\\ModelloLog")
