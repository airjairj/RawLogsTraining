from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
from datasets import load_dataset


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Esempio di testo da classificare
text_to_classify = input("Inserisci il log che vuoi classificare: \n")

# Tokenizza il testo di input
encoded_input = tokenizer(text_to_classify, padding="max_length", truncation=True, return_tensors="pt")

# Prende il modello che ho addestrato dalla directory
model = AutoModelForSequenceClassification.from_pretrained("C:\\Users\\GAMING EDGE\\Desktop\\LAUREA\\Modelli\\ModelloLog")

# Assicurarsi che il modello sia in modalità valutazione (è già il caso se hai completato l'addestramento)
model.eval()

# Effettua la previsione
with torch.no_grad():
    outputs = model(**encoded_input)

# Recupera i logit prodotti dal modello
logits = outputs.logits

# Ottieni le previsioni effettive (classe predetta)
predictions = torch.argmax(logits, dim=1)

# Stampa le previsioni
print("Classe predetta:", predictions.item())

#aggiungere il log per intero
dataset = load_dataset("json", data_files={"train": "C:\\Users\\GAMING EDGE\\Desktop\\LAUREA\\Datasets\\V4\\dataset-involved_services-raw_logs-10000-with_labels.json"})

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
            
print(dataset["train"][predictions.item()]["label"])