from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch


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