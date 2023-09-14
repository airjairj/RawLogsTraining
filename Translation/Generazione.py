#
# INFERENZA
#
in_text = input("Scrivi il testo da tradurre oppure lascia vuoto per uno random:")
#region RANDOM
dataset = load_dataset("json", data_files={"train": "/content/drive/MyDrive/Colab Notebooks/dataset-involved_services-raw_logs-10000-with_labels.json"})

def trova_labels_unici(dataset):
    labels = []
    for val in dataset:

        # Estrai la stringa dal JSON
        data_string = dataset[val]["label"]

        # Rimuovi le stringhe duplicate mantenendo l'ordine dell'apparizione
        labels = list(dict.fromkeys(data_string))

    return labels

temp_list = trova_labels_unici(dataset)

# Se ha lasciato vuoto prendi log random
if(in_text == ""):
    x = np.random.randint(0,len(temp_list))
    in_text = dataset["train"][x]["label"]
    print("Ho preso random:",x,":", in_text)
#endregion
text = "translate raw_logs to label:"+in_text
#
from transformers import pipeline

translator = pipeline("translation", model="MODELLO")
translator(text)
#
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/content/MODELLO")
inputs = tokenizer(text, return_tensors="pt").input_ids
#
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("/content/MODELLO")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
#
tokenizer.decode(outputs[0], skip_special_tokens=True)
#