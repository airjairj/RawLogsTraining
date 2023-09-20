#
# INFERENZA
#
in_text = input("Scrivi il testo da tradurre oppure lascia vuoto per uno random:")
#region RANDOM
dataset = load_dataset("json", data_files={"train": "SOSTITUIRE CON PATH DEL DATASET"})

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
    print("Ho preso random:",x,"\n", in_text)
    print(dataset["train"][x]["raw_logs"])
#endregion


text = "translate raw_logs to label:"+in_text
#
from transformers import pipeline

translator = pipeline("translation",model="SOSTITUIRE CON PATH DEL MODELLO")
translator(text)
#
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SOSTITUIRE CON PATH DEL MODELLO")
inputs = tokenizer(text, return_tensors="pt").input_ids
#
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("SOSTITUIRE CON PATH DEL MODELLO")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
#
tokenizer.decode(outputs[0], skip_special_tokens=True)

#region traduci lettere a parole pt1
mappa_sostituzione_inversa = { 'A':'webservice1', 'B':'redisservice1', 'C':'mobservice1', 'D':'logservice1', 'E':'dbservice1', 'F':'redisservice2', 'G':'logservice2', 'H':'mobservice2',  'I':'dbservice2', 'J':'webservice2'}

stringa_input = tokenizer.decode(outputs[0], skip_special_tokens=True)
lista_lettere = stringa_input.split("--")

lista_parole = [mappa_sostituzione_inversa[lettera] for lettera in lista_lettere]
stringa_output = "--".join(lista_parole)

print(stringa_output)
#endregion

#