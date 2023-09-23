#
# INFERENZA
#
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

in_text = input("Scrivi il testo da tradurre oppure lascia vuoto per uno random:")
#region RANDOM
dataset = load_dataset("json", data_files={"train": "SOSTITUIRE CON IL PATH DEL DATASET"})

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

translator = pipeline("text2text-generation",model="SOSITUIRE CON IL PATH DOVE SI TROVA IL MODELLO")
translator(text)

tokenizer = AutoTokenizer.from_pretrained("SOSITUIRE CON IL PATH DOVE SI TROVA IL MODELLO")
inputs = tokenizer(text, return_tensors="pt").input_ids

model = AutoModelForSeq2SeqLM.from_pretrained("SOSITUIRE CON IL PATH DOVE SI TROVA IL MODELLO")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

tokenizer.decode(outputs[0], skip_special_tokens=True)

#region traduci lettere a parole
mappa_sostituzione_inversa = { 'A':'webservice1', 'B':'redisservice1', 'C':'mobservice1', 'D':'logservice1', 'E':'dbservice1', 'F':'redisservice2', 'G':'logservice2', 'H':'mobservice2',  'I':'dbservice2', 'J':'webservice2'}

stringa_input = tokenizer.decode(outputs[0], skip_special_tokens=True)
lista_lettere = stringa_input.split("--")

lista_parole = [mappa_sostituzione_inversa[lettera] for lettera in lista_lettere]
stringa_output = "--".join(lista_parole)
#endregion

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

MAPPATO_dataset = dataset.map(trova_servizi_unici)
#endregion

#region trova log con cui confrontare
def trova_stringa_simile(input_string, d_set):
    min_distance = 1000  # Inizializziamo con un valore elevato
    stringa_simile = None
    print("GENERATO:", input_string)
    for candidata in d_set:
        distance = Levenshtein.distance(input_string, candidata["label"], score_cutoff=min_distance)
        if distance < min_distance:
            min_distance = distance
            stringa_simile = candidata["raw_logs"]
            print("CANDIDATO:", candidata["label"])
            print("MIN DIST:", min_distance)
    return stringa_simile

# Esempio di utilizzo
stringa_simile = trova_stringa_simile(stringa_input, MAPPATO_dataset["train"])

if stringa_simile is not None:
    print(f"La stringa più simile a '{stringa_output}' è \n'{stringa_simile}'")
else:
    print("Nessuna stringa simile trovata nella lista.")
#endregion