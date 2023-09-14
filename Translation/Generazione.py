#
# INFERENZA
#
in_text = input("Scrivi il testo da tradurre:")
text = "translate English to French:"+in_text
#
from transformers import pipeline

translator = pipeline("translation_en_to_fr", model="MODELLO")
translator(text)
#
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("MODELLO")
inputs = tokenizer(text, return_tensors="pt").input_ids
#
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("MODELLO")
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
#
tokenizer.decode(outputs[0], skip_special_tokens=True)
#