# Consiglio vivamente l'esecuzione su un notebook (Google colab ad esempio)

In google colab è possibile selezionare il tipo di runtime in alto a destra, selezionare una gpu (per ovvi motivi) [connetti -> cambia tipo di runtime -> T4 GPU]

Per l'esecuzione è necessario eseguire in uno o più blocchi di codice:

```python
!pip install datasets
!pip install transformers
!pip install evaluate
!pip install accelerate
!pip install sacrebleu
!pip install torch
!pip install git-lfs
!pip install Levenshtein
!pip install huggingface_hub
```

Inoltre è necessario un token con permessi di scrittura per l'hub di Hugging Face (ed un account ovviamente), verrà chiesto ad ogni esecuzione, ma basta inserirlo la prima volta e poi rilanciarlo dopo l'errore (viene processato solo dopo che il programma si stoppa per qualche motivo)