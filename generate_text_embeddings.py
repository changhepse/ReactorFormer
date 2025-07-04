import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import os




for size in ['gpt2']:
    model = AutoModel.from_pretrained('openai-community/gpt2')
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dirnames = ["CSTR1", "CSTR2", "CSTR3", "BR1", "BR2", "BR3", "PFR1", "PFR2", "PFR3", "ADR1", "ADR2"]

    sents = ["CSTR first order dC_dt=F/V*(C0-C)-kr*C", "CSTR second order dC_dt=F/V*(C0-C)-kr*C**2", "CSTR third order dC_dt=F/V*(C0-C)-kr*C**3",
             "BR first order dC_dt=-kr*C", "BR second order dC_dt=-kr*C**2", "BR third order dC_dt=-kr*C**3",
             "PFR first order dC_dt=-u*dC_dx-kr*C", "PFR second order dC_dt=-u*dC_dx-kr*C**2", "PFR third order dC_dt=-u*dC_dx-kr*C**3",
             "ADR first order dC_dt=-u*dC_dx-kr*C+D*d2C_dx2", "ADR second order dC_dt=-u*dC_dx-kr*C**2+D*d2C_dx2"]
    '''
    sents = ["CSTR first order", "CSTR second order", "CSTR third order",
             "BR first order", "BR second order", "BR third order",
             "PFR first order", "PFR second order", "PFR third order",
             "ADR first order", "ADR second order"]
    '''
    for i, sent in enumerate(sents):
  
        tok = tokenizer(sent, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
        input_ids = tok["input_ids"]

        embeddings = model.get_input_embeddings()(input_ids).detach().cpu().numpy()
        # embeddings = model.text_model.embeddings(input_ids).detach().cpu().numpy()

        print(embeddings.shape) # DS-QWen 1.5B (1, 32, 1536)，Llma3.2-1B (1, 32, 2048)，DS-Llma-8b (1, 32, 4096), Clip-large/roberta-base (1, 32, 768)

        
     
        train_dir = os.path.join("../datasets/mixed_data_train", dirnames[i])
        test_dir = os.path.join("../datasets/mixed_data_test", dirnames[i])
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        np.save(os.path.join(train_dir, size + "_embeddings.npy"), embeddings[0])
        np.save(os.path.join(test_dir, size + "_embeddings.npy"), embeddings[0])

        



