# NOHATES module

# part 1

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class NOHATES_data:
    def __to_df(self, x, y):
        d = {"text": x, "label": y}
        return pd.DataFrame(d)

    def __split_3(self, df, test_size=0.2, valid_size=0.2):
        _df = df.copy().sample(frac=1).reset_index()
        _df = _df[["text", "label"]]
        
        x = df["text"].copy()
        y = df["label"].copy()
        #split train-test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y)
        # split train-valid
        x, y = x_train, y_train
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_size, stratify=y)
        return self.__to_df(x_train, y_train), self.__to_df(x_valid, y_valid), self.__to_df(x_test, y_test)

    def __init__(self):
        tname_data = "./hsd_merge_cleaned_lowered"
        self.data = pd.read_csv(f"{tname_data}.csv")

        train, valid, self.test = self.__split_3(self.data)

# part 2

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn import preprocessing
from transformers import AutoTokenizer
import os
import drivedownload

class NOHATES_model:
    def __init__(self, test):
        self.test = test
        
        self.tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        
        self.__le = preprocessing.LabelEncoder()
        self.__le.fit(list(test['label'].values))
        # encoded_train_labels = le.transform(train_labels)
        # encoded_test_labels = le.transform(test_labels)
        
        tempname = "./beto_aug_model"
        if not os.path.exists(tempname + '_task2a_2.pt'):
            drivedownload.beto()
        self.beto = torch.load(tempname + '_task2a_2.pt', map_location=torch.device('cpu'))
        self.beto.eval()

    def encoder_generator(self, sentences, labels):
        sent_index = []
        input_ids = []
        attention_masks =[]
        
        for index,sent in enumerate(sentences):     
            sent_index.append(index)
            
            encoded_dict = self.tokenizer.encode_plus(sent,
                                                 add_special_tokens=True,
                                                 max_length=50,
                                                 padding='max_length',
                                                 truncation = True,
                                                 return_attention_mask=True,
                                                 return_tensors='pt')
            input_ids.append(encoded_dict['input_ids'])
            
            attention_masks.append(encoded_dict['attention_mask'])
        
        input_ids = torch.cat(input_ids,dim=0)
        attention_masks = torch.cat(attention_masks,dim=0)
        labels = torch.tensor(labels)
        sent_index = torch.tensor(sent_index)
        
        return sent_index,input_ids,attention_masks,labels
    
    def test_text(self, text):
        test_sentences = list([text, ])
        test_labels = list([0, ])        
        encoded_test_labels = self.__le.transform(test_labels)

        test_sent_index, test_input_ids, test_attention_masks, test_encoded_label_tensors = self.encoder_generator(test_sentences,encoded_test_labels)
        # test_dataset = TensorDataset(test_input_ids,test_attention_masks,test_encoded_label_tensors)
        # results
        b_input_ids = test_input_ids
        b_input_mask = test_attention_masks

        predictions = self.beto(b_input_ids,b_input_mask)["logits"] 
        predictions = predictions.detach().numpy()
        pred = np.argmax(predictions,axis=1).flatten()

        return pred[0]

# API module

from fastapi import FastAPI
from pydantic import BaseModel

class ModelRequest(BaseModel):
    text: str

class ModelResponse(BaseModel):
    data: int 

data = NOHATES_data()
model = NOHATES_model(data.test)
app = FastAPI()

@app.get("/")
def root():
    return "NoHateS API running..."

@app.post("/test")
def test(r: ModelRequest):
    ans = model.test_text(r.text)
    print(ans)
    return ModelResponse(data=ans)
