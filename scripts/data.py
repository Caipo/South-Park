from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
from gensim.models import KeyedVectors
import numpy as np
import fasttext
import torch


def get_pandas():
    data_path = Path(r'/home/jin/Data/cleaned.csv')
    data = list() 
    df = pd.read_csv(data_path)
    df.columns = ['Season', 'Episode', 'Character', 'Line'] 
    return df

def clean_data():
    lines = []
    with open(data_path, 'r') as file:
        
        last_line = ''
        for idx, line in enumerate(file):
            line = line.replace('\n', '')

            if idx == 0:
                continue

            if line[0].isnumeric():
                lines.append(last_line)
                last_line = line
                
            else:
                last_line += line 
        
    with open(data_dir / 'cleaned.csv', 'w') as file:
        for line in lines:
            file.write(line + '\n') 

def data_loader(batch_size, train_set = True):
    df = get_pandas()  
    df = df[(df['Character'] == 'Kyle') | (df['Character'] == 'Cartman')]   

    fasttext.FastText.eprint = lambda x: None
    lines = df[['Line', 'Character']]
    model_path = hf_hub_download(
        repo_id="simonschoe/call2vec",
        filename="model.bin"
        )
    
    train, test = train_test_split(df, random_state=42, test_size = 0.2)

    if train_set:
        df = train

    else:
        df = test

    model = fasttext.load_model(model_path)
    character  = 0 

    temp_x = list()
    temp_y = list()
    for idx, row in lines.iterrows():
        vec = model.get_sentence_vector(row['Line']) 

        if row['Character'] == 'Kyle':
            character = 1
        else:
            character = 0

        temp_x.append(vec)
        temp_y.append(character)

        if len(temp_x) == batch_size:
            x = np.array(temp_x)
            x = np.expand_dims(np.array(x), 1)
            x = torch.Tensor(x)

            y = torch.Tensor(temp_y)
            y = np.expand_dims(np.array(y), 1)
            y = torch.Tensor(y)

            yield x, y 
            temp_x = list()
            temp_y = list()
