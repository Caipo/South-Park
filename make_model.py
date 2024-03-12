from pathlib import Path
from sklearn.model_selection import train_test_split
from scripts.data import data_loader 
from model import LSTM
from scripts.evaluate import evaluate_model, plot_losses
import numpy as np
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm
import torch

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.device(device)

    print('Loading Data')
    data = data_loader(batch_size = 1)
    model = LSTM(input_size = 300, 
                 hidden_size = 50, 
                 num_layers = 300, 
                 output_size=1
                 )
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)



    epoch = 1
    losses = list() 
    avg_losses = list()

    print('Training')
    for i in range(epoch):
        for x, y in tqdm(data): 
            y_hat = model(x)
            loss =  loss_fn(y_hat, y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if len(losses) == 100:
                avg_losses.append(np.mean(np.array(losses)))
                print('  Average Loss: ', avg_losses[-1])
                losses = list() 

    print('evaluating model')
    evaluate_model(model)
    plot_losses(avg_losses)
    breakpoint()
