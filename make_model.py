from pathlib import Path
from sklearn.model_selection import train_test_split
from data import data_loader 
from model import LSTM
from evaluate import evaluate_model, plot_losses
import numpy as np
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm
import torch

if __name__ == '__main__':
    data = data_loader(batch_size = 1)

    model = LSTM(input_size = 300, 
                 hidden_size = 100, 
                 num_layers = 400, 
                 output_size=1
                 )
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    epoch = 1
    losses = list() 
    for i in range(epoch):
        for x, y in data: 
            if len(losses) % 100 == 0:
                print(', Loss :', np.mean(np.array(losses)))

            if len(losses) % 5000 == 0:
                break

            y_hat = model(x)
            breakpoint()
            loss =  loss_fn(y_hat.item(), y.item())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    plot_losses(losses)
    evaluate_model(model)
    exit()
