from pathlib import Path
from sklearn.model_selection import train_test_split
from scripts.data_loader import data_loader 
from model import LSTM
from scripts.evaluate import evaluate_model, plot_losses
import numpy as np
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm
import torch

def run_model(epoch, num_layers, hidden_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.device(device)

    print('Loading Data')
    data = data_loader(batch_size = 1)

    print('Making Model')
    model = LSTM(input_size = 200, 
                 hidden_size = hidden_size, 
                 num_layers = num_layers, 
                 output_size=1
                 ).to(device)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)


    losses = list() 
    avg_losses = list()

    print('Training')
    for i in range(epoch):
        data = data_loader(batch_size = 1)
        for x, y in tqdm(data): 
            y_hat = model(x)

            loss = loss_fn(y_hat, y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if len(losses) == 1000:
                avg_losses.append(np.mean(np.array(losses)))
                plot_losses(avg_losses, epoch, hidden_size, num_layers)
                losses = list() 
                print(f" Loss: {round(avg_losses[-1], 4)}")

        print(f'epoch {i}')
        
    print('evaluating model')
    evaluate_model(model, epoch)


if __name__ == '__main__':
    hidden = [10, 30, 60, 120]
    layers = [100, 200, 300, 400]
    
    
    for l in layers:
        for h in hidden:
            run_model(1, l, h)

