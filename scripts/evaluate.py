import matplotlib.pyplot as plt
from scripts.data_loader import data_loader 
import numpy as np
import csv

def plot_losses(losses, epoch, hidden_size, num_layers):
    plt.clf()
    # Loss Plot
    xs = [ round(x / 10, 2) for x in range(len(losses))]

    if len(losses) > 11:

        m,b = np.polyfit(xs[-10:], losses[-10:], 1)
        plt.plot(xs, m*np.array(xs) + b , color = 'red', label = 'best fit for last 10')

    plt.plot(xs, losses, label = 'Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.title("Network Loss")
    plt.show(block = False)
    plt.savefig(f'model_output/{num_layers}-{hidden_size}-losses.png')

def evaluate_model(model, epoch):
    data = data_loader(batch_size = 1, train_set = False)
    right = 0   
    wrong = 0
    
    y_hats = list()
    for x, y in data:
        y = y.item()
        y_hat = round(model(x).item())

        y_hats.append([y_hat, y])
        if y_hat == y:
            right += 1 
        else:
            wrong += 1 

        if right + wrong == 100:
            break

    accuracy = right / (right + wrong)
    layers = model.lstm.num_layers 
    hidden_size = model.lstm.hidden_size
    print(accuracy)
    with open('model_output/meta.csv', 'a') as file:
        file.write(f'{epoch}, {layers}, {hidden_size}, {accuracy} \n')
        


