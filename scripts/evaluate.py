import matplotlib.pyplot as plt
from scripts.data import data_loader 
import csv

def plot_losses(losses):
    # Loss Plot
    xs = [x for x in range(len(losses))]
    plt.plot(xs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title("Network Loss")
    plt.show(block = False)
    plt.savefig(f'model_output/losses.png')

def evaluate_model(model):
    data = data_loader(batch_size = 1, train_set = False)
    right = 0   
    wrong = 0

    for x, y in data:
        y_hat = round(model(x).item())

        if y_hat == y:
            right += 1 
        else:
            wrong += 1 

        if right + wrong == 100:
            break
    
    print('Accuracy: ',  right / (right + wrong))

        


