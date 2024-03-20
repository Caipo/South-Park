# South-Park

A simple natural language processing project to classify lines of text by charters. Using LSTM and pytorch

# Model
Just a simply LSTM model made in pytorch. The hidden layer and layer sizes may varrie but the out put uses a simple sigmoid activation function

input: A torch tensor thats of shape (batch, sentnce, word vector) in this case its (1, 200, 200).
out put: 1 if the sentence is said by kyle, 0 if the sentence is said by cartman.

To find the hyper parmeters we preformed a simple grid search


# Data
https://www.kaggle.com/datasets/tovarischsukhov/southparklines

Preprocessing that added to clean up the data

scripts/clean_data.py  
  - Removeed any symbols that arnt alpha numeric.
  - remove any stop words.
  - stemming.
  - lemmanizing.
  - word list filter.
  - convert all to lower case.

scripts/make_numpy.py
  -  Loads the cleaned csv into a pandas array.
  -  Balence out the data by classes
  -  creates a 80, 20 split.
  -  remove the rows that don't belong to kyle or cartman.
  -  uses https://huggingface.co/fse/glove-twitter-200 model to vectorise each word and put it into a numpy array.
  -  padds any space between the max sentence length and the current length with zeros.
  -  saves as a numpy file.

scripts/data_loader.py
  - shuffles the list of files.
  - reads them to a numpy array.
  - converts to tensor.
  - loads the tensor onto the gpu.
  - yeilds the data to the model.


# Evaluation

I took the data set and tryed to classify 100 sentences, my result was about 71%. Concidering that I watched all the south park epidsiodes and could rember the context for a couple episoides this shall be the best result we can achive with the current data. Of course because we balenced out the data a ranodom geuss would be about 50% right. Hence the goal with this project is to achive an accuracy of 65%.

scripts/evaluate.py
  - A meathod to plot the avrage losses every 1000 iterations.
  - A meathod to save the meata data to model_output/evaluate.py




![South Park](https://www.hollywoodreporter.com/wp-content/uploads/2021/10/south-park-4.jpg?w=1296&h=730&crop=1&resize=1000%2C563)
