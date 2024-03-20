from pathlib import Path
from glob import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import download
import re

syms_to_rmv = ['?', '!', '.', '(', ')', '...', '\n', 
               'ñ', '¿', '%', 'ö', 'ä', '&', '…',  '’', 
                '$', '#', '/', '  ' '+', 'ü',  'й', 'è',  'ú', '\xa0', 
                ':', '5', "'", ';', '—',  '*', ',', '"',  '-', '\''
                ]

#download('wordnet')

ps = PorterStemmer()
lem = WordNetLemmatizer()

stop_words = [
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 
    'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 
    'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'can\'t', 'cannot', 
    'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 
    'doing', 'don\'t', 'down', 'during', 'each', 
    'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 
    'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 
    'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 
    'him', 'himself', 'his', 'how', 'how\'s', 'i', 
    'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 
    'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 
    'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 
    'not', 'of', 'off', 'for', 'on', 'once', 'only', 'or', 
    'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 
    'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 
    'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 
    'that', 'that\'s', 'the', 'their', 'theirs', 
    'them', 'themselves', 'then', 'there', 'there\'s', 'these', 
    'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 
    'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 
    'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 
    'when', 'when\'s', 'where', 'where\'s', 'which', 
    'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 
    'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 
    'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself',
    'yourselves'
]

def remove_muti_space(sentence):
    for i in ['    ', '   ', '  ']:
        sentence = sentence.replace(i, ' ')
    return sentence.strip()

def filter_stop_words(sentence):
    for i in list(set(stop_words) & set(sentence.split(' '))):
        sentence = sentence.replace(' ' + i + ' ', ' ')
    
    sentence = remove_muti_space(sentence)
    return sentence

def stemming(sentence):
    for i in sentence.split(" "):
        sentence.replace(i, ps.stem(i))

    sentence = remove_muti_space(sentence)
    return sentence

def lemmanizing(sentence):
    for i in sentence.split(" "):
        sentence.replace(i, lem.lemmatize(i))

    sentence = remove_muti_space(sentence)
    return sentence

def fileter_symbols(sentence):
    for i in syms_to_rmv:
        sentence = sentence.replace(i, '')

    sentence = remove_muti_space(sentence)
    return sentence

def filter_by_word_list(sentence):
    for i in sentence.split(' '):
        if i not in english_words:
            sentence = sentence.replace(i, ' ')

    sentence = remove_muti_space(sentence)
    return sentence 

def clean_data():
    lines = []

    data_path = Path(r'/home/jin/Data/All-seasons.csv')
    with open(data_path, 'r') as file:
        
        last_line = ''

        for idx, line in enumerate(file):
            
            if 'Season,Episode,Character,Line' in line:
                continue

            if line[0].isnumeric():
                lines.append(last_line)
                last_line = line
           
            else:
                last_line += line

    for indx, line in enumerate(lines): 
        if line == '':
            continue
        
        line = line.replace('\n', '')
        
        try:
            start = line.index('"')
            end = line.rfind('"')

        except:
            continue
        

        sentence = line[start + 1:end].strip()

        sentence = sentence.lower()
        sentence = filter_stop_words(sentence)
        sentence = fileter_symbols(sentence)
        #sentence = filter_by_word_list(sentence)
        #sentence = stemming(sentence)
        sentence = lemmanizing(sentence)

         
        line = line[:start] + '"'  + sentence + '"' 
        lines[indx] = line

    with open(r'/home/jin/Data/cleaned.csv', 'w') as file:
        for line in lines:
                pattern = r'^\d+,\d+,[a-z A-Z]+,\".+"'
                if re.match(pattern, line):
                    file.write(line + '\n')



if __name__ == '__main__':
    english_words = list()

    with open(r'/home/jin/Data/words.txt', 'r') as file:
        for i in file:
            english_words.append(i.lower().replace('\n', ''))

    english_words = set(english_words)

    clean_data()
    #make_numpy() 
    #make_numpy(train_set = False)
