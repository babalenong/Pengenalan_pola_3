from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from pathlib import Path
import numpy as np
import nltk
import pandas as pd
import numbers
from sklearn.feature_extraction.text import CountVectorizer


#Fetch Data
path = Path(__file__).parents[0]
inputFile = str(path) + "\\data.csv"
df = pd.read_csv(inputFile, names=['inputs','predict'], header=0)

#Create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
tokenizer = nltk.RegexpTokenizer(r'\w+')

#Normalization

df.columns = df.columns.str.strip() 
to_stem = lambda x: stemmer.stem(x)
df['inputs'] = df['inputs'].str.lower()
df['predict'] = df['predict'].str.lower()

df['inputs'] = df['inputs'].apply(to_stem)

#Create list of "BoW" vector and append them on dataframe ~ Tokenization included
vectorizer = CountVectorizer()
transformed_vector = np.array(vectorizer.fit_transform(df['inputs']).toarray())  

words_length = len(transformed_vector[0])

for words in range(words_length):
    df.insert(2+words, 'word_{}'.format(words+1), transformed_vector[:,words])

#Split train and test
mask = df['predict'].isnull()
df_test = df[mask]
df_train = df[~mask]

#Define euclidian distance -

def calculate_distance(x, y):
    return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

#Calculate euclidian distance on test (per row) against all of train's data.

df_test.insert(2,'distance', 0.0)
for test_index, test_row in df_test.iterrows():
    distance = []
    vector_test = np.array(test_row[3:].tolist())
    for train_index, train_row in df_train.iterrows():
        vector_train = np.array(train_row[2:].tolist())
        dist = calculate_distance(vector_train, vector_test)
        distance.append([train_row['predict'], dist])
    distance = np.array(distance)
    predicted = distance[np.argmin(distance[:, 1])]
    df_test.at[test_index, 'distance'] = predicted[1]
    df_test.at[test_index, 'predict'] = predicted[0]

df_test.to_csv("out.csv")
        
