import nltk 
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


def plot_target_freq():
    target_col = df["vin"]
    target_col.plot(kind="hist")
    plt.show()

def clean_text(text):

    # remove url
    text = re.sub(r'https\S+', "", text)
    # remove numbers - TODO: Try with an without number
    text = re.sub(r'\d+', '', text)
    # tokenize each text
    word_tokens = word_tokenize(text)

    # remove special chars
    clean_text = []
    for word in word_tokens:
        clean_text.append("".join([e for e in word if e.isalnum()]))


    # remove stop words and lower
    text_wo_stopwrds = [w.lower() for w in clean_text if not w in stopwords.words('english')]

    # do stemming
    stemmer = SnowballStemmer("english")
    stemmed_text = [stemmer.stem(w) for w in text_wo_stopwrds]

    return " ".join(" ".join(stemmed_text).split())



file_path = "./data/vin_w_target.csv"
df = pd.read_csv(file_path)
df.drop(columns=df.columns[0], axis=1, inplace=True)

raw_data = {}
clean_df = pd.DataFrame({'clean_text': df["text"].apply(lambda s: clean_text(s)), 'vin': df["vin"]})
clean_df.to_csv("ml/data/dataclean_df_300.csv")