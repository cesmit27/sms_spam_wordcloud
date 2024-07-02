from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
from collections import Counter

def plot_wordclouds(df):
    ham_messages = ' '.join(df[df['type'] == 'ham']['text'])
    spam_messages = ' '.join(df[df['type'] == 'spam']['text'])   
    
    ham_mask=np.array(Image.open('ham_word.png'))
    #inverted_ham_mask = np.invert(ham_mask)
    spam_mask=np.array(Image.open('spam_word.png'))
    #inverted_spam_mask = np.invert(spam_mask)
    wordcloud_ham = WordCloud(width=1920, height=1080, background_color='white', mask=ham_mask).generate(ham_messages)
    wordcloud_spam = WordCloud(width=1920, height=1080, background_color='white',mask=spam_mask).generate(spam_messages)
    
    plt.figure(figsize=(12, 6), dpi=100)
    
    #Ham
    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud_ham, interpolation='bilinear')
    plt.axis('off')
    
    #Spam
    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud_spam, interpolation='bilinear')
    plt.axis('off')
    
    plt.savefig('word_cloud.png', dpi=1000)
    plt.show()

def plot_word_frequencies(df):
    all_text = ' '.join(df['text'])
    words = all_text.split() 
    word_counts = Counter(words)

    word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['Count'])
    word_counts_df = word_counts_df.sort_values(by='Count', ascending=False).head(20)
    
    #Top 20 word frequencies
    plt.figure(figsize=(10, 6))
    sns.barplot(x=word_counts_df.index, y=word_counts_df['Count'], palette='plasma', hue=word_counts_df.index, legend=False)
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 Word Frequencies')
    plt.savefig('word_frequency.png', dpi=1000)
    plt.show()

df = pd.read_csv('sms_spam.csv')
plot_wordclouds(df)
plot_word_frequencies(df)
