import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer #research
from tqdm.notebook import tqdm #research
from transformers import AutoTokenizer #for pretrained
from transformers import AutoModelForSequenceClassification #for pretrained
from scipy.special import softmax #for pretrained
from transformers import pipeline 


#plot style for matplotlib
plt.style.use('ggplot')
df = pd.read_csv('Reviews.csv')
#made shorter to run faster and less harsh on computer as their is 500000 reviews
df = df.head(200)

#value count gives you number of time each score occurs
ax =df['Score'].value_counts().sort_index().plot(kind = 'bar', title = 'Count of Reviews by Stars',figsize=(10,5))
#plots the graph for us to analyze, and sets the title and x label
ax.set_xlabel('Review Stars') 
ax.set_ylabel('# of reviews')
plt.show()
example = df['Text'][50]
tokens = nltk.word_tokenize(example)
#splits the sentence into words and parts of it
tokens[:10]
#gives each word a code, classifying it as a noun, adjective, etc.
tagged = nltk.pos_tag(tokens) 
#chunks the words into sentences 
#pretty print: prints each characteristic in its own line to print everything seperately and show data better
entities = nltk.chunk.ne_chunk(tagged) 

"""VADER stands for Valence Aware Dictionary and sEntiment Reasoner

This uses a bag of words approach:
1. Stop words are removed
2. each word is scored and combined to a total score."""

# scales the text based on the input based on negative, neutral, and positive, and also compound from -1 to 1 to rate it overall
sia = SentimentIntensityAnalyzer()


res = {}
#interrows: interates each row in the df, putting this in a loop, and does it until it reached the length of df, which is now df.head[500]
for i, row in tqdm(df.iterrows(), total=len(df)): 
    text = row['Text'] 
    myid = row['Id']
    #scores the ID of each row into the polarity scores with neu, neg, pos, and compound with the text
    res[myid] = sia.polarity_scores(text) 
   
#makes df better by sorting it better and easier to see with the pd dataframe layout and T flips it vertically  
vaders = pd.DataFrame(res).T 
vaders = vaders.reset_index().rename(columns={'index':'Id'})
#gives us sentiment positivity and metadata
vaders = vaders.merge(df,how='left')

#plotting vader results
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound score by Amazon Star Review') 

fig, axs = plt.subplots(1,3, figsize=(15,5))
#makes 3 different plots within one run, which is why it is a sub plot
#Each plot shows positive, neutral, and negative value with the figsize
sns.barplot(data = vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data = vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data = vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
#makes it in one picture
plt.tight_layout()

#The downside of vader model is it wont pick up on sarcastic things, which is why there is a model called roberta by huggingface that will react to it
#trained on twitter, which is similar to amazon in a few ways so it is possible to use
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, clean_up_tokenization_spaces=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def polarity_scores_roberta(example):
    #taking the text and putting it into computer language (encoded)
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    #this enchanced the quality of the review, and said it is sure is is negative while vader wasnt sure
    scores_dict = {'roberta_neg' : scores[0], 'roberta_neu' : scores[1], 'roberta_pos' : scores[2]}
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)): 
    try:
        #interates each row in the df, putting this in a loop, and does it until it reached the length of df, which is now df.head[500]
        text = row['Text'] 
        myid = row['Id'] 
        #prints text and ID
        vader_result = sia.polarity_scores(text) 
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
            #scores the ID of each row into the polarity scores with neu, neg, pos, and compound with the text
            roberta_result = polarity_scores_roberta(text)
            both = {**vader_result_rename, **roberta_result}
            res[myid] = both
    except RuntimeError:
            print(f'Broke for id {myid}')
results_df = pd.DataFrame(res).T
#basically makes df better by sorting it better and easier to see with the pd dataframe layout and T flips it vertically
results_df = results_df.reset_index().rename(columns={'index':'Id'}) 
results_df = results_df.merge(df,how='left')

sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()

print(results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0])

#negative sentiment 5 star reviews
print(results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0])
