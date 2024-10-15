Sentiment Analysis on Amazon Reviews Overview This project performs sentiment analysis on Amazon reviews by leveraging both the VADER sentiment analysis tool from the nltk package and a RoBERTa model fine-tuned for sentiment analysis on tweets, from Hugging Face's transformers library.

REQUIREMENTS: Download https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews in order to run this program as this is the dataset being used
    You will also need 
    
    ``` pip install requirements
    ```
    

The project explores how these two models (VADER and RoBERTa) perform on Amazon reviews by comparing their sentiment classification results across a sample of 200 reviews.

Files Reviews.csv: A dataset containing Amazon product reviews, including features like Id, Score (star rating), and Text (review content). We use the first 200 rows for faster computation. The code in this project loads the dataset, processes the review text using both sentiment models, and visualizes the results. Key Libraries pandas: Data manipulation and analysis. numpy: Numerical computations. matplotlib & seaborn: Data visualization. nltk: Natural language processing, specifically VADER sentiment analysis. transformers: Pre-trained models for advanced NLP tasks. scipy: For softmax calculations. tqdm: Progress bar for loops. Sentiment Analysis Methods

VADER (Valence Aware Dictionary and sEntiment Reasoner):
  A rule-based model for general sentiment analysis. Measures the text's positive, neutral, negative, and compound sentiment. Quick and efficient but limited in its ability to detect sarcasm or complex emotions.

RoBERTa (Robustly Optimized BERT Pretraining Approach):
  A transformer-based pre-trained model that is fine-tuned for sentiment analysis on Twitter data. Can capture more nuanced sentiment (e.g., sarcasm) compared to VADER. Outputs probabilities for negative, neutral,     and positive sentiment using softmax. Steps

Data Preparation:
  1. Load the dataset and visualize the distribution of review scores (star ratings).
     
  2. Tokenize the review text using NLTK.

VADER Sentiment Analysis:

  3. Apply VADER to each review and store its sentiment scores (negative, neutral, positive, compound).

RoBERTa Sentiment Analysis:

  4. Apply the RoBERTa model to each review and store its sentiment probabilities (negative, neutral, positive). Comparison & Visualization:

  5. Merge the VADER and RoBERTa results into a single dataframe.

  6. Visualize the sentiment scores (positive, neutral, negative) for different review star ratings using bar plots.

  7. Pairwise plot of VADER and RoBERTa sentiment scores to show their relationship across all reviews. Example Queries:

8. Extract examples where there is a mismatch between the review's star rating and the sentiment model’s classification, e.g., negative sentiment on 5-star reviews. Results Visualizations help explore how both models interpret reviews with different star ratings. We can query specific reviews with sentiment discrepancies (e.g., a 5-star review that has a highly negative sentiment according to RoBERTa). Usage

9. Install the required packages: pip install -r requirements.txt

10. Run the code:

  Execute the Python script to load the dataset, perform sentiment analysis, and generate visualizations. Modify the dataset:

  If you want to analyze more reviews, adjust the df.head(200) line to include more rows, but be mindful of computation time. Dependencies Ensure the following libraries are installed before running the project:

  Acknowledgments This project uses the VADER sentiment analysis model from NLTK. RoBERTa is powered by the transformers library from Hugging Face. Dataset: Amazon reviews on Kaggle.com
