import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import pandas as pd
import re
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.tokenize import sent_tokenize

# 1. 모델 초기화
print("Initializing Models...")
# VADER
vader_analyzer = SentimentIntensityAnalyzer()
# RoBERTa
roberta_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
# DistilBERT
bert_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2. 테스트 리뷰 데이터
reviews = [
    "heart soul fill space left dead left hole edit made feel fair past month",
    "decent game get repetitive definitely worth money would recommend getting sale",
    "dont bother glorified ld horrible rng mechanic game balance team crack based first",
]

# 3. 문장 단위 분할 함수
def split_sentences(text):
    sentences = sent_tokenize(text)
    split_sentences = []
    for sentence in sentences:
        sub_sentences = re.split(r'\s+(but|and|however|although|yet)\s+', sentence, flags=re.IGNORECASE)
        split_sentences.extend([s.strip() for s in sub_sentences if s.strip()])
    return split_sentences

# 4. 모델별 감성 분석 수행 함수
def analyze_sentiment(sentences):
    results = []
    for sentence in sentences:
        # VADER
        vader_score = vader_analyzer.polarity_scores(sentence)['compound']
        vader_sentiment = 'positive' if vader_score > 0.05 else 'negative' if vader_score < -0.05 else 'neutral'

        # RoBERTa
        roberta_result = roberta_analyzer(sentence)[0]
        roberta_sentiment = roberta_result['label'].lower()
        roberta_score = roberta_result['score']

        # DistilBERT
        bert_result = bert_analyzer(sentence)[0]
        bert_sentiment = bert_result['label'].lower()
        bert_score = bert_result['score']

        # 결과 저장
        results.append({
            "Sentence": sentence,
            "VADER_Score": vader_score, "VADER_Sentiment": vader_sentiment,
            "RoBERTa_Score": roberta_score, "RoBERTa_Sentiment": roberta_sentiment,
            "DistilBERT_Score": bert_score, "DistilBERT_Sentiment": bert_sentiment
        })
    return pd.DataFrame(results)

# 5. 전체 리뷰를 문장 단위로 분할하고 분석 실행
print("Analyzing Sentiments...")
all_sentences = []
for review in reviews:
    all_sentences.extend(split_sentences(review))

# 감성 분석 실행
results_df = analyze_sentiment(all_sentences)

# 6. 결과 출력 및 저장
print("\nSentiment Analysis Results:")
print(results_df)
results_df.to_csv("sentiment_model_comparison.csv", index=False)
print("Results saved to 'sentiment_model_comparison.csv'.")
