import nltk
from transformers import pipeline
from nltk.tokenize import word_tokenize
import pandas as pd
import matplotlib.pyplot as plt

# 감성 분석 모델 (BERT 기반)
bert_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# NLTK 단어 토큰화 준비
nltk.download('punkt')

# 예시 문장 리스트
reviews = [
    "heart soul fill space left dead left hole edit made feel fair past month",
    "decent game get repetitive definitely worth money would recommend getting sale",
    "dont bother glorified ld horrible rng mechanic game balance team crack based first"
]

# 단어 단위 감성 분석 함수
def analyze_word_level_sentiment(text_list):
    results = []
    for review in text_list:
        words = word_tokenize(review)  # 단어 단위로 토큰화
        for word in words:
            if word.isalnum():  # 기호 제외, 단어만 분석
                sentiment = bert_analyzer(word)[0]
                results.append({
                    "Word": word,
                    "Sentiment": sentiment['label'],
                    "Score": sentiment['score']
                })
    return pd.DataFrame(results)

# 감성 분석 수행
word_sentiment_results = analyze_word_level_sentiment(reviews)

# 결과 출력
print("Word-Level Sentiment Analysis Results:")
print(word_sentiment_results)

# 감성 결과 시각화
def visualize_word_sentiments(df):
    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'red'])
    plt.title("Word-Level Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

# 시각화 실행
visualize_word_sentiments(word_sentiment_results)
