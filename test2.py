import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import nltk

# NLTK 감성 분석기 다운로드
nltk.download('vader_lexicon')

# 감성 분석 모델 초기화
sia = SentimentIntensityAnalyzer()  # VADER
transformer_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")  # Transformer

# 원본 문장
original_sentences = [
    "Heart and soul filled the space Left 4 Dead left. This game made me feel okay over the past month.",
    "It's a decent game but gets repetitive. Definitely worth the money, but only on sale.",
    "Don't bother with this game. It's a glorified L4D with horrible RNG mechanics and unbalanced gameplay.",
]

# 전처리된 문장
processed_sentences = [
    "heart soul fill space left dead left hole edit made feel fair past month",
    "decent game get repetitive definitely worth money would recommend getting sale",
    "dont bother glorified ld horrible rng mechanic game balance team crack based first",
]

# 감성 분석 함수
def combined_analysis(original_sentences, processed_sentences):
    results = []

    for orig, proc in zip(original_sentences, processed_sentences):
        # Transformer (원본 데이터)
        transformer_result = transformer_analyzer(orig)[0]
        transformer_score = transformer_result['score']
        transformer_sentiment = transformer_result['label'].lower()

        # VADER (전처리된 데이터)
        vader_score = sia.polarity_scores(proc)['compound']
        vader_sentiment = 'positive' if vader_score > 0.05 else 'negative' if vader_score < -0.05 else 'neutral'

        # 결과 저장
        results.append({
            'Original_Sentence': orig,
            'Processed_Sentence': proc,
            'Transformer_Score': transformer_score,
            'Transformer_Sentiment': transformer_sentiment,
            'VADER_Score': vader_score,
            'VADER_Sentiment': vader_sentiment
        })

    return pd.DataFrame(results)

# 분석 실행
combined_results = combined_analysis(original_sentences, processed_sentences)

# 결합 로직: 결과 비교
def resolve_sentiment(row):
    if row['Transformer_Sentiment'] == row['VADER_Sentiment']:
        return row['Transformer_Sentiment']  # 동일한 결과라면 해당 감정 선택
    else:
        return row['Transformer_Sentiment']  # Transformer 결과 우선

combined_results['Final_Sentiment'] = combined_results.apply(resolve_sentiment, axis=1)

# 결과 확장: 불일치 패턴 분석
combined_results['Disagreement'] = combined_results['Transformer_Sentiment'] != combined_results['VADER_Sentiment']

# 결과 출력
print(combined_results)

# 결과 CSV로 저장
combined_results.to_csv("combined_sentiment_results.csv", index=False)
print("Results saved to 'combined_sentiment_results.csv'.")

# 결과 시각화
def plot_disagreement(combined_results):
    # 불일치 여부 분포 시각화
    sns.countplot(x='Disagreement', data=combined_results)
    plt.title('Disagreement Between Transformer and VADER')
    plt.xlabel('Disagreement (True = Mismatch)')
    plt.ylabel('Count')
    plt.show()

    # Transformer와 VADER 점수 비교
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=combined_results, x='Transformer_Score', y='VADER_Score', hue='Disagreement', palette='coolwarm')
    plt.title('Transformer vs. VADER Sentiment Scores')
    plt.xlabel('Transformer Score')
    plt.ylabel('VADER Score')
    plt.legend(title='Disagreement')
    plt.show()

# 시각화 실행
plot_disagreement(combined_results)
