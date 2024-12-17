import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import nltk

# NLTK 다운로드
nltk.download('vader_lexicon')

# 감성 분석 모델 초기화
sia = SentimentIntensityAnalyzer()  # VADER
transformer_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")  # Transformer

# 고유명사 리스트
proper_nouns = {"left", "dead", "fill", "hole", "edit", "space", "fair"}

# 고유명사 필터링 함수
def filter_proper_nouns(sentence, proper_nouns):
    filtered_words = [word for word in sentence.split() if word not in proper_nouns]
    return ' '.join(filtered_words)

# 감정 분석 및 불일치 검출 함수
def analyze_and_compare(original_sentences, processed_sentences):
    results = []
    for orig, proc in zip(original_sentences, processed_sentences):
        filtered_proc = filter_proper_nouns(proc, proper_nouns)

        # Transformer 결과
        transformer_result = transformer_analyzer(orig)[0]
        transformer_sentiment = transformer_result['label'].lower()
        transformer_score = transformer_result['score']

        # VADER 결과
        vader_score = sia.polarity_scores(filtered_proc)['compound']
        vader_sentiment = 'positive' if vader_score > 0.05 else 'negative' if vader_score < -0.05 else 'neutral'

        # 결과 저장
        results.append({
            'Original_Sentence': orig,
            'Processed_Sentence': proc,
            'Filtered_Sentence': filtered_proc,
            'Transformer_Sentiment': transformer_sentiment,
            'Transformer_Score': transformer_score,
            'VADER_Sentiment': vader_sentiment,
            'VADER_Score': vader_score,
            'Disagreement': transformer_sentiment != vader_sentiment
        })

    return pd.DataFrame(results)

# 불일치 비율 분석
def analyze_disagreement_rate(df):
    total = len(df)
    disagreements = df['Disagreement'].sum()
    agreement_rate = (total - disagreements) / total * 100
    disagreement_rate = disagreements / total * 100
    print(f"Total Sentences: {total}")
    print(f"Disagreement Count: {disagreements}")
    print(f"Agreement Rate: {agreement_rate:.2f}%")
    print(f"Disagreement Rate: {disagreement_rate:.2f}%")

# 감정 강도 시각화
def plot_sentiment_scores(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Transformer_Score'], df['VADER_Score'], c=df['Disagreement'].map({True: 'red', False: 'blue'}))
    plt.title('Sentiment Score Comparison: Transformer vs VADER')
    plt.xlabel('Transformer Score')
    plt.ylabel('VADER Score')
    plt.legend(['Disagreement', 'Agreement'])
    plt.grid()
    plt.show()

# 예제 문장
original_sentences = [
    "Heart and soul filled the space Left 4 Dead left. This game made me feel okay over the past month.",
    "It's a decent game but gets repetitive. Definitely worth the money, but only on sale.",
    "Don't bother with this game. It's a glorified L4D with horrible RNG mechanics and unbalanced gameplay.",
]

processed_sentences = [
    "heart soul fill space left dead left hole edit made feel fair past month",
    "decent game get repetitive definitely worth money would recommend getting sale",
    "dont bother glorified ld horrible rng mechanic game balance team crack based first",
]

# 실행
comparison_df = analyze_and_compare(original_sentences, processed_sentences)

# 결과 출력
print("Sentiment Analysis Comparison:")
print(comparison_df)

# 불일치 비율 분석
analyze_disagreement_rate(comparison_df)

# 감정 강도 시각화
plot_sentiment_scores(comparison_df)
