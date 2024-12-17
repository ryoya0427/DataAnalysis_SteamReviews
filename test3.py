import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from wordcloud import WordCloud

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

# 감성 분석 및 불일치 문장 추출
def analyze_and_compare(original_sentences, processed_sentences):
    results = []
    for orig, proc in zip(original_sentences, processed_sentences):
        # Transformer 결과
        transformer_result = transformer_analyzer(orig)[0]
        transformer_sentiment = transformer_result['label'].lower()

        # VADER 결과
        vader_score = sia.polarity_scores(proc)['compound']
        vader_sentiment = 'positive' if vader_score > 0.05 else 'negative' if vader_score < -0.05 else 'neutral'

        # 결과 저장
        results.append({
            'Original_Sentence': orig,
            'Processed_Sentence': proc,
            'Transformer_Sentiment': transformer_sentiment,
            'VADER_Sentiment': vader_sentiment,
            'Disagreement': transformer_sentiment != vader_sentiment
        })

    return pd.DataFrame(results)

# 결과 분석
comparison_df = analyze_and_compare(original_sentences, processed_sentences)

# 불일치 문장만 추출
disagreement_df = comparison_df[comparison_df['Disagreement'] == True]

print("Disagreement Analysis:")
print(disagreement_df)

# WordCloud 생성 함수
def generate_wordcloud(sentences, title):
    text = " ".join(sentences)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=14)
    plt.show()

# 불일치 문장의 원본과 전처리된 데이터 비교
if not disagreement_df.empty:
    generate_wordcloud(disagreement_df['Original_Sentence'], "WordCloud of Original Sentences with Disagreement")
    generate_wordcloud(disagreement_df['Processed_Sentence'], "WordCloud of Processed Sentences with Disagreement")
else:
    print("No disagreements found to generate WordClouds.")
