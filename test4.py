import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from wordcloud import WordCloud
from collections import Counter
import nltk

# NLTK 다운로드
nltk.download('vader_lexicon')

# 고유명사 리스트
proper_nouns = {"left", "dead", "fill", "hole", "edit", "space", "fair"}

# 감성 분석 모델 초기화
sia = SentimentIntensityAnalyzer()  # VADER
transformer_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")  # Transformer

# 고유명사 필터링 함수
def filter_proper_nouns(sentence, proper_nouns):
    filtered_words = [word for word in sentence.split() if word not in proper_nouns]
    return ' '.join(filtered_words)

# 감성 분석 및 불일치 문장 추출
def analyze_and_compare(original_sentences, processed_sentences):
    results = []
    for orig, proc in zip(original_sentences, processed_sentences):
        # 고유명사 필터링
        filtered_proc = filter_proper_nouns(proc, proper_nouns)

        # Transformer 결과
        transformer_result = transformer_analyzer(orig)[0]
        transformer_sentiment = transformer_result['label'].lower()

        # VADER 결과 (필터링된 문장)
        vader_score = sia.polarity_scores(filtered_proc)['compound']
        vader_sentiment = 'positive' if vader_score > 0.05 else 'negative' if vader_score < -0.05 else 'neutral'

        # 결과 저장
        results.append({
            'Original_Sentence': orig,
            'Processed_Sentence': proc,
            'Filtered_Sentence': filtered_proc,
            'Transformer_Sentiment': transformer_sentiment,
            'VADER_Sentiment': vader_sentiment,
            'Disagreement': transformer_sentiment != vader_sentiment
        })

    return pd.DataFrame(results)

# 실행
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

comparison_df = analyze_and_compare(original_sentences, processed_sentences)

# 불일치 문장 출력
disagreement_df = comparison_df[comparison_df['Disagreement'] == True]
print("Disagreement Analysis:")
print(disagreement_df)

# 고유명사 필터링 후 WordCloud
if not disagreement_df.empty:
    filtered_sentences = disagreement_df['Filtered_Sentence']
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(filtered_sentences))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("WordCloud of Filtered Disagreement Sentences")
    plt.show()
else:
    print("No disagreements found for further analysis.")
