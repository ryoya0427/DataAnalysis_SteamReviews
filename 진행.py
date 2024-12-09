import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import torch

# 데이터 불러오기
merged_file_path = './output/merged_reviews.csv'
df = pd.read_csv(merged_file_path)

# 데이터 전처리
df['review'] = df['review'].fillna('').astype(str)  # 결측치 처리 및 문자열 형 변환
print("데이터프레임 준비 완료.")

# GPU 설정 및 감성 분석 파이프라인 로드
device = 0 if torch.cuda.is_available() else -1  # GPU 사용 가능 시 GPU로 설정
print(f"사용 중인 디바이스: {'GPU' if device == 0 else 'CPU'}")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

# 단어 단위 감성 분석을 위한 간단한 감성 사전
positive_words = ["amazing", "stunning", "immersive", "great", "fun", "excellent"]
negative_words = ["boring", "frustrating", "bugs", "laggy", "bad", "terrible"]

# 문장 단위 감성 분석 함수
def analyze_sentiment_batch(reviews, batch_size=32):
    results = []
    for i in tqdm(range(0, len(reviews), batch_size), desc="문장 단위 감성 분석 진행 중"):
        batch = reviews[i:i + batch_size]
        try:
            # 텍스트가 리스트 형식인지 확인
            batch_results = sentiment_pipeline(batch, truncation=True, max_length=512)
            results.extend([res['label'].lower() for res in batch_results])
        except Exception as e:
            print(f"배치 처리 중 오류 발생: {e}")
            results.extend(['unknown'] * len(batch))
    return results

# 단어 단위 감성 분석 함수
def analyze_word_sentiment(review):
    tokens = word_tokenize(review.lower())
    pos_count = sum(1 for word in tokens if word in positive_words)
    neg_count = sum(1 for word in tokens if word in negative_words)
    return {"positive_count": pos_count, "negative_count": neg_count}

# 감성 분석 실행
print("감성 분석 시작...")

# 문장 단위 감성 분석
df['overall_sentiment'] = analyze_sentiment_batch(df['review'].tolist(), batch_size=32)

# 단어 단위 감성 분석
print("단어 단위 감성 분석 시작...")
df['word_sentiment'] = df['review'].apply(analyze_word_sentiment)

# 단어 단위 감성 분석 결과를 분리
df['positive_word_count'] = df['word_sentiment'].apply(lambda x: x['positive_count'])
df['negative_word_count'] = df['word_sentiment'].apply(lambda x: x['negative_count'])
df = df.drop(columns=['word_sentiment'])

# 결과 확인
print("감성 분석 결과:")
print(df[['review', 'overall_sentiment', 'positive_word_count', 'negative_word_count']].head())

# 감성 라벨 분포 확인 (문장 단위)
print("문장 단위 감성 라벨 분포:")
print(df['overall_sentiment'].value_counts())

# 시각화: 문장 단위 감성 분석 결과
plt.figure(figsize=(8, 6))
sns.countplot(x='overall_sentiment', data=df, order=['positive', 'negative', 'unknown'])
plt.title('문장 단위 감성 라벨별 리뷰 수')
plt.show()

# 시각화: 단어 단위 긍정/부정 단어 빈도 분포
plt.figure(figsize=(8, 6))
sns.histplot(df['positive_word_count'], kde=False, color='green', label='Positive Words')
sns.histplot(df['negative_word_count'], kde=False, color='red', label='Negative Words')
plt.title('긍정/부정 단어 빈도 분포')
plt.legend()
plt.show()
