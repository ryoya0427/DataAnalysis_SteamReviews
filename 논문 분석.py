import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# NLTK 감성 분석 도구 다운로드
nltk.download('vader_lexicon')

# 데이터 로드
input_path = "./output/reviews.json"  # 전처리된 데이터 경로
merged_df = pd.read_json(input_path, lines=True)

# 날짜 변환
merged_df['date'] = pd.to_datetime(merged_df['timestamp_created'], unit='s')

# NLTK 감성 분석기 초기화
sia = SentimentIntensityAnalyzer()

# 감성 점수 계산
merged_df['sentiment_score'] = merged_df['processed_review'].apply(lambda x: sia.polarity_scores(x)['compound'])
merged_df['sentiment'] = merged_df['sentiment_score'].apply(
    lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral'
)

# 1. 리뷰 수 분포
def plot_review_distribution():
    series_review_counts = merged_df['series'].value_counts()
    series_review_counts.plot(kind='bar', title='Number of Reviews by Series', color='skyblue')
    plt.ylabel('Number of Reviews')
    plt.show()

# 2. 긍정/부정 리뷰 비율
def plot_sentiment_ratio():
    sentiment_counts = merged_df['sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', title='Sentiment Ratio (Positive vs Negative)', color=['green', 'red', 'grey'])
    plt.ylabel('Number of Reviews')
    plt.show()

# 3. 워드클라우드 생성
def generate_wordcloud(sentiment_filter, title):
    filtered_reviews = " ".join(merged_df[merged_df['sentiment'] == sentiment_filter]['processed_review'])
    wordcloud = WordCloud(background_color='white', max_words=100).generate(filtered_reviews)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# 4. 시간 흐름에 따른 리뷰 감성 변화
def plot_time_sentiment():
    time_sentiment = merged_df.groupby(merged_df['date'].dt.to_period('M'))['voted_up'].mean()
    time_sentiment.plot(kind='line', title='Positive Review Trend Over Time', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Positive Review Ratio')
    plt.show()

# 5. 업데이트 전후 분석
def compare_pre_post_update():
    update_date = pd.Timestamp('2023-01-01')  # 업데이트 날짜
    pre_update_reviews = merged_df[merged_df['date'] < update_date]
    post_update_reviews = merged_df[merged_df['date'] >= update_date]
    pre_update_sentiment = pre_update_reviews['voted_up'].mean()
    post_update_sentiment = post_update_reviews['voted_up'].mean()
    print(f"업데이트 전 긍정 리뷰 비율: {pre_update_sentiment:.2%}")
    print(f"업데이트 후 긍정 리뷰 비율: {post_update_sentiment:.2%}")

# 6. 카테고리별 긍정 리뷰 비율
def plot_category_sentiment():
    category_sentiment = merged_df.groupby('category')['voted_up'].mean()
    category_sentiment.plot(kind='bar', title='Positive Review Ratio by Category', color='purple')
    plt.ylabel('Positive Review Ratio')
    plt.show()

# 7. 플레이 시간과 리뷰 감정의 관계
def plot_playtime_sentiment():
    merged_df['playtime_group'] = pd.cut(
        merged_df['playtime_at_review'], 
        bins=[0, 10, 50, 100, 500, 1000], 
        labels=['0-10h', '10-50h', '50-100h', '100-500h', '500+h']
    )
    playtime_sentiment = merged_df.groupby('playtime_group')['voted_up'].mean()
    playtime_sentiment.plot(kind='bar', title='Positive Review Ratio by Playtime', color='orange')
    plt.ylabel('Positive Review Ratio')
    plt.show()

# 실행
if __name__ == "__main__":
    print("1. 리뷰 수 분포")
    plot_review_distribution()
    
    print("2. 긍정/부정 리뷰 비율")
    plot_sentiment_ratio()
    
    print("3. 긍정적 리뷰 워드클라우드")
    generate_wordcloud('positive', 'Positive Reviews WordCloud')
    
    print("4. 부정적 리뷰 워드클라우드")
    generate_wordcloud('negative', 'Negative Reviews WordCloud')
    
    print("5. 시간 흐름에 따른 리뷰 감성 변화")
    plot_time_sentiment()
    
    print("6. 업데이트 전후 분석")
    compare_pre_post_update()
    
    print("7. 카테고리별 긍정 리뷰 비율")
    plot_category_sentiment()
    
    print("8. 플레이 시간과 리뷰 감정의 관계")
    plot_playtime_sentiment()
