import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# 한글 폰트 설정 (예: NanumGothic 폰트를 사용하는 경우)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
font_path = 'C:/Windows/Fonts/NanumGothic.ttf'  # 실제 경로로 변경 필요


# 데이터 불러오기
review_df = pd.read_json("processed_reviews.json", lines=True)
author_df = pd.read_json("left4dead2_seperate_authors.json", lines=True)




# 1. 리뷰의 긍정/부정 비율
plt.figure(figsize=(6, 4))
sns.countplot(data=review_df, x='voted_up')
plt.title("긍정/부정 리뷰 비율")
ax = sns.countplot(data=review_df, x='voted_up')
plt.xlabel("긍정 리뷰 여부")
plt.ylabel("리뷰 수")
total_reviews = len(review_df)
for p in ax.patches:
    height = p.get_height()
    percentage = height / total_reviews * 100
    ax.annotate(f'{height}개 ({percentage:.1f}%)', 
                (p.get_x() + p.get_width() / 2, height), 
                ha='center', va='bottom', fontsize=12, color='black')
plt.show()

# 2. 언어별 리뷰 수
plt.figure(figsize=(10, 6))
sns.countplot(data=review_df, x='language', order=review_df['language'].value_counts().index)
plt.title("언어별 리뷰 수")
plt.xlabel("언어")
plt.ylabel("리뷰 수")
plt.xticks(rotation=45)
plt.show()

# 3. 작성자의 평균 플레이 시간 분포
plt.figure(figsize=(8, 6))
sns.histplot(author_df['playtime_forever'], bins=30, kde=True)
plt.title("작성자의 총 플레이 시간 분포")
plt.xlabel("총 플레이 시간 (분)")
plt.ylabel("빈도")
plt.show()

# 4. 리뷰 작성 시간 시계열
review_df['timestamp_created'] = pd.to_datetime(review_df['timestamp_created'], unit='s')
review_df.set_index('timestamp_created').resample('ME').size().plot(figsize=(12, 6))
plt.title("리뷰 작성 시간 분포 (월별)")
plt.xlabel("시간")
plt.ylabel("리뷰 수")
plt.show()

# 5. 리뷰 투표 수 분포
plt.figure(figsize=(8, 6))
sns.histplot(review_df['votes_up'], bins=30, kde=True, color='blue', label='Votes Up')
sns.histplot(review_df['votes_funny'], bins=30, kde=True, color='green', label='Votes Funny')
plt.title("리뷰 투표 수 분포")
plt.xlabel("투표 수")
plt.ylabel("빈도")
plt.legend()
plt.show()

# 6. 언어별 긍정/부정 리뷰 비율
language_sentiment = review_df.groupby(['language', 'voted_up']).size().unstack(fill_value=0)
language_sentiment.columns = ['negative_reviews', 'positive_reviews']
language_sentiment['total_reviews'] = language_sentiment['positive_reviews'] + language_sentiment['negative_reviews']
language_sentiment['positive_ratio'] = language_sentiment['positive_reviews'] / language_sentiment['total_reviews'] * 100
language_sentiment['negative_ratio'] = language_sentiment['negative_reviews'] / language_sentiment['total_reviews'] * 100

# 상위 언어/국가만 선택 (리뷰가 많은 상위 10개 국가)
top_languages = language_sentiment.sort_values(by='total_reviews', ascending=False).head(10)

# 시각화 - 언어별 긍정 리뷰 비율
plt.figure(figsize=(12, 8))
sns.barplot(x=top_languages.index, y=top_languages['positive_ratio'], color='blue', label='Positive')
sns.barplot(x=top_languages.index, y=top_languages['negative_ratio'], color='red', label='Negative', bottom=top_languages['positive_ratio'])
plt.title("언어별 긍정/부정 리뷰 비율")
plt.xlabel("언어")
plt.ylabel("비율 (%)")
plt.legend(["Positive", "Negative"])
plt.xticks(rotation=45)
plt.show()

# 7. 언어/국가별 긍정 비율과 평균 플레이 시간 계산
positive_reviews = review_df[review_df['voted_up'] == True]
language_playtime = positive_reviews.merge(author_df, on='steamid')

language_playtime = language_playtime.groupby('language').agg({
    'voted_up': 'size',
    'playtime_forever': 'mean'
}).rename(columns={'voted_up': 'positive_reviews', 'playtime_forever': 'avg_playtime'})

# 시각화 - 언어/국가별 긍정 리뷰 수와 평균 플레이 시간 관계
plt.figure(figsize=(12, 8))
sns.scatterplot(data=language_playtime, x='avg_playtime', y='positive_reviews', hue=language_playtime.index, s=100, palette='viridis')
plt.title("국가별 긍정 리뷰 수와 평균 플레이 시간 관계")
plt.xlabel("평균 플레이 시간 (분)")
plt.ylabel("긍정 리뷰 수")
plt.legend(title="언어")
plt.show()

# 8. 소유한 게임 수와 작성한 리뷰 수에 따른 관계 (voted_up 제외)
plt.figure(figsize=(12, 6))
sns.scatterplot(data=author_df, x='num_games_owned', y='num_reviews', alpha=0.6)
plt.title("소유 게임 수와 작성 리뷰 수 관계")
plt.xlabel("소유한 게임 수")
plt.ylabel("작성한 리뷰 수")
plt.show()

# 9. 부정적 리뷰에서 자주 언급되는 실패 요소 (워드클라우드)
# 불용어에 추가할 단어 목록 정의
additional_stopwords = {
    "game", "valve", "play", "playing", "time", "will", "bad", "people", 
    "one", "even", "still", "now", "hour", "really", "dont", "community", 
    "want", "better", "played", "new", "actually", "every", "something", 
    "reason", "without", "make", "thing", "lot", "everyone", "Left", "Dead", 
    "enough", "mean", "many", "another", "bought", "original", "point", 
    "give", "back", "say", "review", "anymore", "content", "either", "full", 
    "great", "재밌어", "너무", "너무 재밌어", "재밌음", "게임", "진짜 재밌다", "재밌습니다", 
    "이 게임을", "잼있다", "정말", "게임입니다", "제일", "이게", "겜", "없는", "하지만", 
    "수 있다", "많이", "존나", "다시", "보면", "무조건", "근데", "왜"
}

# WordCloud의 기본 불용어 세트에 추가
stopwords = STOPWORDS.union(additional_stopwords)


# 언어가 영어이고 부정적인 리뷰만 필터링
negative_reviews = review_df[(review_df['voted_up'] == False) & (review_df['language'] == 'english')]

# 부정적 영어 리뷰의 텍스트 결합
negative_text = " ".join(review for review in negative_reviews['processed_review'])

# 워드 클라우드 생성
plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(negative_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("부정적 영어 리뷰에서 자주 언급되는 실패 요소")
plt.show()

# 언어가 영어이고 긍정적인 리뷰만 필터링
positive_reviews = review_df[(review_df['voted_up'] == True) & (review_df['language'] == 'english')]

# 부정적 영어 리뷰의 텍스트 결합
positive_text = " ".join(review for review in positive_reviews['processed_review'])

# 워드 클라우드 생성
plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(positive_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("긍정적 영어 리뷰에서 자주 언급되는 성공 요소")
plt.show()

# 긍정적 한국어 리뷰에서 자주 언급되는 요소
positive_reviews_korean = review_df[(review_df['voted_up'] == True) & (review_df['language'] == 'koreana')]
positive_text_korean = " ".join(review for review in positive_reviews_korean['processed_review'])

plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path, stopwords=stopwords).generate(positive_text_korean)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("긍정적 한국어 리뷰에서 자주 언급되는 성공 요소")
plt.show()

# 불용어 설정
custom_stopwords = set(["connection", "failed", "retrying", "server", "error"])

# 워드클라우드 생성 시 불용어 제거 적용
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, font_path=font_path).generate(positive_text)


# 부정적 한국어 리뷰에서 자주 언급되는 요소
negative_reviews_korean = review_df[(review_df['voted_up'] == False) & (review_df['language'] == 'koreana')]
negative_text_korean = " ".join(review for review in negative_reviews_korean['processed_review'])

plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path, stopwords=stopwords).generate(negative_text_korean)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("부정적 한국어 리뷰에서 자주 언급되는 실패 요소")
plt.show()



merged_df = review_df.merge(author_df[['steamid', 'playtime_forever']], on='steamid', how='left')
# 10. 플레이 시간에 따른 긍정/부정 리뷰 비율
plt.figure(figsize=(10, 6))
sns.histplot(data=merged_df, x='playtime_forever', hue='voted_up', kde=True, bins=30, palette={True: 'blue', False: 'red'})
plt.title("플레이 시간에 따른 긍정/부정 리뷰 비율")
plt.xlabel("총 플레이 시간 (분)")
plt.ylabel("빈도")
plt.legend(["긍정 리뷰", "부정 리뷰"])
plt.show()

# 11. 언어별 리뷰 길이 분포
review_df['review_length'] = review_df['review'].apply(lambda x: len(x.split()))

plt.figure(figsize=(12, 6))
sns.boxplot(data=review_df, x='language', y='review_length')
plt.title("언어별 리뷰 길이 분포")
plt.xlabel("언어")
plt.ylabel("리뷰 길이 (단어 수)")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=merged_df, x='playtime_forever', hue='voted_up', kde=True, bins=30, palette={True: 'blue', False: 'red'})
plt.xscale('log')
plt.title("플레이 시간에 따른 긍정/부정 리뷰 비율 (로그 스케일)")
plt.xlabel("총 플레이 시간 (분, 로그 스케일)")
plt.ylabel("빈도")
plt.legend(["긍정 리뷰", "부정 리뷰"])
plt.show()

# 상위 5% 이상치 제거
max_playtime = merged_df['playtime_forever'].quantile(0.95)
filtered_df = merged_df[merged_df['playtime_forever'] <= max_playtime]

plt.figure(figsize=(10, 6))
sns.histplot(data=filtered_df, x='playtime_forever', hue='voted_up', kde=True, bins=30, palette={True: 'blue', False: 'red'})
plt.title("플레이 시간에 따른 긍정/부정 리뷰 비율 (상위 5% 제거)")
plt.xlabel("총 플레이 시간 (분)")
plt.ylabel("빈도")
plt.legend(["긍정 리뷰", "부정 리뷰"])
plt.show()



# 리뷰 길이 계산 후 review_df에 추가
review_df['review_length'] = review_df['review'].apply(lambda x: len(x.split()))

# 데이터 병합 (steamid를 기준으로 병합)
merged_df = review_df.merge(author_df[['steamid', 'playtime_forever']], on='steamid', how='left')

# 플레이 시간에 따른 리뷰 길이 및 긍정/부정 리뷰 분포 시각화
plt.figure(figsize=(12, 6))
sns.scatterplot(data=merged_df, x='playtime_forever', y='review_length', hue='voted_up', alpha=0.6, palette={True: 'blue', False: 'red'})
plt.title("플레이 시간에 따른 리뷰 길이 및 긍정/부정 리뷰 분포")
plt.xlabel("총 플레이 시간 (분)")
plt.ylabel("리뷰 길이 (단어 수)")
plt.xscale('log')  # 로그 스케일 사용
plt.legend(["긍정 리뷰", "부정 리뷰"])
plt.show()


# 소유한 게임 수와 작성한 리뷰 수에 따른 긍정/부정 리뷰 비율
plt.figure(figsize=(10, 6))
sns.scatterplot(data=author_df, x='num_games_owned', y='num_reviews', hue=review_df['voted_up'], alpha=0.6, palette={True: 'blue', False: 'red'})
plt.title("소유한 게임 수와 작성한 리뷰 수에 따른 긍정/부정 리뷰 분포")
plt.xlabel("소유한 게임 수")
plt.ylabel("작성한 리뷰 수")
plt.legend(["긍정 리뷰", "부정 리뷰"])
plt.show()


# 긍정 리뷰만 필터링하여 언어별로 평균 플레이 시간 계산
positive_reviews = merged_df[merged_df['voted_up'] == True]
language_playtime = positive_reviews.groupby('language').agg({
    'playtime_forever': 'mean',
    'voted_up': 'size'
}).rename(columns={'playtime_forever': 'avg_playtime', 'voted_up': 'positive_reviews'})

# 언어별 긍정 리뷰 비율과 평균 플레이 시간 시각화
plt.figure(figsize=(12, 8))
sns.scatterplot(data=language_playtime, x='avg_playtime', y='positive_reviews', hue=language_playtime.index, s=100, palette='viridis')
plt.title("언어별 긍정 리뷰 수와 평균 플레이 시간 관계")
plt.xlabel("평균 플레이 시간 (분)")
plt.ylabel("긍정 리뷰 수")
plt.legend(title="언어")
plt.show()


# 리뷰 작성 시간 형식 변환 및 필터링 (상위 5% 제거)
merged_df['timestamp_created'] = pd.to_datetime(merged_df['timestamp_created'], unit='s')
max_playtime = merged_df['playtime_forever'].quantile(0.95)
filtered_df = merged_df[merged_df['playtime_forever'] <= max_playtime]

# 시계열 시각화
plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered_df, x='timestamp_created', y='playtime_forever', hue='voted_up', palette={True: 'blue', False: 'red'})
plt.title("시간 경과에 따른 플레이 시간 및 긍정/부정 리뷰 변화")
plt.xlabel("리뷰 작성 시간")
plt.ylabel("플레이 시간 (분)")
plt.legend(["긍정 리뷰", "부정 리뷰"])
plt.show()
# 리뷰 길이 계산
review_df['review_length'] = review_df['review'].apply(lambda x: len(x.split()))

# 언어별 긍정/부정 리뷰 길이 분포
plt.figure(figsize=(12, 6))
sns.boxplot(data=review_df, x='language', y='review_length', hue='voted_up', palette={True: 'blue', False: 'red'})
plt.title("언어별 리뷰 길이와 긍정/부정 리뷰 분포")
plt.xlabel("언어")
plt.ylabel("리뷰 길이 (단어 수)")
plt.xticks(rotation=45)
plt.show()