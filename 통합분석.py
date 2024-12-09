import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
merged_file_path = './output/merged_reviews.csv'  # 병합된 리뷰 데이터 경로
df = pd.read_csv(merged_file_path)

# 한글 폰트 설정 (Windows의 경우)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 'Malgun Gothic' 폰트 사용
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# timestamp_created를 datetime 형식으로 변환하고 'year_month' 열 생성
df['timestamp_created'] = pd.to_datetime(df['timestamp_created'], unit='s')
df['year_month'] = df['timestamp_created'].dt.to_period('M')

# 월별 각 게임의 추천/비추천 증가 수 분석
monthly_recommendation = df.groupby(['game_title', 'year_month'])['voted_up'].value_counts().unstack().fillna(0)
monthly_recommendation.columns = ['disliked', 'liked']

# 그래프: 월별 각 게임의 추천/비추천 증가 수
for game_title in df['game_title'].unique():
    game_data = monthly_recommendation.loc[game_title]
    game_data.plot(kind='bar', stacked=True, figsize=(12, 6), title=f'월별 추천/비추천 수 증가 - {game_title}')
    plt.xlabel('월')
    plt.ylabel('리뷰 수')
    plt.legend(['비추천', '추천'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 게임별 추천/비추천 분포
game_recommendation_dist = df.groupby(['game_title'])['voted_up'].value_counts(normalize=True).unstack().fillna(0)
game_recommendation_dist.columns = ['Disliked', 'Liked']

# 그래프: 게임별 추천/비추천 분포
game_recommendation_dist.plot(kind='bar', stacked=True, figsize=(12, 6), title='게임별 추천/비추천 비율')
plt.xlabel('게임')
plt.ylabel('비율')
plt.legend(['비추천', '추천'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 상관관계 분석을 위한 숫자형 데이터만 선택
numeric_columns = df.select_dtypes(include='number')
correlation_matrix = numeric_columns.corr()

# 그래프: 상관관계 히트맵
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('데이터 간의 상관관계 히트맵')
plt.tight_layout()
plt.show()

# 추천/비추천 수와 기타 속성 간의 상관관계 (예: 'votes_up', 'votes_funny' 등)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='votes_up', y='votes_funny', hue='voted_up')
plt.title('추천 여부에 따른 추천수(votes_up)와 재미있는 수(votes_funny) 간의 관계')
plt.xlabel('추천 수')
plt.ylabel('재미있는 수')
plt.tight_layout()
plt.show()

# 게임별 총 추천/비추천 수
total_recommendations = df.groupby('game_title')['voted_up'].value_counts().unstack().fillna(0)
total_recommendations.columns = ['Disliked', 'Liked']

# 그래프: 게임별 총 추천/비추천 수
total_recommendations.plot(kind='bar', stacked=True, figsize=(12, 6), title='게임별 총 추천/비추천 수')
plt.xlabel('게임')
plt.ylabel('리뷰 수')
plt.legend(['비추천', '추천'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 게임별 평균 리뷰 길이 비교
df['review_length'] = df['review'].str.len()
avg_review_length = df.groupby('game_title')['review_length'].mean()

# 그래프: 게임별 평균 리뷰 길이
avg_review_length.plot(kind='bar', figsize=(12, 6), title='게임별 평균 리뷰 길이')
plt.xlabel('게임')
plt.ylabel('평균 리뷰 길이')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 게임별 추천/비추천 비율에 따른 평균 플레이타임 비교
avg_playtime = df.groupby(['game_title', 'voted_up'])['author_playtime_forever'].mean().unstack().fillna(0)
avg_playtime.columns = ['Disliked', 'Liked']

# 그래프: 게임별 추천 여부에 따른 평균 플레이타임
avg_playtime.plot(kind='bar', figsize=(12, 6), title='게임별 추천 여부에 따른 평균 플레이타임')
plt.xlabel('게임')
plt.ylabel('평균 플레이타임 (분)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 게임별 리뷰 수 분석
review_counts = df['game_title'].value_counts()

# 그래프: 게임별 리뷰 수
review_counts.plot(kind='bar', figsize=(12, 6), title='게임별 리뷰 수')
plt.xlabel('게임')
plt.ylabel('리뷰 수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()