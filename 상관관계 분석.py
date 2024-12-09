import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
review_df = pd.read_json("left4dead2_seperate_reviews.json", lines=True)
author_df = pd.read_json("left4dead2_seperate_authors.json", lines=True)

# 리뷰 데이터와 author 데이터를 'steamid'를 기준으로 병합
merged_df = pd.merge(review_df, author_df, on='steamid', how='inner')

# 수치형 데이터만 선택
numeric_data = merged_df.select_dtypes(include=['float64', 'int64'])

# 상관관계 계산
correlation_matrix = numeric_data.corr()

# 상관관계 히트맵 그리기
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("상관관계 히트맵")
plt.show()
