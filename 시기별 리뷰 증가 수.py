import pandas as pd
import matplotlib.pyplot as plt
# 한글 폰트 설정 (NanumGothic 사용 예시)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 (파일 경로와 형식에 맞게 조정)
review_df = pd.read_json("left4dead2_seperate_reviews.json", lines=True)

# 타임스탬프를 datetime 형식으로 변환
review_df['timestamp_created'] = pd.to_datetime(review_df['timestamp_created'], unit='s')

# 긍정과 부정 리뷰로 데이터 분리
positive_reviews = review_df[review_df['voted_up'] == True]
negative_reviews = review_df[review_df['voted_up'] == False]

# 월별 전체 리뷰 수, 긍정 리뷰 수, 부정 리뷰 수 집계
monthly_reviews = review_df.set_index('timestamp_created').resample('M').size()
monthly_positive = positive_reviews.set_index('timestamp_created').resample('M').size()
monthly_negative = negative_reviews.set_index('timestamp_created').resample('M').size()

# 그래프 시각화
plt.figure(figsize=(12, 6))
plt.plot(monthly_reviews, label="전체 리뷰", color="gray", linewidth=1.5)
plt.plot(monthly_positive, label="긍정 리뷰", color="blue", linewidth=1)
plt.plot(monthly_negative, label="부정 리뷰", color="red", linewidth=1)
plt.title("시기별 리뷰 작성 수 및 긍정/부정 리뷰 변화")
plt.xlabel("시간")
plt.ylabel("리뷰 수")
plt.legend()
plt.show()
