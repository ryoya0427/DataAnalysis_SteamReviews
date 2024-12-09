import pandas as pd
import json
import re
import nltk
import os
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# NLTK 불용어 및 기타 데이터 다운로드
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# 데이터 폴더 설정
data_folder = "./preprocessData/"  # JSON 파일이 저장된 폴더 경로
output_folder = "./output/"  # 출력 파일을 저장할 폴더 경로

# 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 카테고리별 폴더 탐색
categories = ['fps', 'rpg', 'strategy']
category_paths = {category: os.path.join(data_folder, category) for category in categories}

# NLTK 설정
stop_words_en = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 리뷰 텍스트 전처리 함수 정의
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"n't", " not", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words_en]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# 게임 시리즈 결정 함수
def determine_series(processed_titles):
    series_map = {}  # 시리즈 이름을 저장할 딕셔너리
    for title in processed_titles:
        prefix = title[:3]  # 앞 3글자 추출
        if prefix not in series_map:
            similar_titles = [t for t in processed_titles if t.startswith(prefix)]
            if len(similar_titles) > 1:
                print(f"게임 '{title}'와 유사한 게임들: {similar_titles}")
                series_name = input(f"'{title}'의 시리즈 이름을 입력하세요 (없다면 'NONE' 입력): ").strip()
                if series_name.upper() == "NONE":
                    series_map[prefix] = None
                else:
                    series_map[prefix] = series_name
            else:
                series_map[prefix] = None
    return series_map

# 모든 리뷰 데이터 처리
all_reviews = []
labels = []
merged_df_list = []

for category, path in category_paths.items():
    if not os.path.exists(path):
        print(f"카테고리 폴더 {path}가 존재하지 않습니다. 건너뜁니다.")
        continue

    json_files = [f for f in os.listdir(path) if f.endswith('.json')]
    processed_titles = [json_file.split('_reviews.json')[0] for json_file in json_files]

    # 시리즈 결정
    series_map = determine_series(processed_titles)

    for json_file in json_files:
        input_path = os.path.join(path, json_file)
        game_title = json_file.split('_reviews.json')[0]
        series = series_map[game_title[:3]]  # 시리즈 이름 가져오기

        # JSON 데이터 읽기
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                if content.strip().startswith('['):
                    data = json.loads(content)
                else:
                    for line in content.splitlines():
                        if line.strip():
                            data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSON 디코딩 오류: {json_file} - {e}")
                continue

        if not data:
            print(f"파일 {json_file}에 유효한 데이터가 없습니다. 건너뜁니다.")
            continue

        # DataFrame 변환
        df = pd.json_normalize(data)
        if 'playtime_at_review' in df.columns:
            df = df.dropna(subset=['playtime_at_review'])

        # author 데이터 분리
        author_columns = [col for col in df.columns if col.startswith('author_')]
        author_df = df[author_columns].copy()
        author_df.columns = [col.replace('author_', '') for col in author_df.columns]

        # 리뷰 데이터 처리
        if 'review' in df.columns and 'language' in df.columns:
            review_df = df[df['language'] == 'english']
            review_df['processed_review'] = review_df['review'].apply(preprocess_text)
            review_df = review_df[review_df['review'].str.len() < 10000]  # 10,000자 이상 리뷰 제거

            # 병합 데이터를 위한 필드 추가
            review_df['category'] = category
            review_df['series'] = series
            review_df['game_title'] = game_title

            # 리뷰 및 추천 여부 저장
            all_reviews.extend(review_df['processed_review'].tolist())
            labels.extend(review_df['voted_up'].tolist())
            merged_df_list.append(review_df)

# 병합된 리뷰 데이터를 스트림 방식으로 저장
if merged_df_list:
    merged_df = pd.concat(merged_df_list, ignore_index=True)
    merged_output_path_json = os.path.join(output_folder, "merged_reviews.json")

    # 스트리밍 방식으로 JSON 파일 저장
    with open(merged_output_path_json, 'w', encoding='utf-8') as f:
        for _, row in merged_df.iterrows():
            f.write(row.to_json(force_ascii=False) + '\n')  # 한 줄씩 JSON 저장
    print(f"병합된 데이터를 스트리밍 방식으로 저장했습니다: {merged_output_path_json}")

# 전처리 성능 평가 수행
def evaluate_model(reviews, labels):
    X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"모델의 정확도: {accuracy:.2f}")

# 모델 성능 평가
if all_reviews and labels:
    evaluate_model(all_reviews, labels)
