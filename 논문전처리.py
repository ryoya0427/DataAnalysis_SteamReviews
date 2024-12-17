import pandas as pd
import json
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# NLTK 불용어 및 표제어 추출기 다운로드
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# NLTK 설정
stop_words_en = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 데이터 폴더 및 출력 폴더 설정
data_folder = "./preprocessData/nonmun/"  # JSON 파일이 저장된 폴더 경로
output_folder = "./output/"  # 전처리된 데이터 저장 폴더
os.makedirs(output_folder, exist_ok=True)

# 시리즈 이름 결정 함수
def determine_series(processed_titles):
    series_map = {}
    for title in processed_titles:
        prefix = title[:3]  # 앞 3글자 추출
        if prefix not in series_map:
            similar_titles = [t for t in processed_titles if t.startswith(prefix)]
            if len(similar_titles) > 1:
                print(f"게임 '{title}'와 유사한 게임들: {similar_titles}")
                series_name = input(f"'{title}'의 시리즈 이름을 입력하세요 (없다면 'NONE' 입력): ").strip()
                if series_name.upper() == "NONE":
                    series_map[prefix] = "Unknown"
                else:
                    series_map[prefix] = series_name
            else:
                series_map[prefix] = title  # 기본적으로 title 자체를 시리즈 이름으로 사용
    return series_map

# 리뷰 텍스트 전처리 함수 정의
def preprocess_text(text):
    # 고유명사 보호
    game_titles = ["Left 4 Dead", "Back 4 Blood", "Call of Duty", "Assassin's Creed","Dark Souls", "Battlefield", "Cyberpunk 2077", "L4D2"]
    for title in game_titles:
        text = text.replace(title, title.replace(" ", "_"))
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words_en]
    return ' '.join(tokens)

# 리뷰 데이터 전처리 함수
def process_reviews(json_files, output_path, categories):
    all_reviews = []

    # 카테고리별 폴더 탐색
    category_paths = {category: os.path.join(data_folder, category) for category in categories}

    for category, path in category_paths.items():
        if not os.path.exists(path):
            print(f"카테고리 폴더 {path}가 존재하지 않습니다. 건너뜁니다.")
            continue

        json_files = [f for f in os.listdir(path) if f.endswith('.json')]
        processed_titles = [json_file.split('_reviews.json')[0] for json_file in json_files]
        series_map = determine_series(processed_titles)

        for json_file in json_files:
            title = json_file.replace("_reviews.json", "")  # 게임명 추출
            series = series_map.get(title[:3], "Unknown")  # 시리즈 이름 결정, 기본값 추가
            input_path = os.path.join(path, json_file)

            try:
                # JSON 데이터 로드
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # DataFrame 변환
                df = pd.json_normalize(data)

                # author 내부 데이터 분리
                if 'author' in df.columns:
                    author_df = pd.json_normalize(df['author'])
                    author_df.columns = [f"author_{col}" for col in author_df.columns]
                    df = pd.concat([df.drop(columns=['author']), author_df], axis=1)

                # 필요한 필드 선택 및 결측값 처리
                fields = [
                    "review", "language", "voted_up", "timestamp_created",
                    "playtime_at_review", "votes_up", "votes_funny",
                    "steam_purchase", "written_during_early_access", "author_steamid"
                ]

                for field in fields:
                    if field not in df.columns:
                        df[field] = 0 if field in ["playtime_at_review", "votes_up", "votes_funny"] else None

                df = df[fields]

                # 영어 리뷰만 필터링
                df = df[df['language'] == 'english']

                # 리뷰 텍스트 전처리
                df['processed_review'] = df['review'].apply(preprocess_text)

                # 게임명(title), 시리즈(series), 카테고리(category) 추가
                df['title'] = title
                df['series'] = series
                df['category'] = category

                # 필드 정리
                processed_df = df[
                    ['title', 'series', 'category', 'processed_review', 'voted_up',
                     'timestamp_created', 'playtime_at_review', 'votes_up', 
                     'votes_funny', 'steam_purchase']
                ]
                all_reviews.append(processed_df)

            except Exception as e:
                print(f"파일 처리 중 오류 발생: {json_file} - {e}")

    # 모든 리뷰 병합
    if all_reviews:
        merged_df = pd.concat(all_reviews, ignore_index=True)
        # JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in merged_df.iterrows():
                f.write(row.to_json(force_ascii=False) + '\n')  # 한 줄씩 JSON 저장
        print(f"전처리된 데이터를 저장했습니다: {output_path}")
    else:
        print("처리할 데이터가 없습니다.")

# JSON 파일 목록 가져오기
categories = ['fps']  # 카테고리 정의
output_path = f"{output_folder}/reviews.json"

# 리뷰 데이터 전처리 실행
process_reviews([], output_path, categories)
