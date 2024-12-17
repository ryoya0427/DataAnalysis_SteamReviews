import json
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def extract_combined_reviews(file_path, target_game):
    """ 특정 게임의 리뷰에서 긍정/부정 단어를 하나의 문장으로 합쳐서 반환 """
    combined_reviews = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record['title'] == target_game:
                pos_words = " ".join(record.get('positive_words', []))
                neg_words = " ".join(record.get('negative_words', []))
                # 긍정+부정 단어를 합쳐 하나의 문장으로
                combined = pos_words + " " + neg_words
                combined_reviews.append(combined.strip())
    return combined_reviews

def clean_text(text):
    # 간단한 전처리: 알파벳만 남기고 소문자 변환
    return re.sub('[^a-zA-Z]', ' ', text).lower()

if __name__ == "__main__":
    input_file = "sentiment_reviews.jsonl"
    game = "Call of Duty\u00ae_ Modern Warfare\u00ae III"  # 분석할 게임 (후작)

    # 리뷰 로드
    reviews = extract_combined_reviews(input_file, game)
    # 리뷰 전처리
    stop_words = list(set(stopwords.words('english')) | {'game', 'good', 'play', 'get', 'like', 'yes', 'one', 'still', 'better', 'fun','great','best','nice','mw','modern','warfare','love','amazing','ever','awesome','buy','really','cod','call','duty','callofduty','ive','never','dont'})
    cleaned_reviews = [clean_text(r) for r in reviews]

    # 빈 리뷰 제거
    cleaned_reviews = [r for r in cleaned_reviews if r.strip()]
    if not cleaned_reviews:
        print("No reviews found for the given game.")
        exit()

    # CountVectorizer로 단어 카운트 벡터화
    vectorizer = CountVectorizer(stop_words=stop_words, min_df=5)  # 너무 적게 등장하는 단어 제외
    dtm = vectorizer.fit_transform(cleaned_reviews)
    vocab = vectorizer.get_feature_names_out()

    # LDA 토픽 모델링
    n_topics = 5  # 원하는 토픽 수
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    def print_top_words(model, feature_names, n_top_words=10):
        for topic_idx, topic in enumerate(model.components_):
            print(f"Topic #{topic_idx}:")
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            print(" ".join(top_features))
            print()

    print_top_words(lda, vocab, n_top_words=10)

    # 예시) 이전에 전작과 후작에 대해 TF-IDF 변화량을 pos_changes, neg_changes로 계산했다고 가정
    # pos_changes = [(word, change), ...]
    # neg_changes = [(word, change), ...]
    # 여기서는 임의의 예시를 듬
    pos_changes = [("gun", 10.0), ("server", 5.0), ("dlc", -2.0)]  # 가상의 예시
    neg_changes = [("zombie", 15.0), ("trash", -10.0), ("map", 1.0)]  # 가상의 예시

    # pos_changes, neg_changes를 합쳐 dict로 만듦
    changes = dict(pos_changes + neg_changes)

    # 사용자 정의 토픽-범주 매핑 (lda 결과를 해석한 뒤 수동으로 할당)
    topic_to_category = {
        0: "gameplay",   # 예: Topic #0 -> 게임플레이 관련
        1: "service",    # 예: Topic #1 -> 서비스/매칭 관련
        2: "gameplay",   # 예: Topic #2 -> 게임플레이 관련
        3: "business",   # 예: Topic #3 -> 가격/비즈니스 관련
        4: "business"    # 예: Topic #4 -> DLC/가격 정책 관련
    }

    doc_topic_dist = lda.transform(dtm)  # shape: (n_docs, n_topics)
    dominant_topics = doc_topic_dist.argmax(axis=1)
    dominant_categories = [topic_to_category[t] for t in dominant_topics]

    category_counts = Counter(dominant_categories)
    total_docs = len(dominant_categories)

    print("=== Document Distribution by Category ===")
    for cat, cnt in category_counts.items():
        print(f"{cat}: {cnt} docs ({cnt/total_docs*100:.2f}%)")
    print()

    def get_topic_top_words(lda_model, feature_names, n_words=30):
        topic_words = {}
        for topic_idx, topic in enumerate(lda_model.components_):
            top_features_ind = topic.argsort()[:-n_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topic_words[topic_idx] = set(top_features)
        return topic_words

    topic_words = get_topic_top_words(lda, vocab, n_words=100)

    word_category_map = {}
    for word, chg in changes.items():
        assigned_cat = "other"
        for t_idx, wset in topic_words.items():
            if word in wset:
                assigned_cat = topic_to_category[t_idx]
                break
        word_category_map[word] = assigned_cat

    category_changes = {}
    for word, chg in changes.items():
        cat = word_category_map[word]
        if cat not in category_changes:
            category_changes[cat] = []
        category_changes[cat].append(chg)

    print("=== TF-IDF Changes by Category ===")
    for cat, ch_list in category_changes.items():
        avg_change = np.mean(ch_list) if ch_list else 0.0
        print(f"{cat}: avg change = {avg_change:.2f}, count={len(ch_list)}")

    cats = list(category_changes.keys())
    avg_changes = [np.mean(category_changes[c]) for c in cats]

    plt.figure(figsize=(8,5))
    bars = plt.bar(cats, avg_changes, color=['green' if x>0 else 'red' for x in avg_changes])
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Average TF-IDF Change by Category")
    plt.xlabel("Category")
    plt.ylabel("Avg TF-IDF Change")
    plt.show()
