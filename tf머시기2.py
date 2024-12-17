import json
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_combined_reviews(file_path, target_game):
    """ 특정 게임의 리뷰에서 긍정/부정 단어를 합쳐 반환 """
    combined_reviews = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record['title'] == target_game:
                pos_words = " ".join(record.get('positive_words', []))
                neg_words = " ".join(record.get('negative_words', []))
                combined_reviews.append((pos_words + " " + neg_words).strip())
    return combined_reviews

def clean_text(text):
    """ 알파벳만 남기고 소문자 변환 """
    return re.sub('[^a-zA-Z]', ' ', text).lower()

def compute_tfidf(reviews, stop_words):
    """ TF-IDF 계산 """
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(reviews)
    feature_names = vectorizer.get_feature_names_out()
    scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    return dict(zip(feature_names, scores))

def perform_lda(reviews, stop_words, n_topics=5):
    """ LDA 토픽 모델링 수행 """
    vectorizer = CountVectorizer(stop_words=stop_words, min_df=5)
    dtm = vectorizer.fit_transform(reviews)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    return lda, vectorizer.get_feature_names_out()

def get_topic_words(model, feature_names, n_top_words=10):
    """ LDA 결과에서 토픽별 상위 단어 반환 """
    topic_words = {}
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topic_words[topic_idx] = set(top_features)
    return topic_words

if __name__ == "__main__":
    input_file = "sentiment_reviews.jsonl"
    games = [
        "Call of Duty\u00ae_ Modern Warfare\u00ae",      # 전작
        "Call of Duty\u00ae_ Modern Warfare\u00ae III"   # 후작
    ]

    stop_words = list(set(stopwords.words('english')) | 
                  {'game', 'good', 'play', 'get', 'like', 'yes', 'one', 'still', 
                   'better', 'fun', 'great', 'best', 'nice', 'mw', 'modern', 
                   'warfare', 'love', 'amazing', 'ever', 'awesome', 'buy', 
                   'really', 'cod', 'call', 'duty', 'callofduty', 'ive', 
                   'never', 'dont','much','worst','bad','make','need','also',
                   'im','ive','made','would','back','full','review','way','fuck',
                   'worth','mp','mwiii','ok','lvl','mwii','worse','crap','butt','new',
                   'feel','first','bought','got','sbmm','every','even','cant','ops',
                   'pretty','year','old','last','actually','since','know','lot','day',
                   'hour','hm','overall','want','thing','use','doo','lol'})

    # 리뷰 추출 및 LDA 수행
    series_reviews = {}
    lda_results, topic_words_results = {}, {}
    for game in games:
        reviews = extract_combined_reviews(input_file, game)
        cleaned_reviews = [clean_text(r) for r in reviews if r.strip()]
        series_reviews[game] = cleaned_reviews
        lda, feature_names = perform_lda(cleaned_reviews, stop_words)
        lda_results[game] = lda
        topic_words_results[game] = get_topic_words(lda, feature_names, n_top_words=20)

    # TF-IDF 변화량 계산
    tfidf_results = {game: compute_tfidf(reviews, stop_words) for game, reviews in series_reviews.items()}
    game1, game2 = games
    changes = {word: tfidf_results[game2].get(word, 0) - tfidf_results[game1].get(word, 0) 
               for word in set(tfidf_results[game1]) | set(tfidf_results[game2])}

    # 사용자 정의 토픽-범주 매핑
    topic_to_category = {
        0: "gameplay",
        1: "service",
        2: "gameplay",
        3: "business",
        4: "business"
    }

    # 변화량을 후작 토픽에 매핑
    topic_changes = defaultdict(list)
    for word, change in changes.items():
        topic_idx = None
        for idx, words in topic_words_results[game2].items():
            if word in words:
                topic_idx = idx
                break
        if topic_idx is not None:
            topic_changes[topic_idx].append((word, change))

    # 토픽별 시각화
    for topic_idx, changes_list in topic_changes.items():
        changes_list.sort(key=lambda x: x[1], reverse=True)
        improved = changes_list[:10]  # 개선된 상위 5단어
        declined = changes_list[-10:]  # 악화된 하위 5단어

        words = [w for w, _ in improved + declined]
        scores = [s for _, s in improved + declined]
        colors = ['green' if s > 0 else 'red' for s in scores]

        plt.figure(figsize=(10, 6))
        plt.barh(words, scores, color=colors)
        plt.axvline(0, color='black', linewidth=1)
        plt.title(f"Topic #{topic_idx} ({topic_to_category.get(topic_idx, 'other')}) - Word Changes")
        plt.xlabel("Change in TF-IDF Score")
        plt.ylabel("Words")
        plt.gca().invert_yaxis()
        plt.show()
