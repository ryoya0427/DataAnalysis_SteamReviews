import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter

def extract_reviews_by_sentiment(file_path, target_game):
    """ 특정 게임의 긍정 및 부정 단어를 결합한 리뷰를 추출합니다. """
    positive_reviews = []
    negative_reviews = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record['title'] == target_game:
                pos_words = " ".join(record.get('positive_words', []))
                neg_words = " ".join(record.get('negative_words', []))
                if pos_words:
                    positive_reviews.append(pos_words)
                if neg_words:
                    negative_reviews.append(neg_words)
    return positive_reviews, negative_reviews

def compute_tfidf(reviews, stop_words, top_n=50):
    """ TF-IDF를 계산하고 상위 단어를 반환합니다. """
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(reviews)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    word_scores = dict(zip(feature_names, scores))
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return dict(sorted_words)

def compare_tfidf(tfidf1, tfidf2):
    """ 두 TF-IDF 결과를 비교하여 변화된 단어를 반환합니다. """
    all_words = set(tfidf1.keys()).union(set(tfidf2.keys()))
    changes = {word: tfidf2.get(word, 0) - tfidf1.get(word, 0) for word in all_words}
    return sorted(changes.items(), key=lambda x: x[1], reverse=True)

def visualize_word_changes(changes, title):
    """ 단어 변화량을 시각화합니다. """
    words, scores = zip(*changes)
    plt.figure(figsize=(10, 6))
    plt.barh(words, scores, color=['green' if s > 0 else 'red' for s in scores])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Change in TF-IDF Score")
    plt.show()

if __name__ == "__main__":
    input_file = "sentiment_reviews.jsonl"  # 데이터 파일 경로
    stop_words = list(set(stopwords.words('english')) | {'game', 'good', 'play', 'get', 'like', 'yes', 'one', 'still', 'better', 'fun','great','best','nice','mw','modern','warfare','love','amazing','ever','awesome','buy','really','cod','call','duty','callofduty','ive','never','dont'})

    # 전작과 후작 설정
    game1 = "Call of Duty\u00ae_ Modern Warfare\u00ae"
    game2 = "Call of Duty\u00ae_ Modern Warfare\u00ae III"

    # 리뷰 데이터 추출
    pos_reviews1, neg_reviews1 = extract_reviews_by_sentiment(input_file, game1)
    pos_reviews2, neg_reviews2 = extract_reviews_by_sentiment(input_file, game2)

    # TF-IDF 계산
    tfidf_pos1 = compute_tfidf(pos_reviews1, stop_words, top_n=50)
    tfidf_pos2 = compute_tfidf(pos_reviews2, stop_words, top_n=50)

    tfidf_neg1 = compute_tfidf(neg_reviews1, stop_words, top_n=50)
    tfidf_neg2 = compute_tfidf(neg_reviews2, stop_words, top_n=50)

    # 긍정 단어 변화 비교
    print("\nChanges in Positive Words:")
    pos_changes = compare_tfidf(tfidf_pos1, tfidf_pos2)
    visualize_word_changes(pos_changes, "Positive Word Changes (Modern Warfare I -> III)")

    # 부정 단어 변화 비교
    print("\nChanges in Negative Words:")
    neg_changes = compare_tfidf(tfidf_neg1, tfidf_neg2)
    visualize_word_changes(neg_changes, "Negative Word Changes (Modern Warfare I -> III)")
