import json
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# 불용어 설정 (영어)
stop_words = set(stopwords.words('english'))
additional_stopwords = {"game", "good", "play", "like", "fun", "best", "great", "bad", "ok", "cool"}  # 추가 불용어
stop_words.update(additional_stopwords)

def load_game_sentiment_data(file_path, target_games):
    """
    특정 게임별 긍정/부정 단어를 집계합니다. 불용어를 제외합니다.
    target_games: 비교 대상 게임 리스트 (전작과 후작)
    """
    game_data = {game: {'positive': Counter(), 'negative': Counter()} for game in target_games}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            title = record['title']
            if title in target_games:
                positive_words = [word for word in record.get('positive_words', []) if word not in stop_words]
                negative_words = [word for word in record.get('negative_words', []) if word not in stop_words]

                game_data[title]['positive'].update(positive_words)
                game_data[title]['negative'].update(negative_words)
    return game_data

def compare_word_counts(game_data, top_n=30):
    """
    전작과 후작 간의 상위 긍정/부정 단어 비교
    """
    for game, data in game_data.items():
        print(f"\n[Game: {game}]")
        print("Top Positive Words:")
        for word, count in data['positive'].most_common(top_n):
            print(f"  {word}: {count}")
        print("Top Negative Words:")
        for word, count in data['negative'].most_common(top_n):
            print(f"  {word}: {count}")

def visualize_word_comparison(game_data, top_n=30):
    """
    전작과 후작의 상위 긍정/부정 단어를 나란히 시각화합니다.
    """
    games = list(game_data.keys())
    for sentiment in ['positive', 'negative']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Top {top_n} {sentiment.capitalize()} Words Comparison", fontsize=16)

        for idx, game in enumerate(games):
            word_counts = game_data[game][sentiment].most_common(top_n)
            words, counts = zip(*word_counts) if word_counts else ([], [])
            axes[idx].barh(words, counts, color='green' if sentiment == 'positive' else 'red')
            axes[idx].invert_yaxis()
            axes[idx].set_title(game)
            axes[idx].set_xlabel('Word Count')

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

if __name__ == "__main__":
    input_file = "sentiment_reviews.jsonl"  # 입력 파일 경로
    target_games = [
        "Call of Duty\u00ae_ Modern Warfare\u00ae",      # 전작
        "Call of Duty\u00ae_ Modern Warfare\u00ae III"  # 후작
    ]

    # 데이터 로드 및 분석
    game_data = load_game_sentiment_data(input_file, target_games)

    # 상위 30개 긍정/부정 단어 비교
    compare_word_counts(game_data, top_n=30)

    # 시각화
    visualize_word_comparison(game_data, top_n=30)
