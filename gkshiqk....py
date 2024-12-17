import json
from collections import Counter

def count_word_in_sentiments(file_path, target_game, target_word):
    """
    특정 게임의 긍정 및 부정 단어 리스트에서 target_word 등장 횟수를 세는 함수.
    """
    positive_count = 0
    negative_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record['title'] == target_game:
                # 긍정 및 부정 단어 리스트에서 target_word를 카운트
                positive_count += record.get('positive_words', []).count(target_word)
                negative_count += record.get('negative_words', []).count(target_word)
    
    return positive_count, negative_count

if __name__ == "__main__":
    input_file = "sentiment_reviews.jsonl"  # 입력 파일 경로
    target_games = [
        "Call of Duty\u00ae_ Modern Warfare\u00ae",      # 전작
        "Call of Duty\u00ae_ Modern Warfare\u00ae III"  # 후작
    ]
    target_word = "campaign"  # 확인하고 싶은 단어

    for game in target_games:
        positive_count, negative_count = count_word_in_sentiments(input_file, game, target_word)
        print(f"Game: {game}")
        print(f"  Positive '{target_word}': {positive_count}")
        print(f"  Negative '{target_word}': {negative_count}")
