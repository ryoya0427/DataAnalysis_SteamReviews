import json
from tqdm import tqdm
import torch
from transformers import pipeline

# CPU 스레드 수 조정 (필요한 경우)
torch.set_num_threads(8)

# 감정 분석 파이프라인 초기화 (CPU 사용)
distilbert_pipeline = pipeline("sentiment-analysis", 
                               model="distilbert-base-uncased-finetuned-sst-2-english",
                               device=-1)

def analyze_sentiment(words, batch_size=256):
    """
    배치 처리를 통해 감정 분석을 수행하는 함수.
    """
    positive_words = []
    negative_words = []
    
    # 배치 단위로 모델에 전달
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        results = distilbert_pipeline(batch, truncation=True)
        for word, res in zip(batch, results):
            if res['label'] == 'POSITIVE':
                positive_words.append(word)
            elif res['label'] == 'NEGATIVE':
                negative_words.append(word)

    return positive_words, negative_words

def batch_analyze_reviews(reviews, batch_size=256):
    """
    여러 리뷰를 한 번에 분석하는 함수.
    reviews: [{"processed_review": ..., "recommendationid": ...}, ...]
    """
    for rev in reviews:
        words = rev.get('processed_review', "").split()
        # 감정 분석 수행 (캐싱 제거)
        positive_words, negative_words = analyze_sentiment(words, batch_size=batch_size)
        rev['positive_words'] = positive_words
        rev['negative_words'] = negative_words

    return reviews

def process_reviews_in_batches(input_file, output_file, chunk_size=5000, batch_size=256):
    """
    대규모 리뷰 데이터를 부분(chunk) 단위로 처리하며,
    ujson을 사용해 빠르게 JSON 파싱을 하고,
    배치 처리를 통해 속도 최적화.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        buffer = []
        line_count = 0
        total_lines = 0

        # 전체 라인 수 계산
        for _ in open(input_file, 'r', encoding='utf-8'):
            total_lines += 1
        infile.seek(0)

        pbar = tqdm(total=total_lines, desc="Processing reviews in batches")

        for line in infile:
            buffer.append(line)
            if len(buffer) >= chunk_size:
                reviews = [json.loads(l) for l in buffer]
                processed = batch_analyze_reviews(reviews, batch_size=batch_size)
                for rev in processed:
                    outfile.write(json.dumps(rev) + '\n')
                buffer.clear()
            pbar.update(1)

        # 남은 라인 처리
        if buffer:
            reviews = [json.loads(l) for l in buffer]
            processed = batch_analyze_reviews(reviews, batch_size=batch_size)
            for rev in processed:
                outfile.write(json.dumps(rev) + '\n')
            buffer.clear()

    print(f"Processed data saved to: {output_file}")

if __name__ == "__main__":
    input_file = "./output/reviews.json"  # 입력 파일
    output_file = "sentiment_reviews.jsonl"  # 출력 파일

    # chunk_size와 batch_size를 환경에 맞게 조정하여 최적의 성능 찾기
    process_reviews_in_batches(input_file, output_file, chunk_size=100, batch_size=256)
