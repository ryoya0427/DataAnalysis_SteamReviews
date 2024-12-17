import json
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize
from networkx.algorithms.community import greedy_modularity_communities
import random

def build_sentiment_network(file_path, target_game, top_n=100, min_freq=5):
    stop_words = set(stopwords.words('english'))
    stop_words |= {'game', 'play', 'good', 'time', 'like', 'im', 'get'}

    word_pairs = Counter()
    word_weights = Counter()
    positive_set = set()
    negative_set = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record['title'] == target_game:
                words = word_tokenize(record['processed_review'])
                filtered_words = [w for w in words if w not in stop_words and w.isalpha()]
                positive_words = set(record.get('positive_words', []))
                negative_words = set(record.get('negative_words', []))

                # 단어 가중치 및 긍정/부정 단어 집합 업데이트
                for word in filtered_words:
                    if word in positive_words:
                        word_weights[word] += 3
                        positive_set.add(word)
                    elif word in negative_words:
                        word_weights[word] += 2
                        negative_set.add(word)

                # 모든 단어 쌍 업데이트
                for combo in combinations(filtered_words, 2):
                    if combo[0] != combo[1]:
                        word_pairs.update([combo])

    # 그래프 생성
    G = nx.Graph()

    # 긍정 단어끼리만 동시 출현한 단어쌍 추가
    positive_word_pairs = [(pair, count) for pair, count in word_pairs.most_common() 
                           if pair[0] in positive_set and pair[1] in positive_set and count >= min_freq]

    # top_n 적용
    positive_word_pairs = positive_word_pairs[:top_n]

    for (word1, word2), count in positive_word_pairs:
        G.add_edge(word1, word2, weight=count)

    # 노드 크기 설정 (기존 설정과 동일)
    node_sizes = [min(word_weights.get(node, 1) * 10, 300) for node in G.nodes()]

    # 엣지 두께 설정 (기존 설정과 동일)
    edge_widths = [G[u][v]['weight'] * 0.01 for u, v in G.edges()]

    # 커뮤니티 탐지 후 커뮤니티별 색상 부여
    communities = list(greedy_modularity_communities(G))
    node_community_map = {}
    for i, c in enumerate(communities):
        for node in c:
            node_community_map[node] = i

    community_count = len(communities)
    colors = plt.cm.get_cmap('tab10', community_count)
    node_colors = [colors(node_community_map[node]) for node in G.nodes()]

    # 그래프 시각화
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, seed=42, k=1.0)

    nx.draw(G, pos, with_labels=True, node_size=node_sizes, font_size=10,
            node_color=node_colors, edge_color='gray', width=edge_widths, linewidths=1, edgecolors='black')

    plt.title(f"Sentiment Network (Positive words only) for '{target_game}'", fontsize=16)
    plt.show()

if __name__ == "__main__":
    input_file = "sentiment_reviews.jsonl"
    target_games = [
        "Call of Duty\u00ae_ Modern Warfare\u00ae",
        "Call of Duty\u00ae_ Modern Warfare\u00ae III"
    ]

    for game in target_games:
        print(f"Building sentiment network for {game}...")
        build_sentiment_network(input_file, game, top_n=100, min_freq=5)
