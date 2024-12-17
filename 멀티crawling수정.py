import os
import requests
import json
import time
import re

def get_game_name(app_id):
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get(app_id) and data[app_id].get('data') and data[app_id]['data'].get('name'):
            return data[app_id]['data']['name']
    return None

def sanitize_filename(filename):
    # 파일 이름에서 특수 문자 제거
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def get_steam_reviews(app_id, cursor="*", max_reviews=None):
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    reviews = []
    params = {
        "json": 1,
        "cursor": cursor,
        "filter": "all",
        "language": "english",
        "day_range": 9223372036854775807,
        "review_type": "all",
        "num_per_page": 100,
    }
    retry_count = 0
    max_retries = 5

    while True:
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                print(f"APP_ID {app_id} - 오류 응답 코드: {response.status_code}")
                raise Exception("데이터 가져오기 실패")
            data = response.json()

            # 리뷰 데이터가 없으면 종료
            if not data.get("reviews") or len(data["reviews"]) == 0:
                print(f"APP_ID {app_id} - 리뷰 데이터가 없습니다. 이 게임을 건너뜁니다.")
                break

            reviews.extend(data["reviews"])

            if max_reviews and len(reviews) >= max_reviews:
                reviews = reviews[:max_reviews]
                break

            new_cursor = data.get("cursor")
            if not new_cursor or new_cursor == params["cursor"]:
                print(f"APP_ID {app_id} - 커서가 더 이상 갱신되지 않습니다. 수집을 종료합니다.")
                break

            params["cursor"] = new_cursor

            if max_reviews:
                progress = min(len(reviews), max_reviews) / max_reviews * 100
                print(f"APP_ID {app_id} - 진행 상황: {progress:.2f}% ({len(reviews)} / {max_reviews})")
            else:
                print(f"APP_ID {app_id} - 진행 상황: {len(reviews)}개의 리뷰 수집 완료")

            time.sleep(1)
            retry_count = 0

        except Exception as e:
            retry_count += 1
            print(f"APP_ID {app_id} - 요청 실패: {e}, 재시도 중... ({retry_count}/{max_retries})")
            if retry_count >= max_retries:
                print(f"APP_ID {app_id} - 최대 재시도 횟수 초과. 이 게임을 건너뜁니다.")
                break
            time.sleep(5)

    return reviews[:max_reviews] if max_reviews else reviews

def save_reviews_to_json(data, filename="reviews.json"):
    # 디렉토리 확인 및 생성
    if not os.path.exists("data"):
        os.makedirs("data")  # "data" 폴더 생성

    filepath = os.path.join("data", filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_reviews_for_multiple_games(app_ids, max_reviews=None):
    for app_id in app_ids:
        game_name = get_game_name(app_id)
        if not game_name:
            print(f"APP_ID {app_id} - 잘못된 APP_ID이거나 게임 이름을 찾을 수 없습니다. 건너뜁니다.")
            continue

        print(f"\n{game_name}의 리뷰 데이터를 수집합니다...")
        reviews_data = get_steam_reviews(app_id, max_reviews=max_reviews)

        if reviews_data:
            filename = sanitize_filename(f"{game_name}_reviews.json")
            save_reviews_to_json(reviews_data, filename)
            print(f"APP_ID {app_id} - 리뷰 데이터를 {filename} 파일로 저장했습니다.")
        else:
            print(f"APP_ID {app_id} - 리뷰 데이터를 가져오지 못했습니다.")

# 예시로 여러 게임의 리뷰 데이터를 가져오기
app_ids = [
    "812140"
    ]
#"7940", "10090", "10180", "42700", "202970", "209160","209650", "311210", "292730", "393080", "476600", "1962660", "1938090","1962663", "1985810", "1985820", "2000950", "2519060", "2933620"
max_reviews = None
get_reviews_for_multiple_games(app_ids, max_reviews=max_reviews)
