# elo.py
# Elo Rating 알고리즘 구현

# ── 상수 ──────────────────────────────────────────────────
INITIAL_RATING = 1500   # 초기 점수
K_FACTOR = 32           # 점수 변동 폭 (클수록 변동 크게)


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    A가 B를 이길 확률 계산
    - 두 점수 차이가 클수록 강한 쪽 승리 확률 높아짐
    """
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_ratings(rating_a: float, rating_b: float, winner: str):
    """
    경기 결과에 따라 두 플레이어 점수 업데이트
    winner: "A" | "B" | "TIE"
    반환: (새로운 rating_a, 새로운 rating_b)
    """
    expected_a = expected_score(rating_a, rating_b)
    expected_b = expected_score(rating_b, rating_a)

    if winner == "A":
        actual_a, actual_b = 1.0, 0.0
    elif winner == "B":
        actual_a, actual_b = 0.0, 1.0
    else:  # TIE
        actual_a, actual_b = 0.5, 0.5

    new_rating_a = rating_a + K_FACTOR * (actual_a - expected_a)
    new_rating_b = rating_b + K_FACTOR * (actual_b - expected_b)

    return round(new_rating_a, 1), round(new_rating_b, 1)


class EloSystem:
    """
    여러 라운드 걸쳐 Elo 점수를 관리하는 클래스
    사용자 1명 vs AI 경쟁자 3종(상/중/하) 구조
    """

    def __init__(self):
        self.ratings = {
            "user":             INITIAL_RATING,
            "competitor_high":  INITIAL_RATING,  # 상
            "competitor_mid":   INITIAL_RATING,  # 중
            "competitor_low":   INITIAL_RATING,  # 하
        }
        self.history = []  # 경기 기록

    def record_match(self, competitor_level: str, winner: str, question: str, normalized_score_user: float, normalized_score_competitor: float):
        """
        경기 결과 기록 + Elo 점수 업데이트
        competitor_level: "high" | "mid" | "low"
        winner: "A"(사용자) | "B"(경쟁자) | "TIE"
        """
        competitor_key = f"competitor_{competitor_level}"

        old_user = self.ratings["user"]
        old_competitor = self.ratings[competitor_key]

        new_user, new_competitor = update_ratings(old_user, old_competitor, winner)

        self.ratings["user"] = new_user
        self.ratings[competitor_key] = new_competitor

        self.history.append({
            "question": question,
            "competitor_level": competitor_level,
            "winner": winner,
            "normalized_score_user": normalized_score_user,
            "normalized_score_competitor": normalized_score_competitor,
            "rating_change_user": round(new_user - old_user, 1),
            "rating_user_after": new_user,
            "rating_competitor_after": new_competitor,
        })

    def get_rankings(self) -> list:
        """현재 Elo 점수 기준 순위 반환"""
        sorted_ratings = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        rankings = []
        for rank, (player, rating) in enumerate(sorted_ratings, start=1):
            rankings.append({
                "rank": rank,
                "player": player,
                "rating": rating,
            })
        return rankings

    def get_user_level(self) -> str:
        """
        사용자 Elo 점수 기반 수준 판정
        경쟁자 점수와 비교해서 상대적 위치 반환
        """
        user_rating = self.ratings["user"]
        high = self.ratings["competitor_high"]
        mid = self.ratings["competitor_mid"]
        low = self.ratings["competitor_low"]

        if user_rating >= high:
            return "상위권 (AI 고수준 경쟁자 초과)"
        elif user_rating >= mid:
            return "중상위권 (AI 중수준 ~ 고수준 사이)"
        elif user_rating >= low:
            return "중위권 (AI 저수준 ~ 중수준 사이)"
        else:
            return "하위권 (AI 저수준 경쟁자 미달)"


# ── 테스트용 ──────────────────────────────────────────────
if __name__ == "__main__":
    import json

    elo = EloSystem()
    print("초기 점수:", elo.ratings)

    # 라운드 1: 사용자가 하수준 경쟁자에게 승리
    elo.record_match("low", "A", "Transformer 설명", 93.3, 40.0)
    # 라운드 2: 사용자가 중수준 경쟁자에게 패배
    elo.record_match("mid", "B", "BERT vs GPT", 55.0, 72.0)
    # 라운드 3: 사용자가 중수준 경쟁자와 무승부
    elo.record_match("mid", "TIE", "Fine-tuning vs Prompt", 65.0, 65.0)

    print("\n경기 후 점수:")
    print(json.dumps(elo.ratings, ensure_ascii=False, indent=2))

    print("\n최종 순위:")
    for r in elo.get_rankings():
        print(f"  {r['rank']}위: {r['player']} ({r['rating']}점)")

    print("\n사용자 수준:", elo.get_user_level())
