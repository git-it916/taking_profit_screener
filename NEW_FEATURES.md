# 새로운 기능 추가 완료!

## CSV 파일 종목 분석 기능

문서 폴더에 있는 CSV 파일들을 자동으로 분석하여 **RVOL, 20일선, 윅 패턴**을 상세하게 보여줍니다!

---

## 주요 기능

### 1. 파일명만 입력하면 자동 분석
```bash
python quick_analyze.py sm.csv
```

### 2. 여러 종목 동시 분석
```bash
python quick_analyze.py sm.csv samsung.csv apple.csv kakao.csv
```

### 3. 상세한 분석 리포트

#### 20일선 분석
- 현재가와 20일선의 거리 (원 및 %)
- 20일선 상태 (위/아래, 멀리/근접)
- 추세 하락 여부 확인

#### RVOL 분석
- 상대 거래량 (평소 대비 몇 배)
- 강도 분류 (매우 강함/강함/보통/약함)
- 방향 (상승/하락, 전일 대비 변화량)

#### 윅 패턴 분석
- 윗꼬리 비율
- 거부 패턴 여부

#### 종합 판정
- 3가지 조건 충족 여부
- 신호 분류 (강력 매도, 주의, 관찰, 정상 등)

---

## 실행 예시

### 단일 종목 분석

```bash
python quick_analyze.py sm.csv
```

**결과**:
```
================================================================================
종목 분석 리포트: SM
================================================================================

날짜: 2026-01-05 17:48:37
현재가: 90.25

--------------------------------------------------------------------------------
[1] 20일 이동평균선 분석
--------------------------------------------------------------------------------
  - 20일선: 92.01
  - 20일선 대비 거리: -1.76원 (-1.91%)
  - 상태: 20일선 아래 (근접)
  - 조건 1 충족 여부: [O] 충족 (추세 하락)

--------------------------------------------------------------------------------
[2] RVOL (상대 거래량) 분석
--------------------------------------------------------------------------------
  - RVOL: 2.39배
  - 강도: 보통 (2~2.5배)
  - 방향: 상승 (전일 대비 +1.53)
  - 조건 3 충족 여부: [O] 충족 (거래량 확인)

--------------------------------------------------------------------------------
[3] 윅 패턴 분석
--------------------------------------------------------------------------------
  - 윅 비율: 0.71 (70.8%)
  - 조건 2 충족 여부: [O] 충족 (거부 패턴)

================================================================================
[종합 판정]
================================================================================
  신호: SELL
  분류: 강력 매도 (3개 조건 모두 충족)
  근거: Valid technical breakdown confirmed by high volume.
================================================================================
```

### 여러 종목 동시 분석

```bash
python quick_analyze.py sm.csv samsung.csv apple.csv kakao.csv
```

**요약 결과**:
```
================================================================================
전체 요약
================================================================================

     종목 현재가  20일선 RVOL   신호                  상태
     SM  90 -1.9% 2.4배 SELL 강력 매도 (3개 조건 모두 충족)
SAMSUNG  85 -2.4% 0.8배 HOLD              추세 하락만
  APPLE 110 +1.2% 1.1배 HOLD               윅 패턴만
  KAKAO 120 +1.6% 2.7배 HOLD             거래량 폭증만


주의: 1개 종목에서 강력 매도 신호 발견!
  - SM: Valid technical breakdown confirmed by high volume.
```

---

## 추가된 파일

| 파일 | 설명 |
|------|------|
| `src/analyzer.py` | 종목 상세 분석 모듈 (핵심) |
| `quick_analyze.py` | 빠른 분석 실행 스크립트 |
| `analyze_stocks.py` | 대화형 분석 프로그램 |
| `create_sample_data.py` | 샘플 데이터 생성 |
| `CSV_ANALYSIS_GUIDE.md` | 사용법 가이드 |

---

## 사용 방법

### 1단계: 샘플 데이터로 테스트 (선택)

```bash
python create_sample_data.py
```

문서 폴더에 4개의 샘플 CSV 파일 생성:
- sm.csv (SELL 신호)
- samsung.csv (추세 하락)
- apple.csv (정상)
- kakao.csv (거래량 폭증)

### 2단계: 빠른 분석

```bash
# 단일 종목
python quick_analyze.py sm.csv

# 여러 종목
python quick_analyze.py sm.csv samsung.csv
```

### 3단계: 대화형 분석 (선택)

```bash
python analyze_stocks.py
```

메뉴:
1. 단일 종목 분석 (파일명만 입력)
2. 여러 종목 일괄 분석
3. 사용자 지정 경로로 분석

---

## Python 코드로 사용

```python
from src import analyze_stock_from_csv, batch_analyze_stocks, StockAnalyzer

# 단일 종목 분석
result = analyze_stock_from_csv("sm.csv")

# 결과 확인
print(f"종목: {result['ticker']}")
print(f"20일선 대비: {result['ma_distance_percent']:.2f}%")
print(f"RVOL: {result['rvol']:.2f}배")
print(f"신호: {result['signal']}")

# 상세 리포트 출력
analyzer = StockAnalyzer()
report = analyzer.format_analysis_report(result)
print(report)

# 여러 종목 일괄 분석
results_df = batch_analyze_stocks(["sm.csv", "samsung.csv", "apple.csv"])

# SELL 신호만 필터링
sell_signals = results_df[results_df['signal'] == 'SELL']
print(sell_signals)
```

---

## 분석 항목 상세 설명

### 20일선 근접도

**거리 계산**:
```
거리(%) = (현재가 - 20일선) / 20일선 × 100
```

**상태 분류**:
- `20일선 위 (멀리)`: +5% 이상
- `20일선 위 (근접)`: +1% ~ +5%
- `20일선 근처`: -1% ~ +1%
- `20일선 아래 (근접)`: -5% ~ -1% ← **주의 구간**
- `20일선 아래 (멀리)`: -5% 미만 ← **위험 구간**

### RVOL 강도

**계산**:
```
RVOL = 현재 거래량 / 최근 20일 평균 거래량
```

**강도 분류**:
- `매우 강함 (3배 이상)`: RVOL >= 3.0 ← **폭발적**
- `강함 (2.5~3배)`: RVOL >= 2.5 ← **매우 강함**
- `보통 (2~2.5배)`: RVOL >= 2.0 ← **신호 기준**
- `약함 (1.5~2배)`: RVOL >= 1.5
- `매우 약함 (1.5배 미만)`: RVOL < 1.5

### RVOL 방향

**의미**:
- `상승`: 거래량이 전일보다 증가 → 신호 강화
- `하락`: 거래량이 전일보다 감소 → 신호 약화

### 신호 분류

| 분류 | 조건 | 의미 | 행동 |
|------|------|------|------|
| **강력 매도** | 3개 모두 충족 | 가장 강한 매도 신호 | 즉시 검토 필요 |
| **주의** | 추세하락 + 윅패턴 | 거래량만 부족 | 관찰, 거래량 증가 시 주의 |
| **관찰** | 추세하락 + 거래량 | 윅패턴만 부족 | 관찰, 추가 하락 가능성 |
| **거짓 신호 가능** | 윅패턴 + 거래량 | 추세는 양호 | 단기 조정, 상승 추세 유지 |
| **추세 하락만** | 조건 1만 | 완만한 하락 | 장기 관찰 |
| **윅 패턴만** | 조건 2만 | 일시적 거부 | 단기 관찰 |
| **거래량 폭증만** | 조건 3만 | 급등/급락 가능 | 변동성 주의 |
| **정상** | 모두 미충족 | 안정적 | 정상 보유 |

---

## 실전 활용 예시

### 1. 매일 아침 보유 종목 점검

**daily_check.bat** (Windows):
```batch
@echo off
python quick_analyze.py 삼성전자.csv SK하이닉스.csv NAVER.csv 카카오.csv > daily_report.txt
notepad daily_report.txt
```

### 2. SELL 신호만 즉시 확인

```python
from src import batch_analyze_stocks

# 보유 종목 전체 분석
results = batch_analyze_stocks([
    "삼성전자.csv", "SK하이닉스.csv", "NAVER.csv", "카카오.csv"
])

# SELL 신호만 필터링
sell_signals = results[results['signal'] == 'SELL']

if len(sell_signals) > 0:
    print(f"\n⚠ 주의: {len(sell_signals)}개 종목에서 매도 신호!")
    print(sell_signals[['ticker', 'close_price', 'ma_distance_percent',
                        'rvol', 'signal_category']])
else:
    print("\n✓ 모든 종목 정상")
```

### 3. 조건별 종목 분류

```python
from src import batch_analyze_stocks

results = batch_analyze_stocks(["sm.csv", "samsung.csv", "apple.csv"])

# 강력 매도
strong_sell = results[
    results['condition_1_trend_breakdown'] &
    results['condition_2_rejection_pattern'] &
    results['condition_3_volume_confirmation']
]

# 추세 하락 + 거래량
trend_vol = results[
    results['condition_1_trend_breakdown'] &
    results['condition_3_volume_confirmation']
]

# RVOL 3배 이상
high_rvol = results[results['rvol'] >= 3.0]

print(f"강력 매도: {len(strong_sell)}개")
print(f"추세 하락 + 거래량: {len(trend_vol)}개")
print(f"RVOL 3배 이상: {len(high_rvol)}개")
```

---

## 장점

1. **파일명만 입력**: 경로 자동 처리
2. **상세한 분석**: 20일선, RVOL, 윅 패턴 모두 확인
3. **조건별 분류**: 어떤 조건이 충족되었는지 명확히 표시
4. **방향성 파악**: RVOL 상승/하락 추세 확인
5. **즉시 활용**: Python 코드 없이 명령어만으로 사용 가능

---

## 다음 단계

1. **샘플 테스트**
   ```bash
   python create_sample_data.py
   python quick_analyze.py sm.csv
   ```

2. **실제 데이터 분석**
   ```bash
   python quick_analyze.py 내종목.csv
   ```

3. **상세 가이드 확인**
   - [CSV_ANALYSIS_GUIDE.md](CSV_ANALYSIS_GUIDE.md)

---

**이제 `python quick_analyze.py <파일명>`으로 시작하세요!**
