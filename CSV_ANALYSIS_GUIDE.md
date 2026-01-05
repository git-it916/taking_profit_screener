# XLSX 파일 종목 분석 가이드

문서 폴더에 있는 XLSX 파일들을 자동으로 분석하여 RVOL, 20일선, 윅 패턴 등을 상세하게 보여줍니다.

## 빠른 시작

### 1. 샘플 데이터 생성 (테스트용)

```bash
python create_sample_data.py
```

문서 폴더에 4개의 샘플 XLSX 파일이 생성됩니다:
- `sm.xlsx` - SELL 신호 (3개 조건 모두 충족)
- `samsung.xlsx` - 추세 하락만
- `apple.xlsx` - 정상
- `kakao.xlsx` - 거래량 폭증만

### 2. 빠른 분석 (추천!)

```bash
# 단일 종목 분석
python quick_analyze.py sm.xlsx

# 여러 종목 동시 분석
python quick_analyze.py sm.xlsx samsung.xlsx apple.xlsx
```

### 3. 대화형 분석

```bash
python analyze_stocks.py
```

메뉴에서 선택:
1. 단일 종목 분석
2. 여러 종목 일괄 분석
3. 사용자 지정 경로로 분석

## 분석 결과 예시

### 단일 종목 상세 분석

```
================================================================================
종목 분석 리포트: SM
================================================================================

날짜: 2026-01-05 17:48:37.216400
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

### 여러 종목 요약

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

## 분석 항목 설명

### 1. 20일 이동평균선 분석

| 항목 | 설명 |
|------|------|
| **20일선** | 최근 20일 종가의 평균 |
| **20일선 대비 거리** | 현재가 - 20일선 (원 및 %) |
| **상태** | 20일선과의 위치 관계 |
| **조건 1** | 현재가 < 20일선 (추세 하락) |

**상태 분류**:
- `20일선 위 (멀리)`: +5% 이상
- `20일선 위 (근접)`: +1% ~ +5%
- `20일선 근처`: -1% ~ +1%
- `20일선 아래 (근접)`: -5% ~ -1%
- `20일선 아래 (멀리)`: -5% 미만

### 2. RVOL (상대 거래량) 분석

| 항목 | 설명 |
|------|------|
| **RVOL** | 현재 거래량 / 20일 평균 거래량 |
| **강도** | RVOL 크기에 따른 분류 |
| **방향** | 전일 대비 상승/하락 |
| **조건 3** | RVOL >= 2.0 (거래량 확인) |

**강도 분류**:
- `매우 강함 (3배 이상)`: RVOL >= 3.0
- `강함 (2.5~3배)`: RVOL >= 2.5
- `보통 (2~2.5배)`: RVOL >= 2.0
- `약함 (1.5~2배)`: RVOL >= 1.5
- `매우 약함 (1.5배 미만)`: RVOL < 1.5

### 3. 윅 패턴 분석

| 항목 | 설명 |
|------|------|
| **윅 비율** | 윗꼬리 길이 / 전체 캔들 길이 |
| **조건 2** | 윅 비율 >= 0.5 (거부 패턴) |

윗꼬리가 길수록 = 상승 시도 후 강하게 거부당함 = 매도 압력 강함

### 4. 종합 판정

**신호 분류**:
- `강력 매도 (3개 조건 모두 충족)`: SELL 신호, 즉시 주의 필요
- `주의 (추세하락 + 윅패턴, 거래량 부족)`: 거래량만 부족, 관찰 필요
- `관찰 (추세하락 + 거래량, 윅패턴 없음)`: 윅패턴만 부족
- `거짓 신호 가능 (윅패턴 + 거래량, 추세 양호)`: 추세는 양호, 단기 조정 가능
- `추세 하락만`: 조건 1만 충족
- `윅 패턴만`: 조건 2만 충족
- `거래량 폭증만`: 조건 3만 충족
- `정상 (신호 없음)`: 모든 조건 미충족

## XLSX 파일 형식

필수 컬럼:
- `Date`: 날짜 (YYYY-MM-DD 형식 권장)
- `Open`: 시가
- `High`: 고가
- `Low`: 저가
- `Close`: 종가
- `Volume`: 거래량

예시:
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,100.5,102.3,99.8,101.2,1500000
2024-01-02,101.2,103.0,100.5,102.5,2000000
...
```

## Python 스크립트로 사용하기

```python
from src import analyze_stock_from_csv, batch_analyze_stocks, StockAnalyzer

# 단일 종목 분석
result = analyze_stock_from_csv(
    "sm.xlsx",
    base_path=r"C:\Users\10845\OneDrive - 이지스자산운용\문서"
)

# 결과 출력
analyzer = StockAnalyzer()
print(analyzer.format_analysis_report(result))

# 여러 종목 일괄 분석
results_df = batch_analyze_stocks(
    ["sm.xlsx", "samsung.xlsx", "apple.xlsx"],
    base_path=r"C:\Users\10845\OneDrive - 이지스자산운용\문서"
)

# SELL 신호만 필터링
sell_signals = results_df[results_df['signal'] == 'SELL']
print(sell_signals)
```

## 실전 사용 예시

### 1. 보유 종목 일괄 점검

```bash
python quick_analyze.py 삼성전자.xlsx SK하이닉스.xlsx NAVER.xlsx 카카오.xlsx
```

### 2. 특정 종목 상세 분석

```bash
python analyze_stocks.py
# 메뉴에서 1 선택
# 파일명 입력: 삼성전자.xlsx
```

### 3. 결과 XLSX로 저장

```python
from src import batch_analyze_stocks

# 여러 종목 분석
results = batch_analyze_stocks(["종목1.xlsx", "종목2.xlsx", "종목3.xlsx"])

# XLSX로 저장
results.to_csv("analysis_results.xlsx", index=False, encoding='utf-8-sig')
```

## 자주 묻는 질문

### Q1: XLSX 파일을 어디에 둬야 하나요?
**A**: 기본값은 `C:\Users\10845\OneDrive - 이지스자산운용\문서` 입니다.
다른 경로를 사용하려면:
- `analyze_stocks.py` 실행 후 메뉴 3번 선택
- 또는 Python 코드에서 `base_path` 파라미터 변경

### Q2: 날짜 컬럼명이 "Date"가 아닌데요?
**A**: `date_column` 파라미터로 지정 가능합니다:
```python
analyze_stock_from_csv("sm.xlsx", date_column="날짜")
```

### Q3: SELL 신호가 나왔는데 어떻게 해야 하나요?
**A**:
1. 분석 결과를 자세히 확인
2. 20일선, RVOL, 윅 패턴이 모두 강하게 나타났는지 확인
3. 다른 보조 지표와 함께 종합적으로 판단
4. 실제 매도는 본인의 투자 원칙에 따라 결정

### Q4: "추세 하락만" 또는 "거래량 폭증만"은 무슨 의미인가요?
**A**:
- **추세 하락만**: 20일선 아래이지만 윅패턴과 거래량이 부족 → 관찰 필요
- **거래량 폭증만**: 거래량은 많지만 추세는 양호 → 급등 후 일시 조정 가능
- **강력 매도**: 3가지 모두 충족 → 가장 강한 신호

### Q5: 여러 종목을 자동으로 매일 분석할 수 있나요?
**A**: 네, 배치 스크립트를 만들면 됩니다:

**daily_check.bat** (Windows):
```batch
@echo off
cd /d "C:\Users\10845\OneDrive - 이지스자산운용\문서\quant_project\taking_profit_screener"
python quick_analyze.py 종목1.xlsx 종목2.xlsx 종목3.xlsx > daily_report.txt
echo 분석 완료! daily_report.txt 확인
pause
```

## 고급 기능

### 커스텀 파라미터로 분석

```python
from src import StockAnalyzer
from src.screener import load_data_from_csv

# 50일선, 30일 RVOL 사용
analyzer = StockAnalyzer(ma_period=50, rvol_period=30)

df = load_data_from_csv("sm.xlsx")
result = analyzer.analyze_latest(df, ticker="SM")

# 결과 출력
report = analyzer.format_analysis_report(result)
print(report)
```

### 특정 조건만 필터링

```python
from src import batch_analyze_stocks

# 여러 종목 분석
results = batch_analyze_stocks(["sm.xlsx", "samsung.xlsx", "apple.xlsx"])

# RVOL 3배 이상만
high_rvol = results[results['rvol'] >= 3.0]

# 20일선 아래 & RVOL 2배 이상
trend_down_high_vol = results[
    results['condition_1_trend_breakdown'] &
    (results['rvol'] >= 2.0)
]

print(trend_down_high_vol[['ticker', 'ma_distance_percent', 'rvol', 'signal_category']])
```

## 요약

| 명령어 | 설명 | 사용 시기 |
|--------|------|-----------|
| `python create_sample_data.py` | 샘플 데이터 생성 | 처음 사용, 테스트 |
| `python quick_analyze.py sm.xlsx` | 빠른 단일 분석 | 간단한 확인 |
| `python quick_analyze.py sm.xlsx samsung.xlsx` | 빠른 일괄 분석 | 여러 종목 빠른 비교 |
| `python analyze_stocks.py` | 대화형 분석 | 상세 분석, XLSX 저장 필요 |

---

**이제 `python quick_analyze.py <파일명>`으로 시작하세요!**
