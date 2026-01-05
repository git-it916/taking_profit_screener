# Taking Profit Screener

**IGIS Asset Management - Quantitative Technical Analysis Tool**

Volume-Confirmed Rejection 전략을 기반으로 한 익절 신호 스크리닝 도구입니다.

## 빠른 시작

### 방법 1: 간편 분석 (가장 추천!)

```bash
# 1. 라이브러리 설치
pip install -r requirements.txt

# 2. start.py 실행 - 파일명만 입력하면 자동 분석!
python start.py
```

파일명을 입력하면 자동으로 RVOL, 20일선, 윗꼬리 패턴을 분석하고 상세 리포트를 제공합니다.

**입력 예시**:
- 단일 종목: `sm.xlsx`
- 여러 종목: `sm.xlsx, samsung.xlsx, apple.xlsx`
- CSV 가능: `old_data.csv`

**상세 가이드**: [CSV_ANALYSIS_GUIDE.md](CSV_ANALYSIS_GUIDE.md)

### 방법 2: 명령줄 도구

```bash
# tools 폴더의 스크립트 사용
python tools/quick_analyze.py sm.xlsx
python tools/quick_analyze.py sm.xlsx samsung.xlsx apple.xlsx
python tools/analyze_stocks.py  # 대화형 분석
```

## 프로젝트 구조

```
taking_profit_screener/
├── start.py                 ← 메인 실행 파일 (파일명만 입력!)
│
├── src/                     ← 핵심 모듈
│   ├── screener.py         (신호 스크리닝)
│   ├── optimizer.py        (파라미터 최적화)
│   ├── analyzer.py         (종목 상세 분석)
│   └── __init__.py         (패키지 초기화)
│
├── tools/                   ← 유틸리티 스크립트
│   ├── quick_analyze.py    (명령줄 빠른 분석)
│   ├── analyze_stocks.py   (대화형 분석)
│   └── create_sample_data.py (샘플 데이터 생성)
│
├── examples/                ← 예시 코드
├── docs/                    ← 상세 문서
│   ├── README.md           (상세 설명)
│   └── USAGE_GUIDE.md      (사용법 가이드)
├── CSV_ANALYSIS_GUIDE.md    ← XLSX/CSV 분석 가이드
└── requirements.txt
```

## 주요 기능

### 1. 종목 분석 (start.py)

**파일명만 입력하면 자동으로 분석합니다**:
- 단일 종목: 상세 리포트 제공
- 여러 종목: 조건별 분류 및 요약

**분석 항목**:
- 20일 이동평균선 대비 거리 및 상태
- RVOL (상대 거래량) 강도 및 방향
- 윗꼬리 패턴 분석
- 종합 판정: SELL, HOLD 신호

**조건별 분류**:
- 강력 매도 (3개 조건 모두 충족)
- 추세 하락만
- 거래량 폭증만
- 주의 필요 (추세하락 + 윅패턴)

### 2. 파라미터 최적화

**Python 코드로 사용**:
- Random Search 최적화
- Walk-Forward 최적화
- 과최적화 방지
- 실전 트레이딩용 파라미터 탐색

상세 예시: [examples/](examples/) 폴더 참고

## 전략 설명

### Volume-Confirmed Rejection

**3가지 조건이 모두 충족되면 SELL 신호 발생**:

1. **추세 하락** (Trend Breakdown): 현재가 < 20일 이동평균
2. **거부 패턴** (Rejection Pattern): 윗꼬리 비율 ≥ 50%
3. **거래량 확인** (Volume Confirmation): RVOL ≥ 2.0

## 입력 데이터 형식

XLSX 또는 CSV 파일은 다음 컬럼을 포함해야 합니다:

| 컬럼 | 설명 |
|------|------|
| Date | 날짜 (YYYY-MM-DD 권장) |
| Open | 시가 |
| High | 고가 |
| Low | 저가 |
| Close | 종가 |
| Volume | 거래량 |

**기본 경로**: `C:\Users\10845\OneDrive - 이지스자산운용\문서`

예시:
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,100.5,102.3,99.8,101.2,1500000
2024-01-02,101.2,103.0,100.5,102.5,2000000
...
```

## 출력 결과

| Date | Current_Price | MA20 | Wick_Ratio | RVOL | Signal | Reasoning |
|------|--------------|------|------------|------|--------|-----------|
| 2024-03-15 | 175.50 | 180.20 | 0.65 | 2.3 | SELL | Valid technical breakdown confirmed by high volume. |

## 예시 코드

### 방법 1: 간단한 분석 (추천)

```python
from src import analyze_stock_from_csv, StockAnalyzer

# XLSX 파일 분석
result = analyze_stock_from_csv("sm.xlsx")

# 결과 확인
print(f"종목: {result['ticker']}")
print(f"20일선 대비: {result['ma_distance_percent']:.2f}%")
print(f"RVOL: {result['rvol']:.2f}배")
print(f"신호: {result['signal']}")

# 상세 리포트 출력
analyzer = StockAnalyzer()
report = analyzer.format_analysis_report(result)
print(report)
```

### 방법 2: 여러 종목 일괄 분석

```python
from src import batch_analyze_stocks

# 여러 종목 분석
results = batch_analyze_stocks(["sm.xlsx", "samsung.xlsx", "apple.xlsx"])

# SELL 신호만 필터링
sell_signals = results[results['signal'] == 'SELL']
print(sell_signals[['ticker', 'ma_distance_percent', 'rvol', 'signal_category']])

# 결과 저장
results.to_csv('analysis_results.csv', index=False, encoding='utf-8-sig')
```

### 방법 3: 고급 스크리닝

```python
from src import ExitSignalScreener, load_data_from_csv

# 1. XLSX/CSV 파일 로드
df = load_data_from_csv('my_data.xlsx', date_column='Date')

# 2. 스크리너 초기화
screener = ExitSignalScreener(ma_period=20, rvol_period=20)

# 3. 분석 실행
filtered_data = screener.apply_filters(df)
output = screener.generate_screening_output(filtered_data, ticker="AAPL")

# 4. SELL 신호만 추출
sell_signals = output[output['Signal'] == 'SELL']
print(sell_signals)
```

### 파라미터 최적화

```python
from src import ParameterOptimizer, load_data_from_csv

# 1. 데이터 로드
df = load_data_from_csv('my_data.csv', date_column='Date')

# 2. 옵티마이저 초기화
optimizer = ParameterOptimizer(df, ticker="AAPL", evaluation_metric='sharpe_ratio')

# 3. Random Search 실행
best_params, best_score = optimizer.random_search(
    n_iterations=50,
    param_ranges={
        'ma_period': (15, 60),
        'rvol_period': (10, 30),
        'wick_threshold': (0.4, 0.7),
        'rvol_threshold': (1.5, 3.5)
    }
)

print(f"최적 파라미터: {best_params}")
print(f"Sharpe Ratio: {best_score:.4f}")
```

## 더 알아보기

- **상세 설명**: [docs/README.md](docs/README.md)
- **사용법 가이드**: [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)
- **예시 코드**: [examples/](examples/) 폴더 참고

## 요구사항

- Python 3.7+
- pandas >= 2.0.0
- numpy >= 1.24.0
- openpyxl >= 3.0.0 (XLSX 지원)

## 라이선스

MIT License

## 문의

IGIS Asset Management
