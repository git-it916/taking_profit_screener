# Taking Profit Screener

**IGIS Asset Management - Quantitative Technical Analysis Tool**

매일 아침 300개 펀드 종목의 20일 이동평균선 하회 여부와 거래량 분석을 통한 모니터링 도구입니다.

## 빠른 시작

```bash
# 1. 라이브러리 설치
pip install -r requirements.txt

# 2. start.py 실행 - 파일명만 입력하면 자동 분석!
python start.py
```

**입력 예시**:
- 단일 종목: `samsung.xlsx`
- 여러 종목: `samsung.xlsx, lg.xlsx, sk.xlsx`
- CSV도 가능: `old_data.csv`

## 주요 기능

### 1. 20일선 하회/상회 추적
- **가장 최근 20일선 하회일** 기록
- **가장 최근 20일선 상회일** 기록
- **20일선 아래 경과일** 계산
- **20일선 가격** 명시적 표시

### 2. RVOL (상대 거래량) 분석
- 현재 거래량 / 20일 평균 거래량
- 강도 분류: 매우 강함(3배+), 강함(2.5~3배), 보통(2~2.5배)
- 전일 대비 방향 (상승/하락)

## 출력 예시

### 요약 테이블
```
종목    현재가  전일비  20일선  괴리율    하회일      RVOL  신호
SM      89    -2.4%   92    -3.3%  2025-10-18  2.3배  SELL
SAMSUNG 86    +0.3%   87    -1.9%  2025-10-18  1.1배  HOLD
APPLE   111   +1.0%   109   +1.6%  2025-11-28  1.0배  HOLD
```

**컬럼 설명**:
- **현재가**: 오늘 종가
- **전일비**: 어제 대비 변동률 (%)
- **20일선**: 20일 이동평균선 가격
- **괴리율**: 현재가와 20일선 간 괴리율 (%)
- **하회일**: 가장 최근 20일선 하회 날짜
- **RVOL**: 상대 거래량 (배수)
- **신호**: SELL / HOLD

### 상세 리포트
```
================================================================================
종목 분석 리포트: SAMSUNG
================================================================================

날짜: 2026-01-06
현재가: 88.92
어제 종가: 91.07 (전일대비 -2.15원, -2.36%)

[1] 20일 이동평균선 분석
  - 20일선 가격: 91.93
  - 현재가 대비: -3.01원 (-3.27%)
  - 상태: 20일선 아래 (근접)
  - 최근 20일선 하회일: 2025-10-18
  - 최근 20일선 상회일: 없음
  - 20일선 아래 경과일: 81일
  - 조건 1 충족 여부: [O] 충족 (추세 하락)

[2] RVOL (상대 거래량) 분석
  - RVOL: 2.22배
  - 강도: 보통 (2~2.5배)
  - 방향: 상승 (전일 대비 +1.33)
  - 조건 2 충족 여부: [O] 충족 (거래량 확인)

[종합 판정]
  신호: SELL
  분류: 강력 매도 (20일선 하회 + 거래량 폭증)
```

## 프로젝트 구조

```
taking_profit_screener/
├── start.py                 ← 메인 실행 파일 (여기만 실행하면 됨!)
│
├── src/                     ← 핵심 모듈
│   ├── screener.py         (20일선, RVOL, 윗꼬리 계산)
│   ├── optimizer.py        (파라미터 최적화)
│   └── analyzer.py         (종목 상세 분석)
│
├── tools/                   ← 유틸리티 스크립트
│   ├── quick_analyze.py    (명령줄 빠른 분석)
│   └── create_sample_data.py (샘플 데이터 생성)
│
├── examples/                ← 예시 코드
└── docs/                    ← 상세 문서
```

## 전략 설명

### 20일선 + RVOL 전략

**2가지 조건이 모두 충족되면 SELL 신호 발생**:

1. **추세 하락** (Trend Breakdown): 현재가 < 20일 이동평균
2. **거래량 확인** (Volume Confirmation): RVOL ≥ 2.0

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

## Python 코드로 사용하기

### 단일 종목 분석

```python
from src import analyze_stock_from_csv, StockAnalyzer

# XLSX 파일 분석
result = analyze_stock_from_csv("samsung.xlsx")

# 결과 확인
print(f"종목: {result['ticker']}")
print(f"현재가: {result['close_price']:.2f}")
print(f"어제 종가: {result['prev_close']:.2f}")
print(f"20일선 가격: {result['ma20']:.2f}")
print(f"20일선 괴리율: {result['ma_distance_percent']:.2f}%")
print(f"20일선 하회일: {result['last_ma20_break_below']}")
print(f"RVOL: {result['rvol']:.2f}배")
print(f"신호: {result['signal']}")

# 상세 리포트 출력
analyzer = StockAnalyzer()
report = analyzer.format_analysis_report(result)
print(report)
```

### 여러 종목 일괄 분석 (300개 펀드 종목)

```python
from src import batch_analyze_stocks

# 300개 파일 분석
filenames = ["stock1.xlsx", "stock2.xlsx", ..., "stock300.xlsx"]
results = batch_analyze_stocks(filenames)

# 20일선 하회 종목만 필터링
breakdown_stocks = results[results['condition_1_trend_breakdown'] == True]
print(f"20일선 하회 종목: {len(breakdown_stocks)}개")

# 최근 3일 이내 하회 종목
recent_breakdown = results[
    (results['condition_1_trend_breakdown'] == True) &
    (results['days_below_ma20'] <= 3)
]
print(f"최근 3일 이내 하회: {len(recent_breakdown)}개")

# SELL 신호만
sell_signals = results[results['signal'] == 'SELL']
print(sell_signals[['ticker', 'last_ma20_break_below', 'ma_distance_percent', 'rvol']])

# 결과 저장
results.to_csv('전체_분석결과.csv', index=False, encoding='utf-8-sig')
```

### 고급 스크리닝

```python
from src import ExitSignalScreener, load_data_from_csv

# 1. XLSX/CSV 파일 로드
df = load_data_from_csv('samsung.xlsx', date_column='Date')

# 2. 스크리너 초기화
screener = ExitSignalScreener(ma_period=20, rvol_period=20)

# 3. 분석 실행
filtered_data = screener.apply_filters(df)

# 4. 20일선 하회 확인
latest = filtered_data.iloc[-1]
print(f"20일선 하회일: {latest['Last_MA20_Break_Below']}")
print(f"경과일: {latest['Days_Below_MA20']}일")
print(f"RVOL: {latest['RVOL']:.2f}배")

# 5. SELL 신호만 추출
output = screener.generate_screening_output(filtered_data, ticker="SAMSUNG")
sell_signals = output[output['Signal'] == 'SELL']
print(sell_signals)
```

## 활용 사례

### 매일 아침 300개 펀드 종목 모니터링

```bash
# 1. 블룸버그/HTS에서 전날 데이터 다운로드 (향후 자동화 예정)
# → 300개 XLSX 파일을 문서 폴더에 저장

# 2. start.py 실행
python start.py

# 3. 파일명 입력 (콤마로 구분)
파일명 입력: stock1.xlsx, stock2.xlsx, ..., stock300.xlsx

# 4. 요약 테이블 확인
# → 하회일, RVOL 등을 한눈에 확인

# 5. CSV로 저장 후 엑셀에서 정렬/필터링
# → 하회일 기준 정렬: 최근 돌파 종목 파악
# → 경과일 기준 정렬: 장기 하락 종목 파악
# → RVOL 기준 정렬: 거래량 급증 종목 파악
```

### 모니터링 우선순위

**Level 1: 강력 매도 신호** (즉시 주의)
- 2가지 조건 모두 충족
- 신호 = SELL

**Level 2: 주의 필요**
- 20일선 하회, 거래량 부족
- 거래량 늘어나면 SELL로 전환

**Level 3: 관찰**
- 20일선 하회만
- 하회일과 경과일 확인
- RVOL 추이 관찰

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
