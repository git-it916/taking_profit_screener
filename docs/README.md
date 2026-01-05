# Taking Profit Screener

**IGIS Asset Management - Quantitative Technical Analysis Tool**

## Overview

Volume-Confirmed Rejection 전략을 기반으로 한 익절 신호 스크리닝 도구입니다.
엄격한 3가지 기술적 조건을 통해 백테스팅과 실시간 스크리닝을 지원합니다.

## Strategy: Volume-Confirmed Rejection

### 3가지 필수 조건

1. **Trend Breakdown**: 현재가 < 20일 이동평균선
2. **Rejection Pattern**: 상단 윅 비율 >= 0.5 (윅이 캔들 전체 길이의 50% 이상)
3. **Volume Confirmation**: RVOL (상대 거래량) >= 2.0

### 시그널 규칙

- **SELL**: 3가지 조건이 모두 충족될 때만 발생
  - Reasoning: "Valid technical breakdown confirmed by high volume."

- **HOLD**: 조건 중 하나라도 미충족 시
  - 특수 케이스: Price < 20MA + Wick >= 0.5 이지만 RVOL < 2.0인 경우
  - Reasoning: "Pattern detected but lacks volume confirmation (Potential Trap)."

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. 기본 사용법

```python
from exit_signal_screener import ExitSignalScreener, load_data_from_csv

# CSV 파일 로드 (필수 컬럼: Date, Open, High, Low, Close, Volume)
df = load_data_from_csv('your_data.csv', date_column='Date')

# 스크리너 초기화
screener = ExitSignalScreener(ma_period=20, rvol_period=20)

# 지표 계산 및 필터 적용
filtered_data = screener.apply_filters(df)

# 스크리닝 결과 생성
output = screener.generate_screening_output(filtered_data, ticker="AAPL")

# SELL 신호만 추출
sell_signals = output[output['Signal'] == 'SELL']
print(sell_signals)
```

### 2. 백테스팅 요약

```python
# 백테스트 통계 확인
summary = screener.backtest_summary(filtered_data)
for key, value in summary.items():
    print(f"{key}: {value}")
```

### 3. 결과 저장

```python
# CSV 파일로 저장
output.to_csv('screening_results.csv', index=False, encoding='utf-8-sig')
```

## File Structure

```
taking_profit_screener/
├── exit_signal_screener.py   # 메인 스크리너 클래스
├── parameter_optimizer.py     # 파라미터 최적화 클래스
├── example_usage.py           # 스크리너 사용 예시
├── optimization_examples.py   # 최적화 사용 예시
├── requirements.txt           # 필요 라이브러리
└── README.md                  # 문서
```

## Input Data Format

CSV 파일은 다음 컬럼을 포함해야 합니다:

```csv
Date,Open,High,Low,Close,Volume
2024-01-01,100.5,102.3,99.8,101.2,1500000
2024-01-02,101.2,103.0,100.5,102.5,2000000
...
```

## Output Format

스크리닝 결과는 다음 형식으로 출력됩니다:

```
Ticker, Date, Current_Price, MA20, Wick_Ratio, RVOL, Signal, Reasoning
```

### 출력 예시:

```
Ticker: AAPL
Date: 2024-03-15
Current_Price: 175.50
MA20: 180.20
Wick_Ratio: 0.65
RVOL: 2.3
Signal: SELL
Reasoning: Valid technical breakdown confirmed by high volume.
```

## Key Features

### 1. 지표 계산 함수

- `calculate_sma()`: 단순 이동평균선
- `calculate_wick_ratio()`: 상단 윅 비율
- `calculate_rvol()`: 상대 거래량

### 2. 필터링

- `apply_filters()`: 모든 조건 체크 및 시그널 생성

### 3. 백테스팅

- `backtest_summary()`: 조건별 충족률, 시그널 발생 빈도 등

## Examples

[example_usage.py](example_usage.py)에서 다양한 사용 예시를 확인할 수 있습니다:

1. **Example 1**: 기본 사용법
2. **Example 2**: 여러 종목 동시 스크리닝
3. **Example 3**: 커스텀 파라미터 (MA50, RVOL30 등)
4. **Example 4**: 조건별 상세 분석
5. **Example 5**: CSV 결과 저장

실행:
```bash
python example_usage.py
```

## Advanced Usage

### 커스텀 파라미터

```python
# 50일 이동평균, 30일 RVOL 사용
screener = ExitSignalScreener(ma_period=50, rvol_period=30)
```

### 여러 종목 동시 분석

```python
tickers = ['AAPL', 'MSFT', 'GOOGL']
all_results = []

for ticker in tickers:
    df = load_data_from_csv(f'{ticker}_data.csv', date_column='Date')
    filtered_data = screener.apply_filters(df)
    output = screener.generate_screening_output(filtered_data, ticker=ticker)
    all_results.append(output[output['Signal'] == 'SELL'])

combined = pd.concat(all_results, ignore_index=True)
```

## Parameter Optimization (파라미터 최적화)

그리드 서치 외에도 다양한 최적화 방법을 제공합니다. 각 방법의 장단점을 고려하여 선택하세요.

### 최적화 가능한 파라미터

- `ma_period`: 이동평균선 기간 (10~100)
- `rvol_period`: 상대 거래량 계산 기간 (10~50)
- `wick_threshold`: 윅 비율 임계값 (0.3~0.8)
- `rvol_threshold`: RVOL 임계값 (1.5~4.0)

### 평가 지표

- `sharpe_ratio`: 샤프 비율 (위험 대비 수익률) - **추천**
- `total_return`: 총 수익률
- `win_rate`: 승률 (SELL 신호의 정확도)
- `profit_factor`: Profit Factor (총 이익 / 총 손실)

### 1. Random Search (랜덤 서치)

**장점**: 간단하고 빠름, 그리드 서치보다 효율적

```python
from parameter_optimizer import ParameterOptimizer

# 옵티마이저 초기화
optimizer = ParameterOptimizer(df, ticker="AAPL", evaluation_metric='sharpe_ratio')

# Random Search 실행
best_params, best_score = optimizer.random_search(
    n_iterations=50,
    param_ranges={
        'ma_period': (15, 60),
        'rvol_period': (10, 30),
        'wick_threshold': (0.4, 0.7),
        'rvol_threshold': (1.5, 3.5)
    }
)

print(f"Best Sharpe Ratio: {best_score:.4f}")
print(f"Optimal Parameters: {best_params}")
```

### 2. Bayesian Optimization (베이지안 최적화)

**장점**: 이전 결과를 학습하여 효율적 탐색, 적은 시도로 좋은 결과

```python
# Bayesian Optimization 실행
best_params, best_score = optimizer.bayesian_optimization(
    n_iterations=40,
    n_initial_points=10
)
```

### 3. Genetic Algorithm (유전 알고리즘)

**장점**: 전역 최적화에 강함, local minima 회피

```python
# Genetic Algorithm 실행
best_params, best_score = optimizer.genetic_algorithm(
    population_size=20,
    n_generations=15,
    mutation_rate=0.2
)
```

### 4. Walk-Forward Optimization (워크포워드 최적화)

**장점**: 과최적화 방지, 실전에 가장 가까운 방법 - **강력 추천**

```python
# Walk-Forward Optimization 실행
results = optimizer.walk_forward_optimization(
    train_window=252,      # 1년 학습
    test_window=63,        # 3개월 테스트
    optimization_method='random_search',  # 'random_search', 'bayesian', 'genetic'
    n_iterations=30
)

# 각 윈도우별 결과 확인
for r in results:
    print(f"Window {r['window']}: Train={r['train_score']:.4f}, Test={r['test_score']:.4f}")
```

### 최적화 히스토리 저장

```python
# 모든 시도 기록을 CSV로 저장
optimizer.export_optimization_history('optimization_history.csv')
```

### 실행 예시

```bash
# 다양한 최적화 예시 실행
python optimization_examples.py
```

[optimization_examples.py](optimization_examples.py)에서 8가지 최적화 예시를 확인할 수 있습니다:
1. Random Search 기본 사용
2. Bayesian Optimization
3. Genetic Algorithm
4. Walk-Forward Optimization
5. 여러 방법 비교
6. 다양한 평가 지표 사용
7. 최적화 히스토리 분석
8. 특정 파라미터만 튜닝

### 방법 선택 가이드

| 상황 | 추천 방법 | 이유 |
|------|-----------|------|
| 빠른 프로토타이핑 | Random Search | 간단하고 빠름 |
| 적은 시도로 최적화 | Bayesian Optimization | 효율적인 탐색 |
| 복잡한 탐색 공간 | Genetic Algorithm | 전역 최적화 |
| 실전 트레이딩 준비 | Walk-Forward | 과최적화 방지 |
| 확신이 없을 때 | Walk-Forward + Random Search | 안정적이고 빠름 |

## Technical Details

### Upper Wick Calculation

```
Upper Wick = High - max(Open, Close)
Total Candle Length = High - Low
Wick Ratio = Upper Wick / Total Candle Length
```

### RVOL Calculation

```
RVOL = Current Volume / Average Volume (20 periods)
```

## Notes

- 도지 캔들 (High = Low)의 경우 Wick Ratio는 0으로 처리됩니다
- 초기 N일은 MA 및 RVOL 계산을 위해 NaN 값을 가질 수 있습니다
- 모든 조건은 AND 연산으로 결합됩니다 (엄격한 필터링)

## License

MIT License

## Contact

IGIS Asset Management
