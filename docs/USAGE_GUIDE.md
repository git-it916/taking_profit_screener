# 사용법 가이드 - Taking Profit Screener

## 목차
1. [기본 스크리닝 (파라미터 튜닝 없이)](#1-기본-스크리닝)
2. [파라미터 최적화 - 빠른 방법](#2-파라미터-최적화---빠른-방법)
3. [파라미터 최적화 - 실전용](#3-파라미터-최적화---실전용)
4. [자주 묻는 질문](#자주-묻는-질문)

---

## 1. 기본 스크리닝

**목적**: 기본 파라미터로 SELL 신호 찾기 (파라미터 튜닝 X)

### 단계 1: 코드 작성

`my_basic_screening.py` 파일을 만들고 아래 코드를 복사하세요:

```python
from exit_signal_screener import ExitSignalScreener, load_data_from_csv

# 1. CSV 파일 로드
df = load_data_from_csv('your_data.csv', date_column='Date')

# 2. 스크리너 초기화 (기본 파라미터: MA20, RVOL20, Wick>=0.5, RVOL>=2.0)
screener = ExitSignalScreener(ma_period=20, rvol_period=20)

# 3. 필터 적용
filtered_data = screener.apply_filters(df)

# 4. 결과 생성
output = screener.generate_screening_output(filtered_data, ticker="MY_STOCK")

# 5. SELL 신호만 추출
sell_signals = output[output['Signal'] == 'SELL']

# 6. 결과 출력
print("\n=== SELL 신호 ===")
print(sell_signals[['Date', 'Current_Price', 'MA20', 'Wick_Ratio', 'RVOL', 'Reasoning']])

# 7. CSV로 저장
sell_signals.to_csv('sell_signals.csv', index=False, encoding='utf-8-sig')
print(f"\n결과 저장 완료: sell_signals.csv ({len(sell_signals)}개 신호)")
```

### 단계 2: 실행

```bash
python my_basic_screening.py
```

**결과**: `sell_signals.csv` 파일이 생성되고 SELL 신호가 화면에 출력됩니다.

---

## 2. 파라미터 최적화 - 빠른 방법

**목적**: Random Search로 빠르게 최적 파라미터 찾기 (추천: 프로토타이핑)

### 단계 1: 코드 작성

`my_quick_optimization.py` 파일을 만들고 아래 코드를 복사하세요:

```python
from parameter_optimizer import ParameterOptimizer
from exit_signal_screener import load_data_from_csv

# 1. 데이터 로드
df = load_data_from_csv('your_data.csv', date_column='Date')

# 2. 옵티마이저 초기화
optimizer = ParameterOptimizer(
    df,
    ticker="MY_STOCK",
    evaluation_metric='sharpe_ratio'  # 샤프 비율로 평가
)

# 3. Random Search 실행 (50번 시도)
print("\n최적화 시작... (약 10초 소요)\n")
best_params, best_score = optimizer.random_search(
    n_iterations=50,
    param_ranges={
        'ma_period': (15, 60),          # MA 15~60일
        'rvol_period': (10, 30),        # RVOL 10~30일
        'wick_threshold': (0.4, 0.7),   # 윅 비율 0.4~0.7
        'rvol_threshold': (1.5, 3.5)    # RVOL 임계값 1.5~3.5
    }
)

# 4. 결과 출력
print("\n" + "="*80)
print("최적화 완료!")
print("="*80)
print(f"\n최적 파라미터:")
print(f"  • MA Period: {best_params['ma_period']}일")
print(f"  • RVOL Period: {best_params['rvol_period']}일")
print(f"  • Wick Threshold: {best_params['wick_threshold']:.3f}")
print(f"  • RVOL Threshold: {best_params['rvol_threshold']:.3f}")
print(f"\n샤프 비율: {best_score:.4f}")

# 5. 최적화 히스토리 저장
optimizer.export_optimization_history('optimization_history.csv')
print("\n모든 시도 기록이 'optimization_history.csv'에 저장되었습니다.")
```

### 단계 2: 실행

```bash
python my_quick_optimization.py
```

**결과**:
- 화면에 최적 파라미터 출력
- `optimization_history.csv`에 모든 시도 기록 저장

---

## 3. 파라미터 최적화 - 실전용

**목적**: Walk-Forward Optimization으로 과최적화 방지 (추천: 실제 트레이딩)

### 단계 1: 코드 작성

`my_production_optimization.py` 파일을 만들고 아래 코드를 복사하세요:

```python
from parameter_optimizer import ParameterOptimizer
from exit_signal_screener import load_data_from_csv

# 1. 데이터 로드 (최소 1년 이상 데이터 필요)
df = load_data_from_csv('your_data.csv', date_column='Date')

print(f"데이터 기간: {len(df)}일")
print(f"시작일: {df.index[0]}, 종료일: {df.index[-1]}\n")

# 2. 옵티마이저 초기화
optimizer = ParameterOptimizer(
    df,
    ticker="MY_STOCK",
    evaluation_metric='sharpe_ratio'
)

# 3. Walk-Forward Optimization 실행
print("Walk-Forward Optimization 시작...")
print("(학습 1년 → 테스트 3개월 반복)\n")

results = optimizer.walk_forward_optimization(
    train_window=252,      # 1년 학습 (252 거래일)
    test_window=63,        # 3개월 테스트 (63 거래일)
    optimization_method='random_search',  # 'random_search', 'bayesian', 'genetic' 중 선택
    n_iterations=30        # 각 윈도우에서 30번 시도
)

# 4. 결과 요약
print("\n" + "="*80)
print("최종 결과 요약")
print("="*80)

best_window = max(results, key=lambda x: x['test_score'])

print(f"\n가장 좋은 성능을 보인 윈도우: {best_window['window']}")
print(f"  테스트 점수: {best_window['test_score']:.4f}")
print(f"\n추천 파라미터:")
params = best_window['best_params']
print(f"  • MA Period: {params['ma_period']}일")
print(f"  • RVOL Period: {params['rvol_period']}일")
print(f"  • Wick Threshold: {params['wick_threshold']:.3f}")
print(f"  • RVOL Threshold: {params['rvol_threshold']:.3f}")

# 5. 각 윈도우별 상세 결과
print("\n" + "="*80)
print("윈도우별 상세 결과")
print("="*80)
for r in results:
    print(f"\nWindow {r['window']}:")
    print(f"  Train Score: {r['train_score']:.4f}")
    print(f"  Test Score: {r['test_score']:.4f}")
    print(f"  Difference: {r['test_score'] - r['train_score']:.4f}")
```

### 단계 2: 실행

```bash
python my_production_optimization.py
```

**결과**:
- 각 윈도우별 Train/Test 성능 비교
- 과최적화 여부 확인 가능
- 실전에 사용할 최적 파라미터 추천

---

## 4. 최적화된 파라미터로 스크리닝하기

최적화로 찾은 파라미터를 사용해서 실제 스크리닝을 하는 방법:

```python
from exit_signal_screener import ExitSignalScreener, load_data_from_csv

# 1. 데이터 로드
df = load_data_from_csv('your_data.csv', date_column='Date')

# 2. 최적화로 찾은 파라미터 사용
screener = ExitSignalScreener(
    ma_period=35,      # 최적화로 찾은 값
    rvol_period=22     # 최적화로 찾은 값
)

# 3. 필터 적용
filtered_data = screener.apply_filters(df)

# 4. 임계값 커스터마이징 (최적화로 찾은 값 적용)
wick_threshold = 0.55   # 최적화로 찾은 값
rvol_threshold = 2.3    # 최적화로 찾은 값

filtered_data['Custom_Condition_2'] = filtered_data['Wick_Ratio'] >= wick_threshold
filtered_data['Custom_Condition_3'] = filtered_data['RVOL'] >= rvol_threshold

filtered_data['Custom_Signal'] = (
    filtered_data['Condition_1_Trend_Breakdown'] &
    filtered_data['Custom_Condition_2'] &
    filtered_data['Custom_Condition_3']
).apply(lambda x: 'SELL' if x else 'HOLD')

# 5. SELL 신호 추출
sell_signals = filtered_data[filtered_data['Custom_Signal'] == 'SELL']

print(f"SELL 신호 {len(sell_signals)}개 발견!")
print(sell_signals[['Close', 'MA20', 'Wick_Ratio', 'RVOL']])
```

---

## 자주 묻는 질문

### Q1: 어떤 방법을 선택해야 하나요?

| 상황 | 추천 방법 |
|------|-----------|
| 처음 사용해보는 경우 | **기본 스크리닝** (1번) |
| 빠르게 파라미터 튜닝하고 싶을 때 | **Random Search** (2번) |
| 실제 트레이딩에 사용할 때 | **Walk-Forward** (3번) ⭐ 필수! |

### Q2: 데이터가 얼마나 필요한가요?

- **기본 스크리닝**: 최소 40일 (MA20 + RVOL20 계산용)
- **Random Search**: 100일 이상 권장
- **Walk-Forward**: 500일 이상 권장 (1년 학습 + 3개월 테스트 반복)

### Q3: 평가 지표는 무엇을 써야 하나요?

```python
evaluation_metric='sharpe_ratio'  # 추천! 위험 대비 수익률
evaluation_metric='win_rate'      # SELL 신호의 정확도
evaluation_metric='total_return'  # 총 수익률
evaluation_metric='profit_factor' # 총 이익 / 총 손실
```

**추천**: `sharpe_ratio` - 위험을 고려한 수익률로 가장 균형잡힌 지표

### Q4: 최적화 시간이 얼마나 걸리나요?

- **Random Search (50회)**: 약 10~20초
- **Bayesian Optimization (40회)**: 약 15~30초
- **Genetic Algorithm (20세대)**: 약 30~60초
- **Walk-Forward (3윈도우)**: 약 1~3분

### Q5: 파라미터 범위는 어떻게 정하나요?

**기본 추천 범위**:
```python
param_ranges = {
    'ma_period': (15, 60),         # 단기~중기 추세
    'rvol_period': (10, 30),       # 거래량 평균 계산 기간
    'wick_threshold': (0.4, 0.7),  # 윅 비율 (너무 낮으면 잡음, 너무 높으면 신호 없음)
    'rvol_threshold': (1.5, 3.5)   # RVOL (너무 낮으면 신호 과다, 너무 높으면 신호 없음)
}
```

특정 종목 특성에 맞게 조정 가능합니다.

### Q6: CSV 파일이 없는데 어떻게 하나요?

예시 데이터 생성 코드:
```python
import pandas as pd
import numpy as np

# 예시 데이터 생성
dates = pd.date_range('2023-01-01', periods=300, freq='D')
df = pd.DataFrame({
    'Date': dates,
    'Open': 100 + np.random.randn(300).cumsum(),
    'High': 102 + np.random.randn(300).cumsum(),
    'Low': 98 + np.random.randn(300).cumsum(),
    'Close': 100 + np.random.randn(300).cumsum(),
    'Volume': np.random.randint(1000000, 5000000, 300)
})

# High/Low 보정
df['High'] = df[['Open', 'High', 'Close']].max(axis=1) + 1
df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) - 1

# CSV 저장
df.to_csv('sample_data.csv', index=False)
print("sample_data.csv 생성 완료!")
```

---

## 빠른 테스트용 - 올인원 코드

데이터 생성부터 최적화까지 한번에 실행:

```python
import pandas as pd
import numpy as np
from parameter_optimizer import ParameterOptimizer

# 1. 테스트 데이터 생성
print("테스트 데이터 생성 중...")
dates = pd.date_range('2023-01-01', periods=300, freq='D')
trend = np.linspace(100, 120, 300)
noise = np.random.randn(300).cumsum() * 0.5
close_prices = trend + noise

df = pd.DataFrame({
    'Open': close_prices + np.random.randn(300) * 0.5,
    'Close': close_prices,
    'Volume': np.random.randint(1000000, 5000000, 300)
})
df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.randn(300) * 1.5)
df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.randn(300) * 1.5)
df.index = dates

# 2. 최적화 실행
print("\n최적화 시작!\n")
optimizer = ParameterOptimizer(df, ticker="TEST", evaluation_metric='sharpe_ratio')
best_params, best_score = optimizer.random_search(n_iterations=30)

# 3. 결과 출력
print("\n" + "="*80)
print("✓ 최적화 완료!")
print("="*80)
print(f"\n최적 파라미터:")
print(f"  MA Period: {best_params['ma_period']}")
print(f"  RVOL Period: {best_params['rvol_period']}")
print(f"  Wick Threshold: {best_params['wick_threshold']:.3f}")
print(f"  RVOL Threshold: {best_params['rvol_threshold']:.3f}")
print(f"\nSharpe Ratio: {best_score:.4f}")
```

위 코드를 `test_all.py`로 저장하고 실행:
```bash
python test_all.py
```

---

## 요약

1. **처음 사용**: 기본 스크리닝으로 시작 → SELL 신호 확인
2. **파라미터 튜닝**: Random Search로 빠르게 최적 파라미터 찾기
3. **실전 배포**: Walk-Forward로 과최적화 방지하고 안정적인 파라미터 결정

**가장 중요한 것**: 실제 트레이딩에 사용하려면 **Walk-Forward Optimization**을 꼭 사용하세요!
