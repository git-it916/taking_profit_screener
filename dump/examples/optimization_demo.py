"""
파라미터 최적화 사용 예시
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src import ParameterOptimizer, load_data_from_csv
import matplotlib.pyplot as plt


def generate_realistic_data(days=500):
    """테스트용 현실적인 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=days, freq='D')

    # 트렌드 + 노이즈
    trend = np.linspace(100, 120, days)
    noise = np.random.randn(days).cumsum() * 0.5
    close_prices = trend + noise

    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(days) * 0.5,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, days)
    })

    # High와 Low 추가 (현실적인 범위)
    df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.randn(days) * 1.5)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.randn(days) * 1.5)

    df.index = dates
    return df


def example_1_random_search():
    """예시 1: Random Search - 가장 간단하고 빠른 방법"""
    print("\n" + "="*80)
    print("Example 1: Random Search Optimization")
    print("="*80)
    print("\n랜덤 서치는 그리드 서치보다 효율적이면서도 구현이 간단합니다.")
    print("적은 시도로 좋은 결과를 찾을 수 있어 빠른 프로토타이핑에 적합합니다.\n")

    df = generate_realistic_data(days=300)

    # Sharpe Ratio 최대화
    optimizer = ParameterOptimizer(df, ticker="EXAMPLE1", evaluation_metric='sharpe_ratio')

    # 커스텀 파라미터 범위 지정
    param_ranges = {
        'ma_period': (15, 60),           # MA 15~60일
        'rvol_period': (10, 30),         # RVOL 10~30일
        'wick_threshold': (0.4, 0.7),    # 윅 비율 0.4~0.7
        'rvol_threshold': (1.5, 3.5)     # RVOL 임계값 1.5~3.5
    }

    best_params, best_score = optimizer.random_search(
        n_iterations=50,
        param_ranges=param_ranges
    )

    print(f"\n최적 파라미터:")
    print(f"  MA Period: {best_params['ma_period']}")
    print(f"  RVOL Period: {best_params['rvol_period']}")
    print(f"  Wick Threshold: {best_params['wick_threshold']:.3f}")
    print(f"  RVOL Threshold: {best_params['rvol_threshold']:.3f}")
    print(f"  Sharpe Ratio: {best_score:.4f}")


def example_2_bayesian_optimization():
    """예시 2: Bayesian Optimization - 효율적인 탐색"""
    print("\n" + "="*80)
    print("Example 2: Bayesian Optimization")
    print("="*80)
    print("\n베이지안 최적화는 이전 결과를 학습하여 다음 탐색 지점을 똑똑하게 선택합니다.")
    print("적은 반복으로 최적해에 가까운 결과를 찾을 수 있습니다.\n")

    df = generate_realistic_data(days=300)

    # Win Rate 최대화
    optimizer = ParameterOptimizer(df, ticker="EXAMPLE2", evaluation_metric='win_rate')

    best_params, best_score = optimizer.bayesian_optimization(
        n_iterations=40,
        n_initial_points=10
    )

    print(f"\n최적 파라미터:")
    print(f"  MA Period: {best_params['ma_period']}")
    print(f"  RVOL Period: {best_params['rvol_period']}")
    print(f"  Wick Threshold: {best_params['wick_threshold']:.3f}")
    print(f"  RVOL Threshold: {best_params['rvol_threshold']:.3f}")
    print(f"  Win Rate: {best_score:.4f}")


def example_3_genetic_algorithm():
    """예시 3: Genetic Algorithm - 전역 최적화"""
    print("\n" + "="*80)
    print("Example 3: Genetic Algorithm")
    print("="*80)
    print("\n유전 알고리즘은 생물의 진화를 모방하여 전역 최적해를 찾습니다.")
    print("복잡한 탐색 공간에서도 local minima에 빠지지 않고 좋은 해를 찾을 수 있습니다.\n")

    df = generate_realistic_data(days=300)

    # Profit Factor 최대화
    optimizer = ParameterOptimizer(df, ticker="EXAMPLE3", evaluation_metric='profit_factor')

    best_params, best_score = optimizer.genetic_algorithm(
        population_size=20,
        n_generations=15,
        mutation_rate=0.2
    )

    print(f"\n최적 파라미터:")
    print(f"  MA Period: {best_params['ma_period']}")
    print(f"  RVOL Period: {best_params['rvol_period']}")
    print(f"  Wick Threshold: {best_params['wick_threshold']:.3f}")
    print(f"  RVOL Threshold: {best_params['rvol_threshold']:.3f}")
    print(f"  Profit Factor: {best_score:.4f}")


def example_4_walk_forward():
    """예시 4: Walk-Forward Optimization - 과최적화 방지"""
    print("\n" + "="*80)
    print("Example 4: Walk-Forward Optimization")
    print("="*80)
    print("\n워크포워드 최적화는 과최적화를 방지하는 가장 강력한 방법입니다.")
    print("학습 기간에서 최적화하고, 테스트 기간에서 검증하는 과정을 반복합니다.")
    print("실제 트레이딩 환경과 가장 유사한 방식입니다.\n")

    df = generate_realistic_data(days=500)

    optimizer = ParameterOptimizer(df, ticker="EXAMPLE4", evaluation_metric='sharpe_ratio')

    results = optimizer.walk_forward_optimization(
        train_window=252,      # 1년 학습
        test_window=63,        # 3개월 테스트
        optimization_method='random_search',  # 'random_search', 'bayesian', 'genetic'
        n_iterations=30
    )

    # 각 윈도우의 최적 파라미터 확인
    print("\n각 윈도우별 최적 파라미터:")
    for r in results:
        params = r['best_params']
        print(f"\nWindow {r['window']}:")
        print(f"  MA: {params['ma_period']}, RVOL_Period: {params['rvol_period']}, "
              f"Wick: {params['wick_threshold']:.2f}, RVOL_Thresh: {params['rvol_threshold']:.2f}")
        print(f"  Train Score: {r['train_score']:.4f}, Test Score: {r['test_score']:.4f}")


def example_5_compare_methods():
    """예시 5: 여러 방법 비교"""
    print("\n" + "="*80)
    print("Example 5: Compare Different Optimization Methods")
    print("="*80)
    print("\n동일한 데이터에서 여러 최적화 방법을 비교합니다.\n")

    df = generate_realistic_data(days=300)

    results = {}

    # 1. Random Search
    print("\n[1/3] Running Random Search...")
    optimizer = ParameterOptimizer(df, ticker="COMPARE", evaluation_metric='sharpe_ratio')
    params_random, score_random = optimizer.random_search(n_iterations=30)
    results['Random Search'] = {'params': params_random, 'score': score_random}

    # 2. Bayesian Optimization
    print("\n[2/3] Running Bayesian Optimization...")
    optimizer = ParameterOptimizer(df, ticker="COMPARE", evaluation_metric='sharpe_ratio')
    params_bayes, score_bayes = optimizer.bayesian_optimization(n_iterations=30, n_initial_points=10)
    results['Bayesian'] = {'params': params_bayes, 'score': score_bayes}

    # 3. Genetic Algorithm
    print("\n[3/3] Running Genetic Algorithm...")
    optimizer = ParameterOptimizer(df, ticker="COMPARE", evaluation_metric='sharpe_ratio')
    params_genetic, score_genetic = optimizer.genetic_algorithm(population_size=15, n_generations=10)
    results['Genetic'] = {'params': params_genetic, 'score': score_genetic}

    # 결과 비교
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    for method, data in results.items():
        print(f"\n{method}:")
        print(f"  Score (Sharpe Ratio): {data['score']:.4f}")
        print(f"  MA Period: {data['params']['ma_period']}")
        print(f"  RVOL Period: {data['params']['rvol_period']}")
        print(f"  Wick Threshold: {data['params']['wick_threshold']:.3f}")
        print(f"  RVOL Threshold: {data['params']['rvol_threshold']:.3f}")

    # 최고 방법 찾기
    best_method = max(results.items(), key=lambda x: x[1]['score'])
    print(f"\n최고 성능: {best_method[0]} (Score: {best_method[1]['score']:.4f})")


def example_6_different_metrics():
    """예시 6: 다양한 평가 지표 사용"""
    print("\n" + "="*80)
    print("Example 6: Optimization with Different Metrics")
    print("="*80)
    print("\n목적에 따라 다른 평가 지표를 사용할 수 있습니다:")
    print("- sharpe_ratio: 위험 대비 수익률")
    print("- total_return: 총 수익률")
    print("- win_rate: 승률 (SELL 신호가 정확한 비율)")
    print("- profit_factor: 총 이익 / 총 손실\n")

    df = generate_realistic_data(days=300)

    metrics = ['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor']

    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"Optimizing for: {metric}")
        print(f"{'='*60}")

        optimizer = ParameterOptimizer(df, ticker="METRIC_TEST", evaluation_metric=metric)
        best_params, best_score = optimizer.random_search(n_iterations=20)

        print(f"Best {metric}: {best_score:.4f}")
        print(f"Parameters: MA={best_params['ma_period']}, RVOL_Period={best_params['rvol_period']}, "
              f"Wick={best_params['wick_threshold']:.2f}, RVOL_Thresh={best_params['rvol_threshold']:.2f}")


def example_7_export_history():
    """예시 7: 최적화 히스토리 저장 및 분석"""
    print("\n" + "="*80)
    print("Example 7: Export and Analyze Optimization History")
    print("="*80)
    print("\n최적화 과정의 모든 시도를 저장하여 나중에 분석할 수 있습니다.\n")

    df = generate_realistic_data(days=300)

    optimizer = ParameterOptimizer(df, ticker="HISTORY", evaluation_metric='sharpe_ratio')
    best_params, best_score = optimizer.random_search(n_iterations=50)

    # 히스토리 저장
    optimizer.export_optimization_history('my_optimization_history.csv')

    # 히스토리 분석
    history_df = pd.DataFrame(optimizer.optimization_history)

    print("\n[Optimization History Statistics]")
    print(f"Total trials: {len(history_df)}")
    print(f"Best score: {history_df['score'].max():.4f}")
    print(f"Worst score: {history_df['score'].min():.4f}")
    print(f"Average score: {history_df['score'].mean():.4f}")
    print(f"Std deviation: {history_df['score'].std():.4f}")

    # 파라미터별 상관관계 분석
    print("\n[Parameter Correlation with Score]")
    for param in ['ma_period', 'rvol_period', 'wick_threshold', 'rvol_threshold']:
        corr = history_df[param].corr(history_df['score'])
        print(f"  {param}: {corr:.3f}")


def example_8_custom_ranges():
    """예시 8: 특정 파라미터만 튜닝"""
    print("\n" + "="*80)
    print("Example 8: Optimize Specific Parameters Only")
    print("="*80)
    print("\n특정 파라미터는 고정하고 일부만 튜닝할 수 있습니다.\n")

    df = generate_realistic_data(days=300)

    # MA는 20으로 고정, 다른 파라미터만 튜닝
    param_ranges = {
        'ma_period': (20, 21),           # 사실상 고정
        'rvol_period': (10, 30),
        'wick_threshold': (0.4, 0.7),
        'rvol_threshold': (1.5, 3.5)
    }

    optimizer = ParameterOptimizer(df, ticker="CUSTOM", evaluation_metric='sharpe_ratio')
    best_params, best_score = optimizer.random_search(
        n_iterations=30,
        param_ranges=param_ranges
    )

    print(f"\n최적 파라미터 (MA는 20으로 고정):")
    print(f"  MA Period: {best_params['ma_period']} (fixed)")
    print(f"  RVOL Period: {best_params['rvol_period']}")
    print(f"  Wick Threshold: {best_params['wick_threshold']:.3f}")
    print(f"  RVOL Threshold: {best_params['rvol_threshold']:.3f}")
    print(f"  Score: {best_score:.4f}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PARAMETER OPTIMIZATION EXAMPLES")
    print("="*80)
    print("\n그리드 서치 외에 다양한 최적화 방법을 사용할 수 있습니다:")
    print("1. Random Search - 간단하고 빠름")
    print("2. Bayesian Optimization - 효율적인 탐색")
    print("3. Genetic Algorithm - 전역 최적화")
    print("4. Walk-Forward - 과최적화 방지")
    print("\n실행할 예시를 선택하거나, 전체 실행을 위해 코드를 수정하세요.\n")

    # 원하는 예시만 주석 해제하여 실행
    example_1_random_search()
    # example_2_bayesian_optimization()
    # example_3_genetic_algorithm()
    # example_4_walk_forward()
    # example_5_compare_methods()
    # example_6_different_metrics()
    # example_7_export_history()
    # example_8_custom_ranges()

    print("\n" + "="*80)
    print("EXAMPLES COMPLETED!")
    print("="*80)
