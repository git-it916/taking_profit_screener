"""
빠른 최적화 비교 테스트
3가지 방법을 빠르게 비교합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src import ParameterOptimizer
import time


def generate_test_data(days=300):
    """테스트 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=days, freq='D')

    # 현실적인 가격 데이터
    trend = np.linspace(100, 120, days)
    noise = np.random.randn(days).cumsum() * 0.5
    close_prices = trend + noise

    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(days) * 0.5,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, days)
    })

    df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.randn(days) * 1.5)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.randn(days) * 1.5)
    df.index = dates

    return df


def quick_comparison():
    """3가지 방법 빠른 비교"""
    print("\n" + "="*80)
    print("QUICK PARAMETER OPTIMIZATION COMPARISON")
    print("="*80)

    df = generate_test_data(days=300)

    results = {}
    timings = {}

    # 1. Random Search
    print("\n[1/3] Running Random Search...")
    start = time.time()
    optimizer = ParameterOptimizer(df, ticker="TEST", evaluation_metric='sharpe_ratio')
    params_random, score_random = optimizer.random_search(n_iterations=20)
    timings['Random Search'] = time.time() - start
    results['Random Search'] = {'params': params_random, 'score': score_random}

    # 2. Bayesian Optimization
    print("\n[2/3] Running Bayesian Optimization...")
    start = time.time()
    optimizer = ParameterOptimizer(df, ticker="TEST", evaluation_metric='sharpe_ratio')
    params_bayes, score_bayes = optimizer.bayesian_optimization(n_iterations=20, n_initial_points=5)
    timings['Bayesian'] = time.time() - start
    results['Bayesian'] = {'params': params_bayes, 'score': score_bayes}

    # 3. Genetic Algorithm
    print("\n[3/3] Running Genetic Algorithm...")
    start = time.time()
    optimizer = ParameterOptimizer(df, ticker="TEST", evaluation_metric='sharpe_ratio')
    params_genetic, score_genetic = optimizer.genetic_algorithm(population_size=10, n_generations=5)
    timings['Genetic'] = time.time() - start
    results['Genetic'] = {'params': params_genetic, 'score': score_genetic}

    # 결과 출력
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    print(f"\n{'Method':<20} {'Score':<12} {'Time (s)':<12} {'MA':<6} {'RVOL_P':<8} {'Wick':<8} {'RVOL_T':<8}")
    print("-"*80)

    for method in ['Random Search', 'Bayesian', 'Genetic']:
        data = results[method]
        p = data['params']
        print(f"{method:<20} {data['score']:<12.4f} {timings[method]:<12.2f} "
              f"{p['ma_period']:<6} {p['rvol_period']:<8} "
              f"{p['wick_threshold']:<8.3f} {p['rvol_threshold']:<8.3f}")

    # 최고 점수 찾기
    best_method = max(results.items(), key=lambda x: x[1]['score'])
    fastest_method = min(timings.items(), key=lambda x: x[1])

    print("\n" + "="*80)
    print(f"최고 성능: {best_method[0]} (Score: {best_method[1]['score']:.4f})")
    print(f"가장 빠름: {fastest_method[0]} (Time: {fastest_method[1]:.2f}s)")
    print("="*80)

    # 추천
    print("\n[Recommendation]")
    print(f"- Quick Prototyping: Random Search")
    print(f"- Best Performance: {best_method[0]}")
    print(f"- Production Use: Walk-Forward Optimization (prevents overfitting)")


if __name__ == "__main__":
    quick_comparison()
