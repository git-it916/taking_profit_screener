"""
Exit Signal Screener 사용 예시
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src import ExitSignalScreener, load_data_from_csv


def example_1_basic_usage():
    """기본 사용법 - CSV 파일에서 데이터 로드"""
    print("\n" + "="*80)
    print("Example 1: Basic Usage with CSV File")
    print("="*80)

    # CSV 파일 로드 (파일명을 실제 파일로 변경하세요)
    # df = load_data_from_csv('your_data.csv', date_column='Date')

    # 예시를 위한 샘플 데이터 생성
    df = generate_sample_data(days=60)

    # 스크리너 초기화
    screener = ExitSignalScreener(ma_period=20, rvol_period=20)

    # 지표 계산 및 필터 적용
    filtered_data = screener.apply_filters(df)

    # 결과 출력 형식으로 변환
    output = screener.generate_screening_output(filtered_data, ticker="AAPL")

    # SELL 신호만 추출
    sell_signals = output[output['Signal'] == 'SELL']

    print("\n[SELL Signals Detected]")
    print(sell_signals[['Date', 'Current_Price', 'MA20', 'Wick_Ratio', 'RVOL', 'Signal', 'Reasoning']])

    # 요약 통계
    print("\n[Summary Statistics]")
    summary = screener.backtest_summary(filtered_data)
    for key, value in summary.items():
        print(f"  {key}: {value}")


def example_2_multiple_tickers():
    """여러 종목 동시 스크리닝"""
    print("\n" + "="*80)
    print("Example 2: Multiple Tickers Screening")
    print("="*80)

    tickers = ['AAPL', 'MSFT', 'GOOGL']
    screener = ExitSignalScreener()

    all_results = []

    for ticker in tickers:
        # 실제로는 각 종목의 데이터를 로드
        # df = load_data_from_csv(f'{ticker}_data.csv', date_column='Date')

        df = generate_sample_data(days=60)
        filtered_data = screener.apply_filters(df)
        output = screener.generate_screening_output(filtered_data, ticker=ticker)

        # SELL 신호만 수집
        sell_signals = output[output['Signal'] == 'SELL']
        all_results.append(sell_signals)

    # 모든 결과 합치기
    combined_results = pd.concat(all_results, ignore_index=True)

    print(f"\nTotal SELL signals across {len(tickers)} tickers: {len(combined_results)}")
    print("\n[Combined Results]")
    print(combined_results[['Ticker', 'Date', 'Current_Price', 'MA20', 'Wick_Ratio', 'RVOL']])


def example_3_custom_parameters():
    """커스텀 파라미터 사용"""
    print("\n" + "="*80)
    print("Example 3: Custom Parameters")
    print("="*80)

    df = generate_sample_data(days=100)

    # 50일 이동평균, 30일 RVOL 사용
    screener = ExitSignalScreener(ma_period=50, rvol_period=30)

    filtered_data = screener.apply_filters(df)
    output = screener.generate_screening_output(filtered_data, ticker="CUSTOM")

    sell_signals = output[output['Signal'] == 'SELL']

    print(f"\nSELL signals with MA50 and RVOL30: {len(sell_signals)}")
    print(sell_signals[['Date', 'Current_Price', 'MA20', 'Wick_Ratio', 'RVOL']].head())


def example_4_detailed_analysis():
    """상세 분석 - 각 조건별 분석"""
    print("\n" + "="*80)
    print("Example 4: Detailed Condition Analysis")
    print("="*80)

    df = generate_sample_data(days=60)
    screener = ExitSignalScreener()
    filtered_data = screener.apply_filters(df)

    # 조건별 통계
    print("\n[Condition Analysis]")
    print(f"Total Days: {len(filtered_data)}")
    print(f"\nCondition 1 (Price < MA20): {filtered_data['Condition_1_Trend_Breakdown'].sum()} days")
    print(f"Condition 2 (Wick >= 0.5): {filtered_data['Condition_2_Rejection_Pattern'].sum()} days")
    print(f"Condition 3 (RVOL >= 2.0): {filtered_data['Condition_3_Volume_Confirmation'].sum()} days")
    print(f"\nAll Conditions Met (SELL): {filtered_data['All_Conditions_Met'].sum()} days")

    # 포텐셜 트랩 케이스 (Price < MA20, Wick >= 0.5, but RVOL < 2.0)
    potential_trap = filtered_data[
        filtered_data['Condition_1_Trend_Breakdown'] &
        filtered_data['Condition_2_Rejection_Pattern'] &
        ~filtered_data['Condition_3_Volume_Confirmation']
    ]

    print(f"\n[Potential Trap Cases]")
    print(f"Pattern detected but lacks volume: {len(potential_trap)} days")
    if len(potential_trap) > 0:
        print(potential_trap[['Close', 'MA20', 'Wick_Ratio', 'RVOL', 'Reasoning']].head())


def example_5_export_results():
    """결과를 CSV로 저장"""
    print("\n" + "="*80)
    print("Example 5: Export Results to CSV")
    print("="*80)

    df = generate_sample_data(days=100)
    screener = ExitSignalScreener()

    filtered_data = screener.apply_filters(df)
    output = screener.generate_screening_output(filtered_data, ticker="EXPORT_TEST")

    # 전체 결과 저장
    output.to_csv('full_screening_results.csv', index=False, encoding='utf-8-sig')
    print("Saved: full_screening_results.csv")

    # SELL 신호만 저장
    sell_only = output[output['Signal'] == 'SELL']
    sell_only.to_csv('sell_signals_only.csv', index=False, encoding='utf-8-sig')
    print("Saved: sell_signals_only.csv")

    print(f"\nTotal records: {len(output)}")
    print(f"SELL signals: {len(sell_only)}")


def generate_sample_data(days=100):
    """테스트용 샘플 데이터 생성"""
    np.random.seed(None)  # 매번 다른 데이터 생성
    dates = pd.date_range('2024-01-01', periods=days, freq='D')

    df = pd.DataFrame({
        'Open': 100 + np.random.randn(days).cumsum() * 0.5,
        'Close': 100 + np.random.randn(days).cumsum() * 0.5,
        'Volume': np.random.randint(1000000, 5000000, days)
    })

    # High와 Low 계산
    df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.randn(days) * 2)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.randn(days) * 2)

    df.index = dates
    return df


if __name__ == "__main__":
    # 모든 예시 실행
    example_1_basic_usage()
    example_2_multiple_tickers()
    example_3_custom_parameters()
    example_4_detailed_analysis()
    example_5_export_results()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
