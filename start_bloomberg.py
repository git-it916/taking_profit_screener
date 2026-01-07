"""
Taking Profit Screener - Bloomberg 버전

Bloomberg Terminal에서 직접 데이터를 받아 분석합니다.
로컬에 파일 저장 없이 실시간으로 분석 가능!
"""

import os
import sys
import pandas as pd
from tabulate import tabulate

# 현재 스크립트의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src import StockAnalyzer, ExitSignalScreener
from src.bloomberg import download_bloomberg_data


def analyze_from_bloomberg(ticker: str, period: str = '1Y') -> dict:
    """
    Bloomberg에서 데이터를 받아 분석

    Parameters:
    -----------
    ticker : str
        Bloomberg 티커 (예: "005930 KS")
    period : str
        데이터 기간 (기본값: '1Y')

    Returns:
    --------
    dict : 분석 결과
    """
    try:
        # ================================================================
        # [1단계] Bloomberg에서 데이터 다운로드
        # ================================================================
        print(f"\n[분석 중] {ticker}...")
        df = download_bloomberg_data(ticker, period=period)

        if df is None or len(df) == 0:
            print(f"  ✗ 데이터 없음")
            return None

        # ================================================================
        # [2단계] 스크리너로 분석
        # ================================================================
        screener = ExitSignalScreener()
        df_analyzed = screener.apply_filters(df)

        # ================================================================
        # [3단계] 상세 분석
        # ================================================================
        analyzer = StockAnalyzer(screener=screener)
        result = analyzer.analyze_latest(ticker, df_analyzed)

        print(f"  ✓ 분석 완료")
        return result

    except Exception as e:
        print(f"  ✗ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """메인 함수"""
    print("="*80)
    print("TAKING PROFIT SCREENER - BLOOMBERG 버전")
    print("Bloomberg Terminal에서 직접 데이터를 받아 분석합니다")
    print("="*80)

    print("\n⚠️  주의사항:")
    print("  1. Bloomberg Terminal이 실행 중이어야 합니다")
    print("  2. Bloomberg에 로그인되어 있어야 합니다")

    # ====================================================================
    # 티커 입력
    # ====================================================================
    print("\n" + "="*80)
    print("Bloomberg 티커를 입력하세요 (쉼표로 구분)")
    print("="*80)
    print("\n티커 형식:")
    print("  - 한국 주식: 005930 KS (삼성전자), 000660 KS (SK하이닉스)")
    print("  - 미국 주식: AAPL US (애플), MSFT US (마이크로소프트)")
    print("  - 예시: 005930 KS, 000660 KS, AAPL US")

    try:
        user_input = input("\n티커 입력: ").strip()

        if not user_input:
            print("\n[에러] 티커를 입력해주세요")
            return

        # 티커 리스트 파싱
        tickers = [t.strip() for t in user_input.split(',')]
        print(f"\n입력된 티커: {tickers}")

        # ================================================================
        # 데이터 기간 입력
        # ================================================================
        print("\n데이터 기간을 선택하세요:")
        print("  1: 1년 (기본값)")
        print("  2: 6개월")
        print("  3: 3개월")

        period_choice = input("\n선택 (엔터=1년): ").strip()

        if period_choice == '2':
            period = '6M'
        elif period_choice == '3':
            period = '3M'
        else:
            period = '1Y'

        # ================================================================
        # 분석 실행
        # ================================================================
        print("\n" + "="*80)
        print(f"총 {len(tickers)}개 종목 분석 시작")
        print("="*80)

        results = []
        for ticker in tickers:
            result = analyze_from_bloomberg(ticker, period=period)

            if result:
                results.append(result)

        if not results:
            print("\n[에러] 분석 결과가 없습니다")
            return

        # ================================================================
        # 결과 출력
        # ================================================================
        print("\n" + "="*80)
        print("분석 결과 요약")
        print("="*80)

        # 요약 테이블 생성
        summary_data = []
        for row in results:
            # 전일비 계산
            if row.get('prev_close') and row['prev_close'] > 0:
                price_change_pct = row['price_change_percent']
                price_change_str = f"{price_change_pct:+.1f}%"
            else:
                price_change_str = "-"

            # 하회일 포맷팅
            break_below = row.get('last_ma20_break_below')
            if pd.notna(break_below):
                break_below_str = str(break_below).split()[0]
            else:
                break_below_str = "없음"

            summary_data.append({
                '종목': row['ticker'],
                '현재가': f"{row['close_price']:.0f}",
                '전일비': price_change_str,
                '20일선': f"{row['ma20']:.0f}",
                '괴리율': f"{row['ma_distance_percent']:+.1f}%",
                '하회일': break_below_str,
                'RVOL': f"{row['rvol']:.1f}배",
                '신호': row['signal']
            })

        # tabulate로 예쁘게 출력
        summary_df = pd.DataFrame(summary_data)
        print("\n" + tabulate(summary_df, headers='keys', tablefmt='simple', showindex=False))

        # ================================================================
        # 조건별 분류
        # ================================================================
        results_df = pd.DataFrame(results)

        print("\n" + "="*80)
        print("조건별 분류")
        print("="*80)

        # SELL 신호
        sell_stocks = results_df[results_df['signal'] == 'SELL']
        print(f"\n[강력 매도 신호] {len(sell_stocks)}개 종목:")
        if len(sell_stocks) > 0:
            for _, stock in sell_stocks.iterrows():
                print(f"  - {stock['ticker']}: {stock['reasoning']}")
        else:
            print("  없음")

        # 20일선 하회 + 거래량 부족
        caution = results_df[
            results_df['condition_1_trend_breakdown'] &
            ~results_df['condition_2_volume_confirmation']
        ]
        print(f"\n[주의 필요] {len(caution)}개 종목 (20일선 하회, 거래량 부족):")
        if len(caution) > 0:
            for _, stock in caution.iterrows():
                break_below = stock.get('last_ma20_break_below')
                break_below_str = str(break_below).split()[0] if pd.notna(break_below) else "데이터 없음"
                print(f"  - {stock['ticker']}: 20일선 대비 {stock['ma_distance_percent']:.2f}%, 하회일: {break_below_str}, RVOL {stock['rvol']:.2f}배")
        else:
            print("  없음")

        # 거래량 폭증만
        rvol_surge = results_df[
            ~results_df['condition_1_trend_breakdown'] &
            results_df['condition_2_volume_confirmation']
        ]
        print(f"\n[거래량 폭증만] {len(rvol_surge)}개 종목 (20일선 위):")
        if len(rvol_surge) > 0:
            for _, stock in rvol_surge.iterrows():
                print(f"  - {stock['ticker']}: RVOL {stock['rvol']:.2f}배 ({stock['rvol_strength']})")
        else:
            print("  없음")

        # ================================================================
        # 상세 리포트 출력 (선택)
        # ================================================================
        print("\n" + "="*80)
        detail_choice = input("\n종목별 상세 리포트를 보시겠습니까? (y/n): ").strip().lower()

        if detail_choice == 'y':
            analyzer = StockAnalyzer()
            for result in results:
                print("\n" + "="*80)
                report = analyzer.format_analysis_report(result)
                print(report)

        # ================================================================
        # CSV 저장 (선택)
        # ================================================================
        print("\n" + "="*80)
        save_choice = input("\n전체 결과를 CSV로 저장하시겠습니까? (y/n): ").strip().lower()

        if save_choice == 'y':
            output_filename = "블룸버그_분석결과.csv"
            results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"\n[저장 완료] {output_filename}")

    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n[에러] 분석 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
