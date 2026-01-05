"""
종목 분석 스크립트

문서 폴더에 있는 CSV 파일들을 읽어서 분석합니다.

사용법:
    python analyze_stocks.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import StockAnalyzer, analyze_stock_from_csv, batch_analyze_stocks
import pandas as pd


def analyze_single_stock():
    """단일 종목 분석"""
    print("\n" + "=" * 80)
    print("단일 종목 분석")
    print("=" * 80)

    # 사용자 입력
    csv_filename = input("\n파일명을 입력하세요 (예: sm.xlsx, samsung.xlsx): ").strip()

    # 기본 경로
    base_path = r"C:\Users\10845\OneDrive - 이지스자산운용\문서"

    # 날짜 컬럼명 입력 (선택)
    date_column = input("날짜 컬럼명을 입력하세요 (기본값: Date, Enter로 건너뛰기): ").strip()
    if not date_column:
        date_column = "Date"

    print(f"\n분석 중: {csv_filename}...")

    try:
        # 분석 실행
        result = analyze_stock_from_csv(csv_filename, base_path, date_column)

        # 결과 출력
        analyzer = StockAnalyzer()
        report = analyzer.format_analysis_report(result)
        print("\n" + report)

        # 결과 저장 여부
        save_choice = input("\n\n결과를 CSV로 저장하시겠습니까? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_filename = f"{result['ticker']}_analysis.csv"
            df_result = pd.DataFrame([result])
            df_result.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"\n결과가 '{output_filename}'에 저장되었습니다.")

    except FileNotFoundError:
        print(f"\n에러: 파일을 찾을 수 없습니다.")
        print(f"경로: {base_path}\\{csv_filename}")
        print("파일명과 경로를 확인해주세요.")
    except Exception as e:
        print(f"\n에러 발생: {e}")
        import traceback
        traceback.print_exc()


def analyze_multiple_stocks():
    """여러 종목 일괄 분석"""
    print("\n" + "=" * 80)
    print("여러 종목 일괄 분석")
    print("=" * 80)

    # 사용자 입력
    print("\n파일명들을 쉼표로 구분하여 입력하세요.")
    print("예: sm.xlsx, samsung.xlsx, apple.xlsx")
    filenames_input = input("\n파일명 입력: ").strip()

    # 파일명 리스트로 변환
    csv_filenames = [f.strip() for f in filenames_input.split(',')]

    # 기본 경로
    base_path = r"C:\Users\10845\OneDrive - 이지스자산운용\문서"

    # 날짜 컬럼명
    date_column = input("\n날짜 컬럼명을 입력하세요 (기본값: Date, Enter로 건너뛰기): ").strip()
    if not date_column:
        date_column = "Date"

    print(f"\n{len(csv_filenames)}개 파일 분석 시작...\n")

    try:
        # 일괄 분석 실행
        results_df = batch_analyze_stocks(csv_filenames, base_path, date_column)

        # 결과 요약 출력
        print("\n" + "=" * 80)
        print("분석 결과 요약")
        print("=" * 80)

        # 간단한 요약표
        summary_columns = [
            'ticker', 'date', 'close_price', 'ma_distance_percent',
            'rvol', 'rvol_strength', 'signal', 'signal_category'
        ]
        print("\n" + results_df[summary_columns].to_string(index=False))

        # 조건별 분류
        print("\n" + "=" * 80)
        print("조건별 분류")
        print("=" * 80)

        # SELL 신호
        sell_stocks = results_df[results_df['signal'] == 'SELL']
        print(f"\n[강력 매도 신호] {len(sell_stocks)}개 종목:")
        if len(sell_stocks) > 0:
            for _, stock in sell_stocks.iterrows():
                print(f"  • {stock['ticker']}: {stock['signal_category']}")
        else:
            print("  없음")

        # 추세 하락만
        trend_down = results_df[
            results_df['condition_1_trend_breakdown'] &
            ~results_df['condition_2_rejection_pattern'] &
            ~results_df['condition_3_volume_confirmation']
        ]
        print(f"\n[추세 하락만] {len(trend_down)}개 종목:")
        if len(trend_down) > 0:
            for _, stock in trend_down.iterrows():
                print(f"  • {stock['ticker']}: 20일선 대비 {stock['ma_distance_percent']:.2f}%")
        else:
            print("  없음")

        # RVOL 폭증만
        rvol_surge = results_df[
            ~results_df['condition_1_trend_breakdown'] &
            ~results_df['condition_2_rejection_pattern'] &
            results_df['condition_3_volume_confirmation']
        ]
        print(f"\n[거래량 폭증만] {len(rvol_surge)}개 종목:")
        if len(rvol_surge) > 0:
            for _, stock in rvol_surge.iterrows():
                print(f"  • {stock['ticker']}: RVOL {stock['rvol']:.2f}배 ({stock['rvol_strength']})")
        else:
            print("  없음")

        # 결과 저장
        save_choice = input("\n\n전체 결과를 CSV로 저장하시겠습니까? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_filename = "batch_analysis_results.csv"
            results_df.to_csv(output_filename, index=False, encoding='utf-utf-sig')
            print(f"\n전체 결과가 '{output_filename}'에 저장되었습니다.")

    except Exception as e:
        print(f"\n에러 발생: {e}")
        import traceback
        traceback.print_exc()


def custom_path_analysis():
    """사용자 지정 경로로 분석"""
    print("\n" + "=" * 80)
    print("사용자 지정 경로 분석")
    print("=" * 80)

    # 경로 입력
    custom_path = input("\n파일 전체 경로를 입력하세요 (xlsx 또는 csv): ").strip()

    # 날짜 컬럼명
    date_column = input("날짜 컬럼명을 입력하세요 (기본값: Date): ").strip() or "Date"

    print(f"\n분석 중...")

    try:
        from src.screener import load_data_from_csv

        # 종목명 추출
        import os
        filename = os.path.basename(custom_path)
        ticker = filename.replace('.xlsx', '').replace('.xls', '').replace('.csv', '').upper()

        # 데이터 로드
        df = load_data_from_csv(custom_path, date_column=date_column)

        # 분석 실행
        analyzer = StockAnalyzer(ma_period=20, rvol_period=20)
        result = analyzer.analyze_latest(df, ticker=ticker)

        # 결과 출력
        report = analyzer.format_analysis_report(result)
        print("\n" + report)

    except Exception as e:
        print(f"\n에러 발생: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 함수"""
    try:
        while True:
            print("\n" + "=" * 80)
            print("종목 분석 도구")
            print("=" * 80)
            print("\n메뉴:")
            print("  1. 단일 종목 분석 (파일명만 입력)")
            print("  2. 여러 종목 일괄 분석")
            print("  3. 사용자 지정 경로로 분석")
            print("  q. 종료")

            choice = input("\n선택: ").strip()

            if choice == 'q':
                print("\n프로그램을 종료합니다.")
                break
            elif choice == '1':
                analyze_single_stock()
            elif choice == '2':
                analyze_multiple_stocks()
            elif choice == '3':
                custom_path_analysis()
            else:
                print("\n잘못된 선택입니다.")

            input("\n\n계속하려면 Enter를 누르세요...")

    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n에러 발생: {e}")


if __name__ == "__main__":
    main()
