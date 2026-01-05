"""
Taking Profit Screener - 통합 분석 도구

사용법:
    python start.py

파일명만 입력하면 자동으로 분석합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import analyze_stock_from_csv, batch_analyze_stocks, StockAnalyzer
import pandas as pd


def main():
    """메인 함수 - 파일명 입력받아 자동 분석"""
    print("\n" + "=" * 80)
    print("TAKING PROFIT SCREENER")
    print("Volume-Confirmed Rejection Strategy")
    print("=" * 80)

    print("\n문서 폴더의 XLSX/CSV 파일을 분석합니다.")
    print("기본 경로: C:\\Users\\10845\\OneDrive - 이지스자산운용\\문서")

    # 파일명 입력
    print("\n파일명을 입력하세요 (여러 개는 쉼표로 구분)")
    print("예시:")
    print("  - 단일 종목: sm.xlsx")
    print("  - 여러 종목: sm.xlsx, samsung.xlsx, apple.xlsx")
    print("  - CSV 가능: old_data.csv, new_data.xlsx")

    user_input = input("\n파일명 입력: ").strip()

    if not user_input:
        print("\n파일명을 입력하지 않았습니다. 프로그램을 종료합니다.")
        return

    # 파일명 파싱
    filenames = [f.strip() for f in user_input.split(',')]

    print("\n" + "=" * 80)
    print(f"총 {len(filenames)}개 파일 분석 시작")
    print("=" * 80)

    if len(filenames) == 1:
        # 단일 종목 분석
        analyze_single(filenames[0])
    else:
        # 여러 종목 분석
        analyze_multiple(filenames)


def analyze_single(filename):
    """단일 종목 상세 분석"""
    print(f"\n[분석 중] {filename}...\n")

    try:
        # 분석 실행
        result = analyze_stock_from_csv(filename)

        # 상세 리포트 출력
        analyzer = StockAnalyzer()
        report = analyzer.format_analysis_report(result)
        print(report)

        # 결과 저장 여부
        save_choice = input("\n\n결과를 CSV로 저장하시겠습니까? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_filename = f"{result['ticker']}_분석결과.csv"
            df_result = pd.DataFrame([result])
            df_result.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"\n[저장 완료] {output_filename}")

    except FileNotFoundError:
        print(f"\n[에러] 파일을 찾을 수 없습니다: {filename}")
        print("파일 경로와 이름을 확인해주세요.")
    except Exception as e:
        print(f"\n[에러] 분석 실패: {e}")


def analyze_multiple(filenames):
    """여러 종목 일괄 분석"""
    print(f"\n{len(filenames)}개 파일 분석 중...\n")

    try:
        # 일괄 분석
        results = batch_analyze_stocks(filenames)

        if len(results) == 0:
            print("\n분석된 종목이 없습니다.")
            return

        # 요약 테이블 출력
        print("\n" + "=" * 80)
        print("분석 결과 요약")
        print("=" * 80)

        summary_data = []
        for _, row in results.iterrows():
            summary_data.append({
                '종목': row['ticker'],
                '현재가': f"{row['close_price']:.0f}",
                '20일선': f"{row['ma_distance_percent']:+.1f}%",
                'RVOL': f"{row['rvol']:.1f}배",
                '신호': row['signal'],
                '상태': row['signal_category']
            })

        df_summary = pd.DataFrame(summary_data)
        print("\n" + df_summary.to_string(index=False))

        # 조건별 분류
        print("\n" + "=" * 80)
        print("조건별 분류")
        print("=" * 80)

        # SELL 신호
        sell_stocks = results[results['signal'] == 'SELL']
        print(f"\n[강력 매도 신호] {len(sell_stocks)}개 종목:")
        if len(sell_stocks) > 0:
            for _, stock in sell_stocks.iterrows():
                print(f"  - {stock['ticker']}: {stock['reasoning']}")
        else:
            print("  없음")

        # 추세 하락만
        trend_down = results[
            results['condition_1_trend_breakdown'] &
            ~results['condition_2_rejection_pattern'] &
            ~results['condition_3_volume_confirmation']
        ]
        print(f"\n[추세 하락만] {len(trend_down)}개 종목:")
        if len(trend_down) > 0:
            for _, stock in trend_down.iterrows():
                print(f"  - {stock['ticker']}: 20일선 대비 {stock['ma_distance_percent']:.2f}%")
        else:
            print("  없음")

        # RVOL 폭증만
        rvol_surge = results[
            ~results['condition_1_trend_breakdown'] &
            ~results['condition_2_rejection_pattern'] &
            results['condition_3_volume_confirmation']
        ]
        print(f"\n[거래량 폭증만] {len(rvol_surge)}개 종목:")
        if len(rvol_surge) > 0:
            for _, stock in rvol_surge.iterrows():
                print(f"  - {stock['ticker']}: RVOL {stock['rvol']:.2f}배 ({stock['rvol_strength']})")
        else:
            print("  없음")

        # 주의 필요 (추세하락 + 윅패턴)
        caution = results[
            results['condition_1_trend_breakdown'] &
            results['condition_2_rejection_pattern'] &
            ~results['condition_3_volume_confirmation']
        ]
        print(f"\n[주의 필요] {len(caution)}개 종목 (추세하락 + 윅패턴, 거래량만 부족):")
        if len(caution) > 0:
            for _, stock in caution.iterrows():
                print(f"  - {stock['ticker']}: 윅비율 {stock['wick_ratio']:.2f}, RVOL {stock['rvol']:.2f}배")
        else:
            print("  없음")

        # 전체 결과 저장
        print("\n" + "=" * 80)
        save_choice = input("\n전체 결과를 CSV로 저장하시겠습니까? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_filename = "전체_분석결과.csv"
            results.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"\n[저장 완료] {output_filename}")

    except Exception as e:
        print(f"\n[에러] 분석 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n에러 발생: {e}")
        import traceback
        traceback.print_exc()
