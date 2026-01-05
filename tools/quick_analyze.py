"""
빠른 종목 분석 - 간편 버전

파일명만 입력하면 바로 분석 결과를 보여줍니다. (XLSX 또는 CSV 지원)

사용법:
    python quick_analyze.py sm.xlsx
    python quick_analyze.py samsung.xlsx apple.xlsx microsoft.xlsx
    python quick_analyze.py sm.csv  (CSV도 가능)
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import StockAnalyzer, analyze_stock_from_csv
import pandas as pd


def quick_analyze(csv_filename: str):
    """빠른 분석 실행"""
    base_path = r"C:\Users\10845\OneDrive - 이지스자산운용\문서"

    try:
        # 분석 실행
        result = analyze_stock_from_csv(csv_filename, base_path, date_column="Date")

        # 결과 출력
        analyzer = StockAnalyzer()
        report = analyzer.format_analysis_report(result)
        print(report)
        print("\n")

        return result

    except FileNotFoundError:
        print(f"\n에러: '{csv_filename}' 파일을 찾을 수 없습니다.")
        print(f"경로: {base_path}")
        return None
    except Exception as e:
        print(f"\n에러: {csv_filename} 분석 실패 - {e}")
        return None


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("\n사용법:")
        print("  python quick_analyze.py <파일명1> <파일명2> ...")
        print("\n예시:")
        print("  python quick_analyze.py sm.xlsx")
        print("  python quick_analyze.py sm.xlsx samsung.xlsx apple.xlsx")
        print("  python quick_analyze.py sm.csv  (CSV도 가능)")
        print("\n또는 대화형 모드로 실행:")
        print("  python analyze_stocks.py")
        return

    csv_filenames = sys.argv[1:]

    print("\n" + "=" * 80)
    print(f"총 {len(csv_filenames)}개 파일 분석")
    print("=" * 80)

    results = []
    for filename in csv_filenames:
        result = quick_analyze(filename)
        if result:
            results.append(result)

    # 요약 출력
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("전체 요약")
        print("=" * 80)

        summary_data = []
        for r in results:
            summary_data.append({
                '종목': r['ticker'],
                '현재가': f"{r['close_price']:.0f}",
                '20일선': f"{r['ma_distance_percent']:+.1f}%",
                'RVOL': f"{r['rvol']:.1f}배",
                '신호': r['signal'],
                '상태': r['signal_category']
            })

        df_summary = pd.DataFrame(summary_data)
        print("\n" + df_summary.to_string(index=False))

        # SELL 신호 종목만 필터링
        sell_stocks = [r for r in results if r['signal'] == 'SELL']
        if sell_stocks:
            print(f"\n\n주의: {len(sell_stocks)}개 종목에서 강력 매도 신호 발견!")
            for stock in sell_stocks:
                print(f"  - {stock['ticker']}: {stock['reasoning']}")


if __name__ == "__main__":
    main()
