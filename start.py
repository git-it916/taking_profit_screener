"""
Taking Profit Screener - 통합 분석 도구

사용법:
    python start.py

한글 종목명 입력 → Bloomberg 티커 변환 → 데이터 다운로드 → 분석까지
모든 과정을 한 번에 실행합니다.
"""

import sys
from pathlib import Path

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import analyze_stock_from_csv, batch_analyze_stocks, StockAnalyzer
from src.ticker_converter import convert_names_to_tickers, load_ticker_mapping_from_excel, KOREAN_STOCK_MAP
from src.bloomberg import download_bloomberg_data
import pandas as pd
from tabulate import tabulate


def main():
    """메인 함수 - 통합 워크플로우"""
    print("\n" + "=" * 80)
    print("TAKING PROFIT SCREENER - 통합 분석 도구")
    print("Volume-Confirmed Rejection Strategy")
    print("=" * 80)

    # ====================================================================
    # [1단계] 데이터 소스 선택
    # ====================================================================
    print("\n데이터 소스를 선택하세요:")
    print("  1: Bloomberg Terminal (실시간 다운로드)")
    print("  2: 로컬 파일 (XLSX/CSV)")

    source_choice = input("\n선택 (1/2): ").strip()

    if source_choice == '1':
        # Bloomberg 워크플로우
        analyze_from_bloomberg()
    elif source_choice == '2':
        # 로컬 파일 워크플로우
        analyze_from_local_files()
    else:
        print("\n[에러] 잘못된 선택입니다")
        return


def analyze_from_bloomberg():
    """Bloomberg Terminal에서 데이터를 받아 분석"""
    print("\n" + "=" * 80)
    print("BLOOMBERG 분석 모드")
    print("=" * 80)

    print("\n주의사항:")
    print("  1. Bloomberg Terminal이 실행 중이어야 합니다")
    print("  2. Bloomberg에 로그인되어 있어야 합니다")

    # ====================================================================
    # [2단계] 입력 방법 선택
    # ====================================================================
    print("\n" + "=" * 80)
    print("입력 방법을 선택하세요:")
    print("  1: 한글 종목명 입력 (자동으로 Bloomberg 티커 변환)")
    print("  2: Bloomberg 티커 직접 입력")
    print("  3: 파일에서 읽기 (.txt, .xlsx)")
    print("=" * 80)

    input_choice = input("\n선택 (1/2/3): ").strip()

    tickers = []

    # --------------------------------------------------------------------
    # 옵션 1: 한글 종목명 입력
    # --------------------------------------------------------------------
    if input_choice == '1':
        print("\n" + "=" * 80)
        print("한글 종목명을 입력하세요 (쉼표로 구분)")
        print("=" * 80)
        print("\n예시: 삼성전자, SK하이닉스, LG에너지솔루션, 네이버, 카카오")
        print(f"\n등록된 종목 수: {len(KOREAN_STOCK_MAP)}개")
        print("(등록된 종목 목록: utils/convert_tickers.py 실행 → 옵션 3)")

        user_input = input("\n종목명 입력: ").strip()

        if not user_input:
            print("\n[에러] 종목명을 입력해주세요")
            return

        # 종목명 파싱
        names = [n.strip() for n in user_input.split(',')]
        print(f"\n입력된 종목: {len(names)}개")

        # 사용자 정의 매핑 파일 (선택)
        print("\n추가 매핑 파일이 있습니까? (ticker_mapping.xlsx)")
        mapping_choice = input("매핑 파일 경로 (없으면 엔터): ").strip()

        custom_mapping = None
        if mapping_choice:
            try:
                custom_mapping = load_ticker_mapping_from_excel(mapping_choice)
                print(f"  [OK] {len(custom_mapping)}개 추가 매핑 로드")
            except Exception as e:
                print(f"  [WARNING] 매핑 파일 로드 실패: {e}")

        # 한글 → Bloomberg 티커 변환
        print("\nBloomberg 티커로 변환 중...")
        tickers = convert_names_to_tickers(names, custom_mapping=custom_mapping)
        print(f"  [OK] {len(tickers)}개 티커 변환 완료")
        print(f"\n변환된 티커: {', '.join(tickers[:5])}" + (f" ... 외 {len(tickers)-5}개" if len(tickers) > 5 else ""))

    # --------------------------------------------------------------------
    # 옵션 2: Bloomberg 티커 직접 입력
    # --------------------------------------------------------------------
    elif input_choice == '2':
        print("\n" + "=" * 80)
        print("Bloomberg 티커를 입력하세요 (쉼표로 구분)")
        print("=" * 80)
        print("\n티커 형식:")
        print("  - 한국: 005930 KS, 000660 KS")
        print("  - 미국: AAPL US, MSFT US")

        user_input = input("\n티커 입력: ").strip()

        if not user_input:
            print("\n[에러] 티커를 입력해주세요")
            return

        tickers = [t.strip() for t in user_input.split(',')]
        print(f"\n입력된 티커: {len(tickers)}개")

    # --------------------------------------------------------------------
    # 옵션 3: 파일에서 읽기
    # --------------------------------------------------------------------
    elif input_choice == '3':
        print("\n" + "=" * 80)
        print("파일 형식:")
        print("  - 텍스트: 한 줄에 하나씩 또는 쉼표 구분")
        print("  - 엑셀: 첫 번째 컬럼")
        print("  - 한글 종목명 또는 Bloomberg 티커 가능")
        print("=" * 80)

        file_path = input("\n파일 경로: ").strip()

        try:
            # 파일 읽기
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
                items = df.iloc[:, 0].astype(str).tolist()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if ',' in content:
                    items = [i.strip() for i in content.split(',')]
                else:
                    items = [i.strip() for i in content.split('\n')]

            items = [i for i in items if i and i != 'nan']
            print(f"\n  [OK] {len(items)}개 항목 로드")

            # 한글 종목명인지 티커인지 확인
            sample = items[0]
            is_korean = any(ord(c) > 127 for c in sample)  # 한글 포함 여부

            if is_korean:
                print("\n한글 종목명으로 감지되어 자동 변환합니다...")
                tickers = convert_names_to_tickers(items)
            else:
                print("\nBloomberg 티커로 감지")
                tickers = items

        except FileNotFoundError:
            print(f"\n[에러] 파일을 찾을 수 없습니다: {file_path}")
            return
        except Exception as e:
            print(f"\n[에러] 파일 읽기 실패: {e}")
            return

    else:
        print("\n[에러] 잘못된 선택입니다")
        return

    if not tickers:
        print("\n[에러] 분석할 티커가 없습니다")
        return

    # ====================================================================
    # [3단계] 데이터 기간 선택
    # ====================================================================
    print("\n" + "=" * 80)
    print("데이터 기간을 선택하세요:")
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

    # ====================================================================
    # [4단계] Bloomberg 다운로드 및 분석
    # ====================================================================
    print("\n" + "=" * 80)
    print(f"총 {len(tickers)}개 종목 분석 시작 (기간: {period})")
    print("=" * 80)

    results = []
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}...")

        try:
            # Bloomberg에서 데이터 다운로드
            df = download_bloomberg_data(ticker, period=period)

            if df is None or len(df) == 0:
                print(f"  >> 데이터 없음")
                continue

            # 상세 분석
            analyzer = StockAnalyzer()
            result = analyzer.analyze_latest(df, ticker)

            results.append(result)
            print(f"  >> 분석 완료")

        except Exception as e:
            print(f"  >> 분석 실패: {e}")
            continue

    if not results:
        print("\n[에러] 분석 결과가 없습니다")
        return

    # ====================================================================
    # [5단계] 결과 출력
    # ====================================================================
    print_analysis_results(results)


def analyze_from_local_files():
    """로컬 파일에서 데이터를 읽어 분석"""
    print("\n" + "=" * 80)
    print("로컬 파일 분석 모드")
    print("=" * 80)

    print("\n문서 폴더의 XLSX/CSV 파일을 분석합니다.")
    print("기본 경로: C:\\Users\\10845\\OneDrive - 이지스자산운용\\문서")

    # 파일명 입력
    print("\n파일명을 입력하세요 (여러 개는 쉼표로 구분)")
    print("예시:")
    print("  - 단일 종목: sm.xlsx")
    print("  - 여러 종목: sm.xlsx, samsung.xlsx, apple.xlsx")

    user_input = input("\n파일명 입력: ").strip()

    if not user_input:
        print("\n[에러] 파일명을 입력하지 않았습니다")
        return

    # 파일명 파싱
    filenames = [f.strip() for f in user_input.split(',')]

    print("\n" + "=" * 80)
    print(f"총 {len(filenames)}개 파일 분석 시작")
    print("=" * 80)

    if len(filenames) == 1:
        # 단일 종목 분석
        analyze_single_file(filenames[0])
    else:
        # 여러 종목 분석
        analyze_multiple_files(filenames)


def analyze_single_file(filename):
    """단일 종목 상세 분석 (로컬 파일)"""
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
        import traceback
        traceback.print_exc()


def analyze_multiple_files(filenames):
    """여러 종목 일괄 분석 (로컬 파일)"""
    print(f"\n{len(filenames)}개 파일 분석 중...\n")

    try:
        # 일괄 분석
        results = batch_analyze_stocks(filenames)

        if len(results) == 0:
            print("\n[에러] 분석된 종목이 없습니다")
            return

        # DataFrame → dict list 변환
        results_list = results.to_dict('records')

        # 결과 출력
        print_analysis_results(results_list)

    except Exception as e:
        print(f"\n[에러] 분석 실패: {e}")
        import traceback
        traceback.print_exc()


def print_analysis_results(results):
    """
    분석 결과를 보기 좋게 출력

    Parameters:
    -----------
    results : list of dict
        분석 결과 리스트
    """
    # ====================================================================
    # 요약 테이블 출력
    # ====================================================================
    print("\n" + "=" * 80)
    print("분석 결과 요약")
    print("=" * 80)

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

    # ====================================================================
    # 조건별 분류
    # ====================================================================
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("조건별 분류")
    print("=" * 80)

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

    # ====================================================================
    # 상세 리포트 출력 (선택)
    # ====================================================================
    print("\n" + "=" * 80)
    detail_choice = input("\n종목별 상세 리포트를 보시겠습니까? (y/n): ").strip().lower()

    if detail_choice == 'y':
        analyzer = StockAnalyzer()
        for result in results:
            print("\n" + "=" * 80)
            report = analyzer.format_analysis_report(result)
            print(report)

    # ====================================================================
    # CSV 저장 (선택)
    # ====================================================================
    print("\n" + "=" * 80)
    save_choice = input("\n전체 결과를 CSV로 저장하시겠습니까? (y/n): ").strip().lower()

    if save_choice == 'y':
        output_filename = "전체_분석결과.csv"
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n[저장 완료] {output_filename}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n[에러] {e}")
        import traceback
        traceback.print_exc()
