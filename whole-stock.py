"""
Taking Profit Screener - 전종목 분석 (코스피 + 코스닥)

py -3.12 whole-stock.py

Bloomberg Terminal에서 전종목 데이터를 받아 분석합니다.
10일선 위 + RVOL≥1.3 종목만 필터링하여 표시합니다.
"""
import os
import sys
import pandas as pd
from datetime import datetime
from tabulate import tabulate

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 현재 스크립트의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src import StockAnalyzer
from src.bloomberg import download_bloomberg_data, get_multiple_security_names
from src.visualizer import create_trend_heatmap


def get_tickers_from_excel(file_path: str) -> list:
    """
    엑셀 파일에서 티커 리스트 읽기

    Parameters:
    -----------
    file_path : str
        엑셀 파일 경로

    Returns:
    --------
    list : 티커 리스트

    엑셀 파일 형식:
    - 'bloomberg_ticker' 컬럼에 티커 리스트 (예: 005930 KS, 000660 KQ)
    - 또는 첫 번째 컬럼에 티커 코드 (예: 005930, 000660)
    - 두 번째 컬럼에 거래소 코드 (KS 또는 KQ) - 선택사항
    """
    import pandas as pd

    print(f"\n[엑셀 파일에서 티커 읽기]")
    print(f"  파일: {file_path}")

    try:
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path)

        print(f"  ✓ 파일 로드 완료")
        print(f"  컬럼: {list(df.columns)}")
        print(f"  행 수: {len(df)}")

        tickers = []

        # bloomberg_ticker 컬럼이 있는지 확인 (우선순위)
        if 'bloomberg_ticker' in df.columns:
            print(f"  ✓ 'bloomberg_ticker' 컬럼 발견")
            target_col = 'bloomberg_ticker'
        else:
            # 첫 번째 컬럼 사용
            target_col = df.columns[0]
            print(f"  ℹ️  'bloomberg_ticker' 컬럼 없음, 첫 번째 컬럼 사용: {target_col}")

        for idx, row in df.iterrows():
            ticker_value = str(row[target_col]).strip()

            # 빈 값 무시
            if not ticker_value or ticker_value == 'nan':
                continue

            # 이미 "005930 KS" 형식인 경우
            if ' ' in ticker_value:
                tickers.append(ticker_value)
            else:
                # 티커만 있는 경우, 거래소 코드 추가
                # 두 번째 컬럼에 거래소 코드가 있는지 확인
                if len(df.columns) > 1:
                    exchange = str(row[df.columns[1]]).strip().upper()
                    if exchange in ['KS', 'KQ']:
                        tickers.append(f"{ticker_value} {exchange}")
                    else:
                        # 거래소 코드가 없으면 6자리 숫자로 판단
                        if len(ticker_value) == 6 and ticker_value.isdigit():
                            # 기본값: KS (코스피)
                            tickers.append(f"{ticker_value} KS")
                        else:
                            tickers.append(ticker_value)
                else:
                    # 컬럼이 하나뿐이면 기본값 사용
                    if len(ticker_value) == 6 and ticker_value.isdigit():
                        tickers.append(f"{ticker_value} KS")
                    else:
                        tickers.append(ticker_value)

        # 중복 제거
        tickers = list(set(tickers))

        print(f"\n  총 {len(tickers)}개 티커 로드 완료")

        # 샘플 출력
        print(f"\n  샘플 티커 (처음 5개):")
        for ticker in tickers[:5]:
            print(f"    - {ticker}")

        return tickers

    except Exception as e:
        print(f"  ✗ 파일 읽기 실패: {e}")
        import traceback
        traceback.print_exc()
        return []


def analyze_from_bloomberg(ticker: str, period: str = '2M') -> dict:
    """
    Bloomberg에서 데이터를 받아 분석

    Parameters:
    -----------
    ticker : str
        Bloomberg 티커
    period : str
        데이터 기간 (기본값: '2M' - 2개월)

    Returns:
    --------
    dict : 분석 결과
    """
    try:
        # Bloomberg에서 데이터 다운로드 (verbose=False로 메시지 숨김)
        df = download_bloomberg_data(ticker, period=period, verbose=False)

        if df is None or len(df) == 0:
            return None

        # 상세 분석
        analyzer = StockAnalyzer()
        result = analyzer.analyze_latest(df, ticker)

        return result

    except Exception as e:
        return None


def analyze_tickers_batch(tickers: list, period: str = '2M', batch_size: int = 50) -> list:
    """
    배치 방식으로 여러 티커 분석 (Bloomberg API 효율적 사용)

    Parameters:
    -----------
    tickers : list
        분석할 티커 리스트
    period : str
        데이터 기간
    batch_size : int
        배치 크기 (기본값: 50)

    Returns:
    --------
    list : 분석 결과 리스트
    """
    results = []
    failed_count = 0

    print(f"\n총 {len(tickers)}개 종목 분석 시작...")
    print(f"배치 크기: {batch_size}개씩 처리")
    print(f"진행 상황은 10개마다 표시됩니다.\n")

    start_time = datetime.now()

    for i, ticker in enumerate(tickers, 1):
        result = analyze_from_bloomberg(ticker, period=period)

        if result:
            results.append(result)
        else:
            failed_count += 1

        # 진행 상황 표시 (10개마다)
        if i % 10 == 0 or i == len(tickers):
            elapsed = datetime.now() - start_time
            progress = i / len(tickers) * 100
            rate = i / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
            remaining = (len(tickers) - i) / rate if rate > 0 else 0

            # 한 줄로 출력 (이전 줄 덮어쓰기)
            print(f"\r[진행중] {i}/{len(tickers)} ({progress:.1f}%) "
                  f"| 경과: {str(elapsed).split('.')[0]} | 속도: {rate:.2f}종목/초 | "
                  f"남은시간: ~{int(remaining/60)}분 {int(remaining%60)}초", end='', flush=True)

    print()  # 줄바꿈
    total_time = datetime.now() - start_time

    print(f"\n분석 완료 - 소요시간: {str(total_time).split('.')[0]}")
    print(f"성공: {len(results)}개, 실패: {failed_count}개")

    return results


def filter_recent_breakout_stocks(results: list, days: int = 5, rvol_threshold: float = 1.3) -> pd.DataFrame:
    """
    10일선 위 + RVOL≥threshold 종목 필터링

    Parameters:
    -----------
    results : list
        분석 결과 리스트
    days : int
        (사용 안 함, 하위 호환성 유지)
    rvol_threshold : float
        RVOL 최소 기준 (기본값: 1.3)

    Returns:
    --------
    pd.DataFrame : 필터링된 결과
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # 조건 1: 현재 10일선 위에 있음 (상승세)
    condition_above = df['current_position'] == 'above'

    # 조건 2: RVOL >= threshold (거래량 증가)
    condition_rvol = df['rvol'] >= rvol_threshold

    # 두 조건을 만족하는 종목 필터링
    filtered = df[condition_above & condition_rvol].copy()

    # RVOL 기준 내림차순 정렬
    filtered = filtered.sort_values('rvol', ascending=False)

    return filtered


def main():
    """메인 함수"""
    print("="*80)
    print("TAKING PROFIT SCREENER - 전종목 분석 (코스피 + 코스닥)")
    print("="*80)

    print("\n⚠️  주의사항:")
    print("  1. Bloomberg Terminal이 실행 중이어야 합니다")
    print("  2. Bloomberg에 로그인되어 있어야 합니다")
    print("  3. 전종목 분석은 약 10~15분 소요됩니다 (1개월 데이터)")

    # ====================================================================
    # 엑셀 파일에서 티커 로드
    # ====================================================================
    print("\n" + "="*80)
    print("티커 리스트 파일 입력")
    print("="*80)

    # 기본 파일 경로
    default_file = "bloomberg_ticker.xlsx"

    print(f"\n기본 파일: {default_file}")
    print("다른 파일을 사용하려면 경로를 입력하세요 (엔터=기본 파일 사용)")
    print("\n엑셀 파일 형식:")
    print("  - 'bloomberg_ticker' 컬럼에 티커 리스트 (예: 005930 KS, 000660 KQ)")

    user_input = input("\n파일 경로 (엔터=기본): ").strip()

    # 사용자 입력이 없으면 기본 파일 사용
    if not user_input:
        file_path = default_file
        print(f"✓ 기본 파일 사용: {default_file}")
    else:
        # 따옴표 제거 (사용자가 경로를 따옴표로 감쌀 수 있음)
        file_path = user_input.strip('"').strip("'")

    # 엑셀 파일에서 티커 읽기
    all_tickers = get_tickers_from_excel(file_path)

    if not all_tickers:
        print("\n[에러] 티커를 읽을 수 없습니다")
        return

    # 사용자 확인
    print(f"\n총 {len(all_tickers)}개 종목을 분석합니다.")
    proceed = input("계속 진행하시겠습니까? (y/n): ").strip().lower()

    if proceed != 'y':
        print("\n분석을 취소했습니다.")
        return

    # ====================================================================
    # 전종목 분석 실행 (순차 처리)
    # ====================================================================
    print("\n" + "="*80)
    print("전종목 분석 시작 (1개월 데이터)")
    print("="*80)

    results = analyze_tickers_batch(all_tickers, period='1M')

    if not results:
        print("\n[에러] 분석 결과가 없습니다")
        return

    # ====================================================================
    # 필터링: 10일선 위 + RVOL≥1.3
    # ====================================================================
    print("\n" + "="*80)
    print("스크리닝: 10일선 위 + RVOL≥1.3 종목")
    print("="*80)

    filtered_df = filter_recent_breakout_stocks(results, rvol_threshold=1.3)

    if filtered_df.empty:
        print("\n조건을 만족하는 종목이 없습니다.")
        return

    print(f"\n✓ {len(filtered_df)}개 종목이 조건을 만족합니다")

    # ====================================================================
    # 종목명 조회
    # ====================================================================
    print("\n[종목명 조회 중...]")
    filtered_tickers = filtered_df['ticker'].tolist()

    try:
        ticker_names = get_multiple_security_names(filtered_tickers)
    except Exception as e:
        print(f"⚠️  종목명 조회 실패: {e}")
        ticker_names = {ticker: ticker for ticker in filtered_tickers}

    # ====================================================================
    # 결과 출력
    # ====================================================================
    print("\n" + "="*80)
    print("스크리닝 결과 (RVOL 높은 순)")
    print("="*80)

    # 요약 테이블 생성
    summary_data = []
    for _, row in filtered_df.iterrows():
        ticker = row['ticker']
        security_name = ticker_names.get(ticker, ticker)

        # 전일비 계산
        if row.get('prev_close') and row['prev_close'] > 0:
            price_change_pct = row['price_change_percent']
            price_change_str = f"{price_change_pct:+.1f}%"
        else:
            price_change_str = "-"

        summary_data.append({
            '종목': security_name[:30],  # 30자 제한
            '티커': ticker,
            '현재가': f"{row['close_price']:.0f}",
            '전일비': price_change_str,
            '10일선': f"{row['ma10']:.0f}",
            '괴리율': f"{row['ma_distance_percent']:+.1f}%",
            'RVOL': f"{row['rvol']:.1f}배",
            '돌파일': row.get('last_break_above', '?'),
        })

    # tabulate로 예쁘게 출력
    summary_df = pd.DataFrame(summary_data)
    print("\n" + tabulate(summary_df, headers='keys', tablefmt='simple', showindex=False))

    # ====================================================================
    # 상세 정보 출력
    # ====================================================================
    print("\n" + "="*80)
    print("상세 정보")
    print("="*80)

    for _, row in filtered_df.iterrows():
        ticker = row['ticker']
        security_name = ticker_names.get(ticker, ticker)

        name_padded = f"{security_name:<30}"
        trend_info = row['trend_detail']
        rvol_str = f"RVOL {row['rvol']:.1f}배"

        print(f"  {name_padded}  {trend_info}, {rvol_str}")

    # ====================================================================
    # CSV 저장 (선택)
    # ====================================================================
    print("\n" + "="*80)
    save_choice = input("\n결과를 CSV로 저장하시겠습니까? (y/n): ").strip().lower()

    if save_choice == 'y':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"전종목_스크리닝_{timestamp}.csv"

        # 저장용 DataFrame 생성
        save_df = filtered_df[[
            'ticker', 'close_price', 'prev_close', 'price_change_percent',
            'ma10', 'ma_distance_percent', 'rvol',
            'last_break_above', 'last_break_below',
            'trend_direction', 'trend_detail', 'signal'
        ]].copy()

        # 종목명 추가
        save_df.insert(0, 'security_name', save_df['ticker'].map(ticker_names))

        save_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n[저장 완료] {output_filename}")

    # ====================================================================
    # 시각화 저장 (선택)
    # ====================================================================
    print("\n" + "="*80)
    viz_choice = input("\n분석 결과를 히트맵으로 저장하시겠습니까? (y/n): ").strip().lower()

    if viz_choice == 'y':
        try:
            print("\n[시각화 생성 중...]")
            saved_path = create_trend_heatmap(filtered_df.to_dict('records'))
            print(f"✓ 히트맵 저장 완료: {saved_path}")
        except Exception as e:
            print(f"✗ 시각화 생성 실패: {e}")
            import traceback
            traceback.print_exc()

    # ====================================================================
    # 조건별 맞춤 필터링 (자동 실행)
    # ====================================================================
    print("\n" + "="*80)
    print("조건별 맞춤 필터링")
    print("="*80)
    print(f"\n현재 필터링된 종목 (10일선 위 + RVOL≥1.3): {len(filtered_df)}개")
    print(f"전체 분석 성공한 종목: {len(results)}개")

    all_results_df = pd.DataFrame(results)

    print("\n1. RVOL 최소값을 입력하세요 (예: 1.3)")
    rvol_min_input = input("   RVOL ≥ ").strip()
    rvol_min = float(rvol_min_input) if rvol_min_input else 1.3

    print("\n2. 10일선 위치를 선택하세요")
    print("   1: 10일선 위만")
    print("   2: 10일선 아래만")
    print("   3: 전체")
    position_choice = input("   선택 (1-3): ").strip()

    # 필터링
    filtered = all_results_df[all_results_df['rvol'] >= rvol_min]

    if position_choice == '1':
        filtered = filtered[filtered['current_position'] == 'above']
        position_desc = "10일선 위"
    elif position_choice == '2':
        filtered = filtered[filtered['current_position'] == 'below']
        position_desc = "10일선 아래"
    else:
        position_desc = "전체"

    filtered = filtered.sort_values('rvol', ascending=False)

    print(f"\n[필터링 결과] {len(filtered)}개 종목 (RVOL ≥ {rvol_min}, {position_desc})")
    print("="*80)

    for idx, (_, row) in enumerate(filtered.iterrows(), 1):
        ticker = row['ticker']
        rvol = row['rvol']
        position = "↑10일선위" if row['current_position'] == 'above' else "↓10일선아래"
        ma_dist = row['ma_distance_percent']
        close = row['close_price']
        print(f"  {idx:3d}. {ticker:12s}  {position}  현재가: {close:8.0f}  괴리율: {ma_dist:+6.1f}%  RVOL: {rvol:5.1f}배")

    # CSV 저장 옵션
    print("\n" + "="*80)
    save_custom = input("\n이 결과를 CSV로 저장하시겠습니까? (y/n): ").strip().lower()
    if save_custom == 'y':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"맞춤필터_RVOL{rvol_min}_{position_desc}_{timestamp}.csv"
        filtered.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✓ 저장 완료: {filename}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n[에러] 분석 실패: {e}")
        import traceback
        traceback.print_exc()
