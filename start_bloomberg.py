"""
Taking Profit Screener - Bloomberg 버전

py -3.12 start_bloomberg.py

Bloomberg Terminal에서 직접 데이터를 받아 분석합니다.
로컬에 파일 저장 없이 실시간으로 분석 가능!
"""
import os
import sys
import pandas as pd
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


def analyze_from_bloomberg(ticker: str, period: str = '1M', show_progress: bool = True) -> dict:
    """
    Bloomberg에서 데이터를 받아 분석

    Parameters:
    -----------
    ticker : str
        Bloomberg 티커 (예: "005930 KS")
    period : str
        데이터 기간 (기본값: '1M')
    show_progress : bool
        진행 상황 표시 여부 (기본값: True)

    Returns:
    --------
    dict : 분석 결과
    """
    try:
        # ================================================================
        # [1단계] Bloomberg에서 데이터 다운로드 (verbose 제어)
        # ================================================================
        df = download_bloomberg_data(ticker, period=period, verbose=show_progress)

        if df is None or len(df) == 0:
            return None

        # ================================================================
        # [1-1단계] 당일 데이터 제외 (장중에만 - 마감 후에는 포함)
        # ================================================================
        from datetime import datetime as dt, time
        now = dt.now()
        today = now.date()
        current_time = now.time()

        # 한국 시장 마감 시간: 오후 3시 30분
        market_close_time = time(15, 30)

        # Date 컬럼을 datetime으로 변환
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['date_only'] = df['Date'].dt.date

            # 장중(마감 전)에만 당일 데이터 제외
            if current_time < market_close_time:
                # 당일 데이터가 있으면 제외 (일봉 미완성)
                if (df['date_only'] == today).any():
                    df = df[df['date_only'] != today].copy()

            # 임시 컬럼 제거
            df = df.drop(columns=['date_only'])

        if len(df) == 0:
            return None

        # ================================================================
        # [2단계] 상세 분석
        # ================================================================
        analyzer = StockAnalyzer()
        result = analyzer.analyze_latest(df, ticker)

        return result

    except Exception as e:
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
        # 종목명 조회 (Bloomberg API)
        # ================================================================
        print("\n[종목명 조회 중...]")
        try:
            ticker_names = get_multiple_security_names(tickers)
            print("✓ 종목명 조회 완료")
            print("\n종목 정보:")
            for ticker in tickers:
                name = ticker_names.get(ticker, ticker)
                print(f"  - {ticker}: {name}")
        except Exception as e:
            print(f"⚠️  종목명 조회 실패 (티커로 표시됩니다): {e}")
            ticker_names = {ticker: ticker for ticker in tickers}

        # ================================================================
        # 데이터 기간 설정 (3개월 고정)
        # ================================================================
        period = '3M'
        print(f"\n데이터 기간: 최근 3개월")

        # ================================================================
        # 분석 실행 (병렬 처리)
        # ================================================================
        print("\n" + "="*80)
        print(f"총 {len(tickers)}개 종목 분석 시작")
        print("="*80)
        print()

        from datetime import datetime as dt
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock

        results = []
        failed_count = 0
        completed_count = 0
        lock = Lock()
        start_time = dt.now()

        # 병렬 처리 워커 수 (5개)
        max_workers = 5
        print(f"병렬 처리: {max_workers}개 동시 실행\n")

        def analyze_single(ticker):
            """단일 티커 분석 (worker thread에서 실행)"""
            try:
                result = analyze_from_bloomberg(ticker, period=period, show_progress=False)
                return (ticker, result, None)
            except Exception as e:
                return (ticker, None, str(e))

        # ThreadPoolExecutor로 병렬 처리
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 모든 작업 제출
            future_to_ticker = {executor.submit(analyze_single, ticker): ticker
                               for ticker in tickers}

            # 완료되는 대로 처리
            for future in as_completed(future_to_ticker):
                ticker, result, error = future.result()

                with lock:
                    completed_count += 1

                    if result:
                        results.append(result)
                    else:
                        failed_count += 1

                    # 진행 상황 표시 (한 줄로 업데이트)
                    elapsed = dt.now() - start_time
                    progress = completed_count / len(tickers) * 100

                    # 프로그레스 바 생성
                    bar_length = 40
                    filled_length = int(bar_length * completed_count // len(tickers))
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)

                    print(f"\r진행: [{bar}] {completed_count}/{len(tickers)} ({progress:.1f}%) | "
                          f"성공: {len(results)} | 실패: {failed_count} | "
                          f"경과: {str(elapsed).split('.')[0]}", end='', flush=True)

        print()  # 줄바꿈
        total_time = dt.now() - start_time
        print(f"\n✓ 분석 완료 - 소요시간: {str(total_time).split('.')[0]}")

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

            # 종목명 가져오기
            ticker = row['ticker']
            security_name = ticker_names.get(ticker, ticker)

            summary_data.append({
                '종목': security_name,
                '현재가': f"{row['close_price']:.0f}",
                '전일비': price_change_str,
                '10일선': f"{row['ma10']:.0f}",
                '괴리율': f"{row['ma_distance_percent']:+.1f}%",
                '추세': row.get('trend_direction', '-'),
                'RVOL': f"{row['rvol']:.1f}배",
                '신호': row['signal']
            })

        # tabulate로 예쁘게 출력
        summary_df = pd.DataFrame(summary_data)
        print("\n" + tabulate(summary_df, headers='keys', tablefmt='simple', showindex=False))

        # ================================================================
        # 추세별 분류 (하락세 vs 상승세)
        # ================================================================
        results_df = pd.DataFrame(results)

        print("\n" + "="*80)
        print("추세별 분류")
        print("="*80)

        # ====================================================================
        # 하락세 종목 (10일선 아래)
        # ====================================================================
        falling_stocks = results_df[results_df['current_position'] == 'below']
        print(f"\n[하락세 종목] {len(falling_stocks)}개 (10일선 아래)")
        print("-" * 80)
        if len(falling_stocks) > 0:
            for _, stock in falling_stocks.iterrows():
                ticker = stock['ticker']
                trend_info = stock['trend_detail']
                rvol_info = f"RVOL {stock['rvol']:.1f}배"
                if stock['condition_2_volume_confirmation']:
                    rvol_info += " [거래량 폭증!]"
                print(f"  - {ticker}: {trend_info}, {rvol_info}")
        else:
            print("  없음")

        # ====================================================================
        # 상승세 종목 (10일선 위)
        # ====================================================================
        rising_stocks = results_df[results_df['current_position'] == 'above']
        print(f"\n[상승세 종목] {len(rising_stocks)}개 (10일선 위)")
        print("-" * 80)
        if len(rising_stocks) > 0:
            for _, stock in rising_stocks.iterrows():
                ticker = stock['ticker']
                trend_info = stock['trend_detail']
                rvol_info = f"RVOL {stock['rvol']:.1f}배"
                if stock['condition_2_volume_confirmation']:
                    rvol_info += " [거래량 폭증!]"
                print(f"  - {ticker}: {trend_info}, {rvol_info}")
        else:
            print("  없음")

        # ====================================================================
        # 조건별 분류
        # ====================================================================
        print("\n" + "="*80)
        print("조건별 분류")
        print("="*80)

        # SELL 신호 (10일선 하회 + 거래량 폭증)
        sell_stocks = results_df[results_df['signal'] == 'SELL']
        print(f"\n[강력 매도 신호] {len(sell_stocks)}개 종목 (10일선 하회 + 거래량 폭증):")
        print("-" * 80)
        if len(sell_stocks) > 0:
            for _, stock in sell_stocks.iterrows():
                ticker = stock['ticker']
                security_name = ticker_names.get(ticker, ticker)

                # 종목명을 30자로 맞춤 (왼쪽 정렬)
                name_padded = f"{security_name:<30}"
                trend_info = stock['trend_detail']
                rvol_str = f"RVOL {stock['rvol']:.1f}배"

                print(f"  {name_padded}  {trend_info}, {rvol_str}, SELL")
        else:
            print("  없음")

        # 10일선 하회 + 거래량 부족
        caution = results_df[
            results_df['condition_1_trend_breakdown'] &
            ~results_df['condition_2_volume_confirmation']
        ]
        print(f"\n[주의 필요] {len(caution)}개 종목 (10일선 하회, 거래량 부족):")
        print("-" * 80)
        if len(caution) > 0:
            for _, stock in caution.iterrows():
                ticker = stock['ticker']
                security_name = ticker_names.get(ticker, ticker)

                # 종목명을 30자로 맞춤 (왼쪽 정렬)
                name_padded = f"{security_name:<30}"
                trend_info = stock['trend_detail']
                rvol_str = f"RVOL {stock['rvol']:.1f}배"

                print(f"  {name_padded}  {trend_info}, {rvol_str}, HOLD")
        else:
            print("  없음")

        # 거래량 폭증만 (10일선 위)
        rvol_surge = results_df[
            ~results_df['condition_1_trend_breakdown'] &
            results_df['condition_2_volume_confirmation']
        ]
        print(f"\n[거래량 폭증] {len(rvol_surge)}개 종목 (10일선 위 + 거래량 폭증):")
        print("-" * 80)
        if len(rvol_surge) > 0:
            for _, stock in rvol_surge.iterrows():
                ticker = stock['ticker']
                security_name = ticker_names.get(ticker, ticker)

                # 종목명을 30자로 맞춤 (왼쪽 정렬)
                name_padded = f"{security_name:<30}"
                trend_info = stock['trend_detail']
                rvol_str = f"RVOL {stock['rvol']:.1f}배"

                print(f"  {name_padded}  {trend_info}, {rvol_str}, WATCH")
        else:
            print("  없음")

        # ================================================================
        # 시각화 저장 (선택)
        # ================================================================
        print("\n" + "="*80)
        viz_choice = input("\n분석 결과를 히트맵으로 저장하시겠습니까? (y/n): ").strip().lower()

        if viz_choice == 'y':
            try:
                print("\n[시각화 생성 중...]")
                saved_path = create_trend_heatmap(results)
                print(f"✓ 히트맵 저장 완료: {saved_path}")
            except Exception as e:
                print(f"✗ 시각화 생성 실패: {e}")
                import traceback
                traceback.print_exc()

        # ================================================================
        # Excel 저장 (조건별 분류만)
        # ================================================================
        print("\n" + "="*80)
        save_choice = input("\n조건별 분류 결과를 Excel로 저장하시겠습니까? (y/n): ").strip().lower()

        if save_choice == 'y':
            import os
            from datetime import datetime as dt

            # 저장 경로 생성
            output_dir = r"C:\Users\Bloomberg\Documents\ssh_project\[오전] start-bloomberg-result"
            os.makedirs(output_dir, exist_ok=True)

            # 파일명에 타임스탬프 추가
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"조건별_분류_{timestamp}.xlsx"
            output_path = os.path.join(output_dir, output_filename)

            # ====================================================================
            # 3가지 카테고리별로 데이터 생성
            # ====================================================================

            # [1] 강력 매도 신호 (10일선 하회 + 거래량 폭증)
            sell_stocks = results_df[results_df['signal'] == 'SELL'].copy()
            sell_data = []
            for _, stock in sell_stocks.iterrows():
                ticker = stock['ticker']
                security_name = ticker_names.get(ticker, ticker)
                sell_data.append({
                    '카테고리': '강력 매도 신호',
                    '종목명': security_name,
                    '티커': ticker,
                    'RVOL': stock['rvol'],
                    '10일선돌파일': stock.get('last_ma10_break_above'),
                    '10일선이탈일': stock.get('last_ma10_break_below'),
                    '추세상세': stock['trend_detail'],
                    '현재가': stock['close_price'],
                    '전일종가': stock['prev_close'],
                    '전일비(%)': stock['price_change_percent'],
                    '10일선': stock['ma10'],
                    '10일선괴리율(%)': stock['ma_distance_percent']
                })

            # [2] 주의 필요 (10일선 하회 + 거래량 부족)
            caution = results_df[
                results_df['condition_1_trend_breakdown'] &
                ~results_df['condition_2_volume_confirmation']
            ].copy()
            caution_data = []
            for _, stock in caution.iterrows():
                ticker = stock['ticker']
                security_name = ticker_names.get(ticker, ticker)
                caution_data.append({
                    '카테고리': '주의 필요',
                    '종목명': security_name,
                    '티커': ticker,
                    'RVOL': stock['rvol'],
                    '10일선돌파일': stock.get('last_ma10_break_above'),
                    '10일선이탈일': stock.get('last_ma10_break_below'),
                    '추세상세': stock['trend_detail'],
                    '현재가': stock['close_price'],
                    '전일종가': stock['prev_close'],
                    '전일비(%)': stock['price_change_percent'],
                    '10일선': stock['ma10'],
                    '10일선괴리율(%)': stock['ma_distance_percent']
                })

            # [3] 거래량 폭증 (10일선 위 + 거래량 폭증)
            rvol_surge = results_df[
                ~results_df['condition_1_trend_breakdown'] &
                results_df['condition_2_volume_confirmation']
            ].copy()
            surge_data = []
            for _, stock in rvol_surge.iterrows():
                ticker = stock['ticker']
                security_name = ticker_names.get(ticker, ticker)
                surge_data.append({
                    '카테고리': '거래량 폭증',
                    '종목명': security_name,
                    '티커': ticker,
                    'RVOL': stock['rvol'],
                    '10일선돌파일': stock.get('last_ma10_break_above'),
                    '10일선이탈일': stock.get('last_ma10_break_below'),
                    '추세상세': stock['trend_detail'],
                    '현재가': stock['close_price'],
                    '전일종가': stock['prev_close'],
                    '전일비(%)': stock['price_change_percent'],
                    '10일선': stock['ma10'],
                    '10일선괴리율(%)': stock['ma_distance_percent']
                })

            # ====================================================================
            # 10일선 이탈일 필터링 함수 정의 (최근 5일 이내)
            # 주의: "거래량 폭증"은 10일선 위에 있어서 이탈일 필터링 제외
            # ====================================================================
            from datetime import date, timedelta

            today = date.today()
            cutoff_date = today - timedelta(days=5)

            def is_recent_breakdown(breakdown_date_str):
                """10일선 이탈일이 최근 5일 이내인지 확인"""
                if pd.isna(breakdown_date_str):
                    return False
                try:
                    # 문자열을 date 객체로 변환
                    breakdown_date = pd.to_datetime(breakdown_date_str).date()
                    # cutoff_date 이후인지 확인 (최근 5일 이내)
                    return breakdown_date >= cutoff_date
                except:
                    return False

            # ====================================================================
            # 이탈일-돌파일 차이 확인 함수 (3일 이상 차이나야 유효)
            # ====================================================================
            def has_valid_date_gap(breakout_date_str, breakdown_date_str):
                """10일선 돌파일과 이탈일의 차이가 3일 이상인지 확인"""
                if pd.isna(breakout_date_str) or pd.isna(breakdown_date_str):
                    return False
                try:
                    breakout_date = pd.to_datetime(breakout_date_str).date()
                    breakdown_date = pd.to_datetime(breakdown_date_str).date()
                    # 이탈일 - 돌파일 >= 3일
                    gap = (breakdown_date - breakout_date).days
                    return gap >= 3
                except:
                    return False

            # ====================================================================
            # 카테고리별로 필터링 적용
            # - "강력 매도 신호", "주의 필요": 10일선 아래 → 이탈일 필터링 + 날짜 차이 확인
            # - "거래량 폭증": 10일선 위 → 필터링 제외 (이탈일 없음)
            # ====================================================================

            # [1] 강력 매도 신호 필터링 (이탈일 최근 5일 + 날짜 차이 3일 이상)
            sell_data_filtered = []
            for item in sell_data:
                if (is_recent_breakdown(item.get('10일선이탈일')) and
                    has_valid_date_gap(item.get('10일선돌파일'), item.get('10일선이탈일'))):
                    sell_data_filtered.append(item)

            # [2] 주의 필요 필터링 (이탈일 최근 5일 + 날짜 차이 3일 이상)
            caution_data_filtered = []
            for item in caution_data:
                if (is_recent_breakdown(item.get('10일선이탈일')) and
                    has_valid_date_gap(item.get('10일선돌파일'), item.get('10일선이탈일'))):
                    caution_data_filtered.append(item)

            # [3] 거래량 폭증은 필터링 제외 (10일선 위에 있음)
            surge_data_filtered = surge_data  # 필터링 안함

            # 필터링 결과 출력
            print(f"\n[필터링] 10일선 이탈일 기준 (최근 5일 이내 + 돌파-이탈 차이 3일 이상)")
            print(f"  - 강력 매도 신호: {len(sell_data)}개 → {len(sell_data_filtered)}개")
            print(f"  - 주의 필요: {len(caution_data)}개 → {len(caution_data_filtered)}개")
            print(f"  - 거래량 폭증: {len(surge_data)}개 (필터링 제외)")

            # 하나의 DataFrame으로 통합 (필터링된 데이터)
            all_category_data = sell_data_filtered + caution_data_filtered + surge_data_filtered
            df_to_save = pd.DataFrame(all_category_data)

            # 소수점 반올림 (RVOL, 전일비(%), 10일선괴리율(%))
            if 'RVOL' in df_to_save.columns:
                df_to_save['RVOL'] = df_to_save['RVOL'].round(1)
            if '전일비(%)' in df_to_save.columns:
                df_to_save['전일비(%)'] = df_to_save['전일비(%)'].round(1)
            if '10일선괴리율(%)' in df_to_save.columns:
                df_to_save['10일선괴리율(%)'] = df_to_save['10일선괴리율(%)'].round(1)

            # 정렬: 10일선 이탈일 내림차순 (최근 먼저) → 10일선 돌파일 오름차순 (오래된 먼저)
            if '10일선돌파일' in df_to_save.columns and '10일선이탈일' in df_to_save.columns:
                df_to_save = df_to_save.sort_values(
                    by=['10일선이탈일', '10일선돌파일'],
                    ascending=[False, True],
                    na_position='last'
                )

            # Excel로 저장 (여러 시트로 분리)
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 전체 데이터 (하나의 시트)
                df_to_save.to_excel(writer, sheet_name='전체', index=False)

                # 카테고리별 시트 (각각 필터링, 반올림 및 정렬 적용)
                # 순서: 강력매도신호 → 주의필요 → 거래량폭증

                # [1] 강력매도신호
                if len(sell_data_filtered) > 0:
                    df_sell = pd.DataFrame(sell_data_filtered)
                    # 반올림
                    if 'RVOL' in df_sell.columns:
                        df_sell['RVOL'] = df_sell['RVOL'].round(1)
                    if '전일비(%)' in df_sell.columns:
                        df_sell['전일비(%)'] = df_sell['전일비(%)'].round(1)
                    if '10일선괴리율(%)' in df_sell.columns:
                        df_sell['10일선괴리율(%)'] = df_sell['10일선괴리율(%)'].round(1)
                    # 정렬
                    if '10일선돌파일' in df_sell.columns and '10일선이탈일' in df_sell.columns:
                        df_sell = df_sell.sort_values(
                            by=['10일선이탈일', '10일선돌파일'],
                            ascending=[False, True],
                            na_position='last'
                        )
                    if len(df_sell) > 0:  # 필터링 후 데이터가 있으면 저장
                        df_sell.to_excel(writer, sheet_name='강력매도신호', index=False)

                # [2] 주의필요 (위로)
                if len(caution_data_filtered) > 0:
                    df_caution = pd.DataFrame(caution_data_filtered)
                    # 반올림
                    if 'RVOL' in df_caution.columns:
                        df_caution['RVOL'] = df_caution['RVOL'].round(1)
                    if '전일비(%)' in df_caution.columns:
                        df_caution['전일비(%)'] = df_caution['전일비(%)'].round(1)
                    if '10일선괴리율(%)' in df_caution.columns:
                        df_caution['10일선괴리율(%)'] = df_caution['10일선괴리율(%)'].round(1)
                    # 정렬
                    if '10일선돌파일' in df_caution.columns and '10일선이탈일' in df_caution.columns:
                        df_caution = df_caution.sort_values(
                            by=['10일선이탈일', '10일선돌파일'],
                            ascending=[False, True],
                            na_position='last'
                        )
                    if len(df_caution) > 0:  # 필터링 후 데이터가 있으면 저장
                        df_caution.to_excel(writer, sheet_name='주의필요', index=False)

                # [3] 거래량폭증 (아래로) - 필터링 제외
                if len(surge_data_filtered) > 0:
                    df_surge = pd.DataFrame(surge_data_filtered)
                    # 반올림
                    if 'RVOL' in df_surge.columns:
                        df_surge['RVOL'] = df_surge['RVOL'].round(1)
                    if '전일비(%)' in df_surge.columns:
                        df_surge['전일비(%)'] = df_surge['전일비(%)'].round(1)
                    if '10일선괴리율(%)' in df_surge.columns:
                        df_surge['10일선괴리율(%)'] = df_surge['10일선괴리율(%)'].round(1)
                    # 정렬
                    if '10일선돌파일' in df_surge.columns and '10일선이탈일' in df_surge.columns:
                        df_surge = df_surge.sort_values(
                            by=['10일선이탈일', '10일선돌파일'],
                            ascending=[False, True],
                            na_position='last'
                        )
                    if len(df_surge) > 0:  # 필터링 후 데이터가 있으면 저장
                        df_surge.to_excel(writer, sheet_name='거래량폭증', index=False)

            print(f"\n✓ Excel 저장 완료: {output_path}")
            print(f"  - 전체 (필터링 후): {len(df_to_save)}개")
            print(f"  - 강력 매도 신호: {len(sell_data)}개")
            print(f"  - 주의 필요: {len(caution_data)}개")
            print(f"  - 거래량 폭증: {len(surge_data)}개")

    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n[에러] 분석 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
