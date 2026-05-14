"""
Taking Profit Screener - 무료 API 버전 (FinanceDataReader)

py -3.12 start_brief.py

start_bloomberg.py와 동일한 로직이지만 Bloomberg 대신 FinanceDataReader(FDR) 사용.
- 한국 주식 (KS, KQ): FDR + KRX 종목 리스트
- 미국 주식 (US): FDR + 티커명 그대로 사용
"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
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
from src.visualizer import create_trend_heatmap

import FinanceDataReader as fdr

# ============================================================================
# 종목명 캐시 (KRX 전체 리스트를 한 번 받아 dict로 보관)
# ============================================================================
_KRX_NAME_CACHE: dict = {}
_CACHE_FILE = os.path.join(current_dir, '.krx_name_cache.json')


def _load_cache_from_disk() -> dict:
    """디스크 캐시 파일에서 종목명 로드"""
    if not os.path.exists(_CACHE_FILE):
        return {}
    try:
        import json
        with open(_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache_to_disk(cache: dict) -> None:
    """종목명 캐시를 디스크에 저장"""
    try:
        import json
        with open(_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass


def _ensure_etf_cache() -> None:
    """ETF 종목명을 한 번 받아 디스크 캐시에 보강"""
    global _KRX_NAME_CACHE
    try:
        etf = fdr.StockListing('ETF/KR')
        added = False
        for col_name in ['Symbol', 'Code']:
            if col_name in etf.columns and 'Name' in etf.columns:
                for _, row in etf.iterrows():
                    code = str(row[col_name]).strip().upper()
                    name = str(row['Name']).strip()
                    if code and name and name != 'nan' and code not in _KRX_NAME_CACHE:
                        _KRX_NAME_CACHE[code] = name
                        added = True
                break
        if added:
            _save_cache_to_disk(_KRX_NAME_CACHE)
    except Exception:
        pass


def _resolve_kr_name(code: str) -> str:
    """단일 한국 종목코드 → 종목명 (캐시/pykrx 단건/ETF 순)"""
    global _KRX_NAME_CACHE
    code = code.strip().upper()

    # 1. 메모리/디스크 캐시
    if code in _KRX_NAME_CACHE:
        return _KRX_NAME_CACHE[code]

    # 2. pykrx 단건 조회
    try:
        from pykrx import stock as pykrx_stock
        nm = pykrx_stock.get_market_ticker_name(code)
        if nm and nm.strip():
            _KRX_NAME_CACHE[code] = nm.strip()
            return nm.strip()
    except Exception:
        pass

    return code


def _build_krx_name_cache() -> dict:
    """디스크 캐시 + ETF 리스트만 우선 보강 (전체 KRX 빌드는 비활성화)"""
    global _KRX_NAME_CACHE
    if _KRX_NAME_CACHE:
        return _KRX_NAME_CACHE

    # 디스크 캐시 우선
    disk_cache = _load_cache_from_disk()
    if disk_cache:
        _KRX_NAME_CACHE = disk_cache

    # ETF 캐시는 가벼우니 한 번 갱신
    _ensure_etf_cache()

    return _KRX_NAME_CACHE


def parse_ticker(ticker: str) -> tuple:
    """
    Bloomberg 형식 티커를 (code, market)으로 분리

    "005930 KS" → ("005930", "KS")
    "AAPL US" → ("AAPL", "US")
    """
    parts = ticker.strip().split()
    if len(parts) >= 2:
        return (parts[0].upper(), parts[1].upper())
    return (parts[0].upper(), None)


def _period_to_start_date(period: str) -> datetime:
    """기간 문자열을 시작 날짜로 변환"""
    end = datetime.now()
    period = period.upper().strip()

    if period == '1M':
        return end - timedelta(days=45)
    if period == '3M':
        return end - timedelta(days=120)
    if period == '6M':
        return end - timedelta(days=210)
    if period == '1Y':
        return end - timedelta(days=400)
    if period == '2Y':
        return end - timedelta(days=760)
    if period == '3Y':
        return end - timedelta(days=1130)
    return end - timedelta(days=120)


def download_data(ticker: str, period: str = '3M', verbose: bool = True) -> pd.DataFrame:
    """
    FDR로 OHLCV 다운로드 (Bloomberg의 download_bloomberg_data 대체)

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    code, market = parse_ticker(ticker)

    if market not in ('KS', 'KQ', 'US'):
        if verbose:
            print(f"  지원하지 않는 시장: {market}")
        return None

    start = _period_to_start_date(period)
    end = datetime.now()

    try:
        df = fdr.DataReader(code, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    except Exception as e:
        if verbose:
            print(f"  데이터 다운로드 실패: {e}")
        return None

    if df is None or len(df) == 0:
        return None

    # FDR 결과: Index=DatetimeIndex, columns=[Open, High, Low, Close, Volume, Change, ...]
    df = df.reset_index()

    # Date 컬럼명 통일
    date_col = None
    for cand in ('Date', 'date', 'index'):
        if cand in df.columns:
            date_col = cand
            break
    if date_col and date_col != 'Date':
        df = df.rename(columns={date_col: 'Date'})

    # 필수 컬럼 확인
    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        if verbose:
            print(f"  컬럼 부족: {missing}, 사용 가능한 컬럼: {list(df.columns)}")
        return None

    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None) if df['Date'].dt.tz is not None else pd.to_datetime(df['Date'])
    return df[required].copy()


def get_security_name(ticker: str) -> str:
    """단일 티커 → 종목명"""
    code, market = parse_ticker(ticker)

    if market in ('KS', 'KQ'):
        _build_krx_name_cache()
        return _resolve_kr_name(code) or ticker

    # 미국 종목은 티커 그대로 표시 (FDR로 종목명 조회는 비효율적)
    return ticker


def get_multiple_security_names(tickers: list) -> dict:
    """다수 티커 → 종목명 dict"""
    _build_krx_name_cache()  # 디스크 캐시 + ETF 보강

    result = {}
    for t in tickers:
        result[t] = get_security_name(t)

    # 새로 받은 한국 종목명 디스크에 저장
    _save_cache_to_disk(_KRX_NAME_CACHE)

    return result


def analyze_from_brief(ticker: str, period: str = '3M', show_progress: bool = True, mode: int = 1) -> dict:
    """
    FDR에서 데이터를 받아 분석 (start_bloomberg.py의 analyze_from_bloomberg과 동일 로직)

    Parameters:
    -----------
    ticker : str
        Bloomberg 형식 티커 (예: "005930 KS")
    period : str
        데이터 기간 (기본값: '3M')
    show_progress : bool
        진행 상황 표시 여부
    mode : int
        1: 보고용 (완성된 일봉)
        2: 실시간 (현재 시점)

    Returns:
    --------
    dict : 분석 결과
    """
    try:
        # ================================================================
        # [1단계] FDR에서 데이터 다운로드
        # ================================================================
        df = download_data(ticker, period=period, verbose=show_progress)

        if df is None or len(df) == 0:
            return None

        # ================================================================
        # [1-1단계] 당일 데이터 제외 (모드 1: 보고용에서만, 장중에만)
        # ================================================================
        from datetime import datetime as dt, time
        now = dt.now()
        today = now.date()
        current_time = now.time()
        market_close_time = time(15, 30)

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['date_only'] = df['Date'].dt.date

            if mode == 1 and current_time < market_close_time:
                if (df['date_only'] == today).any():
                    df = df[df['date_only'] != today].copy()

            df = df.drop(columns=['date_only'])

        if len(df) == 0:
            return None

        # ================================================================
        # [2단계] 상세 분석
        # ================================================================
        analyzer = StockAnalyzer()
        result = analyzer.analyze_latest(df, ticker)

        return result

    except Exception:
        return None


def main():
    """메인 함수 (start_bloomberg.py와 동일 흐름)"""
    print("="*80)
    print("TAKING PROFIT SCREENER - 무료 API 버전 (FinanceDataReader)")
    print("Bloomberg 없이 무료 데이터로 분석합니다")
    print("="*80)

    print("\n참고:")
    print("  - 한국 주식 (KS, KQ): FinanceDataReader + KRX")
    print("  - 미국 주식 (US): FinanceDataReader + Yahoo")

    # ====================================================================
    # 티커 입력
    # ====================================================================
    print("\n" + "="*80)
    print("티커를 입력하세요 (쉼표로 구분)")
    print("="*80)
    print("\n티커 형식 (Bloomberg 형식과 동일):")
    print("  - 한국 주식: 005930 KS (삼성전자), 000660 KS (SK하이닉스)")
    print("  - 미국 주식: AAPL US (애플), MSFT US (마이크로소프트)")
    print("  - 예시: 005930 KS, 000660 KS, AAPL US")

    try:
        user_input = input("\n티커 입력: ").strip()

        if not user_input:
            print("\n[에러] 티커를 입력해주세요")
            return

        tickers = [t.strip() for t in user_input.split(',')]
        print(f"\n입력된 티커: {len(tickers)}개")

        # ================================================================
        # 제외할 티커 입력 (선택사항)
        # ================================================================
        print("\n" + "-"*40)
        exclude_input = input("제외할 티커 입력 (없으면 엔터): ").strip()

        if exclude_input:
            exclude_tickers = [t.strip() for t in exclude_input.split(',')]
            before_count = len(tickers)
            tickers = [t for t in tickers if t not in exclude_tickers]
            after_count = len(tickers)
            print(f"✓ 제외 완료: {before_count}개 → {after_count}개 ({before_count - after_count}개 제외)")

        print(f"\n최종 분석 대상: {len(tickers)}개")

        # ================================================================
        # 모드 선택 (보고용 vs 실시간)
        # ================================================================
        print("\n" + "="*80)
        print("분석 모드 선택")
        print("="*80)
        print("\n[1] 보고용 - 완성된 일봉 데이터 기준")
        print("    → 장중: 전일까지의 데이터 사용")
        print("    → 장마감 후(15:30 이후): 당일 포함")
        print("\n[2] 실시간 - 현재 시점 기준")
        print("    → 장중 미완성 데이터 포함")

        while True:
            mode_input = input("\n모드 선택 (1 또는 2): ").strip()
            if mode_input in ['1', '2']:
                mode = int(mode_input)
                break
            print("1 또는 2를 입력해주세요.")

        mode_name = "보고용 (완성된 일봉)" if mode == 1 else "실시간 (현재 시점)"
        print(f"\n✓ 선택된 모드: {mode_name}")

        # ================================================================
        # 종목명 조회 (KRX 리스트 캐시)
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

        max_workers = 10
        print(f"병렬 처리: {max_workers}개 동시 실행\n")

        def analyze_single(ticker):
            try:
                result = analyze_from_brief(ticker, period=period, show_progress=False, mode=mode)
                return (ticker, result, None)
            except Exception as e:
                return (ticker, None, str(e))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(analyze_single, ticker): ticker
                               for ticker in tickers}

            for future in as_completed(future_to_ticker):
                ticker, result, error = future.result()

                with lock:
                    completed_count += 1

                    if result:
                        results.append(result)
                    else:
                        failed_count += 1

                    elapsed = dt.now() - start_time
                    progress = completed_count / len(tickers) * 100

                    bar_length = 40
                    filled_length = int(bar_length * completed_count // len(tickers))
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)

                    print(f"\r진행: [{bar}] {completed_count}/{len(tickers)} ({progress:.1f}%) | "
                          f"성공: {len(results)} | 실패: {failed_count} | "
                          f"경과: {str(elapsed).split('.')[0]}", end='', flush=True)

        print()
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

        summary_data = []
        for row in results:
            if row.get('prev_close') and row['prev_close'] > 0:
                price_change_pct = row['price_change_percent']
                price_change_str = f"{price_change_pct:+.1f}%"
            else:
                price_change_str = "-"

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

        summary_df = pd.DataFrame(summary_data)
        print("\n" + tabulate(summary_df, headers='keys', tablefmt='simple', showindex=False))

        # ================================================================
        # 추세별 분류 (하락세 vs 상승세)
        # ================================================================
        results_df = pd.DataFrame(results)

        print("\n" + "="*80)
        print("추세별 분류")
        print("="*80)

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

        # ================================================================
        # 조건별 분류
        # ================================================================
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
            from datetime import datetime as dt

            output_dir = r"C:\Users\10845\Documents\quant_project\[오전] start-brief-result"
            os.makedirs(output_dir, exist_ok=True)

            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            mode_suffix = "보고용" if mode == 1 else "실시간"
            output_filename = f"조건별_분류_{mode_suffix}_{timestamp}.xlsx"
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
                    'RSI': stock.get('rsi'),
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
                    '카테고리': 'profit taking',
                    '종목명': security_name,
                    '티커': ticker,
                    'RVOL': stock['rvol'],
                    'RSI': stock.get('rsi'),
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
                    '카테고리': 'upside',
                    '종목명': security_name,
                    '티커': ticker,
                    'RVOL': stock['rvol'],
                    'RSI': stock.get('rsi'),
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
            # ====================================================================
            from datetime import date, timedelta

            today = date.today()
            cutoff_date = today - timedelta(days=5)

            def is_recent_breakdown(breakdown_date_str):
                if pd.isna(breakdown_date_str):
                    return False
                try:
                    breakdown_date = pd.to_datetime(breakdown_date_str).date()
                    return breakdown_date >= cutoff_date
                except Exception:
                    return False

            def has_valid_date_gap(breakout_date_str, breakdown_date_str):
                if pd.isna(breakout_date_str) or pd.isna(breakdown_date_str):
                    return False
                try:
                    breakout_date = pd.to_datetime(breakout_date_str).date()
                    breakdown_date = pd.to_datetime(breakdown_date_str).date()
                    gap = (breakdown_date - breakout_date).days
                    return gap >= 3
                except Exception:
                    return False

            # ====================================================================
            # 카테고리별 필터링
            # ====================================================================
            sell_data_filtered = []
            for item in sell_data:
                if (is_recent_breakdown(item.get('10일선이탈일')) and
                    has_valid_date_gap(item.get('10일선돌파일'), item.get('10일선이탈일'))):
                    sell_data_filtered.append(item)

            caution_data_filtered = []
            for item in caution_data:
                if (is_recent_breakdown(item.get('10일선이탈일')) and
                    has_valid_date_gap(item.get('10일선돌파일'), item.get('10일선이탈일'))):
                    caution_data_filtered.append(item)

            surge_data_filtered = surge_data  # 10일선 위 종목은 필터 제외

            print(f"\n[필터링] 10일선 이탈일 기준 (최근 5일 이내 + 돌파-이탈 차이 3일 이상)")
            print(f"  - 강력 매도 신호: {len(sell_data)}개 → {len(sell_data_filtered)}개")
            print(f"  - 주의 필요: {len(caution_data)}개 → {len(caution_data_filtered)}개")
            print(f"  - 거래량 폭증: {len(surge_data)}개 (필터링 제외)")

            all_category_data = sell_data_filtered + caution_data_filtered + surge_data_filtered
            df_to_save = pd.DataFrame(all_category_data)

            # 소수점 반올림
            for col in ['RVOL', 'RSI', '전일비(%)', '10일선괴리율(%)']:
                if col in df_to_save.columns:
                    df_to_save[col] = df_to_save[col].round(1)

            # 정렬
            if '카테고리' in df_to_save.columns:
                category_order = {'강력 매도 신호': 0, 'profit taking': 1, 'upside': 2}
                df_to_save['카테고리_순서'] = df_to_save['카테고리'].map(category_order).fillna(99)

                if '10일선돌파일' in df_to_save.columns and '10일선이탈일' in df_to_save.columns:
                    df_to_save = df_to_save.sort_values(
                        by=['카테고리_순서', '10일선이탈일', '10일선돌파일'],
                        ascending=[True, False, True],
                        na_position='last'
                    )
                else:
                    df_to_save = df_to_save.sort_values(by=['카테고리_순서'], ascending=[True])

                df_to_save = df_to_save.drop(columns=['카테고리_순서'])

            # ====================================================================
            # Excel 저장 (시트별 분리)
            # ====================================================================
            def _round_and_sort(df_part):
                for col in ['RVOL', 'RSI', '전일비(%)', '10일선괴리율(%)']:
                    if col in df_part.columns:
                        df_part[col] = df_part[col].round(1)
                if '10일선돌파일' in df_part.columns and '10일선이탈일' in df_part.columns:
                    df_part = df_part.sort_values(
                        by=['10일선이탈일', '10일선돌파일'],
                        ascending=[False, True],
                        na_position='last'
                    )
                return df_part

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_to_save.to_excel(writer, sheet_name='전체', index=False)

                if len(sell_data_filtered) > 0:
                    df_sell = _round_and_sort(pd.DataFrame(sell_data_filtered))
                    if len(df_sell) > 0:
                        df_sell.to_excel(writer, sheet_name='강력매도신호', index=False)

                if len(caution_data_filtered) > 0:
                    df_caution = _round_and_sort(pd.DataFrame(caution_data_filtered))
                    if len(df_caution) > 0:
                        df_caution.to_excel(writer, sheet_name='profit taking', index=False)

                if len(surge_data_filtered) > 0:
                    df_surge = _round_and_sort(pd.DataFrame(surge_data_filtered))
                    if len(df_surge) > 0:
                        df_surge.to_excel(writer, sheet_name='upside', index=False)

            print(f"\n✓ Excel 저장 완료: {output_path}")
            print(f"  - 전체 (필터링 후): {len(df_to_save)}개")
            print(f"  - 강력 매도 신호: {len(sell_data)}개")
            print(f"  - profit taking: {len(caution_data)}개")
            print(f"  - upside: {len(surge_data)}개")

    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n[에러] 분석 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
