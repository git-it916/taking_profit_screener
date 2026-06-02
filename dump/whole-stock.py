"""
Taking Profit Screener - 거래량 폭증 종목 스크리닝

py -3.12 whole-stock.py

FinanceDataReader로 전종목 데이터를 받아 분석합니다.
"거래량 폭증" 종목만 필터링: 10일선 돌파 + RVOL≥1.5
"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta, time
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
import FinanceDataReader as fdr


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
    period = period.upper().strip()
    end = datetime.now()

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


def _previous_business_day(day):
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _expected_cache_date(mode: int = 1):
    now = datetime.now()
    today = now.date()
    expected = _previous_business_day(today)

    if mode == 1 and expected == today and now.time() < time(15, 30):
        expected = _previous_business_day(today - timedelta(days=1))

    return expected


def _normalize_ohlcv(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return None

    df = df.reset_index()
    date_col = None
    for cand in ('Date', 'date', 'index'):
        if cand in df.columns:
            date_col = cand
            break
    if date_col and date_col != 'Date':
        df = df.rename(columns={date_col: 'Date'})

    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        if verbose:
            print(f"  컬럼 부족: {missing}, 사용 가능한 컬럼: {list(df.columns)}")
        return None

    date_series = pd.to_datetime(df['Date'])
    df['Date'] = date_series.dt.tz_localize(None) if date_series.dt.tz is not None else date_series
    return df[required].copy()


def _get_cache_path(ticker: str) -> str:
    code, market = parse_ticker(ticker)
    cache_dir = os.path.join(current_dir, 'database', 'fdr_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{code}_{market}.csv")


def download_data(ticker: str, period: str = '3M', verbose: bool = True, mode: int = 1) -> pd.DataFrame:
    code, market = parse_ticker(ticker)
    if market not in ('KS', 'KQ', 'US'):
        if verbose:
            print(f"  지원하지 않는 시장: {market}")
        return None

    requested_start = _period_to_start_date(period)
    end = datetime.now()
    cache_path = _get_cache_path(ticker)
    cached_df = None

    if os.path.exists(cache_path):
        try:
            cached_df = pd.read_csv(cache_path, parse_dates=['Date'])
            cached_df = cached_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception:
            cached_df = None

    if cached_df is not None and not cached_df.empty:
        latest_cached = pd.to_datetime(cached_df['Date']).max()
        earliest_cached = pd.to_datetime(cached_df['Date']).min()
        if earliest_cached <= requested_start and latest_cached.date() >= _expected_cache_date(mode):
            return cached_df[cached_df['Date'] >= requested_start][['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        if earliest_cached > requested_start:
            fetch_start = requested_start
        else:
            fetch_start = max(requested_start, latest_cached)
    else:
        fetch_start = requested_start

    try:
        new_df = fdr.DataReader(code, fetch_start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    except Exception as e:
        if verbose:
            print(f"  데이터 다운로드 실패: {e}")
        new_df = None

    new_df = _normalize_ohlcv(new_df, verbose=verbose)

    frames = [df for df in (cached_df, new_df) if df is not None and not df.empty]
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
    df.to_csv(cache_path, index=False, encoding='utf-8-sig')
    return df[df['Date'] >= requested_start][['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()


def save_accumulated_workbook(output_filename: str, sheets: dict, dedupe_keys: dict, replace_filters: dict = None) -> None:
    replace_filters = replace_filters or {}
    existing_sheets = {}
    if os.path.exists(output_filename):
        try:
            existing_sheets = pd.read_excel(output_filename, sheet_name=None)
        except Exception as e:
            print(f"  기존 누적 파일 읽기 실패, 새로 저장합니다: {e}")

    for sheet_name, new_df in sheets.items():
        new_df = new_df.copy()

        if sheet_name in existing_sheets:
            existing_df = existing_sheets[sheet_name]
            filters = replace_filters.get(sheet_name, {})
            if filters:
                remove_mask = pd.Series(True, index=existing_df.index)
                for col, value in filters.items():
                    if col in existing_df.columns:
                        remove_mask &= existing_df[col].astype(str) == str(value)
                    else:
                        remove_mask &= False
                existing_df = existing_df[~remove_mask].copy()
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        keys = [key for key in dedupe_keys.get(sheet_name, []) if key in combined_df.columns]
        if keys:
            combined_df = combined_df.drop_duplicates(subset=keys, keep='last')

        if '기준일' in combined_df.columns:
            combined_df['_row_order'] = range(len(combined_df))
            combined_df['_기준일_sort'] = pd.to_datetime(combined_df['기준일'], errors='coerce')
            combined_df = combined_df.sort_values(
                by=['_기준일_sort', '_row_order'],
                ascending=[False, True],
                na_position='last'
            ).drop(columns=['_기준일_sort', '_row_order'])

        existing_sheets[sheet_name] = combined_df

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        for sheet_name, df in existing_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def _get_krx_listing() -> pd.DataFrame:
    cache_path = os.path.join(current_dir, 'database', 'krx_listing_cache.csv')
    try:
        listing = fdr.StockListing('KRX')
        if listing is not None and not listing.empty:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            listing.to_csv(cache_path, index=False, encoding='utf-8-sig')
            return listing
    except Exception:
        pass

    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass
    return pd.DataFrame()


def _get_krx_etf_listing() -> pd.DataFrame:
    cache_path = os.path.join(current_dir, 'database', 'krx_etf_listing_cache.csv')
    try:
        listing = fdr.StockListing('ETF/KR')
        if listing is not None and not listing.empty:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            listing.to_csv(cache_path, index=False, encoding='utf-8-sig')
            return listing
    except Exception:
        pass

    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass
    return pd.DataFrame()


def _load_ticker_metadata_from_excel(file_path: str) -> dict:
    metadata = {}
    if not file_path or not os.path.exists(file_path):
        return metadata

    try:
        xl = pd.ExcelFile(file_path)
        for sheet in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            ticker_col = None
            for cand in ('티커', 'bloomberg_ticker'):
                if cand in df.columns:
                    ticker_col = cand
                    break
            if ticker_col is None:
                continue

            name_col = '종목명' if '종목명' in df.columns else None
            for _, row in df.iterrows():
                ticker = str(row.get(ticker_col, '')).strip()
                if not ticker or ticker == 'nan':
                    continue
                if ' ' not in ticker:
                    exchange = str(row[df.columns[1]]).strip().upper() if len(df.columns) > 1 else ''
                    ticker = f"{ticker} {exchange}" if exchange in ('KS', 'KQ', 'US') else ticker

                item = metadata.setdefault(ticker, {})
                if name_col:
                    name = row.get(name_col)
                    if pd.notna(name) and str(name).strip():
                        item['name'] = str(name).strip()
    except Exception:
        pass
    return metadata


def get_security_metadata(tickers: list, excel_metadata: dict = None) -> tuple:
    excel_metadata = excel_metadata or {}
    listing = _get_krx_listing()
    names = {}
    market_caps = {}

    if not listing.empty and {'Code', 'Name'}.issubset(listing.columns):
        listing = listing.copy()
        listing['Code'] = listing['Code'].astype(str).str.upper().str.zfill(6)
        by_code = listing.drop_duplicates('Code').set_index('Code')
    else:
        by_code = pd.DataFrame()

    etf_listing = _get_krx_etf_listing()
    if not etf_listing.empty and {'Symbol', 'Name'}.issubset(etf_listing.columns):
        etf_listing = etf_listing.copy()
        etf_listing['Symbol'] = etf_listing['Symbol'].astype(str).str.upper().str.zfill(6)
        etf_by_code = etf_listing.drop_duplicates('Symbol').set_index('Symbol')
    else:
        etf_by_code = pd.DataFrame()

    for ticker in tickers:
        code, market = parse_ticker(ticker)
        excel_name = excel_metadata.get(ticker, {}).get('name')
        names[ticker] = excel_name or ticker
        market_caps[ticker] = None

        if market in ('KS', 'KQ') and not by_code.empty and code in by_code.index:
            row = by_code.loc[code]
            name = row.get('Name')
            marcap = row.get('Marcap')
            if pd.notna(name) and str(name).strip():
                names[ticker] = str(name).strip()
            if pd.notna(marcap):
                market_caps[ticker] = pd.to_numeric(marcap, errors='coerce')
        elif market in ('KS', 'KQ') and not etf_by_code.empty and code in etf_by_code.index:
            row = etf_by_code.loc[code]
            name = row.get('Name')
            marcap = row.get('MarCap')
            if pd.notna(name) and str(name).strip():
                names[ticker] = str(name).strip()
            if pd.notna(marcap):
                # FinanceDataReader ETF/KR MarCap is in KRW 100M units.
                market_caps[ticker] = pd.to_numeric(marcap, errors='coerce') * 100_000_000

    return names, market_caps


def get_security_name(ticker: str) -> str:
    names, _ = get_security_metadata([ticker])
    return names.get(ticker, ticker)


def get_multiple_security_names(tickers: list) -> dict:
    names, _ = get_security_metadata(tickers)
    return names


def get_multiple_market_caps(tickers: list, excel_metadata: dict = None) -> dict:
    _, market_caps = get_security_metadata(tickers, excel_metadata)
    return market_caps


def get_tickers_from_excel(file_path: str) -> tuple:
    """
    엑셀 파일에서 모든 시트의 티커 리스트 읽기

    읽는 시트:
    - 첫 번째 시트 (기존 국내 주식)
    - 해외ETF 시트 (있으면)
    - 한국ETF 시트 (있으면)

    Returns:
        (all_tickers, etf_tickers_set) 튜플
    """
    import pandas as pd

    print(f"\n[엑셀 파일에서 티커 읽기]")
    print(f"  파일: {file_path}")

    ETF_SHEETS = {'해외ETF', '한국ETF'}

    try:
        # 전체 시트 목록 확인
        xl = pd.ExcelFile(file_path)
        all_sheets = xl.sheet_names
        print(f"  시트 목록: {all_sheets}")

        all_tickers = []
        etf_tickers = []

        def parse_tickers_from_df(df, sheet_name):
            """DataFrame에서 티커 파싱"""
            tickers = []

            # 티커 컬럼 찾기 (bloomberg_ticker > 티커 > 첫 번째 컬럼)
            if 'bloomberg_ticker' in df.columns:
                target_col = 'bloomberg_ticker'
            elif '티커' in df.columns:
                target_col = '티커'
            else:
                target_col = df.columns[0]

            for _, row in df.iterrows():
                ticker_value = str(row[target_col]).strip()

                if not ticker_value or ticker_value == 'nan':
                    continue

                # "005930 KS" 또는 "SPY US" 형식
                if ' ' in ticker_value:
                    tickers.append(ticker_value)
                else:
                    if len(df.columns) > 1:
                        exchange = str(row[df.columns[1]]).strip().upper()
                        if exchange in ['KS', 'KQ']:
                            tickers.append(f"{ticker_value} {exchange}")
                        elif len(ticker_value) == 6 and ticker_value.isdigit():
                            tickers.append(f"{ticker_value} KS")
                        else:
                            tickers.append(ticker_value)
                    else:
                        if len(ticker_value) == 6 and ticker_value.isdigit():
                            tickers.append(f"{ticker_value} KS")
                        else:
                            tickers.append(ticker_value)
            return tickers

        # 각 시트별 읽기
        for sheet in all_sheets:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet)
                tickers = parse_tickers_from_df(df, sheet)
                all_tickers.extend(tickers)
                if sheet in ETF_SHEETS:
                    etf_tickers.extend(tickers)
                print(f"  OK [{sheet}] {len(tickers)}개 로드")
            except Exception as e:
                print(f"  FAIL [{sheet}] 읽기 실패: {e}")

        # 중복 제거
        all_tickers = list(set(all_tickers))
        etf_tickers_set = set(etf_tickers)

        print(f"\n  총 {len(all_tickers)}개 티커 로드 완료 (ETF: {len(etf_tickers_set)}개)")
        print(f"\n  샘플 티커 (처음 5개):")
        for ticker in all_tickers[:5]:
            print(f"    - {ticker}")

        return all_tickers, etf_tickers_set

    except Exception as e:
        print(f"  FAIL 파일 읽기 실패: {e}")
        import traceback
        traceback.print_exc()
        return [], set()


def analyze_from_fdr(ticker: str, period: str = '3M', mode: int = 1) -> dict:
    """
    FDR에서 데이터를 받아 분석

    Parameters:
    -----------
    ticker : str
        Bloomberg 형식 티커
    period : str
        데이터 기간 (기본값: '3M' - 3개월)
    mode : int
        1 = 보고용 (전일 또는 장마감 후 당일 완성된 데이터)
        2 = 실시간 (현재 시점의 미완성 데이터 포함)

    Returns:
    --------
    dict : 분석 결과
    """
    try:
        df = download_data(ticker, period=period, verbose=False, mode=mode)

        if df is None or len(df) == 0:
            return None

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['_date_only'] = df['Date'].dt.date

            from datetime import datetime as _dt, time as _time
            _now = _dt.now()
            _today = _now.date()
            _market_close = _time(15, 30)

            if mode == 1 and _now.time() < _market_close:
                is_today = df['_date_only'] == _today
                if is_today.any():
                    df = df[~is_today].copy()

            df = df.drop(columns=['_date_only'])

        if len(df) == 0:
            return None

        analyzer = StockAnalyzer()
        result = analyzer.analyze_latest(df, ticker)

        return result

    except Exception:
        return None


def analyze_tickers_parallel(tickers: list, period: str = '3M', max_workers: int = 30, mode: int = 1) -> list:
    """
    병렬 방식으로 여러 티커 분석 (FinanceDataReader 병렬 호출)

    Parameters:
    -----------
    tickers : list
        분석할 티커 리스트
    period : str
        데이터 기간
    max_workers : int
        동시 실행 worker 수 (기본값: 3 - 안전한 수준)
    mode : int
        1 = 보고용 (전일 또는 장마감 후 당일 완성된 데이터)
        2 = 실시간 (현재 시점의 미완성 데이터 포함)

    Returns:
    --------
    list : 분석 결과 리스트
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    results = []
    failed_tickers = []
    completed_count = 0
    lock = Lock()

    mode_name = "보고용 (완성된 일봉)" if mode == 1 else "실시간 (현재 시점)"
    print(f"\n총 {len(tickers)}개 종목 분석 시작...")
    print(f"분석 모드: {mode_name}")
    print(f"병렬 처리: {max_workers}개 동시 실행")
    print(f"⚠️  FinanceDataReader를 통해 데이터를 가져옵니다.\n")

    start_time = datetime.now()

    def analyze_single(ticker):
        """단일 티커 분석 (worker thread에서 실행)"""
        try:
            result = analyze_from_fdr(ticker, period=period, mode=mode)
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
                    failed_tickers.append(ticker)

                # 진행 상황 표시
                elapsed = datetime.now() - start_time
                progress = completed_count / len(tickers) * 100
                rate = completed_count / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                remaining = (len(tickers) - completed_count) / rate if rate > 0 else 0

                # 한 줄로 출력 (이전 줄 덮어쓰기)
                print(f"\r[진행중] {completed_count}/{len(tickers)} ({progress:.1f}%) "
                      f"| 경과: {str(elapsed).split('.')[0]} | 속도: {rate:.2f}종목/초 | "
                      f"남은시간: ~{int(remaining/60)}분 {int(remaining%60)}초", end='', flush=True)

    print()  # 줄바꿈
    total_time = datetime.now() - start_time

    print(f"\n✓ 분석 완료 - 소요시간: {str(total_time).split('.')[0]}")
    print(f"  성공: {len(results)}개, 실패: {len(failed_tickers)}개")

    if failed_tickers:
        print(f"\n⚠️  실패한 종목 ({len(failed_tickers)}개):")
        for ticker in failed_tickers[:10]:  # 처음 10개만 표시
            print(f"  - {ticker}")
        if len(failed_tickers) > 10:
            print(f"  ... 외 {len(failed_tickers) - 10}개")

    return results


def filter_volume_surge_breakout(results: list, rvol_threshold: float = 1.5) -> pd.DataFrame:
    """
    거래량 폭증 종목 필터링: 10일선 돌파 + RVOL≥1.5

    조건:
    - condition_1_trend_breakdown = False (10일선 위)
    - condition_2_volume_confirmation = True (RVOL >= 1.5)

    Parameters:
    -----------
    results : list
        분석 결과 리스트
    rvol_threshold : float
        RVOL 최소 기준 (기본값: 1.5)

    Returns:
    --------
    pd.DataFrame : 필터링된 결과
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # ====================================================================
    # 디버깅: 각 조건별 종목 수 출력
    # ====================================================================
    print(f"\n[디버깅] 전체 분석 결과: {len(df)}개")

    # 10일선 위 종목 (condition_1_trend_breakdown = False)
    above_ma10 = ~df['condition_1_trend_breakdown']
    print(f"[디버깅] 10일선 위 종목: {above_ma10.sum()}개")

    # RVOL >= 1.5 종목 (condition_2_volume_confirmation = True)
    high_rvol = df['condition_2_volume_confirmation']
    print(f"[디버깅] RVOL >= 1.5 종목: {high_rvol.sum()}개")

    # 거래량 폭증 조건 (start_bloomberg.py와 동일)
    # - 10일선 위 (condition_1_trend_breakdown = False)
    # - 거래량 폭증 (condition_2_volume_confirmation = True)
    condition_surge = above_ma10 & high_rvol
    print(f"[디버깅] 두 조건 모두 충족 (10일선 위 + RVOL>=1.5): {condition_surge.sum()}개")

    # 필터링
    filtered = df[condition_surge].copy()

    # trend_detail에서 10일선 위 날짜 추출하여 정렬 (최근 날짜가 위로)
    # 형식: "10일선 아래(2026-01-08) → 10일선 위(2026-01-13)"
    import re

    def extract_crossover_date(trend_detail):
        """10일선 위 날짜를 추출 (10일선 돌파 날짜)"""
        if pd.isna(trend_detail):
            return None
        # "10일선 위(YYYY-MM-DD)" 패턴 추출
        match = re.search(r'10일선 위\((\d{4}-\d{2}-\d{2})\)', trend_detail)
        if match:
            return pd.to_datetime(match.group(1))
        return None

    # 10일선 돌파 날짜 추출
    filtered['crossover_date'] = filtered['trend_detail'].apply(extract_crossover_date)

    # 10일선 돌파 날짜 기준 내림차순 정렬 (최근 날짜가 위로)
    # 날짜가 없는 경우 맨 아래로
    filtered = filtered.sort_values('crossover_date', ascending=False, na_position='last')

    # 임시 컬럼 제거
    filtered = filtered.drop(columns=['crossover_date'])

    return filtered


def filter_below_ma10(results: list) -> pd.DataFrame:
    """
    10일선 하회 종목 필터링 (RVOL 무관)

    조건:
    - condition_1_trend_breakdown = True (10일선 아래)

    Parameters:
    -----------
    results : list
        분석 결과 리스트

    Returns:
    --------
    pd.DataFrame : 필터링된 결과
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # 디버깅 출력
    print(f"\n[디버깅] 전체 분석 결과: {len(df)}개")

    # 10일선 아래 종목 (condition_1_trend_breakdown = True)
    below_ma10 = df['condition_1_trend_breakdown']
    print(f"[디버깅] 10일선 하회 종목: {below_ma10.sum()}개")

    # 필터링
    filtered = df[below_ma10].copy()

    # 10일선 이탈일 기준 정렬 (최근 이탈일이 위로)
    if 'last_ma10_break_below' in filtered.columns:
        filtered = filtered.sort_values('last_ma10_break_below', ascending=False, na_position='last')

    return filtered


def main():
    """메인 함수"""
    print("="*80)
    print("TAKING PROFIT SCREENER - 거래량 폭증 종목 스크리닝")
    print("="*80)

    print("\n⚠️  주의사항:")
    print("  1. 인터넷 연결이 필요합니다")
    print("  2. FinanceDataReader로 데이터를 가져옵니다")
    print("  3. 전종목 분석은 약 10~15분 소요됩니다 (3개월 데이터)")
    print("\n📊 스크리닝 조건:")
    print("  - 10일선 돌파 (10일선 위)")
    print("  - 거래량 폭증 (RVOL ≥ 1.5배)")

    # ====================================================================
    # 분석 모드 선택
    # ====================================================================
    print("\n" + "="*80)
    print("분석 모드 선택")
    print("="*80)
    print("\n[1] 보고용 - 전일 또는 장마감 후 당일 (완성된 일봉)")
    print("    → 장중: 전일까지의 데이터 사용")
    print("    → 장마감 후(15:30 이후): 당일 포함")
    print("\n[2] 실시간 - 현재 시점의 10일선 돌파 및 RVOL 확인")
    print("    → 장중 미완성 데이터 포함")
    print("    → 현재 거래량 기준 RVOL 계산")

    while True:
        mode_input = input("\n모드 선택 (1 또는 2): ").strip()
        if mode_input in ['1', '2']:
            mode = int(mode_input)
            break
        print("1 또는 2를 입력해주세요.")

    mode_name = "보고용 (완성된 일봉)" if mode == 1 else "실시간 (현재 시점)"
    print(f"\n✓ 선택된 모드: {mode_name}")

    # ====================================================================
    # 스크리닝 타입 선택
    # ====================================================================
    print("\n" + "="*80)
    print("스크리닝 타입 선택")
    print("="*80)
    print("\n[A] 10일선 돌파 + RVOL >= 1.5 (거래량 폭증)")
    print("    → 10일선 위 + 거래량 폭증 종목")
    print("\n[B] 10일선 하회만")
    print("    → 10일선 아래 종목 (RVOL 무관)")

    while True:
        screen_type = input("\n스크리닝 타입 선택 (A 또는 B): ").strip().upper()
        if screen_type in ['A', 'B']:
            break
        print("A 또는 B를 입력해주세요.")

    screen_type_name = "10일선 돌파 + RVOL >= 1.5" if screen_type == 'A' else "10일선 하회"
    print(f"\n✓ 선택된 스크리닝: {screen_type_name}")

    # ====================================================================
    # 티커 로드 (타입에 따라 다른 방식)
    # ====================================================================
    if screen_type == 'A':
        # 타입 A: 엑셀 파일에서 티커 로드
        print("\n" + "="*80)
        print("티커 리스트 파일 입력")
        print("="*80)

        default_file = "bloomberg_ticker.xlsx"
        file_path = default_file

        print(f"\n✓ 엑셀 파일: {default_file}")

        all_tickers, etf_tickers_set = get_tickers_from_excel(file_path)
        excel_metadata = _load_ticker_metadata_from_excel(file_path)

        if not all_tickers:
            print("\n[에러] 티커를 읽을 수 없습니다")
            return

        print(f"\n총 {len(all_tickers)}개 종목을 분석합니다.")
    else:
        # 타입 B: 사용자 직접 입력
        print("\n" + "="*80)
        print("티커 직접 입력")
        print("="*80)
        print("\n티커 형식: 005930 KS, 000660 KS (쉼표로 구분)")

        user_input = input("\n티커 입력: ").strip()

        if not user_input:
            print("\n[에러] 티커를 입력해주세요")
            return

        all_tickers = [t.strip() for t in user_input.split(',')]
        etf_tickers_set = set()
        excel_metadata = {}
        print(f"\n총 {len(all_tickers)}개 종목을 분석합니다.")

    # ====================================================================
    # 전종목 분석 실행 (병렬 처리)
    # ====================================================================
    print("\n" + "="*80)
    print("전종목 분석 시작 (3개월 데이터)")
    print("="*80)

    results = analyze_tickers_parallel(all_tickers, period='3M', max_workers=30, mode=mode)

    if not results:
        print("\n[에러] 분석 결과가 없습니다")
        return

    # ====================================================================
    # 필터링: 타입에 따라 다른 필터 적용
    # ====================================================================
    print("\n" + "="*80)
    if screen_type == 'A':
        print("스크리닝: 거래량 폭증 종목 (10일선 돌파 + RVOL≥1.5)")
        print("="*80)
        filtered_df = filter_volume_surge_breakout(results, rvol_threshold=1.5)
    else:
        print("스크리닝: 10일선 하회 종목")
        print("="*80)
        filtered_df = filter_below_ma10(results)

    if filtered_df.empty:
        print("\n조건을 만족하는 종목이 없습니다.")
        return

    print(f"\n✓ {len(filtered_df)}개 종목이 조건을 만족합니다")

    # ====================================================================
    # 종목명 및 시가총액 조회
    # ====================================================================
    print("\n[종목명 조회 중...]")
    filtered_tickers = filtered_df['ticker'].tolist()

    try:
        ticker_names, market_caps = get_security_metadata(filtered_tickers, excel_metadata)
    except Exception as e:
        print(f"⚠️  종목명/시가총액 조회 실패: {e}")
        ticker_names = {ticker: ticker for ticker in filtered_tickers}
        market_caps = {ticker: None for ticker in filtered_tickers}

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
    print("상세 정보 (거래량 폭증 종목)")
    print("="*80)

    for _, row in filtered_df.iterrows():
        ticker = row['ticker']
        security_name = ticker_names.get(ticker, ticker)
        market_cap = market_caps.get(ticker)

        # 한글 종목명은 15자로 제한 (한글 2바이트 고려)
        # 영문 종목명은 30자로 제한
        if any('\uac00' <= char <= '\ud7a3' for char in security_name):
            # 한글이 포함된 경우
            name_padded = f"{security_name:<15}"
        else:
            # 영문인 경우
            name_padded = f"{security_name:<30}"

        trend_info = row['trend_detail']
        rvol_str = f"RVOL {row['rvol']:.1f}배"

        # 시가총액 포맷
        if market_cap is not None:
            market_cap_str = f"시총 {market_cap}"
        else:
            market_cap_str = "시총 N/A"

        print(f"  {name_padded}  {trend_info}, {rvol_str}, {market_cap_str}, WATCH")

    # ====================================================================
    # TOP 5 출력 (10일선 돌파일 & 이탈일 기준)
    # ====================================================================
    print("\n" + "="*80)
    print("TOP 5 종목 (10일선 돌파일 최근순)")
    print("="*80)

    # 10일선 돌파일 내림차순 정렬 (최근이 먼저)
    top5_breakout = filtered_df.sort_values('last_ma10_break_above', ascending=False, na_position='last').head(5)

    for idx, row in top5_breakout.iterrows():
        ticker = row['ticker']
        name = ticker_names.get(ticker, ticker)
        market_cap = market_caps.get(ticker)

        # 시가총액 포맷

    # 10일선 이탈일 오름차순 정렬 (오래된 것이 먼저)
    top5_breakdown = filtered_df.sort_values('last_ma10_break_below', ascending=True, na_position='last').head(5)

    for idx, row in top5_breakdown.iterrows():
        ticker = row['ticker']
        name = ticker_names.get(ticker, ticker)
        market_cap = market_caps.get(ticker)

        # 시가총액 포맷 (Bloomberg 원본)
        cap_str = market_cap if market_cap is not None else "N/A"

        print(f"  {name:<15} | 이탈일: {row['last_ma10_break_below']} | RVOL: {row['rvol']:.1f}배 | 시총: {cap_str}")

    # ====================================================================
    # 엑셀 저장 (자동)
    # ====================================================================
    print("\n" + "="*80)
    print("엑셀 파일 저장 중...")

    # 저장 디렉토리 생성 (모드 및 타입별)
    save_dir = r"C:\Users\10845\Documents\quant_project\[오후] whole-stock"
    os.makedirs(save_dir, exist_ok=True)

    output_date = datetime.now().strftime("%Y%m%d")
    output_filename = os.path.join(save_dir, f"whole-stock_{output_date}.xlsx")

    # 저장용 DataFrame 생성 (추세방향, 신호 제외)
    # RSI 컬럼이 있는지 확인
    base_columns = [
        'ticker', 'rvol',
        'last_ma10_break_above', 'last_ma10_break_below',
        'trend_detail',
        'close_price', 'prev_close', 'price_change_percent',
        'ma10', 'ma_distance_percent'
    ]

    # RSI가 있으면 추가
    if 'rsi' in filtered_df.columns:
        base_columns.insert(2, 'rsi')

    save_df = filtered_df[base_columns].copy()

    # 종목명과 시가총액 추가
    save_df.insert(0, '종목명', save_df['ticker'].map(ticker_names))
    save_df.insert(1, '티커', save_df['ticker'])
    save_df['종목명'] = save_df['종목명'].fillna(save_df['티커'])
    save_df = save_df.drop(columns=['ticker'])

    # 시가총액 추가
    save_df.insert(2, '시가총액', save_df['티커'].map(market_caps))

    # 소수점 반올림 (전일비, 10일선괴리율, RVOL, RSI)
    save_df['price_change_percent'] = save_df['price_change_percent'].round(1)
    save_df['ma_distance_percent'] = save_df['ma_distance_percent'].round(1)
    save_df['rvol'] = save_df['rvol'].round(1)
    if 'rsi' in save_df.columns:
        save_df['rsi'] = save_df['rsi'].round(1)

    # 컬럼명 한글화
    rename_dict = {
        'rvol': 'RVOL',
        'last_ma10_break_above': '10일선돌파일',
        'last_ma10_break_below': '10일선이탈일',
        'trend_detail': '추세상세',
        'close_price': '현재가',
        'prev_close': '전일종가',
        'price_change_percent': '전일비',
        'ma10': '10일선',
        'ma_distance_percent': '10일선괴리율'
    }
    if 'rsi' in save_df.columns:
        rename_dict['rsi'] = 'RSI'
    save_df = save_df.rename(columns=rename_dict)

    # 모드 및 타입에 따른 필터링
    from datetime import date
    today = date.today()

    if screen_type == 'A':
        # 타입 A: 10일선 돌파일이 당일인 종목만
        before_filter = len(save_df)
        save_df['돌파일_date'] = pd.to_datetime(save_df['10일선돌파일']).dt.date
        save_df = save_df[save_df['돌파일_date'] == today].copy()
        save_df = save_df.drop(columns=['돌파일_date'])
        after_filter = len(save_df)
        mode_str = "보고용" if mode == 1 else "실시간"
        print(f"\n[{mode_str}] 10일선 돌파일이 당일인 종목만: {before_filter}개 → {after_filter}개")

        # 돌파일-이탈일 간격 3일 이상 필터 (단기 노이즈·공휴일 제거)
        before_gap = len(save_df)
        def _has_valid_gap(row):
            breakout = row['10일선돌파일']
            breakdown = row['10일선이탈일']
            if pd.isna(breakout) or pd.isna(breakdown):
                return False
            try:
                gap = (pd.to_datetime(breakout) - pd.to_datetime(breakdown)).days
                return gap >= 3
            except Exception:
                return False
        save_df = save_df[save_df.apply(_has_valid_gap, axis=1)].copy()
        after_gap = len(save_df)
        print(f"[갭 필터] 돌파일-이탈일 3일 이상: {before_gap}개 → {after_gap}개")

        # 결과 없으면 정렬 생략
        if not save_df.empty and '시가총액' in save_df.columns:
            # 정렬: 시가총액 내림차순 → 10일선 이탈일 오름차순
            save_df['시가총액_num'] = pd.to_numeric(save_df['시가총액'], errors='coerce')
            save_df = save_df.sort_values(
                by=['시가총액_num', '10일선이탈일'],
                ascending=[False, True],
                na_position='last'
            )
            save_df = save_df.drop(columns=['시가총액_num'])
    else:
        # 타입 B: 10일선 이탈일이 당일인 종목만
        before_filter = len(save_df)
        save_df['이탈일_date'] = pd.to_datetime(save_df['10일선이탈일']).dt.date
        save_df = save_df[save_df['이탈일_date'] == today].copy()
        save_df = save_df.drop(columns=['이탈일_date'])
        after_filter = len(save_df)
        mode_str = "보고용" if mode == 1 else "실시간"
        print(f"\n[{mode_str}] 10일선 이탈일이 당일인 종목만: {before_filter}개 → {after_filter}개")

        # 결과 없으면 정렬 생략
        if not save_df.empty and '시가총액' in save_df.columns:
            # 정렬: 시가총액 내림차순
            save_df['시가총액_num'] = pd.to_numeric(save_df['시가총액'], errors='coerce')
            save_df = save_df.sort_values(
                by=['시가총액_num'],
                ascending=[False],
                na_position='last'
            )
            save_df = save_df.drop(columns=['시가총액_num'])

    final_columns = [
        '종목명', '티커', '시가총액', 'RVOL', 'RSI',
        '10일선돌파일', '10일선이탈일', '추세상세',
        '현재가', '전일종가', '전일비', '10일선', '10일선괴리율'
    ]
    for col in final_columns:
        if col not in save_df.columns:
            save_df[col] = pd.NA
    save_df = save_df[final_columns]

    # ====================================================================
    # ETF 현황 시트 준비 (날짜 필터 없이 전체 ETF 결과)
    # ====================================================================
    etf_sheet_df = None
    if etf_tickers_set and results:
        etf_raw = [r for r in results if r.get('ticker') in etf_tickers_set]
        if etf_raw:
            etf_all = pd.DataFrame(etf_raw)

            etf_base_cols = [c for c in [
                'ticker', 'rsi', 'rvol',
                'last_ma10_break_above', 'last_ma10_break_below',
                'trend_detail',
                'close_price', 'prev_close', 'price_change_percent',
                'ma10', 'ma_distance_percent'
            ] if c in etf_all.columns]

            etf_save = etf_all[etf_base_cols].copy()

            # 종목명 / 시가총액 조회
            etf_ticker_list = etf_all['ticker'].tolist()
            try:
                etf_names = get_multiple_security_names(etf_ticker_list)
            except Exception:
                etf_names = {t: t for t in etf_ticker_list}
            etf_save.insert(0, '종목명', etf_save['ticker'].map(etf_names))
            etf_save.insert(1, '티커', etf_save['ticker'])
            etf_save = etf_save.drop(columns=['ticker'])

            # 10일선 상태 컬럼 추가
            etf_save.insert(3, '10일선상태',
                etf_all['condition_1_trend_breakdown'].map(
                    {False: '10일선 위', True: '10일선 아래'}
                ).values
            )

            # 소수점 반올림
            for col in ['price_change_percent', 'ma_distance_percent', 'rvol']:
                if col in etf_save.columns:
                    etf_save[col] = etf_save[col].round(1)
            if 'rsi' in etf_save.columns:
                etf_save['rsi'] = etf_save['rsi'].round(1)

            etf_save = etf_save.rename(columns={
                'rvol': 'RVOL',
                'rsi': 'RSI',
                'last_ma10_break_above': '10일선돌파일',
                'last_ma10_break_below': '10일선이탈일',
                'trend_detail': '추세상세',
                'close_price': '현재가',
                'prev_close': '전일종가',
                'price_change_percent': '전일비(%)',
                'ma10': '10일선',
                'ma_distance_percent': '10일선괴리율(%)'
            })

            # 정렬: 10일선돌파일 최근순 → 10일선이탈일 오래된순 (모멘텀 강한 ETF 상단)
            etf_save = etf_save.sort_values(
                by=['10일선돌파일', '10일선이탈일'],
                ascending=[False, True],
                na_position='last'
            )
            etf_columns = [
                '종목명', '티커', 'RSI', '10일선상태', 'RVOL',
                '10일선돌파일', '10일선이탈일', '추세상세',
                '현재가', '전일종가', '전일비(%)', '10일선', '10일선괴리율(%)'
            ]
            for col in etf_columns:
                if col not in etf_save.columns:
                    etf_save[col] = pd.NA
            etf_save = etf_save[etf_columns]
            etf_sheet_df = etf_save
            print(f"\n[ETF현황] {len(etf_sheet_df)}개 ETF 분석 완료")

    if etf_sheet_df is not None and not etf_sheet_df.empty:
        print(f"  - 스크리닝결과 시트: {len(save_df)}개")
        print(f"  - ETF현황 시트: {len(etf_sheet_df)}개")

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        save_df.to_excel(writer, sheet_name='스크리닝결과', index=False)
        if etf_sheet_df is not None and not etf_sheet_df.empty:
            etf_sheet_df.to_excel(writer, sheet_name='ETF현황', index=False)
    print(f"\n[저장 완료] {output_filename}")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n[에러] 분석 실패: {e}")
        import traceback
        traceback.print_exc()
