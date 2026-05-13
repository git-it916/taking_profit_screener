"""
20주선(20주선) 하회 종목 스크리닝

py -3.12 under_20w.py

Bloomberg Terminal에서 전종목 데이터를 받아 20주선(≈20주선) 하회 종목을 스크리닝합니다.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date
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

from src.bloomberg import download_bloomberg_data, get_multiple_security_names, get_market_caps


def get_tickers_from_excel(file_path: str) -> list:
    """엑셀 파일에서 모든 시트의 티커 리스트 읽기"""
    print(f"\n[엑셀 파일에서 티커 읽기]")
    print(f"  파일: {file_path}")

    try:
        xl = pd.ExcelFile(file_path)
        all_sheets = xl.sheet_names
        print(f"  시트 목록: {all_sheets}")

        all_tickers = []

        def parse_tickers_from_df(df):
            tickers = []
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

        for sheet in all_sheets:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet)
                tickers = parse_tickers_from_df(df)
                all_tickers.extend(tickers)
                print(f"  OK [{sheet}] {len(tickers)}개 로드")
            except Exception as e:
                print(f"  FAIL [{sheet}] 읽기 실패: {e}")

        all_tickers = list(set(all_tickers))
        print(f"\n  총 {len(all_tickers)}개 티커 로드 완료")

        return all_tickers

    except Exception as e:
        print(f"  FAIL 파일 읽기 실패: {e}")
        import traceback
        traceback.print_exc()
        return []


def analyze_single_ticker(ticker: str, period: str = '1Y', mode: int = 1) -> dict:
    """
    Bloomberg에서 데이터를 받아 20주선 분석

    20주선 = 캘린더 기준 140일(20주 × 7일) 내 영업일 종가 평균

    Returns:
        dict: 분석 결과 (20주선 관련 정보 포함)
    """
    try:
        df = download_bloomberg_data(ticker, period=period, verbose=False)

        if df is None or len(df) == 0:
            return None

        # 모드에 따른 당일 데이터 처리
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

        # DatetimeIndex 설정 (시간 기반 rolling 사용을 위해)
        df = df.copy()
        if 'Date' in df.columns:
            df = df.set_index('Date')
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 데이터가 140일(20주) 미만이면 계산 불가
        date_span = (df.index[-1] - df.index[0]).days
        if date_span < 140:
            return None

        # 20주선 SMA 계산 (당일 제외, 캘린더 140일 윈도우)
        df['MA20W'] = df['Close'].shift(1).rolling('140D').mean()

        # 10일 SMA도 계산 (출력용)
        df['MA10'] = df['Close'].shift(1).rolling(window=10).mean()

        # RVOL 계산
        avg_vol = df['Volume'].shift(1).rolling(window=20).mean()
        df['RVOL'] = df['Volume'] / avg_vol

        # RSI 계산 (14일)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 최신 데이터
        latest = df.iloc[-1]

        if pd.isna(latest['MA20W']):
            return None

        # 20주선 하회 여부
        below_ma20w = latest['Close'] < latest['MA20W']

        if not below_ma20w:
            # 20주선 위에 있으면 스크리닝 대상 아님 — 결과는 반환하되 플래그 설정
            pass

        # 20주선 괴리율
        ma20w_distance_pct = ((latest['Close'] - latest['MA20W']) / latest['MA20W']) * 100

        # 10일선 괴리율
        ma10_distance_pct = None
        if pd.notna(latest.get('MA10')):
            ma10_distance_pct = ((latest['Close'] - latest['MA10']) / latest['MA10']) * 100

        # 20주선 크로스오버 추적
        position = (df['Close'] >= df['MA20W']).astype(int)
        position_shift = position.shift(1)
        break_below_mask = (position_shift == 1) & (position == 0)
        break_above_mask = (position_shift == 0) & (position == 1)

        dates = pd.Series(df.index.strftime('%Y-%m-%d'), index=df.index)

        # 최근 하회일/상회일
        last_break_below = None
        last_break_above = None
        for idx in df.index:
            if break_below_mask.loc[idx]:
                last_break_below = dates.loc[idx]
        for idx in df.index:
            if break_above_mask.loc[idx]:
                last_break_above = dates.loc[idx]

        # 20주선 아래 머문 일수
        days_below = 0
        for idx in reversed(df.index.tolist()):
            if pd.isna(df.loc[idx, 'MA20W']):
                break
            if df.loc[idx, 'Close'] < df.loc[idx, 'MA20W']:
                days_below += 1
            else:
                break

        # 전일비
        prev_close = None
        price_change_pct = 0
        if len(df) >= 2:
            prev_close = df.iloc[-2]['Close']
            price_change_pct = ((latest['Close'] - prev_close) / prev_close) * 100

        # 추세 상세
        current_position = 'below' if latest['Close'] < latest['MA20W'] else 'above'
        if current_position == 'below':
            trend_detail = f"20주선 위({last_break_above or '?'}) → 20주선 아래({last_break_below or '?'})"
        else:
            trend_detail = f"20주선 아래({last_break_below or '?'}) → 20주선 위({last_break_above or '?'})"

        result = {
            'ticker': ticker,
            'close_price': latest['Close'],
            'prev_close': prev_close,
            'price_change_percent': price_change_pct,
            'ma20w': latest['MA20W'],
            'ma20w_distance_percent': ma20w_distance_pct,
            'ma10': latest['MA10'] if pd.notna(latest.get('MA10')) else None,
            'ma10_distance_percent': ma10_distance_pct,
            'below_ma20w': below_ma20w,
            'last_ma20w_break_below': last_break_below,
            'last_ma20w_break_above': last_break_above,
            'days_below_ma20w': days_below,
            'trend_detail': trend_detail,
            'rvol': latest['RVOL'] if pd.notna(latest.get('RVOL')) else None,
            'rsi': float(latest['RSI']) if pd.notna(latest.get('RSI')) else None,
        }

        return result

    except Exception as e:
        return None


def analyze_tickers_parallel(tickers: list, period: str = '1Y', max_workers: int = 15, mode: int = 1) -> list:
    """병렬 방식으로 여러 티커 분석"""
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
    print(f"데이터 기간: {period} (20주선 계산용)\n")

    start_time = datetime.now()

    def _analyze(ticker):
        try:
            result = analyze_single_ticker(ticker, period=period, mode=mode)
            return (ticker, result, None)
        except Exception as e:
            return (ticker, None, str(e))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(_analyze, t): t for t in tickers}

        for future in as_completed(future_to_ticker):
            ticker, result, error = future.result()

            with lock:
                completed_count += 1
                if result:
                    results.append(result)
                else:
                    failed_tickers.append(ticker)

                elapsed = datetime.now() - start_time
                progress = completed_count / len(tickers) * 100
                rate = completed_count / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                remaining = (len(tickers) - completed_count) / rate if rate > 0 else 0

                print(f"\r[진행중] {completed_count}/{len(tickers)} ({progress:.1f}%) "
                      f"| 경과: {str(elapsed).split('.')[0]} | 속도: {rate:.2f}종목/초 | "
                      f"남은시간: ~{int(remaining/60)}분 {int(remaining%60)}초", end='', flush=True)

    print()
    total_time = datetime.now() - start_time

    print(f"\n✓ 분석 완료 - 소요시간: {str(total_time).split('.')[0]}")
    print(f"  성공: {len(results)}개, 실패: {len(failed_tickers)}개")

    if failed_tickers:
        print(f"\n⚠️  실패한 종목 ({len(failed_tickers)}개):")
        for ticker in failed_tickers[:10]:
            print(f"  - {ticker}")
        if len(failed_tickers) > 10:
            print(f"  ... 외 {len(failed_tickers) - 10}개")

    return results


def filter_below_ma20w(results: list) -> pd.DataFrame:
    """20주선(20주선) 하회 종목 필터링"""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    print(f"\n[디버깅] 전체 분석 결과: {len(df)}개")

    below = df['below_ma20w'] == True
    print(f"[디버깅] 20주선 하회 종목: {below.sum()}개")

    filtered = df[below].copy()

    # 20주선 이탈일 기준 정렬 (최근 이탈일이 위로)
    if 'last_ma20w_break_below' in filtered.columns:
        filtered = filtered.sort_values('last_ma20w_break_below', ascending=False, na_position='last')

    return filtered


def main():
    """메인 함수"""
    print("="*80)
    print("20주선(20주선) 하회 종목 스크리닝")
    print("="*80)

    print("\n⚠️  주의사항:")
    print("  1. Bloomberg Terminal이 실행 중이어야 합니다")
    print("  2. Bloomberg에 로그인되어 있어야 합니다")
    print("  3. 전종목 분석은 약 15~20분 소요됩니다 (1년 데이터)")

    # ====================================================================
    # 분석 모드 선택
    # ====================================================================
    print("\n" + "="*80)
    print("분석 모드 선택")
    print("="*80)
    print("\n[1] 보고용 - 전일 또는 장마감 후 당일 (완성된 일봉)")
    print("    → 장중: 전일까지의 데이터 사용")
    print("    → 장마감 후(15:30 이후): 당일 포함")
    print("\n[2] 실시간 - 현재 시점의 미완성 데이터 포함")

    while True:
        mode_input = input("\n모드 선택 (1 또는 2): ").strip()
        if mode_input in ['1', '2']:
            mode = int(mode_input)
            break
        print("1 또는 2를 입력해주세요.")

    mode_name = "보고용 (완성된 일봉)" if mode == 1 else "실시간 (현재 시점)"
    print(f"\n✓ 선택된 모드: {mode_name}")

    # ====================================================================
    # 티커 로드
    # ====================================================================
    print("\n" + "="*80)
    print("티커 리스트 로드")
    print("="*80)

    default_file = "bloomberg_ticker.xlsx"
    all_tickers = get_tickers_from_excel(default_file)

    if not all_tickers:
        print("\n[에러] 티커를 읽을 수 없습니다")
        return

    print(f"\n총 {len(all_tickers)}개 종목을 분석합니다.")

    # ====================================================================
    # 전종목 분석 실행 (1년 데이터)
    # ====================================================================
    print("\n" + "="*80)
    print("전종목 분석 시작 (1년 데이터 - 20주선 계산용)")
    print("="*80)

    results = analyze_tickers_parallel(all_tickers, period='1Y', max_workers=15, mode=mode)

    if not results:
        print("\n[에러] 분석 결과가 없습니다")
        return

    # ====================================================================
    # 필터링: 20주선 하회 종목
    # ====================================================================
    print("\n" + "="*80)
    print("스크리닝: 20주선(20주선) 하회 종목")
    print("="*80)

    filtered_df = filter_below_ma20w(results)

    if filtered_df.empty:
        print("\n조건을 만족하는 종목이 없습니다.")
        return

    print(f"\n✓ {len(filtered_df)}개 종목이 20주선 하회 조건을 만족합니다")

    # ====================================================================
    # 종목명 및 시가총액 조회
    # ====================================================================
    print("\n[종목명 조회 중...]")
    filtered_tickers = filtered_df['ticker'].tolist()

    try:
        ticker_names = get_multiple_security_names(filtered_tickers)
    except Exception as e:
        print(f"⚠️  종목명 조회 실패: {e}")
        ticker_names = {ticker: ticker for ticker in filtered_tickers}

    print("\n[시가총액 조회 중...]")
    try:
        market_caps = get_market_caps(filtered_tickers)
    except Exception as e:
        print(f"⚠️  시가총액 조회 실패: {e}")
        market_caps = {ticker: None for ticker in filtered_tickers}

    # ====================================================================
    # 시가총액 1000억 이상 필터
    # ====================================================================
    before_cap = len(filtered_df)
    filtered_df = filtered_df[filtered_df['ticker'].map(
        lambda t: pd.to_numeric(market_caps.get(t), errors='coerce')
    ) >= 100_000_000_000].copy()
    print(f"\n[시총 필터] 1000억 이상: {before_cap}개 → {len(filtered_df)}개")

    if filtered_df.empty:
        print("\n시가총액 1000억 이상인 20주선 하회 종목이 없습니다.")
        return

    # ====================================================================
    # 결과 출력
    # ====================================================================
    print("\n" + "="*80)
    print("스크리닝 결과 (20주선 괴리율 순)")
    print("="*80)

    summary_data = []
    for _, row in filtered_df.iterrows():
        ticker = row['ticker']
        security_name = ticker_names.get(ticker, ticker)

        if row.get('prev_close') and row['prev_close'] > 0:
            price_change_str = f"{row['price_change_percent']:+.1f}%"
        else:
            price_change_str = "-"

        rvol_str = f"{row['rvol']:.1f}배" if row.get('rvol') is not None and pd.notna(row['rvol']) else "-"

        summary_data.append({
            '종목': security_name[:30],
            '티커': ticker,
            '현재가': f"{row['close_price']:.0f}",
            '전일비': price_change_str,
            '20주선': f"{row['ma20w']:.0f}",
            '괴리율': f"{row['ma20w_distance_percent']:+.1f}%",
            'RVOL': rvol_str,
            '이탈일': row.get('last_ma20w_break_below', '?'),
        })

    summary_df_display = pd.DataFrame(summary_data)
    print("\n" + tabulate(summary_df_display, headers='keys', tablefmt='simple', showindex=False))

    # ====================================================================
    # 상세 정보 출력
    # ====================================================================
    print("\n" + "="*80)
    print("상세 정보 (20주선 하회 종목)")
    print("="*80)

    for _, row in filtered_df.iterrows():
        ticker = row['ticker']
        security_name = ticker_names.get(ticker, ticker)
        market_cap = market_caps.get(ticker)

        if any('\uac00' <= char <= '\ud7a3' for char in security_name):
            name_padded = f"{security_name:<15}"
        else:
            name_padded = f"{security_name:<30}"

        trend_info = row['trend_detail']
        rvol_str = f"RVOL {row['rvol']:.1f}배" if row.get('rvol') is not None and pd.notna(row['rvol']) else "RVOL N/A"

        if market_cap is not None:
            market_cap_str = f"시총 {market_cap}"
        else:
            market_cap_str = "시총 N/A"

        print(f"  {name_padded}  {trend_info}, {rvol_str}, {market_cap_str}")

    # ====================================================================
    # TOP 5 출력
    # ====================================================================
    print("\n" + "="*80)
    print("TOP 5 종목 (20주선 이탈일 최근순)")
    print("="*80)

    top5_recent = filtered_df.sort_values('last_ma20w_break_below', ascending=False, na_position='last').head(5)
    for _, row in top5_recent.iterrows():
        ticker = row['ticker']
        name = ticker_names.get(ticker, ticker)
        cap = market_caps.get(ticker)
        cap_str = cap if cap is not None else "N/A"
        rvol_str = f"{row['rvol']:.1f}배" if row.get('rvol') is not None and pd.notna(row['rvol']) else "N/A"
        print(f"  {name:<15} | 이탈일: {row['last_ma20w_break_below']} | RVOL: {rvol_str} | 시총: {cap_str}")

    print("\n" + "="*80)
    print("TOP 5 종목 (20주선 이탈일 오래된순)")
    print("="*80)

    top5_old = filtered_df.sort_values('last_ma20w_break_below', ascending=True, na_position='last').head(5)
    for _, row in top5_old.iterrows():
        ticker = row['ticker']
        name = ticker_names.get(ticker, ticker)
        cap = market_caps.get(ticker)
        cap_str = cap if cap is not None else "N/A"
        rvol_str = f"{row['rvol']:.1f}배" if row.get('rvol') is not None and pd.notna(row['rvol']) else "N/A"
        print(f"  {name:<15} | 이탈일: {row['last_ma20w_break_below']} | RVOL: {rvol_str} | 시총: {cap_str}")

    # ====================================================================
    # 엑셀 저장
    # ====================================================================
    print("\n" + "="*80)
    print("엑셀 파일 저장 중...")

    save_dir = r"C:\Users\Bloomberg\Documents\ssh_project\[오후] whole-stock-result"
    file_prefix = "20주선하회_보고용" if mode == 1 else "20주선하회_실시간"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(save_dir, f"{file_prefix}_{timestamp}.xlsx")

    # 저장용 DataFrame 생성
    base_columns = [
        'ticker', 'rvol',
        'last_ma20w_break_below', 'last_ma20w_break_above',
        'days_below_ma20w',
        'trend_detail',
        'close_price', 'prev_close', 'price_change_percent',
        'ma20w', 'ma20w_distance_percent',
        'ma10', 'ma10_distance_percent',
    ]

    if 'rsi' in filtered_df.columns:
        base_columns.insert(2, 'rsi')

    save_df = filtered_df[base_columns].copy()

    # 종목명과 시가총액 추가
    save_df.insert(0, '종목명', save_df['ticker'].map(ticker_names))
    save_df.insert(1, '티커', save_df['ticker'])
    save_df = save_df.drop(columns=['ticker'])
    save_df.insert(2, '시가총액', save_df['티커'].map(market_caps))

    # 소수점 반올림
    save_df['price_change_percent'] = save_df['price_change_percent'].round(1)
    save_df['ma20w_distance_percent'] = save_df['ma20w_distance_percent'].round(1)
    if 'ma10_distance_percent' in save_df.columns:
        save_df['ma10_distance_percent'] = save_df['ma10_distance_percent'].round(1)
    if 'rvol' in save_df.columns:
        save_df['rvol'] = save_df['rvol'].round(1)
    if 'rsi' in save_df.columns:
        save_df['rsi'] = save_df['rsi'].round(1)

    # 컬럼명 한글화
    rename_dict = {
        'rvol': 'RVOL',
        'last_ma20w_break_below': '20주선이탈일',
        'last_ma20w_break_above': '20주선돌파일',
        'days_below_ma20w': '하회일수',
        'trend_detail': '추세상세',
        'close_price': '현재가',
        'prev_close': '전일종가',
        'price_change_percent': '전일비(%)',
        'ma20w': '20주선',
        'ma20w_distance_percent': '20주선괴리율(%)',
        'ma10': '10일선',
        'ma10_distance_percent': '10일선괴리율(%)',
    }
    if 'rsi' in save_df.columns:
        rename_dict['rsi'] = 'RSI'
    save_df = save_df.rename(columns=rename_dict)

    # 최근 5일 내 이탈 종목만 필터
    from datetime import timedelta
    today = date.today()
    cutoff = today - timedelta(days=5)
    cutoff_ts = pd.Timestamp(cutoff)
    before_filter = len(save_df)
    save_df['이탈일_date'] = pd.to_datetime(save_df['20주선이탈일'], errors='coerce')
    save_df = save_df[save_df['이탈일_date'] >= cutoff_ts].copy()
    save_df = save_df.drop(columns=['이탈일_date'])
    after_filter = len(save_df)
    mode_str = "보고용" if mode == 1 else "실시간"
    print(f"\n[{mode_str}] 최근 5일 내({cutoff}~{today}) 20주선 이탈 종목: {before_filter}개 → {after_filter}개")

    # 이탈일-돌파일 간격 5일 이상 필터 (단기 노이즈 제거)
    before_gap = len(save_df)
    def _has_valid_gap(row):
        breakdown = row['20주선이탈일']
        breakout = row['20주선돌파일']
        if pd.isna(breakdown) or pd.isna(breakout):
            return False
        try:
            gap = (pd.to_datetime(breakdown) - pd.to_datetime(breakout)).days
            return gap >= 5
        except Exception:
            return False
    save_df = save_df[save_df.apply(_has_valid_gap, axis=1)].copy()
    after_gap = len(save_df)
    print(f"[갭 필터] 이탈일-돌파일 5일 이상: {before_gap}개 → {after_gap}개")

    # 시가총액 내림차순 정렬
    if not save_df.empty and '시가총액' in save_df.columns:
        save_df['시가총액_num'] = pd.to_numeric(save_df['시가총액'], errors='coerce')
        save_df = save_df.sort_values(by=['시가총액_num'], ascending=[False], na_position='last')
        save_df = save_df.drop(columns=['시가총액_num'])

    # 엑셀 저장
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        save_df.to_excel(writer, sheet_name='스크리닝결과', index=False)
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
