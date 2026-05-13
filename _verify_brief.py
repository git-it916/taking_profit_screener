"""start_brief.py 전 단계 무결성 검증 스크립트"""
import sys, os, json, time
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime, timedelta

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\Bloomberg\Documents\ssh_project\taking_profit_screener')

ROOT = r'C:\Users\Bloomberg\Documents\ssh_project\taking_profit_screener'
CACHE = os.path.join(ROOT, '.krx_name_cache.json')
RESULT_DIR = r'C:\Users\Bloomberg\Documents\ssh_project\[오전] start-brief-result'

# 검증용 종목 (대표 + ETF + 다양한 시장)
TICKERS = [
    '005930 KS',  # 삼성전자
    '000660 KS',  # SK하이닉스
    '207940 KS',  # 삼성바이오로직스
    '034020 KS',  # 두산에너빌리티
    '105560 KS',  # KB금융
    '373220 KS',  # LG에너지솔루션
    '267270 KS',  # HD현대중공업
    '069500 KS',  # KODEX 200 (ETF)
    '189300 KQ',  # 인텔리안테크 (KQ)
    '127120 KQ',  # 디엔에이링크 (KQ)
]

print('='*80)
print('start_brief.py 전 단계 무결성 검증')
print('='*80)

# ============================================================================
# [STEP 0] 캐시 디스크 read 정상성 검증
# ============================================================================
print('\n[STEP 0] 디스크 캐시 정상성')
print('-'*80)
if os.path.exists(CACHE):
    with open(CACHE, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    print(f'  ✓ 캐시 파일 존재: {len(cache_data):,}개 종목, {os.path.getsize(CACHE):,} bytes')
    sample_keys = list(cache_data.keys())[:3]
    print(f'  샘플: {[(k, cache_data[k]) for k in sample_keys]}')
else:
    print(f'  ⚠️  캐시 파일 없음 (첫 실행)')

# ============================================================================
# [STEP 1] download_data() — OHLCV 데이터 무결성 검증
# ============================================================================
print('\n[STEP 1] OHLCV 데이터 무결성')
print('-'*80)
from start_brief import download_data

t0 = time.time()
data_results = {}
for t in TICKERS:
    df = download_data(t, period='3M', verbose=False)
    data_results[t] = df

print(f'  소요시간: {time.time()-t0:.1f}초')

# 무결성 검사 (각 DataFrame에 대해)
checks = {'success': 0, 'fail_load': 0, 'date_sorted': 0, 'ohlc_valid': 0,
          'volume_positive': 0, 'no_null': 0, 'last_date_recent': 0}

today = datetime.now().date()
recent_cutoff = today - timedelta(days=10)

for t, df in data_results.items():
    if df is None or len(df) == 0:
        checks['fail_load'] += 1
        continue
    checks['success'] += 1

    # 날짜 정렬 확인
    if df['Date'].is_monotonic_increasing:
        checks['date_sorted'] += 1

    # OHLC 일관성: high >= max(open,close), low <= min(open,close)
    h_ok = ((df['High'] >= df['Open']) & (df['High'] >= df['Close']) & (df['High'] >= df['Low'])).all()
    l_ok = ((df['Low'] <= df['Open']) & (df['Low'] <= df['Close']) & (df['Low'] <= df['High'])).all()
    if h_ok and l_ok:
        checks['ohlc_valid'] += 1

    # 거래량 양수
    if (df['Volume'] >= 0).all():
        checks['volume_positive'] += 1

    # null 없음
    if not df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any():
        checks['no_null'] += 1

    # 마지막 날짜가 최근 영업일인지
    last_d = pd.to_datetime(df['Date'].iloc[-1]).date()
    if last_d >= recent_cutoff:
        checks['last_date_recent'] += 1

total = len(TICKERS)
print(f'  데이터 로드 성공: {checks["success"]}/{total}')
print(f'  데이터 로드 실패: {checks["fail_load"]}/{total}')
n = checks['success']
print(f'  날짜 단조증가:    {checks["date_sorted"]}/{n}')
print(f'  OHLC 일관성:     {checks["ohlc_valid"]}/{n}')
print(f'  Volume >= 0:     {checks["volume_positive"]}/{n}')
print(f'  Null 없음:       {checks["no_null"]}/{n}')
print(f'  최근 영업일:     {checks["last_date_recent"]}/{n}')

# 마지막 날짜 분포
last_dates = {}
for t, df in data_results.items():
    if df is not None and len(df) > 0:
        d = str(pd.to_datetime(df['Date'].iloc[-1]).date())
        last_dates[d] = last_dates.get(d, 0) + 1
print(f'  마지막 날짜 분포: {last_dates}')

# ============================================================================
# [STEP 2] 종목명 매칭 검증
# ============================================================================
print('\n[STEP 2] 종목명 매칭')
print('-'*80)
from start_brief import get_multiple_security_names

t0 = time.time()
names = get_multiple_security_names(TICKERS)
print(f'  소요시간: {time.time()-t0:.1f}초')

KNOWN = {
    '005930 KS': '삼성전자', '000660 KS': 'SK하이닉스', '207940 KS': '삼성바이오로직스',
    '034020 KS': '두산에너빌리티', '105560 KS': 'KB금융', '373220 KS': 'LG에너지솔루션',
    '069500 KS': 'KODEX 200',
}
correct = 0
for t, expected in KNOWN.items():
    actual = names.get(t)
    if actual == expected:
        correct += 1
    else:
        print(f'  ✗ {t}: 예상={expected}, 실제={actual}')
print(f'  대표종목 정확도: {correct}/{len(KNOWN)}')
print(f'  종목명 매칭 (티커가 아닌 한글로 변환): {sum(1 for t,n in names.items() if n != t)}/{len(TICKERS)}')

# ============================================================================
# [STEP 3] StockAnalyzer 계산 검증 (수동 계산과 비교)
# ============================================================================
print('\n[STEP 3] 분석 계산 정확성 (StockAnalyzer)')
print('-'*80)
from start_brief import analyze_from_brief

calc_pass = 0
for t in TICKERS:
    df = data_results.get(t)
    if df is None or len(df) < 11:
        continue

    # 모드 1 보고용 처리: 장중이면 당일 제외
    now = datetime.now()
    if now.time() < dtime(15, 30):
        df = df[df['Date'].dt.date != now.date()].copy()

    if len(df) < 11:
        continue

    # 수동 계산
    last_close = df['Close'].iloc[-1]
    manual_ma10 = df['Close'].iloc[-11:-1].mean()  # 직전 10일
    avg_vol_10 = df['Volume'].iloc[-11:-1].mean()
    manual_rvol = df['Volume'].iloc[-1] / avg_vol_10 if avg_vol_10 > 0 else None

    # StockAnalyzer 결과
    r = analyze_from_brief(t, period='3M', show_progress=False, mode=1)
    if r is None:
        continue

    close_ok = abs(last_close - r['close_price']) < 0.5
    ma_ok = abs(manual_ma10 - r['ma10']) < 0.5
    rvol_ok = abs(manual_rvol - r['rvol']) < 0.001 if manual_rvol else False

    if close_ok and ma_ok and rvol_ok:
        calc_pass += 1
    else:
        nm = names.get(t, t)
        print(f'  ✗ {nm}: close={close_ok}, ma10={ma_ok}, rvol={rvol_ok}')

print(f'  계산 정확성: {calc_pass}/{checks["success"]} (수동계산 ±0.5/±0.001 일치)')

# ============================================================================
# [STEP 4] 모드별 동작 검증 (mode 1 vs 2)
# ============================================================================
print('\n[STEP 4] 모드 1/2 처리 검증')
print('-'*80)
now = datetime.now()
print(f'  현재 시각: {now.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'  장중(15:30 이전): {now.time() < dtime(15, 30)}')
print(f'  → 모드1에서 당일 제외 적용 여부: {now.time() < dtime(15, 30)}')

t = '005930 KS'
r1 = analyze_from_brief(t, period='3M', show_progress=False, mode=1)
r2 = analyze_from_brief(t, period='3M', show_progress=False, mode=2)
print(f'  {names.get(t)} 모드1 종가: {r1["close_price"]:.0f}')
print(f'  {names.get(t)} 모드2 종가: {r2["close_price"]:.0f}')
print(f'  → 동일' if r1["close_price"] == r2["close_price"] else f'  → 다름 (장중이라면 정상)')

# ============================================================================
# [STEP 5] 신호 분류 정확성 검증
# ============================================================================
print('\n[STEP 5] 신호 분류 정확성 (BUY/WATCH/SELL/HOLD)')
print('-'*80)

results = []
for t in TICKERS:
    r = analyze_from_brief(t, period='3M', show_progress=False, mode=1)
    if r:
        results.append(r)

results_df = pd.DataFrame(results)
sig_counts = results_df['signal'].value_counts().to_dict()
print(f'  신호 분포: {sig_counts}')

# 신호 분류 규칙 검증
# BUY: 10일선 위 + RVOL >= 2.0   → 코드는 condition_2(RVOL>=1.5)로 보임
# WATCH: 1.5 <= RVOL < 2.0
# SELL: 10일선 아래 + RVOL >= 2.0
# HOLD: 나머지
sig_rules_ok = 0
sig_rules_fail = []
for _, r in results_df.iterrows():
    sig = r['signal']
    c1 = r['condition_1_trend_breakdown']  # True = 10일선 아래
    c2 = r['condition_2_volume_confirmation']  # True = RVOL >= 1.5
    rvol = r['rvol']
    nm = names.get(r['ticker'], r['ticker'])

    if sig == 'BUY' and (not c1) and c2:
        sig_rules_ok += 1
    elif sig == 'SELL' and c1 and c2:
        sig_rules_ok += 1
    elif sig == 'WATCH' and 1.0 <= rvol < 1.5:
        sig_rules_ok += 1
    elif sig == 'HOLD':
        sig_rules_ok += 1
    else:
        sig_rules_fail.append((nm, sig, c1, c2, rvol))

print(f'  신호 분류 규칙 일치: {sig_rules_ok}/{len(results_df)}')
if sig_rules_fail:
    for nm, sig, c1, c2, rvol in sig_rules_fail[:3]:
        print(f'  ✗ {nm}: signal={sig}, c1={c1}, c2={c2}, rvol={rvol:.2f}')

# ============================================================================
# [STEP 6] Excel 저장 검증 (실제 파일 생성 + 시트 검증)
# ============================================================================
print('\n[STEP 6] Excel 저장 (실제 파일 생성)')
print('-'*80)

# main()에서 사용하는 동일 로직으로 Excel 생성
ticker_str = ','.join(TICKERS)

# Excel 저장 부분만 직접 호출 (main()을 stdin으로 돌리는 대신)
# 카테고리별 데이터 생성
sell_stocks = results_df[results_df['signal'] == 'SELL']
caution_stocks = results_df[
    results_df['condition_1_trend_breakdown'] &
    ~results_df['condition_2_volume_confirmation']
]
surge_stocks = results_df[
    ~results_df['condition_1_trend_breakdown'] &
    results_df['condition_2_volume_confirmation']
]
print(f'  카테고리 분포:')
print(f'    강력 매도 신호: {len(sell_stocks)}개')
print(f'    profit taking: {len(caution_stocks)}개')
print(f'    upside: {len(surge_stocks)}개')

# 실제 main()의 Excel 저장 로직을 흉내내기 위해 main을 직접 실행
# stdin: 티커, 제외(엔터), 모드1, 히트맵n, 엑셀y
print('\n  [실제 main() end-to-end 실행 시뮬레이션]')

stdin_input = f"{ticker_str}\n\n1\nn\ny\n"

import subprocess
proc = subprocess.run(
    ['py', '-3.12', os.path.join(ROOT, 'start_brief.py')],
    input=stdin_input,
    capture_output=True, text=True, encoding='utf-8',
    cwd=ROOT, timeout=300
)

# stdout에서 Excel 저장 결과 추출
out = proc.stdout
saved_match = None
for line in out.split('\n'):
    if 'Excel 저장 완료' in line:
        saved_match = line.strip()
        break

if saved_match:
    print(f'  ✓ {saved_match}')
    # 파일 경로 추출
    saved_path = saved_match.split('Excel 저장 완료: ')[-1].strip()
    if os.path.exists(saved_path):
        sz = os.path.getsize(saved_path)
        print(f'  ✓ 파일 존재: {saved_path}')
        print(f'  ✓ 파일 크기: {sz:,} bytes')

        # Excel 시트 검증
        xl = pd.ExcelFile(saved_path)
        print(f'  ✓ 시트 목록: {xl.sheet_names}')
        for sh in xl.sheet_names:
            sdf = pd.read_excel(xl, sheet_name=sh)
            print(f'    [{sh}] {sdf.shape[0]}행 x {sdf.shape[1]}열 / 컬럼: {list(sdf.columns)[:5]}...')
            if len(sdf) > 0:
                # 데이터 무결성: RVOL/RSI/현재가 numeric 확인
                if 'RVOL' in sdf.columns:
                    rvol_ok = pd.to_numeric(sdf['RVOL'], errors='coerce').notna().all()
                    print(f'      RVOL numeric: {rvol_ok}')
                if '현재가' in sdf.columns:
                    cp_ok = pd.to_numeric(sdf['현재가'], errors='coerce').notna().all()
                    print(f'      현재가 numeric: {cp_ok}')
    else:
        print(f'  ✗ 파일 없음: {saved_path}')
else:
    print(f'  ✗ Excel 저장 결과 없음')
    print(f'  STDERR: {proc.stderr[:500]}')

print('\n' + '='*80)
print('검증 완료')
print('='*80)
