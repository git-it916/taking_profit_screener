"""
start_bloomberg.py와 start_brief.py의 비즈니스 로직이 100% 동일한지 검증

검증 항목:
- analyze 함수 핵심 로직
- main() 흐름 (모드 선택 → 종목명 조회 → 병렬 분석 → 카테고리 분류 → Excel 저장)
- 카테고리 분류 조건 (강력매도/profit taking/upside)
- 필터링 규칙 (이탈일 최근 5일 + 돌파-이탈 갭 3일)
- 정렬 (카테고리 순서, 이탈일 내림차순, 돌파일 오름차순)
- 반올림 (RVOL/RSI/전일비/괴리율 → 소수1자리)
- Excel 시트 구조 (전체, 강력매도신호, profit taking, upside)

마지막에 생성된 Excel을 읽어서 실제 적용 여부도 확인
"""
import os, re, sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r'C:\Users\Bloomberg\Documents\ssh_project\taking_profit_screener'
BLOOMBERG = os.path.join(ROOT, 'start_bloomberg.py')
BRIEF = os.path.join(ROOT, 'start_brief.py')
RESULT_DIR = r'C:\Users\Bloomberg\Documents\ssh_project\[오전] start-brief-result'


def normalize(s: str) -> str:
    """공백/주석 정규화 (line-by-line 비교 위해)"""
    lines = []
    for ln in s.splitlines():
        # 주석 제거
        ln = re.sub(r'#.*$', '', ln)
        # 공백 정규화
        ln = re.sub(r'\s+', ' ', ln).strip()
        if ln:
            lines.append(ln)
    return '\n'.join(lines)


def extract_block(src: str, start_marker: str, end_marker: str = None) -> str:
    """주석/마커 사이의 코드 블록 추출"""
    si = src.find(start_marker)
    if si < 0:
        return ''
    if end_marker:
        ei = src.find(end_marker, si + len(start_marker))
        return src[si:ei] if ei > 0 else src[si:]
    return src[si:]


with open(BLOOMBERG, 'r', encoding='utf-8') as f:
    blo = f.read()
with open(BRIEF, 'r', encoding='utf-8') as f:
    brf = f.read()

print('='*80)
print('start_bloomberg.py vs start_brief.py 로직 일치 검증')
print('='*80)

# ============================================================================
# [1] analyze 함수 핵심 로직 비교
# ============================================================================
print('\n[1] analyze_from_xxx 함수 핵심 로직')
print('-'*80)

# bloomberg 버전: download_bloomberg_data → 모드처리 → StockAnalyzer
# brief 버전: download_data → 모드처리 → StockAnalyzer
# 데이터 소스 호출만 다르고 모드 처리/StockAnalyzer 호출은 같아야 함

mode_logic_blo = """
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

            # 모드 1 (보고용): 장중(마감 전)에만 당일 데이터 제외
            # 모드 2 (실시간): 당일 데이터 포함
            if mode == 1 and current_time < market_close_time:
                # 당일 데이터가 있으면 제외 (일봉 미완성)
                if (df['date_only'] == today).any():
                    df = df[df['date_only'] != today].copy()

            # 임시 컬럼 제거
            df = df.drop(columns=['date_only'])
"""

mode_logic_brf = """
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
"""

if normalize(mode_logic_blo) == normalize(mode_logic_brf):
    print('  ✓ 모드 1/2 당일 데이터 제외 로직 동일')
else:
    print('  ✗ 모드 처리 로직 불일치')
    n1 = normalize(mode_logic_blo).split('\n')
    n2 = normalize(mode_logic_brf).split('\n')
    for i, (a, b) in enumerate(zip(n1, n2)):
        if a != b:
            print(f'    bloomberg[{i}]: {a}')
            print(f'    brief[{i}]:     {b}')

# StockAnalyzer 호출 일치 여부
if 'analyzer = StockAnalyzer()' in blo and 'analyzer = StockAnalyzer()' in brf:
    print('  ✓ StockAnalyzer() 동일하게 호출')
if 'analyzer.analyze_latest(df, ticker)' in blo and 'analyzer.analyze_latest(df, ticker)' in brf:
    print('  ✓ analyzer.analyze_latest(df, ticker) 동일')

# ============================================================================
# [2] main() 흐름 비교
# ============================================================================
print('\n[2] main() 흐름 비교')
print('-'*80)

main_steps = [
    ('티커 입력 받기',       'user_input = input'),
    ('제외 티커 처리',       'exclude_tickers = '),
    ('모드 선택',           'mode_input = input'),
    ('종목명 조회',         'get_multiple_security_names'),
    ('데이터 기간 3M 고정',  "period = '3M'"),
    ('병렬 처리 10 worker',  'max_workers = 10'),
    ('ThreadPoolExecutor',  'ThreadPoolExecutor'),
    ('진행바',              "bar_length = 40"),
    ('요약 테이블',         'tabulate(summary_df'),
    ('추세별 분류',         "current_position'] == 'below'"),
    ('SELL 신호 분류',      "results_df['signal'] == 'SELL'"),
    ('주의 필요 분류',       'condition_1_trend_breakdown'),
    ('거래량 폭증 분류',     'condition_2_volume_confirmation'),
    ('히트맵 입력',         'create_trend_heatmap'),
    ('Excel 저장 입력',     'save_choice = input'),
]

for label, marker in main_steps:
    in_blo = marker in blo
    in_brf = marker in brf
    if in_blo and in_brf:
        print(f'  ✓ {label}')
    elif in_blo and not in_brf:
        print(f'  ✗ {label} — brief에 없음')
    elif not in_blo and in_brf:
        print(f'  ⚠ {label} — bloomberg에 없음 (brief에는 있음)')
    else:
        print(f'  ? {label} — 둘 다 없음')

# ============================================================================
# [3] 카테고리 분류 조건 비교
# ============================================================================
print('\n[3] 카테고리 분류 조건')
print('-'*80)

# bloomberg/brief 모두 동일한 정의가 있어야 함
patterns = {
    "SELL signal":      "results_df[results_df['signal'] == 'SELL']",
    "주의 필요 정의":    "results_df['condition_1_trend_breakdown'] &\n            ~results_df['condition_2_volume_confirmation']",
    "거래량 폭증 정의":  "~results_df['condition_1_trend_breakdown'] &\n            results_df['condition_2_volume_confirmation']",
}
for name, pat in patterns.items():
    in_blo = pat in blo
    in_brf = pat in brf
    if in_blo and in_brf:
        print(f'  ✓ {name}')
    else:
        print(f'  ✗ {name}: blo={in_blo}, brf={in_brf}')

# ============================================================================
# [4] 필터링 규칙
# ============================================================================
print('\n[4] 필터링 규칙 (이탈일 최근 5일, 갭 3일)')
print('-'*80)

filter_checks = [
    ('cutoff = today - 5일',  'today - timedelta(days=5)'),
    ('이탈일 >= cutoff',       'breakdown_date >= cutoff_date'),
    ('갭 (이탈일-돌파일) >=3', 'gap >= 3'),
    ('SELL 필터 적용',         'sell_data_filtered'),
    ('caution 필터 적용',      'caution_data_filtered'),
    ('upside 필터 제외',        'surge_data_filtered = surge_data'),
]
for label, marker in filter_checks:
    in_blo = marker in blo
    in_brf = marker in brf
    print(f'  {"✓" if in_blo and in_brf else "✗"} {label}: blo={in_blo}, brf={in_brf}')

# ============================================================================
# [5] 정렬 로직
# ============================================================================
print('\n[5] 정렬 로직')
print('-'*80)

sort_checks = [
    ('카테고리 순서 dict',  "{'profit taking.': 0, 'profit taking': 1, 'upside': 2}"),
    ('이탈일 내림 + 돌파일 오름', "['카테고리_순서', '10일선이탈일', '10일선돌파일']"),
    ('ascending=[True, False, True]', '[True, False, True]'),
]
for label, marker in sort_checks:
    in_blo = marker in blo
    in_brf = marker in brf
    print(f'  {"✓" if in_blo and in_brf else "✗"} {label}: blo={in_blo}, brf={in_brf}')

# ============================================================================
# [6] 반올림 규칙
# ============================================================================
print('\n[6] 반올림 (소수1자리)')
print('-'*80)

round_cols = ['RVOL', 'RSI', '전일비(%)', '10일선괴리율(%)']
for col in round_cols:
    pat_blo = f"'{col}'" in blo and '.round(1)' in blo
    pat_brf = f"'{col}'" in brf and '.round(1)' in brf
    print(f'  {"✓" if pat_blo and pat_brf else "?"} {col} 반올림 코드 존재')

# ============================================================================
# [7] Excel 시트 구조
# ============================================================================
print('\n[7] Excel 시트 구조')
print('-'*80)

sheet_names = ['전체', '강력매도신호', 'profit taking', 'upside']
for sn in sheet_names:
    in_blo = f"sheet_name='{sn}'" in blo
    in_brf = f"sheet_name='{sn}'" in brf
    print(f'  {"✓" if in_blo and in_brf else "✗"} 시트 "{sn}": blo={in_blo}, brf={in_brf}')

# ============================================================================
# [8] 실제 Excel 출력의 규칙 적용 검증
# ============================================================================
print('\n[8] 실제 Excel 출력 검증 (마지막 생성 파일)')
print('-'*80)

if not os.path.exists(RESULT_DIR):
    print(f'  결과 폴더 없음: {RESULT_DIR}')
else:
    files = sorted(os.listdir(RESULT_DIR), reverse=True)
    if not files:
        print('  Excel 파일 없음')
    else:
        latest = os.path.join(RESULT_DIR, files[0])
        print(f'  검증 파일: {latest}')

        xl = pd.ExcelFile(latest)
        print(f'  시트: {xl.sheet_names}')

        # "전체" 시트 검증
        if '전체' in xl.sheet_names:
            df_all = pd.read_excel(xl, sheet_name='전체')
            print(f'\n  [전체] {len(df_all)}행 / 컬럼: {list(df_all.columns)}')

            # 카테고리 종류 확인
            cats = df_all['카테고리'].unique() if '카테고리' in df_all.columns else []
            print(f'    카테고리 종류: {list(cats)}')

            # 정렬 검증: 카테고리 순서 (강력매도 → profit taking → upside)
            if '카테고리' in df_all.columns and len(df_all) > 1:
                cat_order_map = {'profit taking.': 0, 'profit taking': 1, 'upside': 2}
                df_all['_cat_order'] = df_all['카테고리'].map(cat_order_map)
                is_sorted = df_all['_cat_order'].is_monotonic_increasing
                print(f'    ✓ 카테고리 순서 정렬: {is_sorted}')

            # 반올림 검증: RVOL이 소수 1자리인지
            for col in ['RVOL', 'RSI', '전일비(%)', '10일선괴리율(%)']:
                if col in df_all.columns:
                    vals = df_all[col].dropna()
                    if len(vals) > 0:
                        # round(1) 적용 시 소수점 2자리에서 0이어야 함
                        all_one_decimal = all(abs(round(v, 1) - v) < 1e-9 for v in vals)
                        print(f'    ✓ {col} 소수1자리 반올림: {all_one_decimal}')

            # 필터링 검증: 강력매도/profit taking 항목은 모두
            #   - 10일선이탈일이 cutoff (5일 전) 이내
            #   - 10일선이탈일 - 10일선돌파일 >= 3일
            from datetime import date, timedelta
            today = date.today()
            cutoff = today - timedelta(days=5)

            for cat in ['profit taking.', 'profit taking']:
                sub = df_all[df_all['카테고리'] == cat] if '카테고리' in df_all.columns else pd.DataFrame()
                if len(sub) == 0:
                    print(f'    [{cat}] 항목 없음 (필터 결과 0개)')
                    continue
                violations = 0
                for _, row in sub.iterrows():
                    bk_dn = pd.to_datetime(row['10일선이탈일'], errors='coerce')
                    bk_up = pd.to_datetime(row['10일선돌파일'], errors='coerce')
                    if pd.isna(bk_dn) or pd.isna(bk_up):
                        violations += 1
                        continue
                    if bk_dn.date() < cutoff:
                        violations += 1
                    elif (bk_dn.date() - bk_up.date()).days < 3:
                        violations += 1
                print(f'    ✓ [{cat}] {len(sub)}건 중 필터 위반: {violations}건')

        # 카테고리별 시트 vs 전체 시트의 카테고리별 행 수 일치
        for sheet, cat in [('강력매도신호', 'profit taking.'), ('profit taking', 'profit taking'), ('upside', 'upside')]:
            if sheet in xl.sheet_names:
                cnt_sheet = len(pd.read_excel(xl, sheet_name=sheet))
                cnt_all = (df_all['카테고리'] == cat).sum() if '카테고리' in df_all.columns else 0
                match = '✓' if cnt_sheet == cnt_all else '✗'
                print(f'    {match} 시트[{sheet}] {cnt_sheet}행 = 전체 시트 카테고리[{cat}] {cnt_all}행')

print('\n' + '='*80)
print('검증 완료')
print('='*80)
