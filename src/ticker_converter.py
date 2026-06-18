"""
한글 종목명 → Bloomberg 티커 변환기

한글로 된 종목명을 Bloomberg 티커로 자동 변환합니다.
"""

import pandas as pd
from typing import List, Dict, Optional


# ====================================================================
# 한국 주요 종목 매핑 테이블 (예시)
# ====================================================================
KOREAN_STOCK_MAP = {
    # 대형주
    '삼성전자': '005930 KS',
    '삼성전자우': '005935 KS',
    'SK하이닉스': '000660 KS',
    'LG에너지솔루션': '373220 KS',
    '삼성바이오로직스': '207940 KS',
    'POSCO홀딩스': '005490 KS',
    'NAVER': '035420 KS',
    '네이버': '035420 KS',
    '카카오': '035720 KS',
    'LG화학': '051910 KS',
    '현대차': '005380 KS',
    '기아': '000270 KS',
    '셀트리온': '068270 KS',
    '삼성SDI': '006400 KS',
    '현대모비스': '012330 KS',
    'KB금융': '105560 KS',
    '신한지주': '055550 KS',
    'LG전자': '066570 KS',
    '삼성물산': '028260 KS',
    'SK이노베이션': '096770 KS',

    # 중형주
    '하나금융지주': '086790 KS',
    'SK텔레콤': '017670 KS',
    'KT': '030200 KS',
    'S-Oil': '010950 KS',
    '한국전력': '015760 KS',
    'LG': '003550 KS',
    '포스코퓨처엠': '003670 KS',
    '현대건설': '000720 KS',
    '아모레퍼시픽': '090430 KS',
    '삼성생명': '032830 KS',

    # 2차전지/배터리
    '삼성SDI': '006400 KS',
    'LG에너지솔루션': '373220 KS',
    'SK온': '096770 KS',
    '포스코퓨처엠': '003670 KS',
    '에코프로비엠': '247540 KS',
    '에코프로': '086520 KS',

    # 반도체
    '삼성전자': '005930 KS',
    'SK하이닉스': '000660 KS',
    'SK스퀘어': '402340 KS',

    # 바이오
    '삼성바이오로직스': '207940 KS',
    '셀트리온': '068270 KS',
    '셀트리온헬스케어': '091990 KS',
    '셀트리온제약': '068760 KS',

    # 엔터/미디어
    'HYBE': '352820 KS',
    '하이브': '352820 KS',
    'JYP': '035900 KS',
    'SM': '041510 KS',
    'YG': '122870 KS',

    # 기타
    '카카오뱅크': '323410 KS',
    '크래프톤': '259960 KS',
    '두산에너빌리티': '034020 KS',
}


def load_ticker_mapping_from_excel(filepath: str) -> Dict[str, str]:
    """
    엑셀 파일에서 종목명-티커 매핑 로드

    엑셀 형식:
    | 종목명 | 티커 |
    |--------|------|
    | 삼성전자 | 005930 KS |
    | SK하이닉스 | 000660 KS |

    Parameters:
    -----------
    filepath : str
        엑셀 파일 경로

    Returns:
    --------
    dict : {종목명: 티커} 매핑
    """
    df = pd.read_excel(filepath)

    # 첫 번째 컬럼: 종목명, 두 번째 컬럼: 티커
    if len(df.columns) < 2:
        raise ValueError("엑셀 파일은 최소 2개 컬럼(종목명, 티커)이 필요합니다")

    name_col = df.columns[0]
    ticker_col = df.columns[1]

    mapping = {}
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        ticker = str(row[ticker_col]).strip()

        if name and ticker and name != 'nan' and ticker != 'nan':
            mapping[name] = ticker

    return mapping


def convert_names_to_tickers(
    names: List[str],
    custom_mapping: Optional[Dict[str, str]] = None,
    raise_on_missing: bool = False
) -> List[str]:
    """
    한글 종목명 리스트를 Bloomberg 티커 리스트로 변환

    Parameters:
    -----------
    names : List[str]
        한글 종목명 리스트
    custom_mapping : dict, optional
        사용자 정의 매핑 (우선 적용됨)
    raise_on_missing : bool
        변환 실패시 에러 발생 여부 (기본값: False)

    Returns:
    --------
    List[str] : Bloomberg 티커 리스트

    Examples:
    ---------
    names = ["삼성전자", "SK하이닉스", "LG에너지솔루션"]
    tickers = convert_names_to_tickers(names)
    # ['005930 KS', '000660 KS', '373220 KS']
    """
    # 매핑 테이블 병합 (custom이 우선)
    mapping = KOREAN_STOCK_MAP.copy()
    if custom_mapping:
        mapping.update(custom_mapping)

    tickers = []
    missing = []

    for name in names:
        name = name.strip()

        if name in mapping:
            tickers.append(mapping[name])
        else:
            missing.append(name)
            if not raise_on_missing:
                # 매핑 실패시 원래 이름 유지
                tickers.append(name)

    if missing:
        print(f"\n⚠️  변환 실패 종목 ({len(missing)}개):")
        for name in missing:
            print(f"  - {name}")
        print("\n해결 방법:")
        print("  1. ticker_mapping.xlsx 파일에 추가")
        print("  2. KOREAN_STOCK_MAP에 직접 추가")

    if raise_on_missing and missing:
        raise ValueError(f"변환 실패: {missing}")

    return tickers


def batch_convert_from_file(
    input_file: str,
    output_file: Optional[str] = None,
    mapping_file: Optional[str] = None
) -> List[str]:
    """
    파일에서 종목명을 읽어 티커로 변환

    입력 파일 형식:
    - 텍스트 파일: 한 줄에 하나씩 또는 쉼표로 구분
    - 엑셀 파일: 첫 번째 컬럼에 종목명

    Parameters:
    -----------
    input_file : str
        입력 파일 경로 (.txt, .xlsx)
    output_file : str, optional
        출력 파일 경로 (None이면 화면 출력만)
    mapping_file : str, optional
        사용자 정의 매핑 엑셀 파일

    Returns:
    --------
    List[str] : Bloomberg 티커 리스트

    Examples:
    ---------
    # stocks.txt:
    # 삼성전자
    # SK하이닉스
    # LG에너지솔루션

    tickers = batch_convert_from_file("stocks.txt")
    """
    # 사용자 정의 매핑 로드
    custom_mapping = None
    if mapping_file:
        print(f"사용자 정의 매핑 로드: {mapping_file}")
        custom_mapping = load_ticker_mapping_from_excel(mapping_file)
        print(f"  ✓ {len(custom_mapping)}개 매핑 로드")

    # 입력 파일 읽기
    print(f"\n종목명 파일 읽기: {input_file}")

    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        # 엑셀 파일
        df = pd.read_excel(input_file)
        names = df.iloc[:, 0].astype(str).tolist()
    else:
        # 텍스트 파일
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 쉼표 또는 줄바꿈으로 분리
        if ',' in content:
            names = [n.strip() for n in content.split(',')]
        else:
            names = [n.strip() for n in content.split('\n')]

    # 빈 문자열 제거
    names = [n for n in names if n and n != 'nan']

    print(f"  ✓ {len(names)}개 종목명 로드")

    # 변환
    print(f"\nBloomberg 티커로 변환 중...")
    tickers = convert_names_to_tickers(names, custom_mapping)

    print(f"  ✓ {len(tickers)}개 티커 변환 완료")

    # 출력 파일 저장
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(', '.join(tickers))
        print(f"\n저장 완료: {output_file}")

    return tickers


def print_ticker_list(tickers: List[str], max_per_line: int = 5):
    """
    티커 리스트를 보기 좋게 출력

    Parameters:
    -----------
    tickers : List[str]
        티커 리스트
    max_per_line : int
        한 줄에 출력할 최대 티커 수
    """
    print(f"\n{'='*80}")
    print(f"변환된 Bloomberg 티커 ({len(tickers)}개)")
    print(f"{'='*80}\n")

    for i in range(0, len(tickers), max_per_line):
        chunk = tickers[i:i+max_per_line]
        print(', '.join(chunk))

    print(f"\n{'='*80}")
    print("\n복사해서 start_bloomberg.py에 붙여넣기:")
    print(', '.join(tickers))


# 사용 예시
if __name__ == "__main__":
    # 예시 1: 리스트로 변환
    names = [
        "삼성전자", "SK하이닉스", "LG에너지솔루션",
        "삼성바이오로직스", "NAVER", "카카오"
    ]

    print("="*80)
    print("예시 1: 리스트로 변환")
    print("="*80)
    tickers = convert_names_to_tickers(names)
    print_ticker_list(tickers)

    # 예시 2: 파일에서 변환
    # batch_convert_from_file("stocks.txt", "tickers.txt")
