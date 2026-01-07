"""
Bloomberg API를 통한 데이터 다운로드

xbbg 라이브러리를 사용하여 Bloomberg Terminal에서 직접 데이터를 가져옵니다.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


def download_bloomberg_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = '1Y'
) -> pd.DataFrame:
    """
    Bloomberg에서 OHLCV 데이터 다운로드

    ====================================================================
    사용 전 필수 조건:
    ====================================================================
    1. Bloomberg Terminal이 실행 중이어야 합니다
    2. Bloomberg에 로그인되어 있어야 합니다
    3. xbbg 라이브러리가 설치되어 있어야 합니다

    ====================================================================
    티커 형식:
    ====================================================================
    - 한국 주식: "005930 KS" (삼성전자), "000660 KS" (SK하이닉스)
    - 미국 주식: "AAPL US" (애플), "MSFT US" (마이크로소프트)
    - 일본 주식: "7203 JP" (도요타)

    Parameters:
    -----------
    ticker : str
        Bloomberg 티커 (예: "005930 KS", "AAPL US")
    start_date : str, optional
        시작 날짜 (YYYY-MM-DD 형식)
        지정하지 않으면 period 사용
    end_date : str, optional
        종료 날짜 (YYYY-MM-DD 형식)
        기본값: 오늘
    period : str
        데이터 기간 (start_date 미지정시)
        - '1Y': 1년
        - '6M': 6개월
        - '3M': 3개월
        기본값: '1Y'

    Returns:
    --------
    pd.DataFrame : OHLCV 데이터
        컬럼: Date, Open, High, Low, Close, Volume

    Examples:
    ---------
    # 삼성전자 1년 데이터
    df = download_bloomberg_data("005930 KS")

    # 애플 2024년 데이터
    df = download_bloomberg_data("AAPL US", "2024-01-01", "2024-12-31")

    # LG에너지솔루션 6개월
    df = download_bloomberg_data("373220 KS", period="6M")
    """
    try:
        from xbbg import blp
    except ImportError:
        raise ImportError(
            "xbbg 라이브러리가 설치되어 있지 않습니다.\n"
            "설치: pip install xbbg"
        )

    # ====================================================================
    # [1단계] 날짜 범위 설정
    # ====================================================================
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date is None:
        # period에 따라 시작 날짜 계산
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        if period == '1Y':
            start_dt = end_dt - timedelta(days=365)
        elif period == '6M':
            start_dt = end_dt - timedelta(days=180)
        elif period == '3M':
            start_dt = end_dt - timedelta(days=90)
        else:
            # 기본값: 1년
            start_dt = end_dt - timedelta(days=365)

        start_date = start_dt.strftime('%Y-%m-%d')

    print(f"\n[Bloomberg 다운로드]")
    print(f"  티커: {ticker}")
    print(f"  기간: {start_date} ~ {end_date}")

    # ====================================================================
    # [2단계] Bloomberg에서 데이터 다운로드
    # ====================================================================
    try:
        # xbbg.blp.bdh() 함수로 과거 데이터 다운로드
        df = blp.bdh(
            tickers=ticker,
            flds=['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME'],
            start_date=start_date,
            end_date=end_date
        )

        if df is None or len(df) == 0:
            raise ValueError(f"데이터를 가져올 수 없습니다: {ticker}")

        # ====================================================================
        # [3단계] 데이터 포맷 변환
        # ====================================================================
        # xbbg는 MultiIndex 컬럼을 반환함 (ticker, field)
        # 단일 티커이므로 컬럼 평탄화
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)

        # 인덱스를 Date 컬럼으로 변환
        df = df.reset_index()

        # 컬럼명 변경
        df = df.rename(columns={
            'index': 'Date',
            'date': 'Date',
            'PX_OPEN': 'Open',
            'PX_HIGH': 'High',
            'PX_LOW': 'Low',
            'PX_LAST': 'Close',
            'PX_VOLUME': 'Volume'
        })

        # 필수 컬럼만 선택
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required_cols]

        # NaN 제거
        df = df.dropna()

        print(f"  ✓ {len(df)}개 일봉 데이터 다운로드 완료")

        return df

    except Exception as e:
        print(f"  ✗ 다운로드 실패: {e}")
        raise


def download_multiple_tickers(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = '1Y',
    save_to_file: bool = False,
    output_dir: str = "."
) -> dict:
    """
    여러 티커의 데이터를 한 번에 다운로드

    Parameters:
    -----------
    tickers : List[str]
        Bloomberg 티커 리스트
    start_date : str, optional
        시작 날짜
    end_date : str, optional
        종료 날짜
    period : str
        데이터 기간 (기본값: '1Y')
    save_to_file : bool
        파일로 저장할지 여부 (기본값: False)
    output_dir : str
        저장 디렉토리 (기본값: 현재 디렉토리)

    Returns:
    --------
    dict : {ticker: DataFrame} 형식의 딕셔너리

    Examples:
    ---------
    # 여러 종목 다운로드
    tickers = ["005930 KS", "000660 KS", "AAPL US"]
    data = download_multiple_tickers(tickers)

    # 파일로 저장
    data = download_multiple_tickers(tickers, save_to_file=True, output_dir="./data")
    """
    import os

    results = {}
    failed = []

    print(f"\n{'='*80}")
    print(f"Bloomberg 일괄 다운로드: {len(tickers)}개 종목")
    print(f"{'='*80}")

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")

        try:
            # 데이터 다운로드
            df = download_bloomberg_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                period=period
            )

            # 결과 저장
            results[ticker] = df

            # 파일로 저장 (옵션)
            if save_to_file:
                # 티커에서 파일명 생성 (공백 제거)
                filename = ticker.replace(" ", "_") + ".xlsx"
                filepath = os.path.join(output_dir, filename)

                # 디렉토리 생성
                os.makedirs(output_dir, exist_ok=True)

                # 엑셀로 저장
                df.to_excel(filepath, index=False)
                print(f"  ✓ 저장됨: {filepath}")

        except Exception as e:
            print(f"  ✗ 실패: {e}")
            failed.append(ticker)
            continue

    # 요약
    print(f"\n{'='*80}")
    print(f"다운로드 완료")
    print(f"{'='*80}")
    print(f"성공: {len(results)}개")
    print(f"실패: {len(failed)}개")

    if failed:
        print(f"\n실패한 티커:")
        for ticker in failed:
            print(f"  - {ticker}")

    return results


# 사용 예시
if __name__ == "__main__":
    # 예시 1: 단일 종목
    df = download_bloomberg_data("005930 KS")
    print(df.tail())

    # 예시 2: 여러 종목
    tickers = ["005930 KS", "000660 KS", "373220 KS"]
    data = download_multiple_tickers(tickers, save_to_file=True, output_dir="./data")
