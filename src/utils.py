"""
공통 유틸리티 함수들
"""

import pandas as pd
import numpy as np
from typing import Optional


def generate_test_data(days: int = 300, seed: int = 42, start_date: str = '2023-01-01') -> pd.DataFrame:
    """
    테스트용 주가 데이터 생성

    Parameters:
    -----------
    days : int
        생성할 데이터 일수 (default: 300)
    seed : int
        랜덤 시드 (default: 42)
    start_date : str
        시작 날짜 (default: '2023-01-01')

    Returns:
    --------
    pd.DataFrame
        OHLCV 데이터프레임 (인덱스는 날짜)
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, periods=days, freq='D')

    # 현실적인 가격 움직임 생성
    trend = np.linspace(100, 120, days)
    noise = np.random.randn(days).cumsum() * 0.5
    close_prices = trend + noise

    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(days) * 0.5,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, days)
    })

    # High/Low 보정 (High는 항상 가장 높고, Low는 항상 가장 낮아야 함)
    df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.randn(days) * 1.5)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.randn(days) * 1.5)

    df.index = dates

    return df


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    숫자를 퍼센트 문자열로 변환

    Parameters:
    -----------
    value : float
        변환할 숫자
    decimals : int
        소수점 자릿수 (default: 2)

    Returns:
    --------
    str
        퍼센트 문자열 (예: "12.34%")
    """
    return f"{value:.{decimals}f}%"


def print_separator(char: str = "=", length: int = 80, message: Optional[str] = None):
    """
    구분선 출력

    Parameters:
    -----------
    char : str
        구분선 문자 (default: "=")
    length : int
        구분선 길이 (default: 80)
    message : str, optional
        구분선 중앙에 표시할 메시지
    """
    if message:
        padding = (length - len(message) - 2) // 2
        print(f"{char * padding} {message} {char * padding}")
    else:
        print(char * length)


def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> bool:
    """
    데이터프레임 유효성 검사

    Parameters:
    -----------
    df : pd.DataFrame
        검사할 데이터프레임
    required_columns : list, optional
        필수 컬럼 리스트 (default: ['Open', 'High', 'Low', 'Close', 'Volume'])

    Returns:
    --------
    bool
        유효하면 True, 아니면 False

    Raises:
    -------
    ValueError
        필수 컬럼이 없는 경우
    """
    if required_columns is None:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 데이터 개수 확인
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    return True


def save_results(df: pd.DataFrame, filepath: str, encoding: str = 'utf-8-sig'):
    """
    결과를 CSV 파일로 저장

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 데이터프레임
    filepath : str
        저장 경로
    encoding : str
        인코딩 방식 (default: 'utf-8-sig', Excel 호환용)
    """
    df.to_csv(filepath, index=False, encoding=encoding)
    print(f"\n결과가 '{filepath}'에 저장되었습니다. ({len(df)}개 행)")
