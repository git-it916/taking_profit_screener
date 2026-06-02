"""
테스트용 샘플 XLSX 데이터 생성

실제 종목명으로 샘플 데이터를 생성합니다. (XLSX 형식)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_stock_data(ticker: str, days: int = 100, pattern: str = "normal"):
    """
    샘플 주가 데이터 생성

    Parameters:
    -----------
    ticker : str
        종목명
    days : int
        생성할 데이터 일수
    pattern : str
        'normal' - 정상, 'sell_signal' - SELL 신호, 'trend_down' - 추세하락만
    """
    np.random.seed(hash(ticker) % 1000)

    # 날짜 생성
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(days-1, -1, -1)]

    # 기본 가격 데이터
    if pattern == "sell_signal":
        # SELL 신호가 나오도록 설정
        trend = np.linspace(110, 90, days)  # 하락 추세
        noise = np.random.randn(days) * 0.5
        close_prices = trend + noise

        # 마지막 날에 강한 윗꼬리 + 높은 거래량
        volumes = np.random.randint(800000, 1200000, days)
        volumes[-1] = 2500000  # RVOL 2.5배 이상

    elif pattern == "trend_down":
        # 추세 하락만
        trend = np.linspace(110, 85, days)
        noise = np.random.randn(days) * 0.5
        close_prices = trend + noise
        volumes = np.random.randint(800000, 1200000, days)

    elif pattern == "rvol_surge":
        # 거래량 폭증만
        trend = np.linspace(100, 120, days)  # 상승 추세
        noise = np.random.randn(days) * 0.5
        close_prices = trend + noise
        volumes = np.random.randint(800000, 1200000, days)
        volumes[-1] = 3000000  # 거래량 폭증

    else:  # normal
        trend = np.linspace(100, 110, days)
        noise = np.random.randn(days) * 0.5
        close_prices = trend + noise
        volumes = np.random.randint(800000, 1200000, days)

    # OHLC 데이터 생성
    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices + np.random.randn(days) * 0.5,
        'Close': close_prices,
        'Volume': volumes
    })

    # High/Low 계산
    df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.randn(days) * 1.5)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.randn(days) * 1.5)

    # SELL 신호 패턴인 경우 마지막 날 윗꼬리 추가
    if pattern == "sell_signal":
        last_idx = len(df) - 1
        df.loc[last_idx, 'High'] = df.loc[last_idx, 'Close'] + 5  # 강한 윗꼬리

    return df


def main():
    """샘플 데이터 생성"""
    print("샘플 XLSX 파일 생성 중...\n")

    # 문서 경로
    base_path = r"C:\Users\10845\OneDrive - 이지스자산운용\문서"

    # 다양한 패턴의 샘플 생성
    samples = [
        ("sm.xlsx", "sell_signal", "삼성전자 - SELL 신호"),
        ("samsung.xlsx", "trend_down", "삼성SDI - 추세 하락만"),
        ("apple.xlsx", "normal", "애플 - 정상"),
        ("kakao.xlsx", "rvol_surge", "카카오 - 거래량 폭증"),
    ]

    for filename, pattern, description in samples:
        df = create_sample_stock_data(filename.replace('.xlsx', ''), days=100, pattern=pattern)

        # XLSX 저장
        filepath = f"{base_path}\\{filename}"
        df.to_excel(filepath, index=False, engine='openpyxl')
        print(f"[OK] {filename} 생성 완료 - {description}")
        print(f"  경로: {filepath}")
        print(f"  데이터: {len(df)}일, {df['Date'].iloc[0]} ~ {df['Date'].iloc[-1]}\n")

    print("=" * 80)
    print("모든 샘플 파일 생성 완료!")
    print("=" * 80)
    print("\n다음 명령어로 분석해보세요:")
    print("  python quick_analyze.py sm.xlsx")
    print("  python quick_analyze.py sm.xlsx samsung.xlsx apple.xlsx kakao.xlsx")
    print("  python analyze_stocks.py")


if __name__ == "__main__":
    main()
