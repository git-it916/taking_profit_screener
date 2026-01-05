"""
Volume-Confirmed Rejection Exit Signal Screener
Strict Quantitative Technical Analysis for Backtesting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ExitSignalScreener:
    """
    기술적 지표 기반 익절 신호 스크리너

    Strategy: Volume-Confirmed Rejection
    - Condition 1: Trend Breakdown (Price < 20MA)
    - Condition 2: Rejection Pattern (Upper Wick >= 50% of Total Candle)
    - Condition 3: Volume Confirmation (RVOL >= 2.0)
    """

    def __init__(self, ma_period: int = 20, rvol_period: int = 20):
        """
        Parameters:
        -----------
        ma_period : int
            이동평균선 기간 (default: 20)
        rvol_period : int
            상대 거래량 계산 기간 (default: 20)
        """
        self.ma_period = ma_period
        self.rvol_period = rvol_period

    def calculate_sma(self, df: pd.DataFrame, column: str = 'Close') -> pd.Series:
        """
        단순 이동평균선(SMA) 계산

        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV 데이터
        column : str
            계산할 컬럼명 (default: 'Close')

        Returns:
        --------
        pd.Series : 20일 이동평균선
        """
        return df[column].rolling(window=self.ma_period).mean()

    def calculate_wick_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        상단 윅(Upper Shadow) 비율 계산

        Wick Ratio = Upper Wick Length / Total Candle Length
        - Upper Wick = High - max(Open, Close)
        - Total Candle = High - Low

        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV 데이터 (Open, High, Low, Close 필수)

        Returns:
        --------
        pd.Series : 윅 비율 (0~1 사이 값)
        """
        upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
        total_candle = df['High'] - df['Low']

        # 0으로 나누기 방지 (도지 캔들 등)
        wick_ratio = np.where(
            total_candle > 0,
            upper_wick / total_candle,
            0
        )

        return pd.Series(wick_ratio, index=df.index)

    def calculate_rvol(self, df: pd.DataFrame, volume_column: str = 'Volume') -> pd.Series:
        """
        상대 거래량(RVOL) 계산

        RVOL = Current Volume / Average Volume (N periods)

        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV 데이터
        volume_column : str
            거래량 컬럼명 (default: 'Volume')

        Returns:
        --------
        pd.Series : 상대 거래량
        """
        avg_volume = df[volume_column].rolling(window=self.rvol_period).mean()
        rvol = df[volume_column] / avg_volume

        return rvol

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표를 계산하고 필터 적용

        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV 데이터 (Date, Open, High, Low, Close, Volume 필수)

        Returns:
        --------
        pd.DataFrame : 지표가 추가된 데이터프레임
        """
        # 필수 컬럼 확인
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # 데이터 복사
        result_df = df.copy()

        # 1. 20일 이동평균선 계산
        result_df['MA20'] = self.calculate_sma(result_df)

        # 2. 윅 비율 계산
        result_df['Wick_Ratio'] = self.calculate_wick_ratio(result_df)

        # 3. 상대 거래량 계산
        result_df['RVOL'] = self.calculate_rvol(result_df)

        # 4. 각 조건 체크
        result_df['Condition_1_Trend_Breakdown'] = result_df['Close'] < result_df['MA20']
        result_df['Condition_2_Rejection_Pattern'] = result_df['Wick_Ratio'] >= 0.5
        result_df['Condition_3_Volume_Confirmation'] = result_df['RVOL'] >= 2.0

        # 5. 신호 생성 (모든 조건이 TRUE일 때만 SELL)
        result_df['All_Conditions_Met'] = (
            result_df['Condition_1_Trend_Breakdown'] &
            result_df['Condition_2_Rejection_Pattern'] &
            result_df['Condition_3_Volume_Confirmation']
        )

        result_df['Signal'] = np.where(
            result_df['All_Conditions_Met'],
            'SELL',
            'HOLD'
        )

        # 6. 시그널 설명 추가
        def generate_reasoning(row):
            if row['Signal'] == 'SELL':
                return "Valid technical breakdown confirmed by high volume."
            else:
                # HOLD 이유 세부 분석
                reasons = []
                if not row['Condition_1_Trend_Breakdown']:
                    reasons.append("Price above 20MA")
                if not row['Condition_2_Rejection_Pattern']:
                    reasons.append("No rejection pattern")
                if not row['Condition_3_Volume_Confirmation']:
                    reasons.append("Lacks volume confirmation")

                if row['Condition_1_Trend_Breakdown'] and row['Condition_2_Rejection_Pattern'] and not row['Condition_3_Volume_Confirmation']:
                    return "Pattern detected but lacks volume confirmation (Potential Trap)."

                return f"Pattern incomplete: {', '.join(reasons)}."

        result_df['Reasoning'] = result_df.apply(generate_reasoning, axis=1)

        return result_df

    def generate_screening_output(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> pd.DataFrame:
        """
        스크리닝 결과를 사용자 요구 형식으로 출력

        Output Format:
        Ticker, Current Price, 20MA, Wick Ratio, RVOL, Signal, Reasoning

        Parameters:
        -----------
        df : pd.DataFrame
            필터가 적용된 데이터프레임
        ticker : str
            종목 티커 (default: "UNKNOWN")

        Returns:
        --------
        pd.DataFrame : 스크리닝 결과
        """
        output_df = pd.DataFrame({
            'Ticker': ticker,
            'Date': df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('Date', range(len(df))),
            'Current_Price': df['Close'],
            'MA20': df['MA20'],
            'Wick_Ratio': df['Wick_Ratio'].round(3),
            'RVOL': df['RVOL'].round(2),
            'Signal': df['Signal'],
            'Reasoning': df['Reasoning']
        })

        return output_df

    def backtest_summary(self, df: pd.DataFrame) -> dict:
        """
        백테스팅 요약 통계

        Parameters:
        -----------
        df : pd.DataFrame
            필터가 적용된 데이터프레임

        Returns:
        --------
        dict : 요약 통계
        """
        total_days = len(df)
        sell_signals = (df['Signal'] == 'SELL').sum()
        hold_signals = (df['Signal'] == 'HOLD').sum()

        # 조건별 충족률
        condition_1_met = df['Condition_1_Trend_Breakdown'].sum()
        condition_2_met = df['Condition_2_Rejection_Pattern'].sum()
        condition_3_met = df['Condition_3_Volume_Confirmation'].sum()

        summary = {
            'Total_Trading_Days': total_days,
            'SELL_Signals': sell_signals,
            'HOLD_Signals': hold_signals,
            'SELL_Signal_Rate': f"{(sell_signals / total_days * 100):.2f}%" if total_days > 0 else "N/A",
            'Condition_1_Met_Rate': f"{(condition_1_met / total_days * 100):.2f}%",
            'Condition_2_Met_Rate': f"{(condition_2_met / total_days * 100):.2f}%",
            'Condition_3_Met_Rate': f"{(condition_3_met / total_days * 100):.2f}%",
        }

        return summary


def load_data_from_csv(filepath: str, date_column: Optional[str] = None) -> pd.DataFrame:
    """
    CSV 또는 XLSX 파일에서 OHLCV 데이터 로드

    Parameters:
    -----------
    filepath : str
        CSV 또는 XLSX 파일 경로
    date_column : str, optional
        날짜 컬럼명 (인덱스로 설정)

    Returns:
    --------
    pd.DataFrame : OHLCV 데이터
    """
    # 파일 확장자에 따라 로드 방법 선택
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)

    return df


# 사용 예시
if __name__ == "__main__":
    # 예시 데이터 생성 (실제로는 CSV에서 로드)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': 102 + np.random.randn(100).cumsum(),
        'Low': 98 + np.random.randn(100).cumsum(),
        'Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000000, 5000000, 100)
    })

    # High가 항상 가장 높고 Low가 가장 낮도록 조정
    sample_data['High'] = sample_data[['Open', 'High', 'Close']].max(axis=1) + abs(np.random.randn(100))
    sample_data['Low'] = sample_data[['Open', 'Low', 'Close']].min(axis=1) - abs(np.random.randn(100))

    sample_data.set_index('Date', inplace=True)

    # 스크리너 초기화
    screener = ExitSignalScreener(ma_period=20, rvol_period=20)

    # 필터 적용
    filtered_data = screener.apply_filters(sample_data)

    # 스크리닝 결과 출력
    output = screener.generate_screening_output(filtered_data, ticker="SAMPLE")

    # SELL 신호만 필터링
    sell_signals = output[output['Signal'] == 'SELL']

    print("=" * 80)
    print("EXIT SIGNAL SCREENER - Volume-Confirmed Rejection Strategy")
    print("=" * 80)
    print("\n[SELL SIGNALS]")
    print(sell_signals.to_string(index=False))

    print("\n" + "=" * 80)
    print("[BACKTEST SUMMARY]")
    print("=" * 80)
    summary = screener.backtest_summary(filtered_data)
    for key, value in summary.items():
        print(f"{key}: {value}")

    # CSV 저장
    output.to_csv('screening_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n결과가 'screening_results.csv'에 저장되었습니다.")
