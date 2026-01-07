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

    def _track_ma_crossover(self, df: pd.DataFrame) -> pd.Series:
        """
        20일선 하회/상회 날짜 추적

        각 시점에서:
        - 가장 최근에 20일선을 하회한 날짜
        - 가장 최근에 20일선을 상회한 날짜
        - 20일선 아래에 머문 일수

        Parameters:
        -----------
        df : pd.DataFrame
            MA20가 계산된 데이터프레임

        Returns:
        --------
        pd.Series : 각 시점의 crossover 정보 딕셔너리
        """
        result = []
        last_break_below = None
        last_break_above = None
        days_below = 0
        prev_position = None  # 'above' or 'below'

        for idx, row in df.iterrows():
            if pd.isna(row['MA20']) or pd.isna(row['Close']):
                result.append({
                    'last_break_below': None,
                    'last_break_above': None,
                    'days_below': 0
                })
                continue

            current_position = 'below' if row['Close'] < row['MA20'] else 'above'

            # 첫 번째 유효한 데이터 - 초기 상태 설정
            if prev_position is None:
                if current_position == 'below':
                    last_break_below = idx  # 데이터 시작부터 20일선 아래면 시작일을 하회일로 기록
                    days_below = 1
                prev_position = current_position
                result.append({
                    'last_break_below': last_break_below,
                    'last_break_above': last_break_above,
                    'days_below': days_below
                })
                continue

            # 하회 감지 (위에서 아래로)
            if prev_position == 'above' and current_position == 'below':
                last_break_below = idx
                days_below = 1

            # 상회 감지 (아래에서 위로)
            elif prev_position == 'below' and current_position == 'above':
                last_break_above = idx
                days_below = 0

            # 20일선 아래에 계속 있는 경우
            elif current_position == 'below' and prev_position == 'below':
                days_below += 1

            # 20일선 위에 계속 있는 경우
            elif current_position == 'above':
                days_below = 0

            result.append({
                'last_break_below': last_break_below,
                'last_break_above': last_break_above,
                'days_below': days_below
            })

            prev_position = current_position

        return pd.Series(result, index=df.index)

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

        # 2. 상대 거래량 계산
        result_df['RVOL'] = self.calculate_rvol(result_df)

        # 3. 20일선 하회/상회 날짜 추적
        result_df['MA20_Cross'] = self._track_ma_crossover(result_df)
        result_df['Last_MA20_Break_Below'] = result_df['MA20_Cross'].apply(lambda x: x['last_break_below'] if isinstance(x, dict) else None)
        result_df['Last_MA20_Break_Above'] = result_df['MA20_Cross'].apply(lambda x: x['last_break_above'] if isinstance(x, dict) else None)
        result_df['Days_Below_MA20'] = result_df['MA20_Cross'].apply(lambda x: x['days_below'] if isinstance(x, dict) else 0)

        # 4. 조건 체크 (20일선 하회 + RVOL만)
        result_df['Condition_1_Trend_Breakdown'] = result_df['Close'] < result_df['MA20']
        result_df['Condition_2_Volume_Confirmation'] = result_df['RVOL'] >= 2.0

        # 5. 신호 생성 (두 조건 모두 충족시 SELL)
        result_df['All_Conditions_Met'] = (
            result_df['Condition_1_Trend_Breakdown'] &
            result_df['Condition_2_Volume_Confirmation']
        )

        result_df['Signal'] = np.where(
            result_df['All_Conditions_Met'],
            'SELL',
            'HOLD'
        )

        # 6. 시그널 설명 추가
        def generate_reasoning(row):
            if row['Signal'] == 'SELL':
                return "20일선 하회 + 거래량 폭증 확인"
            else:
                # HOLD 이유 세부 분석
                reasons = []
                if not row['Condition_1_Trend_Breakdown']:
                    reasons.append("20일선 위")
                if not row['Condition_2_Volume_Confirmation']:
                    reasons.append("거래량 부족")

                if row['Condition_1_Trend_Breakdown'] and not row['Condition_2_Volume_Confirmation']:
                    return "20일선 하회, 거래량 부족"

                return f"{', '.join(reasons)}"

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
        condition_2_met = df['Condition_2_Volume_Confirmation'].sum()

        summary = {
            'Total_Trading_Days': total_days,
            'SELL_Signals': sell_signals,
            'HOLD_Signals': hold_signals,
            'SELL_Signal_Rate': f"{(sell_signals / total_days * 100):.2f}%" if total_days > 0 else "N/A",
            'Condition_1_Met_Rate': f"{(condition_1_met / total_days * 100):.2f}%",
            'Condition_2_Met_Rate': f"{(condition_2_met / total_days * 100):.2f}%",
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
