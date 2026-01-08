"""
Volume-Confirmed Rejection Exit Signal Screener
Strict Quantitative Technical Analysis for Backtesting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def convert_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    시간봉 데이터를 일봉으로 변환 (OHLCV 리샘플링)

    ====================================================================
    시간봉 → 일봉 변환 규칙:
    ====================================================================
    - Open: 하루 중 첫 번째 시간의 시가
    - High: 하루 중 최고가
    - Low: 하루 중 최저가
    - Close: 하루 중 마지막 시간의 종가
    - Volume: 하루 거래량 합계

    예시:
    2025-12-30 09:00: Open=369500, High=370000, Low=369000, Close=370000, Volume=1000
    2025-12-30 10:00: Open=370000, High=371000, Low=369500, Close=370500, Volume=1500
    2025-12-30 11:00: Open=370500, High=372000, Low=370000, Close=371000, Volume=2000

    → 일봉 변환:
    2025-12-30: Open=369500 (첫 시가), High=372000 (최고),
                Low=369000 (최저), Close=371000 (마지막 종가),
                Volume=4500 (합계)

    Parameters:
    -----------
    df : pd.DataFrame
        시간봉 OHLCV 데이터 (Date 컬럼 필수)

    Returns:
    --------
    pd.DataFrame : 일봉으로 변환된 데이터
    """
    # Date 컬럼이 있는지 확인
    if 'Date' not in df.columns:
        raise ValueError("Date 컬럼이 필요합니다.")

    # Date를 datetime으로 변환
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])

    # Date를 인덱스로 설정
    df_copy.set_index('Date', inplace=True)

    # ====================================================================
    # 일봉으로 리샘플링 ('D' = Daily)
    # ====================================================================
    daily_df = df_copy.resample('D').agg({
        'Open': 'first',    # 하루 중 첫 번째 Open (장 시작 시가)
        'High': 'max',      # 하루 중 최고가
        'Low': 'min',       # 하루 중 최저가
        'Close': 'last',    # 하루 중 마지막 Close (장 마감 종가)
        'Volume': 'sum'     # 하루 거래량 합계
    })

    # NaN 제거 (거래가 없는 날)
    daily_df.dropna(inplace=True)

    # 인덱스를 Date 컬럼으로 복원
    daily_df.reset_index(inplace=True)

    print(f"[변환 완료] 시간봉 {len(df)}개 → 일봉 {len(daily_df)}개")

    return daily_df


class ExitSignalScreener:
    """
    기술적 지표 기반 익절 신호 스크리너

    Strategy: Volume-Confirmed Rejection
    - Condition 1: Trend Breakdown (Price < 10MA)
    - Condition 2: Rejection Pattern (Upper Wick >= 50% of Total Candle)
    - Condition 3: Volume Confirmation (RVOL >= 2.0)
    """

    def __init__(self, ma_period: int = 10, rvol_period: int = 10):
        """
        Parameters:
        -----------
        ma_period : int
            이동평균선 기간 (default: 10)
        rvol_period : int
            상대 거래량 계산 기간 (default: 10)
        """
        self.ma_period = ma_period
        self.rvol_period = rvol_period

    def calculate_sma(self, df: pd.DataFrame, column: str = 'Close') -> pd.Series:
        """
        단순 이동평균선(SMA) 계산

        ====================================================================
        SMA (단순 이동평균선) 공식:
        ====================================================================
        SMA_20 = (P_1 + P_2 + P_3 + ... + P_20) / 20

        여기서:
        - P_1 = 오늘 종가 (가장 최근)
        - P_2 = 1일 전 종가
        - P_20 = 20일 전 종가 (가장 오래됨)

        각 가격은 동등한 비중(1/20 = 5%)을 가집니다.

        ====================================================================
        계산 예시:
        ====================================================================
        날짜 1~19: MA10 = NaN (데이터 부족)
        날짜 20: MA10 = (1일~20일 종가 합계) / 20
        날짜 21: MA10 = (2일~21일 종가 합계) / 20  ← "이동": 1일 제외, 21일 추가
        날짜 22: MA10 = (3일~22일 종가 합계) / 20  ← "이동": 2일 제외, 22일 추가

        매일 가장 오래된 데이터는 제외되고 새 데이터가 추가되므로 "이동평균"

        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV 데이터
        column : str
            계산할 컬럼명 (default: 'Close')

        Returns:
        --------
        pd.Series : 20일 이동평균선
            - 초기 19일은 NaN (데이터 부족)
            - 20일째부터 값이 계산됨
        """
        # pandas의 rolling().mean()은 표준 SMA 공식과 100% 일치
        # 검증: MA_CALCULATION_VERIFICATION.md 참조 (수동 계산과 차이 0.0000000000)
        return df[column].rolling(window=self.ma_period).mean()


    def calculate_rvol(self, df: pd.DataFrame, volume_column: str = 'Volume') -> pd.Series:
        """
        상대 거래량(RVOL) 계산

        ====================================================================
        RVOL (Relative Volume) 공식:
        ====================================================================
        RVOL = 현재 거래량 / 과거 N일 평균 거래량

        여기서 N = self.rvol_period (기본값: 20일)

        ====================================================================
        계산 예시: (N=20일 기준)
        ====================================================================
        날짜 20:
            평균 거래량 = (1일~20일 거래량 합계) / 20
            RVOL = 20일 거래량 / 평균 거래량

        날짜 21:
            평균 거래량 = (2일~21일 거래량 합계) / 20
            RVOL = 21일 거래량 / 평균 거래량

        실제 예:
            평균 거래량 = 500,000주
            오늘 거래량 = 1,000,000주
            → RVOL = 1,000,000 / 500,000 = 2.0 (평균의 2배!)

        ====================================================================
        RVOL 해석:
        ====================================================================
        - RVOL = 1.0: 평균 수준의 거래량
        - RVOL = 2.0: 평균의 2배 (거래량 폭증!) ← SELL 조건 충족선
        - RVOL = 0.5: 평균의 절반 (거래량 부족)
        - RVOL = 3.0+: 매우 강한 거래량 (특이 이벤트 가능성)

        강도 분류:
        - 매우 강함: RVOL ≥ 3.0
        - 강함: 2.5 ≤ RVOL < 3.0
        - 보통: 2.0 ≤ RVOL < 2.5  ← 조건 2 충족
        - 약함: RVOL < 2.0

        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV 데이터
        volume_column : str
            거래량 컬럼명 (default: 'Volume')

        Returns:
        --------
        pd.Series : 상대 거래량 비율
            - 값 범위: 0 ~ 무한대 (실제로는 0.1~10 사이가 일반적)
            - 초기 19일은 NaN (평균 계산 불가)
        """
        # ====================================================================
        # [1단계] 과거 N일 평균 거래량 계산
        # ====================================================================
        # rolling(window=20).mean() → 최근 20일 거래량의 평균
        avg_volume = df[volume_column].rolling(window=self.rvol_period).mean()

        # DEBUG: 평균 거래량 확인용 (필요시 주석 해제)
        # print(f"평균 거래량 최근 5일:\n{avg_volume.tail()}")

        # ====================================================================
        # [2단계] RVOL 계산 (현재 거래량 / 평균 거래량)
        # ====================================================================
        # 예: 현재 거래량 = 1,000,000, 평균 거래량 = 500,000
        #     → RVOL = 2.0 (평균의 2배! → SELL 조건 충족)
        rvol = df[volume_column] / avg_volume

        # DEBUG: RVOL 확인용 (필요시 주석 해제)
        # print(f"RVOL 최근 5일:\n{rvol.tail()}")

        return rvol

    def _track_ma_crossover(self, df: pd.DataFrame) -> pd.Series:
        """
        10일선 하회/상회 날짜 추적

        각 시점에서:
        - 가장 최근에 10일선을 하회한 날짜
        - 가장 최근에 10일선을 상회한 날짜
        - 10일선 아래에 머문 일수

        Parameters:
        -----------
        df : pd.DataFrame
            MA10가 계산된 데이터프레임

        Returns:
        --------
        pd.Series : 각 시점의 crossover 정보 딕셔너리
        """
        result = []  # 각 날짜별 결과를 저장할 리스트

        # ====================================================================
        # 추적 변수 초기화
        # ====================================================================
        last_break_below = None  # 마지막으로 10일선을 하회한 날짜 (위 → 아래로 내려간 날)
        last_break_above = None  # 마지막으로 10일선을 상회한 날짜 (아래 → 위로 올라간 날)
        days_below = 0           # 10일선 아래에 머문 일수 (연속으로 아래에 있었던 일수)
        prev_position = None     # 이전 시점의 위치 ('above' 또는 'below')

        # ====================================================================
        # 데이터프레임의 모든 행을 순회하며 하회/상회 추적
        # ====================================================================
        for idx, row in df.iterrows():
            # 현재 행의 날짜를 가져옴 (Date 컬럼이 있으면 사용, 없으면 인덱스 사용)
            current_date = row.get('Date', idx)
            if pd.notna(current_date) and hasattr(current_date, 'strftime'):
                current_date = current_date.strftime('%Y-%m-%d')

            # ================================================================
            # [1단계] MA10 또는 Close가 없는 경우 처리
            # ================================================================
            # MA10은 10일치 데이터가 있어야 계산되므로, 초기 9일은 NaN입니다.
            # 이 경우 추적 불가능하므로 None 반환
            if pd.isna(row['MA10']) or pd.isna(row['Close']):
                result.append({
                    'last_break_below': None,
                    'last_break_above': None,
                    'days_below': 0
                })
                continue

            # ================================================================
            # [2단계] 현재 위치 판단
            # ================================================================
            # 종가와 10일선을 비교하여 현재 위치 결정:
            # - Close < MA10 → 'below' (10일선 아래)
            # - Close >= MA10 → 'above' (10일선 위 또는 동일)
            current_position = 'below' if row['Close'] < row['MA10'] else 'above'

            # DEBUG: 현재 위치 확인용 (필요시 주석 해제)
            # print(f"{idx}: Close={row['Close']:.2f}, MA10={row['MA10']:.2f}, Position={current_position}")

            # ================================================================
            # [3단계] 첫 번째 유효한 데이터 처리
            # ================================================================
            # prev_position이 None이면 이것이 첫 번째 유효 데이터입니다.
            if prev_position is None:
                # 데이터 시작부터 이미 10일선 아래에 있으면
                # 시작 날짜를 하회일로 기록합니다.
                if current_position == 'below':
                    last_break_below = current_date  # 첫 날짜를 하회일로 설정
                    days_below = 1
                    # DEBUG: 첫 데이터가 하회 상태
                    # print(f"  → 첫 데이터: 10일선 아래 시작, 하회일={current_date}")

                prev_position = current_position  # 다음 루프를 위해 현재 위치 저장
                result.append({
                    'last_break_below': last_break_below,
                    'last_break_above': last_break_above,
                    'days_below': days_below
                })
                continue

            # ================================================================
            # [4단계] 하회 감지 (위 → 아래)
            # ================================================================
            # 이전: 10일선 위(above), 현재: 10일선 아래(below)
            # → 이 시점에 10일선을 "하회"한 것!
            if prev_position == 'above' and current_position == 'below':
                last_break_below = current_date  # 현재 날짜를 "하회일"로 기록
                days_below = 1          # 하회 경과일 1일로 초기화
                # DEBUG: 하회 감지
                # print(f"  → 하회 감지! {current_date}에 10일선 아래로 내려감")
                # print(f"     Close={row['Close']:.2f} < MA10={row['MA10']:.2f}")

            # ================================================================
            # [5단계] 상회 감지 (아래 → 위)
            # ================================================================
            # 이전: 10일선 아래(below), 현재: 10일선 위(above)
            # → 이 시점에 10일선을 "상회"한 것!
            elif prev_position == 'below' and current_position == 'above':
                last_break_above = current_date  # 현재 날짜를 "상회일"로 기록
                days_below = 0          # 하회 경과일 0으로 리셋 (더 이상 아래에 없음)
                # DEBUG: 상회 감지
                # print(f"  → 상회 감지! {current_date}에 10일선 위로 올라감")
                # print(f"     Close={row['Close']:.2f} >= MA10={row['MA10']:.2f}")

            # ================================================================
            # [6단계] 10일선 아래에 계속 머무는 경우
            # ================================================================
            # 이전: 아래(below), 현재: 아래(below)
            # → 계속 아래에 머물고 있으므로 경과일만 1일 증가
            elif current_position == 'below' and prev_position == 'below':
                days_below += 1  # 하회 경과일 1일 증가
                # DEBUG: 계속 아래에 있음
                # print(f"  → 계속 아래: {days_below}일째 (Close={row['Close']:.2f}, MA10={row['MA10']:.2f})")

            # ================================================================
            # [7단계] 10일선 위에 계속 머무는 경우
            # ================================================================
            # 이전: 위(above), 현재: 위(above)
            # → 계속 위에 있으므로 경과일은 0 유지
            elif current_position == 'above':
                days_below = 0  # 위에 있으면 경과일은 항상 0
                # DEBUG: 계속 위에 있음
                # print(f"  → 계속 위: 경과일 0 (Close={row['Close']:.2f}, MA10={row['MA10']:.2f})")

            # ================================================================
            # [8단계] 현재 날짜의 결과 저장
            # ================================================================
            result.append({
                'last_break_below': last_break_below,  # 가장 최근 하회일
                'last_break_above': last_break_above,  # 가장 최근 상회일
                'days_below': days_below               # 현재까지 하회 경과일
            })

            # ================================================================
            # [9단계] 다음 루프를 위해 현재 위치를 이전 위치로 저장
            # ================================================================
            prev_position = current_position

        # 결과를 pandas Series로 변환하여 반환
        # index는 원본 데이터프레임의 index(날짜)와 동일하게 유지
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

        # 1. 10일 이동평균선 계산
        result_df['MA10'] = self.calculate_sma(result_df)

        # 2. 상대 거래량 계산
        result_df['RVOL'] = self.calculate_rvol(result_df)

        # 3. 10일선 하회/상회 날짜 추적
        result_df['MA10_Cross'] = self._track_ma_crossover(result_df)
        result_df['Last_MA10_Break_Below'] = result_df['MA10_Cross'].apply(lambda x: x['last_break_below'] if isinstance(x, dict) else None)
        result_df['Last_MA10_Break_Above'] = result_df['MA10_Cross'].apply(lambda x: x['last_break_above'] if isinstance(x, dict) else None)
        result_df['Days_Below_MA10'] = result_df['MA10_Cross'].apply(lambda x: x['days_below'] if isinstance(x, dict) else 0)

        # 4. 조건 체크 (10일선 하회 + RVOL만)
        result_df['Condition_1_Trend_Breakdown'] = result_df['Close'] < result_df['MA10']
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
                return "10일선 하회 + 거래량 폭증 확인"
            else:
                # HOLD 이유 세부 분석
                reasons = []
                if not row['Condition_1_Trend_Breakdown']:
                    reasons.append("10일선 위")
                if not row['Condition_2_Volume_Confirmation']:
                    reasons.append("거래량 부족")

                if row['Condition_1_Trend_Breakdown'] and not row['Condition_2_Volume_Confirmation']:
                    return "10일선 하회, 거래량 부족"

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
            'MA10': df['MA10'],
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


def load_data_from_csv(filepath: str, date_column: Optional[str] = None, convert_daily: bool = True) -> pd.DataFrame:
    """
    CSV 또는 XLSX 파일에서 OHLCV 데이터 로드

    ====================================================================
    중요: 시간봉 → 일봉 자동 변환
    ====================================================================
    블룸버그 등에서 다운로드한 시간봉 데이터는 자동으로 일봉으로 변환됩니다.
    - 시간봉 감지 기준: 같은 날짜에 여러 행이 있는 경우
    - 변환 방법: OHLCV 리샘플링 (convert_to_daily 함수)

    Parameters:
    -----------
    filepath : str
        CSV 또는 XLSX 파일 경로
    date_column : str, optional
        날짜 컬럼명 (인덱스로 설정)
    convert_daily : bool
        시간봉을 일봉으로 자동 변환 (기본값: True)

    Returns:
    --------
    pd.DataFrame : OHLCV 데이터 (일봉)
    """
    # ====================================================================
    # [1단계] 파일 로드
    # ====================================================================
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    # ====================================================================
    # [2단계] 시간봉 감지 및 일봉 변환
    # ====================================================================
    # Date 컬럼이 있고, convert_daily=True인 경우 시간봉 여부 확인
    if convert_daily and 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

        # 같은 날짜에 여러 행이 있는지 확인 (시간봉 감지)
        df['Date_Only'] = df['Date'].dt.date
        date_counts = df['Date_Only'].value_counts()

        if (date_counts > 1).any():
            # 시간봉 데이터 감지!
            print(f"\n[시간봉 감지] 같은 날짜에 최대 {date_counts.max()}개 행")
            print(f"   >> 일봉으로 자동 변환합니다...")

            # Date_Only 컬럼 제거
            df.drop('Date_Only', axis=1, inplace=True)

            # 일봉으로 변환
            df = convert_to_daily(df)
        else:
            # 이미 일봉 데이터
            print(f"\n[일봉 확인] {len(df)}일 데이터")
            df.drop('Date_Only', axis=1, inplace=True)

    # ====================================================================
    # [3단계] 날짜 인덱스 설정 (선택적)
    # ====================================================================
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
