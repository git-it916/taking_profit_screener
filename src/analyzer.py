"""
종목별 상세 분석 모듈

RVOL, 20일선 근접도, 신호 강도 등을 분석합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .screener import ExitSignalScreener


class StockAnalyzer:
    """종목 상세 분석 클래스"""

    def __init__(self, ma_period: int = 20, rvol_period: int = 20):
        """
        Parameters:
        -----------
        ma_period : int
            이동평균선 기간 (default: 20)
        rvol_period : int
            RVOL 계산 기간 (default: 20)
        """
        self.ma_period = ma_period
        self.rvol_period = rvol_period
        self.screener = ExitSignalScreener(ma_period=ma_period, rvol_period=rvol_period)

    def analyze_latest(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> Dict:
        """
        최신 데이터 분석 (가장 최근 일자)

        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV 데이터
        ticker : str
            종목명

        Returns:
        --------
        Dict : 분석 결과
        """
        # 필터 적용
        filtered_data = self.screener.apply_filters(df)

        # 최신 데이터 추출
        latest = filtered_data.iloc[-1]

        # 20일선 근접도 계산
        ma_distance_percent = ((latest['Close'] - latest['MA20']) / latest['MA20']) * 100
        ma_distance_points = latest['Close'] - latest['MA20']

        # RVOL 강도 분류
        rvol_strength = self._classify_rvol(latest['RVOL'])

        # RVOL 방향 (상승 vs 하락)
        if len(filtered_data) >= 2:
            prev_rvol = filtered_data.iloc[-2]['RVOL']
            rvol_direction = "상승" if latest['RVOL'] > prev_rvol else "하락"
            rvol_change = latest['RVOL'] - prev_rvol
        else:
            rvol_direction = "정보 부족"
            rvol_change = 0

        # 20일선 돌파/이탈 상태
        ma_status = self._classify_ma_status(ma_distance_percent)

        # 조건 충족 여부 체크
        condition_1 = latest['Condition_1_Trend_Breakdown']
        condition_2 = latest['Condition_2_Rejection_Pattern']
        condition_3 = latest['Condition_3_Volume_Confirmation']

        # 신호 분류
        signal_category = self._classify_signal(condition_1, condition_2, condition_3)

        # 결과 정리
        result = {
            'ticker': ticker,
            'date': latest.name if isinstance(filtered_data.index, pd.DatetimeIndex) else "N/A",
            'close_price': latest['Close'],

            # 20일선 분석
            'ma20': latest['MA20'],
            'ma_distance_percent': ma_distance_percent,
            'ma_distance_points': ma_distance_points,
            'ma_status': ma_status,
            'condition_1_trend_breakdown': condition_1,

            # RVOL 분석
            'rvol': latest['RVOL'],
            'rvol_strength': rvol_strength,
            'rvol_direction': rvol_direction,
            'rvol_change': rvol_change,
            'condition_3_volume_confirmation': condition_3,

            # 윅 패턴 분석
            'wick_ratio': latest['Wick_Ratio'],
            'condition_2_rejection_pattern': condition_2,

            # 종합 신호
            'signal': latest['Signal'],
            'signal_category': signal_category,
            'reasoning': latest['Reasoning']
        }

        return result

    def _classify_rvol(self, rvol: float) -> str:
        """RVOL 강도 분류"""
        if pd.isna(rvol):
            return "정보 없음"
        elif rvol >= 3.0:
            return "매우 강함 (3배 이상)"
        elif rvol >= 2.5:
            return "강함 (2.5~3배)"
        elif rvol >= 2.0:
            return "보통 (2~2.5배)"
        elif rvol >= 1.5:
            return "약함 (1.5~2배)"
        else:
            return "매우 약함 (1.5배 미만)"

    def _classify_ma_status(self, distance_percent: float) -> str:
        """20일선 근접도 분류"""
        if pd.isna(distance_percent):
            return "정보 없음"
        elif distance_percent > 5:
            return "20일선 위 (멀리)"
        elif distance_percent > 1:
            return "20일선 위 (근접)"
        elif distance_percent > -1:
            return "20일선 근처"
        elif distance_percent > -5:
            return "20일선 아래 (근접)"
        else:
            return "20일선 아래 (멀리)"

    def _classify_signal(self, cond1: bool, cond2: bool, cond3: bool) -> str:
        """신호 분류"""
        if cond1 and cond2 and cond3:
            return "강력 매도 (3개 조건 모두 충족)"
        elif cond1 and cond2:
            return "주의 (추세하락 + 윅패턴, 거래량 부족)"
        elif cond1 and cond3:
            return "관찰 (추세하락 + 거래량, 윅패턴 없음)"
        elif cond2 and cond3:
            return "거짓 신호 가능 (윅패턴 + 거래량, 추세 양호)"
        elif cond1:
            return "추세 하락만"
        elif cond2:
            return "윅 패턴만"
        elif cond3:
            return "거래량 폭증만"
        else:
            return "정상 (신호 없음)"

    def format_analysis_report(self, result: Dict) -> str:
        """분석 결과를 보기 좋게 포맷팅"""
        report = []
        report.append("=" * 80)
        report.append(f"종목 분석 리포트: {result['ticker']}")
        report.append("=" * 80)
        report.append(f"\n날짜: {result['date']}")
        report.append(f"현재가: {result['close_price']:.2f}")

        report.append("\n" + "-" * 80)
        report.append("[1] 20일 이동평균선 분석")
        report.append("-" * 80)
        report.append(f"  - 20일선: {result['ma20']:.2f}")
        report.append(f"  - 20일선 대비 거리: {result['ma_distance_points']:+.2f}원 ({result['ma_distance_percent']:+.2f}%)")
        report.append(f"  - 상태: {result['ma_status']}")
        report.append(f"  - 조건 1 충족 여부: {'[O] 충족 (추세 하락)' if result['condition_1_trend_breakdown'] else '[X] 미충족 (추세 양호)'}")

        report.append("\n" + "-" * 80)
        report.append("[2] RVOL (상대 거래량) 분석")
        report.append("-" * 80)
        report.append(f"  - RVOL: {result['rvol']:.2f}배")
        report.append(f"  - 강도: {result['rvol_strength']}")
        report.append(f"  - 방향: {result['rvol_direction']} (전일 대비 {result['rvol_change']:+.2f})")
        report.append(f"  - 조건 3 충족 여부: {'[O] 충족 (거래량 확인)' if result['condition_3_volume_confirmation'] else '[X] 미충족 (거래량 부족)'}")

        report.append("\n" + "-" * 80)
        report.append("[3] 윅 패턴 분석")
        report.append("-" * 80)
        report.append(f"  - 윅 비율: {result['wick_ratio']:.2f} ({result['wick_ratio']*100:.1f}%)")
        report.append(f"  - 조건 2 충족 여부: {'[O] 충족 (거부 패턴)' if result['condition_2_rejection_pattern'] else '[X] 미충족 (윅 부족)'}")

        report.append("\n" + "=" * 80)
        report.append("[종합 판정]")
        report.append("=" * 80)
        report.append(f"  신호: {result['signal']}")
        report.append(f"  분류: {result['signal_category']}")
        report.append(f"  근거: {result['reasoning']}")
        report.append("=" * 80)

        return "\n".join(report)


def analyze_stock_from_csv(csv_filename: str,
                           base_path: str = r"C:\Users\10845\OneDrive - 이지스자산운용\문서",
                           date_column: str = "Date") -> Dict:
    """
    CSV 또는 XLSX 파일에서 종목 데이터를 읽어 분석

    Parameters:
    -----------
    csv_filename : str
        파일명 (예: "sm.xlsx", "samsung.xlsx", "sm.csv")
    base_path : str
        파일이 있는 폴더 경로
    date_column : str
        날짜 컬럼명 (default: "Date")

    Returns:
    --------
    Dict : 분석 결과
    """
    import os
    from .screener import load_data_from_csv

    # 파일 경로 생성
    filepath = os.path.join(base_path, csv_filename)

    # 종목명 추출 (파일명에서 확장자 제거)
    ticker = csv_filename.replace('.xlsx', '').replace('.xls', '').replace('.csv', '').upper()

    # 파일 로드 (CSV 또는 XLSX)
    df = load_data_from_csv(filepath, date_column=date_column)

    # 분석 실행
    analyzer = StockAnalyzer(ma_period=20, rvol_period=20)
    result = analyzer.analyze_latest(df, ticker=ticker)

    return result


def batch_analyze_stocks(csv_filenames: list,
                         base_path: str = r"C:\Users\10845\OneDrive - 이지스자산운용\문서",
                         date_column: str = "Date") -> pd.DataFrame:
    """
    여러 CSV 또는 XLSX 파일을 일괄 분석

    Parameters:
    -----------
    csv_filenames : list
        파일명 리스트 (예: ["sm.xlsx", "samsung.xlsx", "apple.csv"])
    base_path : str
        파일이 있는 폴더 경로
    date_column : str
        날짜 컬럼명

    Returns:
    --------
    pd.DataFrame : 전체 종목 분석 결과
    """
    results = []

    for filename in csv_filenames:
        try:
            result = analyze_stock_from_csv(filename, base_path, date_column)
            results.append(result)
            print(f"[OK] {filename} 분석 완료")
        except Exception as e:
            print(f"[X] {filename} 분석 실패: {e}")

    # DataFrame으로 변환
    df_results = pd.DataFrame(results)

    return df_results
