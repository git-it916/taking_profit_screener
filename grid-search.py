"""
MA 기간 최적화 그리드 서치
=========================

10일선부터 19일선까지 각 MA 기간에 대해:
- 거래량 폭증 신호 발생 시점을 찾고
- 이후 5영업일 이내에 주가가 상승한 비율(승률)을 계산

py -3.12 grid-search.py
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime as dt
from tabulate import tabulate

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 현재 스크립트의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.bloomberg import download_bloomberg_data, get_multiple_security_names


def calculate_sma_with_period(df: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
    """
    지정된 기간의 이동평균선 계산 (당일 제외)

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV 데이터
    period : int
        MA 기간 (10, 11, 12, ..., 19)
    column : str
        계산할 컬럼 (기본값: 'Close')

    Returns:
    --------
    pd.Series : MA 값
    """
    # 당일 제외: shift(1)로 한 행씩 밀어서 계산
    return df[column].shift(1).rolling(window=period).mean()


def calculate_rvol_with_period(df: pd.DataFrame, period: int, volume_column: str = 'Volume') -> pd.Series:
    """
    지정된 기간의 RVOL 계산 (당일 제외)

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV 데이터
    period : int
        RVOL 기간 (10, 11, 12, ..., 19)
    volume_column : str
        거래량 컬럼 (기본값: 'Volume')

    Returns:
    --------
    pd.Series : RVOL 값
    """
    # 과거 N일 평균 거래량 (당일 제외)
    avg_volume = df[volume_column].shift(1).rolling(window=period).mean()

    # RVOL = 당일 거래량 / 평균 거래량
    rvol = df[volume_column] / avg_volume
    return rvol


def find_ma_breakout_dates(df: pd.DataFrame, ma_period: int, rvol_threshold: float = 1.5) -> list:
    """
    이동평균선 돌파 날짜 찾기 (거래량 조건 포함)

    돌파 조건:
    - 전일: 종가 <= MA선 (MA선 아래)
    - 당일: 종가 > MA선 (MA선 돌파)
    - 당일: RVOL >= 1.5배 (거래량 폭증)

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV 데이터
    ma_period : int
        MA 기간
    rvol_threshold : float
        RVOL 임계값 (기본값: 1.5배)

    Returns:
    --------
    list : 돌파 발생 인덱스 리스트
    """
    # MA 계산 (당일 제외)
    df['ma'] = calculate_sma_with_period(df, ma_period)

    # RVOL 계산 (당일 제외)
    df['rvol'] = calculate_rvol_with_period(df, ma_period)

    # 전일 종가와 MA 비교
    df['prev_close'] = df['Close'].shift(1)
    df['prev_ma'] = df['ma'].shift(1)

    # 돌파 조건:
    # 1. 전일 종가 <= 전일 MA (MA선 아래 또는 같음)
    # 2. 당일 종가 > 당일 MA (MA선 돌파)
    # 3. 당일 RVOL >= 1.5배 (거래량 폭증)
    # 4. MA와 이전 데이터가 유효함 (NaN 아님)
    breakout_condition = (
        (df['prev_close'] <= df['prev_ma']) &  # 전일은 MA 아래
        (df['Close'] > df['ma']) &              # 당일은 MA 돌파
        (df['rvol'] >= rvol_threshold) &        # 거래량 폭증 (RVOL >= 1.5)
        (df['ma'].notna()) &                    # MA 값 유효
        (df['prev_ma'].notna()) &               # 전일 MA 값 유효
        (df['rvol'].notna())                    # RVOL 값 유효
    )

    breakout_signals = df[breakout_condition].copy()

    return breakout_signals.index.tolist()


def calculate_win_rate(df: pd.DataFrame, signal_indices: list, lookforward_days: int = 5,
                       target_gain_pct: float = 10.0, target_loss_pct: float = 5.0) -> dict:
    """
    돌파 후 N일 이내 승률 계산 (상승 vs 하락)

    승률 정의:
    - 승리: 5일 내 10% 이상 상승
    - 패배: 5일 내 5% 이상 하락
    - 무승부: -5% ~ +10% 구간 (승률 계산 제외)
    - 승률 = 승리 / (승리 + 패배) × 100

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV 데이터
    signal_indices : list
        돌파 발생 인덱스 리스트
    lookforward_days : int
        미래 관찰 기간 (기본값: 5영업일)
    target_gain_pct : float
        목표 수익률 (기본값: 10.0%)
    target_loss_pct : float
        손실 기준 (기본값: 5.0%)

    Returns:
    --------
    dict : 승률 통계
    """
    if len(signal_indices) == 0:
        return {
            'win_rate': 0.0,
            'achievement_rate': 0.0,
            'total_breakouts': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'avg_max_gain': 0.0,
            'avg_max_loss': 0.0,
            'max_gain_overall': 0.0,
            'max_loss_overall': 0.0
        }

    wins = 0
    losses = 0
    draws = 0
    all_max_gains = []  # 모든 케이스의 최대 수익률
    all_max_losses = []  # 모든 케이스의 최대 손실률

    for signal_idx in signal_indices:
        # 돌파 시점의 종가
        breakout_price = df.loc[signal_idx, 'Close']

        # 돌파 이후 N일간의 데이터 가져오기
        future_data = df.loc[signal_idx:].iloc[1:lookforward_days+1]  # 다음날부터 N일

        if len(future_data) == 0:
            # 미래 데이터가 없으면 제외 (최근 돌파)
            continue

        # N일 이내 최고가/최저가
        max_price = future_data['High'].max()
        min_price = future_data['Low'].min()

        # 최대 수익률/손실률 계산
        max_gain_pct = ((max_price - breakout_price) / breakout_price) * 100
        max_loss_pct = ((min_price - breakout_price) / breakout_price) * 100

        all_max_gains.append(max_gain_pct)
        all_max_losses.append(max_loss_pct)

        # 승/패/무 판정
        if max_gain_pct >= target_gain_pct:
            # 10% 이상 상승 → 승리
            wins += 1
        elif max_loss_pct <= -target_loss_pct:
            # 5% 이상 하락 → 패배
            losses += 1
        else:
            # -5% ~ +10% 구간 → 무승부
            draws += 1

    total = wins + losses + draws

    # 승률 = 승리 / (승리 + 패배) × 100 (무승부 제외)
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0

    # 목표달성률 = 승리 / 총 돌파 횟수 × 100 (무승부 포함)
    achievement_rate = (wins / total * 100) if total > 0 else 0.0

    avg_max_gain = np.mean(all_max_gains) if len(all_max_gains) > 0 else 0.0
    avg_max_loss = np.mean(all_max_losses) if len(all_max_losses) > 0 else 0.0
    max_gain_overall = max(all_max_gains) if len(all_max_gains) > 0 else 0.0
    max_loss_overall = min(all_max_losses) if len(all_max_losses) > 0 else 0.0

    return {
        'win_rate': win_rate,
        'achievement_rate': achievement_rate,
        'total_breakouts': total,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'avg_max_gain': avg_max_gain,
        'avg_max_loss': avg_max_loss,
        'max_gain_overall': max_gain_overall,
        'max_loss_overall': max_loss_overall
    }


def grid_search_ma_period(ticker: str, period: str = '3Y', ma_range: tuple = (10, 21),
                         lookforward_days: int = 5, target_gain_pct: float = 10.0,
                         rvol_threshold: float = 1.5) -> pd.DataFrame:
    """
    MA 기간 그리드 서치 (2단계 분석, 거래량 조건 포함)

    Step 1: 각 MA선(10~20일) 돌파 날짜를 모두 찾기 (RVOL >= 1.5 조건 포함)
    Step 2: 돌파 후 5영업일 내 10% 이상 상승 승률 계산

    Parameters:
    -----------
    ticker : str
        Bloomberg 티커
    period : str
        데이터 기간 (기본값: '3Y' - 3년 데이터 권장)
    ma_range : tuple
        MA 기간 범위 (시작, 끝) - 끝은 포함 안 됨 (기본값: (10, 21) → 10~20일)
    lookforward_days : int
        미래 관찰 기간 (기본값: 5영업일)
    target_gain_pct : float
        목표 수익률 (기본값: 10.0%)
    rvol_threshold : float
        RVOL 임계값 (기본값: 1.5배)

    Returns:
    --------
    pd.DataFrame : 그리드 서치 결과
    """
    try:
        # ================================================================
        # STEP 1: 데이터 다운로드
        # ================================================================
        df = download_bloomberg_data(ticker, period=period, verbose=False)

        if df is None or len(df) == 0:
            return None

        # ================================================================
        # STEP 1-1: 당일 데이터 제외 (일봉 미완성 가능성)
        # ================================================================
        from datetime import datetime as dt
        today = dt.now().date()

        # Date 컬럼을 datetime으로 변환
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['date_only'] = df['Date'].dt.date

            # 당일 데이터가 있으면 제외 (일봉 미완성)
            if (df['date_only'] == today).any():
                df = df[df['date_only'] != today].copy()

            # 임시 컬럼 제거
            df = df.drop(columns=['date_only'])

        if len(df) == 0:
            return None

        # ================================================================
        # STEP 2: MA별 돌파 날짜 찾기 (RVOL 조건 포함)
        # ================================================================
        results = []

        for ma_period in range(ma_range[0], ma_range[1]):
            # 이동평균선 돌파 날짜 찾기 (RVOL >= 1.5 조건 포함)
            breakout_indices = find_ma_breakout_dates(df.copy(), ma_period, rvol_threshold)

            # 승률 계산 (돌파 후 N일 내 목표 수익률 달성)
            win_stats = calculate_win_rate(
                df,
                breakout_indices,
                lookforward_days,
                target_gain_pct
            )

            results.append({
                'MA기간': ma_period,
                '돌파횟수': win_stats['total_breakouts'],
                '상승(10%↑)': win_stats['wins'],
                '하락(5%↓)': win_stats['losses'],
                '무승부': win_stats['draws'],
                '승률(%)': win_stats['win_rate'],
                '목표달성률(%)': win_stats['achievement_rate'],
                '평균최대수익률(%)': win_stats['avg_max_gain'],
                '평균최대손실률(%)': win_stats['avg_max_loss'],
                '최고수익률(%)': win_stats['max_gain_overall'],
                '최대손실률(%)': win_stats['max_loss_overall']
            })

        return pd.DataFrame(results)

    except Exception as e:
        return None


def main():
    """메인 함수"""
    print("="*80)
    print("MA 기간 최적화 그리드 서치 (2단계 분석, 거래량 조건 포함)")
    print("="*80)
    print("\n이 프로그램은 10일선부터 20일선까지 각 MA 기간에 대해:")
    print("  [Step 1] 3년 데이터에서 MA 돌파 날짜를 모두 찾기 (RVOL >= 1.5 조건 포함)")
    print("  [Step 2] 돌파 후 5영업일 내 10% 이상 상승 승률 계산")
    print("  [결과] 최적의 MA 기간 추천")
    print("\n✅ 돌파 조건: MA 돌파 + 거래량 폭증 (RVOL >= 1.5배)")

    print("\n⚠️  주의사항:")
    print("  1. Bloomberg Terminal이 실행 중이어야 합니다")
    print("  2. Bloomberg에 로그인되어 있어야 합니다")
    print("  3. 3년 데이터로 분석합니다 (돌파 케이스 충분히 확보)")
    print("  4. 여러 종목 입력 시 한 종목씩 순차 처리됩니다 (시간 소요)")

    # ====================================================================
    # 티커 입력 (여러 종목 가능, 순차 처리)
    # ====================================================================
    print("\n" + "="*80)
    print("Bloomberg 티커를 입력하세요 (쉼표로 구분)")
    print("="*80)
    print("\n티커 형식:")
    print("  - 한국 주식: 005930 KS (삼성전자), 000660 KS (SK하이닉스)")
    print("  - 미국 주식: AAPL US (애플), MSFT US (마이크로소프트)")
    print("  - 예시: 005930 KS, 000660 KS, AAPL US")
    print("\n⚠️  주의: 여러 종목 입력 시 한 종목씩 순차적으로 처리됩니다")

    try:
        user_input = input("\n티커 입력: ").strip()

        if not user_input:
            print("\n[에러] 티커를 입력해주세요")
            return

        # 티커 리스트 파싱
        tickers = [t.strip() for t in user_input.split(',')]
        print(f"\n입력된 티커: {len(tickers)}개")

        # ================================================================
        # 종목명 조회
        # ================================================================
        print("\n[종목명 조회 중...]")
        try:
            ticker_names = get_multiple_security_names(tickers)
            print("✓ 종목명 조회 완료")
            print("\n종목 정보:")
            for ticker in tickers:
                name = ticker_names.get(ticker, ticker)
                print(f"  - {ticker}: {name}")
        except Exception as e:
            print(f"⚠️  종목명 조회 실패 (티커로 표시됩니다): {e}")
            ticker_names = {ticker: ticker for ticker in tickers}

        # ================================================================
        # 데이터 기간 선택
        # ================================================================
        print("\n데이터 기간을 선택하세요:")
        print("  1: 3년 (권장 - 충분한 돌파 케이스)")
        print("  2: 2년")
        print("  3: 1년")
        print("  4: 6개월")

        period_choice = input("\n선택 (엔터=3년): ").strip()

        if period_choice == '2':
            period = '2Y'
        elif period_choice == '3':
            period = '1Y'
        elif period_choice == '4':
            period = '6M'
        else:
            period = '3Y'

        # ================================================================
        # 그리드 서치 파라미터 설정
        # ================================================================
        print("\n" + "="*80)
        print("그리드 서치 파라미터")
        print("="*80)

        # MA 범위
        ma_start = 10
        ma_end = 21  # 20까지 (21은 포함 안 됨)
        print(f"\nMA 기간 범위: {ma_start}일 ~ {ma_end-1}일")

        # 미래 관찰 기간
        lookforward_days = 5
        print(f"미래 관찰 기간: {lookforward_days}영업일")

        # 목표 수익률
        target_gain_pct = 10.0
        print(f"목표 수익률: {target_gain_pct}% 이상")

        # RVOL 임계값
        rvol_threshold = 1.5
        print(f"RVOL 임계값: {rvol_threshold}배 이상")

        # 손실 기준
        target_loss_pct = 5.0
        print(f"손실 기준: {target_loss_pct}% 이상 하락")

        print(f"\n승률 = 상승(10%↑) / (상승 + 하락) × 100 (무승부 제외)")
        print(f"  - 승리: {lookforward_days}일 내 {target_gain_pct}% 이상 상승")
        print(f"  - 패배: {lookforward_days}일 내 {target_loss_pct}% 이상 하락")
        print(f"  - 무승부: -{target_loss_pct}% ~ +{target_gain_pct}% 구간")
        print(f"\n돌파 조건 = MA 돌파 + RVOL >= {rvol_threshold}배")

        # ================================================================
        # 그리드 서치 실행 (한 종목씩 순차 처리)
        # ================================================================
        print("\n" + "="*80)
        print(f"총 {len(tickers)}개 종목 그리드 서치 시작 (순차 처리)")
        print("="*80)
        print()

        start_time = dt.now()
        all_results = {}
        failed_tickers = []

        for i, ticker in enumerate(tickers, 1):
            security_name = ticker_names.get(ticker, ticker)

            # 진행률 계산
            progress = i / len(tickers) * 100
            bar_length = 60
            filled = int(bar_length * i // len(tickers))
            bar = '█' * filled + '░' * (bar_length - filled)

            # 한 줄로 진행 상황 표시
            print(f"\r[{bar}] {i}/{len(tickers)} ({progress:.1f}%) | "
                  f"성공: {len(all_results)} | 실패: {len(failed_tickers)} | "
                  f"현재: {security_name[:30]}", end='', flush=True)

            result_df = grid_search_ma_period(
                ticker=ticker,
                period=period,
                ma_range=(ma_start, ma_end),
                lookforward_days=lookforward_days,
                target_gain_pct=target_gain_pct,
                rvol_threshold=rvol_threshold
            )

            if result_df is not None and len(result_df) > 0:
                all_results[ticker] = result_df
            else:
                failed_tickers.append(ticker)

        # 줄바꿈
        print()

        total_time = dt.now() - start_time
        print(f"\n✓ 그리드 서치 완료 - 소요시간: {str(total_time).split('.')[0]}")

        if not all_results:
            print("\n[에러] 분석 결과가 없습니다")
            return

        # ================================================================
        # 결과 출력
        # ================================================================
        print("\n" + "="*80)
        print("그리드 서치 결과")
        print("="*80)

        for ticker, result_df in all_results.items():
            security_name = ticker_names.get(ticker, ticker)

            print(f"\n\n{'='*80}")
            print(f"{security_name} ({ticker})")
            print('='*80)

            # 테이블 출력
            print("\n" + tabulate(result_df, headers='keys', tablefmt='simple',
                                 showindex=False, floatfmt='.2f'))

            # 최고 승률 찾기
            if len(result_df) > 0 and result_df['돌파횟수'].sum() > 0:
                # 돌파가 5개 이상인 MA 기간만 고려 (신뢰도 확보)
                valid_results = result_df[result_df['돌파횟수'] >= 5]

                if len(valid_results) > 0:
                    best_row = valid_results.loc[valid_results['승률(%)'].idxmax()]
                    print(f"\n🏆 최적 MA 기간: {int(best_row['MA기간'])}일")
                    print(f"   - 승률: {best_row['승률(%)']:.2f}% (상승 vs 하락)")
                    print(f"   - 돌파 횟수: {int(best_row['돌파횟수'])}개")
                    print(f"   - 상승(10%↑): {int(best_row['상승(10%↑)'])}개")
                    print(f"   - 하락(5%↓): {int(best_row['하락(5%↓)'])}개")
                    print(f"   - 무승부: {int(best_row['무승부'])}개")
                    print(f"   - 목표달성률: {best_row['목표달성률(%)']:.2f}%")
                    print(f"   - 평균 최대 수익률: {best_row['평균최대수익률(%)']:+.2f}%")
                    print(f"   - 평균 최대 손실률: {best_row['평균최대손실률(%)']:+.2f}%")
                else:
                    print("\n⚠️  돌파가 5개 미만인 MA 기간들만 존재합니다 (더 긴 데이터 기간 필요)")
            else:
                print("\n⚠️  분석할 돌파 케이스가 없습니다")

        # ================================================================
        # 종합 요약
        # ================================================================
        print("\n" + "="*80)
        print("종합 요약")
        print("="*80)

        # 각 종목의 최적 MA 기간 요약
        summary_data = []
        for ticker, result_df in all_results.items():
            security_name = ticker_names.get(ticker, ticker)

            if len(result_df) > 0 and result_df['돌파횟수'].sum() > 0:
                valid_results = result_df[result_df['돌파횟수'] >= 5]

                if len(valid_results) > 0:
                    best_row = valid_results.loc[valid_results['승률(%)'].idxmax()]
                    summary_data.append({
                        '종목': security_name,
                        '최적MA': f"{int(best_row['MA기간'])}일",
                        '승률': f"{best_row['승률(%)']:.1f}%",
                        '목표달성률': f"{best_row['목표달성률(%)']:.1f}%",
                        '돌파횟수': int(best_row['돌파횟수']),
                        '상승': int(best_row['상승(10%↑)']),
                        '하락': int(best_row['하락(5%↓)']),
                        '평균수익률': f"{best_row['평균최대수익률(%)']:+.2f}%",
                        '평균손실률': f"{best_row['평균최대손실률(%)']:+.2f}%"
                    })

        if summary_data:
            print("\n" + tabulate(summary_data, headers='keys', tablefmt='simple', showindex=False))

            # 가장 많이 나온 MA 기간 찾기
            ma_periods = [int(row['최적MA'].replace('일', '')) for row in summary_data]
            from collections import Counter
            ma_counter = Counter(ma_periods)
            most_common_ma = ma_counter.most_common(1)[0]

            print(f"\n📊 가장 많이 선택된 MA 기간: {most_common_ma[0]}일 ({most_common_ma[1]}개 종목)")

            # ================================================================
            # 시각화 생성
            # ================================================================
            print("\n" + "="*80)
            viz_choice = input("\n분석 결과를 시각화하시겠습니까? (y/n): ").strip().lower()

            if viz_choice == 'y':
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib
                    matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트
                    matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호

                    # 1. MA 기간 분포 히스토그램
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    fig.suptitle('MA 기간 최적화 그리드 서치 결과', fontsize=16, fontweight='bold')

                    # 1-1. MA 기간 분포
                    axes[0, 0].hist(ma_periods, bins=range(10, 22), edgecolor='black', alpha=0.7)
                    axes[0, 0].set_xlabel('MA 기간 (일)', fontsize=12)
                    axes[0, 0].set_ylabel('종목 수', fontsize=12)
                    axes[0, 0].set_title('최적 MA 기간 분포', fontsize=14)
                    axes[0, 0].grid(True, alpha=0.3)
                    axes[0, 0].axvline(most_common_ma[0], color='red', linestyle='--',
                                      label=f'최빈값: {most_common_ma[0]}일')
                    axes[0, 0].legend()

                    # 1-2. 승률 분포
                    win_rates = [float(row['승률'].replace('%', '')) for row in summary_data]
                    axes[0, 1].hist(win_rates, bins=20, edgecolor='black', alpha=0.7, color='green')
                    axes[0, 1].set_xlabel('승률 (%)', fontsize=12)
                    axes[0, 1].set_ylabel('종목 수', fontsize=12)
                    axes[0, 1].set_title('승률 분포', fontsize=14)
                    axes[0, 1].grid(True, alpha=0.3)
                    axes[0, 1].axvline(sum(win_rates)/len(win_rates), color='red',
                                      linestyle='--', label=f'평균: {sum(win_rates)/len(win_rates):.1f}%')
                    axes[0, 1].legend()

                    # 1-3. MA 기간 vs 승률 산점도
                    axes[1, 0].scatter(ma_periods, win_rates, alpha=0.6, s=100)
                    axes[1, 0].set_xlabel('MA 기간 (일)', fontsize=12)
                    axes[1, 0].set_ylabel('승률 (%)', fontsize=12)
                    axes[1, 0].set_title('MA 기간 vs 승률', fontsize=14)
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].axhline(50, color='red', linestyle='--', alpha=0.5, label='50% 기준선')
                    axes[1, 0].legend()

                    # 1-4. 상위 10개 종목 승률 바 차트
                    sorted_data = sorted(summary_data,
                                       key=lambda x: float(x['승률'].replace('%', '')),
                                       reverse=True)[:10]
                    stock_names = [row['종목'][:15] + '...' if len(row['종목']) > 15
                                  else row['종목'] for row in sorted_data]
                    stock_win_rates = [float(row['승률'].replace('%', '')) for row in sorted_data]

                    bars = axes[1, 1].barh(stock_names, stock_win_rates, color='skyblue', edgecolor='black')
                    axes[1, 1].set_xlabel('승률 (%)', fontsize=12)
                    axes[1, 1].set_title('상위 10개 종목 승률', fontsize=14)
                    axes[1, 1].grid(True, alpha=0.3, axis='x')
                    axes[1, 1].invert_yaxis()

                    # 바 위에 값 표시
                    for i, (bar, wr) in enumerate(zip(bars, stock_win_rates)):
                        axes[1, 1].text(wr + 1, i, f'{wr:.1f}%',
                                       va='center', fontsize=9)

                    plt.tight_layout()

                    # grid_search_database 폴더에 저장
                    import os
                    os.makedirs('grid_search_database', exist_ok=True)

                    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
                    viz_filename = f"grid_search_database/grid_search_visualization_{timestamp}.png"
                    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
                    print(f"✓ 시각화 저장 완료: {viz_filename}")

                    # 그래프 표시
                    plt.show()

                except ImportError:
                    print("✗ matplotlib이 설치되지 않았습니다")
                    print("  설치: pip install matplotlib")
                except Exception as e:
                    print(f"✗ 시각화 생성 실패: {e}")
                    import traceback
                    traceback.print_exc()

        else:
            print("\n⚠️  충분한 돌파 케이스가 있는 종목이 없습니다 (더 긴 데이터 기간 권장)")

        # ================================================================
        # CSV 저장 (전체 800개 row를 하나의 파일로)
        # ================================================================
        print("\n" + "="*80)
        save_choice = input("\n결과를 CSV로 저장하시겠습니까? (y/n): ").strip().lower()

        if save_choice == 'y':
            import os
            os.makedirs('grid_search_database', exist_ok=True)

            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")

            # ============================================================
            # [방법 1] 전체 데이터를 하나의 CSV로 통합 (800개 row)
            # ============================================================
            all_rows = []
            for ticker, result_df in all_results.items():
                security_name = ticker_names.get(ticker, ticker)

                # 각 MA 기간별 결과에 종목 정보 추가
                for _, row in result_df.iterrows():
                    all_rows.append({
                        '티커': ticker,
                        '종목명': security_name,
                        'MA기간': int(row['MA기간']),
                        '돌파횟수': int(row['돌파횟수']),
                        '상승(10%↑)': int(row['상승(10%↑)']),
                        '하락(5%↓)': int(row['하락(5%↓)']),
                        '무승부': int(row['무승부']),
                        '승률(%)': row['승률(%)'],
                        '목표달성률(%)': row['목표달성률(%)'],
                        '평균최대수익률(%)': row['평균최대수익률(%)'],
                        '평균최대손실률(%)': row['평균최대손실률(%)'],
                        '최고수익률(%)': row['최고수익률(%)'],
                        '최대손실률(%)': row['최대손실률(%)']
                    })

            # 하나의 DataFrame으로 통합
            combined_df = pd.DataFrame(all_rows)

            # 파일명: grid_search_all_results_YYYYMMDD_HHMMSS.csv
            combined_filename = f"grid_search_database/grid_search_all_results_{timestamp}.csv"
            combined_df.to_csv(combined_filename, index=False, encoding='utf-8-sig')
            print(f"✓ 전체 결과 저장 완료: {combined_filename}")
            print(f"  총 {len(combined_df)}개 row (종목 {len(all_results)}개 × MA {ma_end - ma_start}개)")

            # ============================================================
            # [방법 2] 종목별 개별 CSV도 저장 (선택)
            # ============================================================
            individual_choice = input("\n종목별 개별 CSV도 저장하시겠습니까? (y/n): ").strip().lower()

            if individual_choice == 'y':
                for ticker, result_df in all_results.items():
                    security_name = ticker_names.get(ticker, ticker).replace('/', '_')
                    filename = f"grid_search_database/grid_search_{ticker.replace(' ', '_')}_{timestamp}.csv"
                    result_df.to_csv(filename, index=False, encoding='utf-8-sig')
                    print(f"✓ 저장 완료: {filename}")

        if failed_tickers:
            print(f"\n⚠️  실패한 종목: {', '.join(failed_tickers)}")

    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n[에러] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
