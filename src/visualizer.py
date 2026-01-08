"""
종목 분석 결과 시각화 모듈

추세별/조건별 분류를 히트맵과 차트로 시각화합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


def setup_korean_font():
    """한글 폰트 설정"""
    # Windows 한글 폰트 설정
    try:
        # 맑은 고딕 폰트 사용
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    except:
        # 폰트 설정 실패시 기본 폰트 사용
        pass


def create_trend_heatmap(results: List[Dict], save_path: str = None) -> str:
    """
    추세별/조건별 분류 히트맵 생성

    Parameters:
    -----------
    results : List[Dict]
        분석 결과 리스트
    save_path : str, optional
        저장 경로 (지정하지 않으면 자동 생성)

    Returns:
    --------
    str : 저장된 파일 경로
    """
    setup_korean_font()

    df = pd.DataFrame(results)

    # 저장 경로 생성
    if save_path is None:
        # database 폴더 생성
        db_folder = Path("database")
        db_folder.mkdir(exist_ok=True)

        # 파일명: 날짜_시간_히트맵.png
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = db_folder / f"{timestamp}_heatmap.png"

    # ====================================================================
    # 히트맵 데이터 준비
    # ====================================================================

    # 추세 방향별 그룹화
    trend_groups = df.groupby('trend_direction').size()

    # 조건별 그룹화
    condition_data = {
        '10일선 하회': df['condition_1_trend_breakdown'].sum(),
        '10일선 위': (~df['condition_1_trend_breakdown']).sum(),
        '거래량 폭증 (RVOL≥2)': df['condition_2_volume_confirmation'].sum(),
        '거래량 부족 (RVOL<2)': (~df['condition_2_volume_confirmation']).sum(),
    }

    # 신호별 그룹화
    signal_counts = df['signal'].value_counts()

    # ====================================================================
    # 시각화 생성
    # ====================================================================
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ====================================================================
    # 1. 추세 방향 파이 차트
    # ====================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    colors_trend = ['#FF6B6B', '#4ECDC4']  # 하락세: 빨강, 상승세: 청록
    wedges, texts, autotexts = ax1.pie(
        trend_groups.values,
        labels=trend_groups.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_trend,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    ax1.set_title(f'추세 방향 분포 (총 {len(df)}개 종목)', fontsize=14, weight='bold', pad=20)

    # ====================================================================
    # 2. 조건별 분류 막대 그래프
    # ====================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    condition_df = pd.DataFrame(list(condition_data.items()), columns=['조건', '종목수'])
    bars = ax2.barh(condition_df['조건'], condition_df['종목수'], color=['#FF6B6B', '#4ECDC4', '#FFD93D', '#95E1D3'])
    ax2.set_xlabel('종목 수', fontsize=12)
    ax2.set_title('조건별 분류', fontsize=14, weight='bold', pad=20)
    ax2.grid(axis='x', alpha=0.3)

    # 막대 위에 숫자 표시
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(width)}개',
                ha='left', va='center', fontsize=10, weight='bold')

    # ====================================================================
    # 3. RVOL 분포 히스토그램
    # ====================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # RVOL 데이터
    falling_rvol = df[df['current_position'] == 'below']['rvol']
    rising_rvol = df[df['current_position'] == 'above']['rvol']

    # 히스토그램
    bins = np.arange(0, max(df['rvol'].max(), 5) + 0.5, 0.5)
    ax3.hist([falling_rvol, rising_rvol], bins=bins, label=['하락세', '상승세'],
            color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
    ax3.axvline(x=2.0, color='red', linestyle='--', linewidth=2, label='RVOL 2.0 기준')
    ax3.set_xlabel('RVOL (상대 거래량)', fontsize=12)
    ax3.set_ylabel('종목 수', fontsize=12)
    ax3.set_title('RVOL 분포', fontsize=14, weight='bold', pad=20)
    ax3.legend(fontsize=11)
    ax3.grid(axis='y', alpha=0.3)

    # ====================================================================
    # 4. 신호별 분류
    # ====================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    signal_colors = {
        'SELL': '#FF6B6B',
        'HOLD': '#95E1D3'
    }
    colors = [signal_colors.get(sig, '#CCCCCC') for sig in signal_counts.index]

    bars = ax4.bar(range(len(signal_counts)), signal_counts.values, color=colors, edgecolor='black')
    ax4.set_xticks(range(len(signal_counts)))
    ax4.set_xticklabels(signal_counts.index, fontsize=11)
    ax4.set_ylabel('종목 수', fontsize=12)
    ax4.set_title('신호별 분류', fontsize=14, weight='bold', pad=20)
    ax4.grid(axis='y', alpha=0.3)

    # 막대 위에 숫자 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}개',
                ha='center', va='bottom', fontsize=11, weight='bold')

    # ====================================================================
    # 전체 제목
    # ====================================================================
    fig.suptitle(f'종목 분석 결과 시각화 - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                fontsize=16, weight='bold', y=0.98)

    # ====================================================================
    # 저장
    # ====================================================================
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return str(save_path)


def create_summary_chart(results: List[Dict], save_path: str = None) -> str:
    """
    간단한 요약 차트 생성 (테이블 형식)

    Parameters:
    -----------
    results : List[Dict]
        분석 결과 리스트
    save_path : str, optional
        저장 경로

    Returns:
    --------
    str : 저장된 파일 경로
    """
    setup_korean_font()

    df = pd.DataFrame(results)

    # 저장 경로 생성
    if save_path is None:
        db_folder = Path("database")
        db_folder.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = db_folder / f"{timestamp}_summary.png"

    # 요약 데이터 생성
    summary_data = {
        '총 종목 수': [len(df)],
        '하락세': [len(df[df['current_position'] == 'below'])],
        '상승세': [len(df[df['current_position'] == 'above'])],
        '강력 매도 신호': [len(df[df['signal'] == 'SELL'])],
        '거래량 폭증': [len(df[df['condition_2_volume_confirmation']])],
    }

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')

    summary_df = pd.DataFrame(summary_data)
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                    cellLoc='center', loc='center',
                    colColours=['#FFD93D']*len(summary_df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 3)

    plt.title(f'분석 결과 요약 - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
             fontsize=16, weight='bold', pad=20)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return str(save_path)
