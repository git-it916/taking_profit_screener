"""
한글 종목명 → Bloomberg 티커 변환 도구

300개 종목명을 한 번에 Bloomberg 티커로 변환합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가 (utils 폴더 상위)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ticker_converter import (
    convert_names_to_tickers,
    batch_convert_from_file,
    print_ticker_list,
    KOREAN_STOCK_MAP
)


def main():
    """메인 함수"""
    print("="*80)
    print("한글 종목명 → Bloomberg 티커 변환기")
    print("="*80)

    print("\n변환 방법을 선택하세요:")
    print("  1: 직접 입력 (복사 & 붙여넣기)")
    print("  2: 파일에서 읽기 (.txt, .xlsx)")
    print("  3: 등록된 종목 목록 보기")

    choice = input("\n선택 (1/2/3): ").strip()

    # ====================================================================
    # 옵션 1: 직접 입력
    # ====================================================================
    if choice == '1':
        print("\n" + "="*80)
        print("한글 종목명을 입력하세요 (쉼표 또는 줄바꿈으로 구분)")
        print("="*80)
        print("\n예시:")
        print("  삼성전자, SK하이닉스, LG에너지솔루션")
        print("\n또는:")
        print("  삼성전자")
        print("  SK하이닉스")
        print("  LG에너지솔루션")
        print("\n입력 완료 후 빈 줄에서 Ctrl+Z (Windows) 또는 Ctrl+D (Mac/Linux)")
        print("-"*80)

        lines = []
        print("\n입력:")
        try:
            while True:
                line = input()
                if line:
                    lines.append(line)
        except EOFError:
            pass

        # 파싱
        text = '\n'.join(lines)
        if ',' in text:
            names = [n.strip() for n in text.split(',')]
        else:
            names = [n.strip() for n in text.split('\n')]

        names = [n for n in names if n]

        if not names:
            print("\n[에러] 종목명이 입력되지 않았습니다")
            return

        print(f"\n입력된 종목: {len(names)}개")

        # 변환
        tickers = convert_names_to_tickers(names)
        print_ticker_list(tickers)

        # 저장 옵션
        save = input("\n파일로 저장하시겠습니까? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("파일명 (기본값: tickers.txt): ").strip() or "tickers.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(', '.join(tickers))
            print(f"\n저장 완료: {filename}")

    # ====================================================================
    # 옵션 2: 파일에서 읽기
    # ====================================================================
    elif choice == '2':
        print("\n" + "="*80)
        print("파일 형식:")
        print("  - 텍스트 파일: 한 줄에 하나씩 또는 쉼표로 구분")
        print("  - 엑셀 파일: 첫 번째 컬럼에 종목명")
        print("="*80)

        input_file = input("\n입력 파일 경로: ").strip()

        if not os.path.exists(input_file):
            print(f"\n[에러] 파일을 찾을 수 없습니다: {input_file}")
            return

        # 사용자 정의 매핑 파일 (선택)
        print("\n사용자 정의 매핑 파일이 있습니까? (엑셀 형식)")
        print("  형식: | 종목명 | 티커 |")
        mapping_file = input("매핑 파일 경로 (없으면 엔터): ").strip()

        if mapping_file and not os.path.exists(mapping_file):
            print(f"\n[경고] 매핑 파일을 찾을 수 없습니다: {mapping_file}")
            mapping_file = None

        # 변환
        tickers = batch_convert_from_file(
            input_file,
            mapping_file=mapping_file
        )

        print_ticker_list(tickers)

        # 저장 옵션
        save = input("\n파일로 저장하시겠습니까? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("파일명 (기본값: tickers_output.txt): ").strip() or "tickers_output.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(', '.join(tickers))
            print(f"\n저장 완료: {filename}")

    # ====================================================================
    # 옵션 3: 등록된 종목 목록
    # ====================================================================
    elif choice == '3':
        print("\n" + "="*80)
        print(f"등록된 종목 ({len(KOREAN_STOCK_MAP)}개)")
        print("="*80)

        for name, ticker in sorted(KOREAN_STOCK_MAP.items()):
            print(f"  {name:<20} → {ticker}")

        print("\n" + "="*80)
        print("\n추가 종목이 필요하면:")
        print("  1. ticker_mapping.xlsx 파일 생성")
        print("  2. | 종목명 | 티커 | 형식으로 작성")
        print("  3. 옵션 2에서 매핑 파일로 지정")

    else:
        print("\n[에러] 잘못된 선택입니다")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n[에러] {e}")
        import traceback
        traceback.print_exc()
