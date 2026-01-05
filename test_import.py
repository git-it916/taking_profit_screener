"""
간단한 import 테스트 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print("Import 테스트 시작")
print("="*80)

try:
    # src 모듈 import 테스트
    print("\n[1/4] src 모듈 import 테스트...")
    from src import ExitSignalScreener, ParameterOptimizer, load_data_from_csv
    print("   [OK] ExitSignalScreener import 성공")
    print("   [OK] ParameterOptimizer import 성공")
    print("   [OK] load_data_from_csv import 성공")

    # utils import 테스트
    print("\n[2/4] utils 모듈 import 테스트...")
    from src.utils import generate_test_data, print_separator
    print("   [OK] generate_test_data import 성공")
    print("   [OK] print_separator import 성공")

    # 기본 기능 테스트
    print("\n[3/4] 기본 기능 테스트...")
    df = generate_test_data(days=50)
    print(f"   [OK] 테스트 데이터 생성 성공 ({len(df)}일)")

    screener = ExitSignalScreener(ma_period=20, rvol_period=20)
    print("   [OK] ExitSignalScreener 초기화 성공")

    filtered_data = screener.apply_filters(df)
    print("   [OK] apply_filters 실행 성공")

    output = screener.generate_screening_output(filtered_data, ticker="TEST")
    sell_signals = output[output['Signal'] == 'SELL']
    print(f"   [OK] SELL 신호 {len(sell_signals)}개 발견")

    # 최적화 테스트
    print("\n[4/4] 옵티마이저 초기화 테스트...")
    optimizer = ParameterOptimizer(df, ticker="TEST", evaluation_metric='sharpe_ratio')
    print("   [OK] ParameterOptimizer 초기화 성공")

    print("\n" + "="*80)
    print("모든 테스트 통과!")
    print("="*80)
    print("\n프로젝트 구조가 정상적으로 설정되었습니다.")
    print("이제 'python start.py'를 실행하여 프로그램을 사용하세요.\n")

except Exception as e:
    print(f"\n에러 발생: {e}")
    import traceback
    traceback.print_exc()
