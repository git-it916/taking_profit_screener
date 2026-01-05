# XLSX 지원 업데이트 완료!

CSV에서 XLSX로 완전히 전환되었습니다!

## 변경 사항

### 파일 형식 변경
- **이전**: CSV 파일만 지원 (.csv)
- **이후**: XLSX 파일 기본 지원 (.xlsx, .xls), CSV도 호환

### 주요 변경 내용

1. **라이브러리 추가**
   - `openpyxl>=3.0.0` 추가 (XLSX 읽기/쓰기)

2. **코어 모듈 업데이트**
   - `src/screener.py`: `load_data_from_csv()` 함수가 XLSX 자동 감지
   - `src/analyzer.py`: XLSX 파일명 처리 추가

3. **실행 스크립트 업데이트**
   - `quick_analyze.py`: XLSX 파일명 예시 변경
   - `analyze_stocks.py`: 사용자 입력 가이드 변경
   - `create_sample_data.py`: XLSX 파일 생성으로 변경

4. **문서 업데이트**
   - `README.md`: XLSX 예시로 변경
   - `CSV_ANALYSIS_GUIDE.md`: XLSX 가이드로 변경 (CSV도 지원)

## 사용 방법

### 1. 라이브러리 설치

```bash
pip install -r requirements.txt
```

이제 `openpyxl`이 자동으로 설치됩니다.

### 2. 샘플 XLSX 파일 생성

```bash
python create_sample_data.py
```

문서 폴더에 4개의 XLSX 파일이 생성됩니다:
- `sm.xlsx` - SELL 신호
- `samsung.xlsx` - 추세 하락만
- `apple.xlsx` - 정상
- `kakao.xlsx` - 거래량 폭증만

### 3. XLSX 파일 분석

```bash
# 단일 종목
python quick_analyze.py sm.xlsx

# 여러 종목
python quick_analyze.py sm.xlsx samsung.xlsx apple.xlsx

# CSV도 가능
python quick_analyze.py old_data.csv
```

## 파일 형식 자동 감지

프로그램이 파일 확장자를 자동으로 감지합니다:

```python
def load_data_from_csv(filepath: str, date_column: Optional[str] = None):
    # 파일 확장자에 따라 로드 방법 선택
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)  # XLSX 로드
    else:
        df = pd.read_csv(filepath)    # CSV 로드

    # 나머지 동일...
```

## 실행 예시

### 테스트 실행

```bash
# 1. 라이브러리 설치
pip install -r requirements.txt

# 2. 샘플 생성
python create_sample_data.py

# 3. 분석
python quick_analyze.py sm.xlsx
```

**결과**:
```
================================================================================
종목 분석 리포트: SM
================================================================================

날짜: 2026-01-05 18:00:04
현재가: 90.32

[1] 20일 이동평균선 분석
  - 20일선: 92.04
  - 20일선 대비 거리: -1.71원 (-1.86%)
  - 상태: 20일선 아래 (근접)
  - 조건 1: [O] 충족 (추세 하락)

[2] RVOL (상대 거래량) 분석
  - RVOL: 2.42배
  - 강도: 보통 (2~2.5배)
  - 방향: 상승 (전일 대비 +1.55)
  - 조건 3: [O] 충족 (거래량 확인)

[3] 윅 패턴 분석
  - 윅 비율: 0.73 (73.4%)
  - 조건 2: [O] 충족 (거부 패턴)

[종합 판정]
  신호: SELL
  분류: 강력 매도 (3개 조건 모두 충족)
```

### 실전 사용

```bash
# 문서 폴더에 XLSX 파일 준비
# 예: 삼성전자.xlsx, SK하이닉스.xlsx 등

# 분석 실행
python quick_analyze.py 삼성전자.xlsx SK하이닉스.xlsx NAVER.xlsx
```

## Python 코드 사용

```python
from src import analyze_stock_from_csv, StockAnalyzer

# XLSX 파일 분석
result = analyze_stock_from_csv("sm.xlsx")

# 결과 확인
print(f"종목: {result['ticker']}")
print(f"20일선 대비: {result['ma_distance_percent']:.2f}%")
print(f"RVOL: {result['rvol']:.2f}배")
print(f"신호: {result['signal']}")

# CSV도 동일하게 동작
result_csv = analyze_stock_from_csv("old_data.csv")
```

## 호환성

### 지원하는 파일 형식
- ✅ `.xlsx` (Excel 2007 이상, 권장)
- ✅ `.xls` (Excel 97-2003)
- ✅ `.csv` (이전 호환성 유지)

### 필수 컬럼
모든 형식 공통:
- `Date`: 날짜
- `Open`: 시가
- `High`: 고가
- `Low`: 저가
- `Close`: 종가
- `Volume`: 거래량

## 기존 CSV 사용자를 위한 안내

### CSV 파일도 그대로 사용 가능

```bash
# 기존 CSV 파일도 동일하게 동작
python quick_analyze.py old_data.csv new_data.xlsx

# 섞어서 사용 가능
python quick_analyze.py stock1.csv stock2.xlsx stock3.xlsx
```

### CSV를 XLSX로 변환 (선택)

```python
import pandas as pd

# CSV 로드
df = pd.read_csv('old_data.csv')

# XLSX로 저장
df.to_excel('new_data.xlsx', index=False, engine='openpyxl')
```

## 장점

1. **Excel 호환성**: Excel에서 바로 열어서 확인 가능
2. **데이터 타입 보존**: 날짜, 숫자 형식 자동 인식
3. **파일 크기**: CSV보다 압축률 높음
4. **범용성**: 대부분의 증권사 HTS/MTS에서 XLSX 지원

## 문제 해결

### Q: openpyxl 설치 오류
```bash
pip install --upgrade pip
pip install openpyxl
```

### Q: "파일을 찾을 수 없습니다" 오류
- 파일 확장자 확인 (.xlsx, .xls, .csv)
- 파일 경로 확인
- 파일명에 특수문자 제거

### Q: CSV 파일도 계속 사용하고 싶어요
- 문제 없습니다! CSV 파일도 그대로 지원합니다.
- `.csv` 확장자면 자동으로 CSV로 읽습니다.

## 요약

| 항목 | 이전 | 이후 |
|------|------|------|
| **파일 형식** | CSV만 | XLSX 우선, CSV 호환 |
| **라이브러리** | pandas, numpy | pandas, numpy, openpyxl |
| **샘플 파일** | sm.csv | sm.xlsx |
| **사용법** | `python quick_analyze.py sm.csv` | `python quick_analyze.py sm.xlsx` |

---

**이제 `python quick_analyze.py <파일명.xlsx>`로 시작하세요!**

CSV 파일도 여전히 사용 가능합니다: `python quick_analyze.py <파일명.csv>`
