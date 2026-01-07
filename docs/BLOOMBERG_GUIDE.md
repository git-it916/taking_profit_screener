# Bloomberg API 사용 가이드

Taking Profit Screener를 Bloomberg Terminal과 연동하여 사용하는 방법입니다.

## 🎯 핵심 장점

✅ **로컬 파일 저장 불필요** - Bloomberg에서 직접 데이터를 가져옴
✅ **300개 종목 일괄 분석** - 티커만 입력하면 자동 분석
✅ **최신 데이터** - 실시간으로 가장 최신 데이터 사용
✅ **자동 일봉 변환** - 시간봉도 자동으로 일봉으로 변환

---

## 📋 Step 1: 사전 준비

### 필수 요구사항

1. **Bloomberg Terminal 설치 및 실행**
   - Bloomberg Terminal이 실행 중이어야 합니다
   - Bloomberg에 로그인되어 있어야 합니다

2. **Python 라이브러리 설치**
   ```bash
   pip install xbbg
   ```

### 확인 방법

```bash
python -c "from xbbg import blp; print('Bloomberg API 준비 완료!')"
```

---

## 🚀 Step 2: 프로그램 실행

### 방법 1: start_bloomberg.py 실행

```bash
python start_bloomberg.py
```

### 방법 2: 배치 파일 사용

`run_bloomberg.bat` 파일 생성:
```batch
@echo off
"C:\Users\10845\AppData\Local\Programs\Python\Python312\python.exe" start_bloomberg.py
pause
```

더블클릭으로 실행!

---

## 📝 Step 3: 티커 입력

### 티커 형식

| 시장 | 형식 | 예시 |
|------|------|------|
| **한국** | 종목코드 KS | `005930 KS` (삼성전자)<br>`000660 KS` (SK하이닉스)<br>`373220 KS` (LG에너지솔루션) |
| **미국** | 티커 US | `AAPL US` (애플)<br>`MSFT US` (마이크로소프트)<br>`TSLA US` (테슬라) |
| **일본** | 종목코드 JP | `7203 JP` (도요타)<br>`6758 JP` (소니) |

### 입력 예시

#### 단일 종목
```
티커 입력: 005930 KS
```

#### 여러 종목 (쉼표로 구분)
```
티커 입력: 005930 KS, 000660 KS, 373220 KS
```

#### 300개 종목 예시
```
티커 입력: 005930 KS, 000660 KS, 035720 KS, 051910 KS, ...
```

---

## 🔄 Step 4: 프로그램 동작 과정

### 자동 처리 흐름

```
1. 티커 입력
   ↓
2. Bloomberg에서 OHLCV 데이터 다운로드
   - 기본: 최근 1년 일봉 데이터
   ↓
3. 시간봉 → 일봉 자동 변환 (필요시)
   ↓
4. 20일선, RVOL 계산
   ↓
5. 매도 신호 판단
   ↓
6. 결과 출력 (요약 테이블 + 상세 리포트)
```

### 실행 화면 예시

```
================================================================================
TAKING PROFIT SCREENER - BLOOMBERG 버전
Bloomberg Terminal에서 직접 데이터를 받아 분석합니다
================================================================================

⚠️  주의사항:
  1. Bloomberg Terminal이 실행 중이어야 합니다
  2. Bloomberg에 로그인되어 있어야 합니다

================================================================================
Bloomberg 티커를 입력하세요 (쉼표로 구분)
================================================================================

티커 형식:
  - 한국 주식: 005930 KS (삼성전자), 000660 KS (SK하이닉스)
  - 미국 주식: AAPL US (애플), MSFT US (마이크로소프트)
  - 예시: 005930 KS, 000660 KS, AAPL US

티커 입력: 005930 KS, 373220 KS

입력된 티커: ['005930 KS', '373220 KS']

데이터 기간을 선택하세요:
  1: 1년 (기본값)
  2: 6개월
  3: 3개월

선택 (엔터=1년):

================================================================================
총 2개 종목 분석 시작
================================================================================

[분석 중] 005930 KS...
[Bloomberg 다운로드]
  티커: 005930 KS
  기간: 2024-01-07 ~ 2026-01-07
  ✓ 252개 일봉 데이터 다운로드 완료
  ✓ 분석 완료

[분석 중] 373220 KS...
[Bloomberg 다운로드]
  티커: 373220 KS
  기간: 2024-01-07 ~ 2026-01-07
  ✓ 248개 일봉 데이터 다운로드 완료
  ✓ 분석 완료

================================================================================
분석 결과 요약
================================================================================

종목       현재가  전일비  20일선  괴리율    하회일      RVOL  신호
005930 KS  55000  -1.2%   56200  -2.1%  2025-12-28  0.8배  HOLD
373220 KS  370000 -2.4%   410550 -9.9%  2025-12-16  0.9배  HOLD

================================================================================
조건별 분류
================================================================================

[강력 매도 신호] 0개 종목:
  없음

[주의 필요] 2개 종목 (20일선 하회, 거래량 부족):
  - 005930 KS: 20일선 대비 -2.13%, 하회일: 2025-12-28, RVOL 0.82배
  - 373220 KS: 20일선 대비 -9.88%, 하회일: 2025-12-16, RVOL 0.95배

[거래량 폭증만] 0개 종목 (20일선 위):
  없음

================================================================================

종목별 상세 리포트를 보시겠습니까? (y/n):
```

---

## 💡 Step 5: 고급 사용법

### 5-1. 파이썬 스크립트로 직접 사용

```python
from src.bloomberg import download_bloomberg_data
from src import ExitSignalScreener, StockAnalyzer

# 1. Bloomberg에서 데이터 다운로드
df = download_bloomberg_data("005930 KS", period="1Y")

# 2. 분석
screener = ExitSignalScreener()
df_analyzed = screener.apply_filters(df)

# 3. 결과 확인
analyzer = StockAnalyzer()
result = analyzer.analyze_latest("삼성전자", df_analyzed)

# 4. 리포트 출력
report = analyzer.format_analysis_report(result)
print(report)
```

### 5-2. 여러 종목 일괄 다운로드 & 저장

```python
from src.bloomberg import download_multiple_tickers

# 티커 리스트
tickers = [
    "005930 KS",  # 삼성전자
    "000660 KS",  # SK하이닉스
    "373220 KS",  # LG에너지솔루션
    # ... 300개
]

# 일괄 다운로드 (파일로 저장)
data = download_multiple_tickers(
    tickers,
    period='1Y',
    save_to_file=True,
    output_dir="./bloomberg_data"
)
```

결과:
```
./bloomberg_data/
├── 005930_KS.xlsx
├── 000660_KS.xlsx
├── 373220_KS.xlsx
└── ...
```

### 5-3. 사용자 정의 기간

```python
# 2024년 전체 데이터
df = download_bloomberg_data(
    "005930 KS",
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# 최근 3개월
df = download_bloomberg_data("AAPL US", period="3M")

# 최근 6개월
df = download_bloomberg_data("MSFT US", period="6M")
```

---

## ⚠️ 문제 해결

### 문제 1: "xbbg 라이브러리가 설치되어 있지 않습니다"

**해결**:
```bash
pip install xbbg
```

### 문제 2: "Bloomberg에 연결할 수 없습니다"

**원인**:
- Bloomberg Terminal이 실행되지 않음
- Bloomberg에 로그인되지 않음

**해결**:
1. Bloomberg Terminal 실행
2. Bloomberg 로그인 확인
3. 다시 시도

### 문제 3: "데이터를 가져올 수 없습니다"

**원인**:
- 잘못된 티커 형식
- 해당 종목의 데이터가 Bloomberg에 없음

**해결**:
1. 티커 형식 확인 (예: `005930 KS` 맞음, `005930` 틀림)
2. Bloomberg에서 해당 티커 검색 가능 여부 확인
3. 공백 포함 여부 확인

### 문제 4: "시간봉 데이터가 다운로드됨"

**해결**:
- 자동으로 일봉 변환됨 (걱정하지 않아도 됨)
- `[변환 완료] 시간봉 XXX개 → 일봉 YYY개` 메시지 확인

---

## 📊 데이터 품질 확인

### Bloomberg 데이터 vs 로컬 파일 비교

```python
from src.bloomberg import download_bloomberg_data
from src import load_data_from_csv

# Bloomberg에서 다운로드
df_bloomberg = download_bloomberg_data("373220 KS")

# 로컬 파일 로드
df_local = load_data_from_csv("LGENSOL.xlsx")

# 비교
print(f"Bloomberg: {len(df_bloomberg)}일")
print(f"로컬 파일: {len(df_local)}일")
print(f"\nBloomberg 최신 데이터: {df_bloomberg.iloc[-1]['Date']}")
print(f"로컬 파일 최신 데이터: {df_local.iloc[-1]['Date']}")
```

---

## 🎓 추가 리소스

### Bloomberg 티커 찾기

1. Bloomberg Terminal에서 `<TICKER>` 입력
2. 종목명 검색
3. "Equity" 탭에서 티커 확인

### 300개 종목 티커 리스트 작성

1. Excel에 티커 리스트 작성:
   ```
   005930 KS
   000660 KS
   373220 KS
   ...
   ```

2. 한 줄로 변환:
   ```python
   # tickers.txt 파일 읽어서 한 줄로
   with open('tickers.txt', 'r') as f:
       tickers = [line.strip() for line in f if line.strip()]

   ticker_string = ", ".join(tickers)
   print(ticker_string)
   ```

3. 복사해서 프로그램에 붙여넣기

---

## 📞 지원

문제가 발생하면:
1. CLAUDE.md 파일 참조
2. Bloomberg Terminal 상태 확인
3. 에러 메시지 확인

---

**Created**: 2026-01-07
**Version**: 1.0.0
**Author**: Taking Profit Screener Team
