# CLAUDE.md

이 문서는 Taking Profit Screener 프로젝트의 개발 과정과 사용자 요구사항을 기록합니다.

## 프로젝트 개요

**Taking Profit Screener**는 Bloomberg Terminal 또는 XLSX/CSV 주가 데이터를 분석하여 10일 이동평균선과 상대 거래량(RVOL)을 기반으로 매도 시점을 포착하는 스크리닝 도구입니다.

**주요 기능**:
- Bloomberg Terminal API를 통한 실시간 데이터 분석 (start_bloomberg.py)
- 로컬 파일(XLSX/CSV) 기반 분석 (start.py)
- 10일선 추세 분류 (하락세/상승세 구분)
- RVOL 기반 거래량 분석 (10일 평균 대비)
- 시각화 히트맵 생성
- Bloomberg 티커 → 한글 종목명 자동 변환

## 사용자 요구사항 (개발 순서)

### 1. 초기 요구사항
- **목표**: CSV/XLSX 파일에서 주가 데이터를 읽어 RVOL과 20일선 분석
- **입력**: 문서 경로(`C:\Users\10845\OneDrive - 이지스자산운용\문서`)에서 XLSX 파일
- **출력**:
  - RVOL 충족 여부
  - 20일선 충족 여부
  - 둘 다 충족 여부 분류
  - RVOL 방향과 강도
  - 20일선과의 거리

### 2. 구조화 요구사항
- **파일 구조 정리**:
  - `start.py`를 메인 진입점으로
  - 나머지 기능은 폴더로 모듈화
  - 의존성을 통한 클래스/함수 구조
- **입력 방식**:
  - 파일명을 여러 개 입력받기
  - 자동으로 분석 실행

### 3. 20일선 하회/상회 추적 기능 추가
- **최근 20일선 하회일** 기록
- **최근 20일선 상회일** 기록
- **20일선 아래 경과일** 계산

### 4. 윗꼬리 분석 제거 요청
> "하락시 아래꼬리, 윗꼬리랑 상승시 아래꼬리, 윗꼬리가 다르잖아. 일단 꼬리분석은 나중에 하자."

- **제거 대상**:
  - 윗꼬리 비율 계산
  - 아래꼬리 비율 계산
  - 매도세 강도 판단
- **유지**: RVOL과 20일선만 사용
- **추가**: README 최신화, 불필요한 파일 삭제

### 5. 20일선 가격 및 어제 종가 표시
- **20일선 가격**: 괴리율뿐만 아니라 실제 가격 값 표시
- **어제 종가**: 전일 종가와 변화량(원, %) 표시

### 6. MA 계산 방식 검증 요청
- **확인 사항**:
  - SMA_20 = (P_1 + P_2 + ... + P_20) / 20
  - 각 가격의 동등한 비중 (1/20)
  - "이동" 평균 원리 (매일 가장 오래된 데이터 제외, 새 데이터 추가)
- **검증 결과**: pandas `rolling(window=20).mean()`이 표준 SMA 공식과 100% 일치 확인

## 최종 전략 구조

### Volume-Confirmed Rejection → 20일선 + RVOL 전략

**SELL 신호 조건** (2가지 모두 충족 시):
1. **조건 1**: 현재가 < 20일 이동평균선 (추세 하락)
2. **조건 2**: RVOL ≥ 2.0 (거래량 폭증)

### 출력 정보

#### 요약 테이블
```
종목    현재가  전일비  20일선  괴리율    하회일      RVOL  신호
SM      89    -1.1%   92    -2.7%  2025-10-18  2.3배  SELL
SAMSUNG 86    +0.2%   88    -2.0%  2025-10-18  0.9배  HOLD
```

#### 상세 리포트
```
[1] 20일 이동평균선 분석
  - 20일선 가격: 91.95
  - 현재가 대비: -2.52원 (-2.74%)
  - 상태: 20일선 아래 (근접)
  - 최근 20일선 하회일: 2025-10-18
  - 최근 20일선 상회일: 없음
  - 20일선 아래 경과일: 81일
  - 조건 1 충족 여부: [O] 충족 (추세 하락)

[2] RVOL (상대 거래량) 분석
  - RVOL: 2.33배
  - 강도: 보통 (2~2.5배)
  - 방향: 상승 (전일 대비 +1.46)
  - 조건 2 충족 여부: [O] 충족 (거래량 확인)

[종합 판정]
  신호: SELL
  분류: 강력 매도 (20일선 하회 + 거래량 폭증)
```

## 프로젝트 구조

```
taking_profit_screener/
├── start.py                ← 메인 진입점
├── run.bat                 ← 실행 배치 파일
├── requirements.txt        ← 의존성 패키지
├── README.md              ← 사용자 가이드
├── CLAUDE.md              ← 개발 히스토리 (이 파일)
├── MA_CALCULATION_VERIFICATION.md  ← MA 계산 검증 문서
│
├── src/                   ← 핵심 로직
│   ├── __init__.py
│   ├── screener.py        ← ExitSignalScreener (MA20, RVOL 계산)
│   ├── analyzer.py        ← StockAnalyzer (상세 분석 및 리포트)
│   └── optimizer.py       ← 파라미터 최적화 (향후 확장)
│
└── tools/                 ← 유틸리티
    ├── quick_analyze.py
    ├── analyze_stocks.py
    └── create_sample_data.py
```

## 핵심 계산 로직

### 1. 20일 단순 이동평균 (SMA)

**공식**:
```
SMA_20 = (P_1 + P_2 + P_3 + ... + P_20) / 20
```

- `P_1`: 오늘 종가
- `P_2`: 1일 전 종가
- `P_20`: 20일 전 종가
- 각 가격은 동등한 비중 (1/20 = 5%)

**구현** (`src/screener.py`):
```python
def calculate_sma(self, df: pd.DataFrame, column: str = 'Close') -> pd.Series:
    return df[column].rolling(window=self.ma_period).mean()
```

**검증 결과**: pandas `rolling().mean()`과 수동 계산의 차이 = 0.0000000000

### 2. 상대 거래량 (RVOL)

**공식**:
```
RVOL = 현재 거래량 / 20일 평균 거래량
```

**구현** (`src/screener.py`):
```python
def calculate_rvol(self, df: pd.DataFrame, volume_column: str = 'Volume') -> pd.Series:
    avg_volume = df[volume_column].rolling(window=self.ma_period).mean()
    rvol = df[volume_column] / avg_volume
    return rvol
```

**강도 분류**:
- 매우 강함: RVOL ≥ 3.0
- 강함: 2.5 ≤ RVOL < 3.0
- 보통: 2.0 ≤ RVOL < 2.5
- 약함: RVOL < 2.0

### 3. 20일선 하회/상회 추적

**로직** (`src/screener.py`):
```python
def _track_ma_crossover(self, df: pd.DataFrame) -> pd.Series:
    """
    20일선 상회/하회 날짜 및 경과일 추적

    Returns:
        dict: {
            'last_break_below': 최근 하회일,
            'last_break_above': 최근 상회일,
            'days_below': 하회 경과일
        }
    """
```

## 제거된 기능들

### 윗꼬리/아래꼬리 분석 (제거됨)
- **제거 이유**: 상승장과 하락장에서 꼬리의 의미가 다르기 때문
- **제거 날짜**: 2026-01-07
- **영향받은 파일**:
  - `src/screener.py`: `calculate_wick_ratio()`, `calculate_lower_wick_ratio()` 삭제
  - `src/analyzer.py`: 윗꼬리 관련 결과 필드 및 리포트 섹션 제거
  - `start.py`: "매도세" 컬럼 제거
  - `README.md`: 윗꼬리 패턴 설명 삭제

### 3개 조건 전략 → 2개 조건 전략
- **이전**: 추세 하락 + 윗꼬리 패턴 + 거래량 확인
- **현재**: 추세 하락 + 거래량 확인

## 기술 스택

### Python 환경
- **Python 3.12.10 (64-bit)** 필수
- **경로**: `C:\Users\10845\AppData\Local\Programs\Python\Python312\python.exe`
- **주의**: 32비트 Python은 pandas 2.x 미지원

### 의존성 패키지
```
pandas>=2.0        # 데이터 분석
openpyxl>=3.0      # XLSX 파일 읽기
numpy>=2.0         # 수치 계산
tabulate>=0.9      # 테이블 포맷팅
```

### 설치 방법
```bash
"C:\Users\10845\AppData\Local\Programs\Python\Python312\python.exe" -m pip install pandas openpyxl numpy tabulate
```

## 사용 방법

### 1. 배치 파일로 실행 (가장 간단)
```
run.bat 더블클릭
```

### 2. PowerShell에서 실행
```powershell
& "C:\Users\10845\AppData\Local\Programs\Python\Python312\python.exe" start.py
```

### 3. VSCode에서 실행
1. Ctrl + Shift + P
2. "Python: Select Interpreter" 검색
3. `Python312\python.exe` 선택
4. F5 실행

### 입력 형식
```
파일명 입력: sm.xlsx, samsung.xlsx, apple.xlsx
```

## 입력 데이터 형식

XLSX/CSV 파일은 다음 컬럼을 포함해야 합니다:

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| Date | 날짜 | 2025-10-18 |
| Open | 시가 | 89.50 |
| High | 고가 | 92.30 |
| Low | 저가 | 88.10 |
| Close | 종가 | 89.44 |
| Volume | 거래량 | 1500000 |

## 모니터링 우선순위

### Level 1: 강력 매도 신호 (즉시 주의)
- 조건 1, 2 모두 충족
- 신호 = SELL
- **의미**: 20일선 하회 + 거래량 폭증 = 강한 매도 압력

### Level 2: 주의 필요
- 20일선 하회, 거래량 부족
- **관찰**: 거래량이 늘어나면 SELL로 전환 가능

### Level 3: 관찰
- 20일선 하회만
- **모니터링**: 하회일과 경과일 확인, RVOL 추이 관찰

## Bloomberg Terminal 연동 설정

### 환경 요구사항
- **Python 버전**: Python 3.12.10 (64-bit) **필수**
  - Python 3.14는 blpapi 미지원
  - Python 3.8-3.12 지원
- **Bloomberg Terminal**: 실행 중이고 로그인 상태여야 함

### 패키지 의존성
```
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.0.0
tabulate>=0.9.0
xbbg>=0.10.0
blpapi==3.25.11.1
```

### Bloomberg API 설치 과정

#### 1단계: Python 3.12 설치
```bash
# Python 3.12.10 다운로드 및 설치
# https://www.python.org/downloads/
# "Add Python to PATH" 체크 필수
```

#### 2단계: 기본 패키지 설치
```bash
py -3.12 -m pip install -r requirements.txt
```

#### 3단계: Bloomberg API (blpapi) 설치

**방법 1: Bloomberg 공식 pip 저장소에서 설치 (권장)**
```bash
py -3.12 -m pip download --no-deps --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi --dest "C:\Users\Bloomberg\Downloads\blpapi_wheels"
py -3.12 -m pip install "C:\Users\Bloomberg\Downloads\blpapi_wheels\blpapi-3.25.11.1-py3-none-win_amd64.whl"
```

**방법 2: Bloomberg Terminal에서 다운로드**
1. Bloomberg Terminal 실행
2. `WAPI<GO>` 입력
3. "APIs" 필터 → "Bloomberg API (BLPAPI)" 선택
4. "Bloomberg API - SDK" 섹션 찾기
5. **Windows + Python** 행의 다운로드 버튼 클릭
6. 다운로드한 wheel 파일 설치:
   ```bash
   py -3.12 -m pip install [다운로드경로]\blpapi-3.25.11.1-py3-none-win_amd64.whl
   ```

**주의사항:**
- **소스 코드 버전**(~340KB)이 아닌 **미리 빌드된 wheel 버전**(~5.6MB)을 다운로드해야 함
- 소스 버전은 C++ 컴파일러와 C++ SDK가 필요하므로 권장하지 않음

#### 4단계: 설치 확인
```bash
py -3.12 -c "import blpapi; print('Bloomberg API installed successfully'); print('Version:', blpapi.__version__)"
```

예상 출력:
```
Bloomberg API installed successfully
Version: 3.25.11.1
```

### 코드 수정 사항

#### Windows 콘솔 인코딩 설정
한글 및 이모지 출력 문제 해결을 위해 `start.py`와 `start_bloomberg.py`에 추가:

```python
# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
```

#### Bloomberg 티커 자동 변환
`src/bloomberg.py`에 티커 형식 자동 변환 추가:

```python
# Bloomberg 티커 형식 확인 및 조정
# "005930 KS" -> "005930 KS Equity"
if not ticker.upper().endswith((' EQUITY', ' INDEX', ' CURNCY', ' COMDTY')):
    ticker = ticker + ' Equity'
```

**이유**: Bloomberg API는 `005930 KS`가 아닌 `005930 KS Equity` 형식을 요구

#### 배치 파일 업데이트
`run.bat`와 `run_bloomberg.bat` 수정:

```bat
@echo off
chcp 65001 >nul
py -3.12 start_bloomberg.py
pause
```

**변경사항:**
- UTF-8 인코딩 설정 (`chcp 65001`)
- Python 3.12 명시적 지정 (`py -3.12`)
- 하드코딩된 경로 제거

### Bloomberg 데이터 다운로드 사용법

#### 티커 형식
```
한국 주식: 005930 KS, 000660 KS
미국 주식: AAPL US, MSFT US
(자동으로 "Equity" 접미사 추가됨)
```

#### 실행 방법
```bash
# 배치 파일 실행
run_bloomberg.bat

# 또는 직접 실행
py -3.12 start_bloomberg.py
```

#### 프로그램 흐름
1. 티커 입력 (쉼표로 구분)
2. 데이터 기간 선택 (1년/6개월/3개월)
3. Bloomberg Terminal에서 데이터 자동 다운로드
4. 분석 실행
5. 결과 출력 (요약 테이블 + 상세 리포트)

### 통합 분석 도구 (start.py)

두 가지 데이터 소스를 지원:
1. **Bloomberg Terminal** - 실시간 데이터 다운로드
2. **로컬 파일** - XLSX/CSV 파일 분석

#### 한글 종목명 → Bloomberg 티커 자동 변환
```python
from src.ticker_converter import convert_names_to_tickers

# 사용 예시
names = ["삼성전자", "SK하이닉스", "LG에너지솔루션"]
tickers = convert_names_to_tickers(names)
# 결과: ['005930 KS', '000660 KS', '373220 KS']
```

## 개발 히스토리

### 2026-01-08 (RVOL 기간 변경 및 시각화 개선)
- ✅ RVOL 계산 기간 변경: 20일 → 10일
- ✅ `src/screener.py` - rvol_period 기본값 변경 (20 → 10)
- ✅ `src/analyzer.py` - rvol_period 기본값 변경 (20 → 10)
- ✅ 시각화에서 종목별 괴리율 히트맵 제거
- ✅ 차트 레이아웃 최적화 (3x2 → 2x2)

### 2026-01-08 (시각화 기능 추가)
- ✅ 히트맵 시각화 모듈 추가 (`src/visualizer.py`)
- ✅ 추세별/조건별 분류를 차트로 시각화
  - 추세 방향 파이 차트
  - 조건별 분류 막대 그래프
  - RVOL 분포 히스토그램
  - 신호별 분류 막대 그래프
- ✅ database 폴더에 자동 저장 (날짜_시간_heatmap.png)
- ✅ requirements.txt에 matplotlib, seaborn 추가
- ✅ 상세 리포트 프롬프트 제거, 시각화 저장 프롬프트로 대체

### 2026-01-08 (추세 방향 분류 기능 추가)
- ✅ 하락세/상승세 자동 판단 로직 추가
- ✅ `src/analyzer.py` - 추세 방향 필드 추가 (trend_direction, trend_detail, current_position)
- ✅ 하락세 종목: "10일선 위(날짜) → 10일선 아래(날짜)" 경로 표시
- ✅ 상승세 종목: "10일선 아래(날짜) → 10일선 위(날짜)" 경로 표시
- ✅ start.py, start_bloomberg.py - 추세별 분류 섹션 추가
- ✅ 요약 테이블에 '추세' 컬럼 추가
- ✅ 리포트에 추세 경로 정보 표시

### 2026-01-08 (이동평균선 기간 변경: 20일 → 10일)
- ✅ 모든 이동평균선 기간을 20일에서 10일로 변경
- ✅ `src/screener.py` - ExitSignalScreener 기본값 수정 (ma_period=10)
- ✅ `src/analyzer.py` - StockAnalyzer 기본값 수정 (ma_period=10)
- ✅ 모든 변수명 업데이트 (MA20 → MA10, ma20 → ma10, 20일선 → 10일선)
- ✅ 컬럼명 업데이트 (Last_MA20_Break → Last_MA10_Break, Days_Below_MA20 → Days_Below_MA10)
- ✅ start.py, start_bloomberg.py 출력 메시지 업데이트
- ✅ 전략 설명 업데이트 (Price < 20MA → Price < 10MA)

### 2026-01-07 (하회일 날짜 포맷 수정)
- ✅ 날짜 표시 오류 수정: 숫자(233.0, 235.0) → 표준 날짜 형식(2024-10-18)
- ✅ `src/screener.py`의 `_track_ma_crossover` 함수 개선
  - DataFrame 인덱스 대신 Date 컬럼의 실제 날짜 사용
  - 날짜를 'YYYY-MM-DD' 형식의 문자열로 저장
- ✅ 하회일/상회일 추적 로직 전체 업데이트

### 2026-01-07 (Bloomberg 연동)
- ✅ Bloomberg API (blpapi) 3.25.11.1 설치
- ✅ Python 3.14 → 3.12 다운그레이드 (blpapi 호환성)
- ✅ Windows 콘솔 UTF-8 인코딩 설정
- ✅ Bloomberg 티커 자동 변환 (`KS` → `KS Equity`)
- ✅ 배치 파일 업데이트 (UTF-8, Python 3.12 지정)
- ✅ `start_bloomberg.py` - Bloomberg 전용 실행 파일
- ✅ `start.py` - 통합 실행 파일 (Bloomberg + 로컬)
- ✅ `src/bloomberg.py` - Bloomberg API 연동 모듈
- ✅ `src/ticker_converter.py` - 한글 종목명 변환
- ✅ requirements.txt 업데이트 (tabulate, xbbg 추가)
- ✅ StockAnalyzer 초기화 오류 수정 (screener 파라미터 제거)

### 2026-01-07 (전략 단순화)
- ✅ 윗꼬리/아래꼬리 분석 완전 제거
- ✅ 3개 조건 → 2개 조건으로 전략 단순화
- ✅ 이동평균선 가격 명시적 표시
- ✅ 어제 종가 및 전일대비 변화 추가
- ✅ MA 계산 방식 검증 (표준 SMA 공식과 100% 일치 확인)
- ✅ Python 환경 문제 해결 (32비트 → 64비트)
- ✅ README.md 최신화
- ✅ 불필요한 문서 파일 삭제 (UPDATE_LOG.md, CSV_ANALYSIS_GUIDE.md 등)
- ✅ run.bat 파일 생성

### 이전 개발 내용 (요약)
- 20일선 하회/상회 날짜 추적 기능 추가
- CSV → XLSX 지원으로 변경
- 파일 구조 모듈화
- 다중 종목 배치 분석 기능
- 상세 리포트 포맷팅

## 향후 확장 가능성

### 1. 백테스팅 기능
- 과거 데이터로 전략 성과 검증
- 승률, 평균 수익률 계산

### 2. 파라미터 최적화
- RVOL 임계값 조정 (현재: 2.0)
- MA 기간 조정 (현재: 20일)
- optimizer.py 활용

### 3. 추가 지표
- RSI (과매도/과매수)
- 볼린저 밴드
- MACD

### 4. 실시간 모니터링
- API 연동
- 자동 알림 (이메일, Slack 등)

### 5. 꼬리 분석 재도입
- 상승장/하락장 구분 로직
- 맥락에 맞는 해석

## 문제 해결 기록

### Bloomberg API 설치 문제

#### 문제 1: Python 3.14에서 blpapi 설치 불가
**증상**:
```
ERROR: Could not find a version that satisfies the requirement blpapi
```

**원인**: Python 3.14는 blpapi가 지원하지 않음 (Python 3.8-3.12만 지원)

**해결**:
1. Python 3.12.10 설치
2. VSCode에서 인터프리터 변경: `Ctrl+Shift+P` → "Python: Select Interpreter" → Python 3.12
3. 패키지 재설치

#### 문제 2: 소스 코드 버전 다운로드 (컴파일 에러)
**증상**:
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**원인**: 소스 코드 버전(~340KB)을 다운로드하여 C++ 컴파일러가 필요함

**해결**: Bloomberg 공식 pip 저장소에서 미리 빌드된 wheel 파일 다운로드
```bash
py -3.12 -m pip download --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi --dest ".\blpapi_wheels"
py -3.12 -m pip install ".\blpapi_wheels\blpapi-3.25.11.1-py3-none-win_amd64.whl"
```

#### 문제 3: 티커 형식 오류
**증상**:
```
ValueError: 데이터를 가져올 수 없습니다: 000660 KS
```

**원인**: Bloomberg API는 `000660 KS Equity` 형식 필요

**해결**: `src/bloomberg.py`에 자동 변환 로직 추가
```python
if not ticker.upper().endswith((' EQUITY', ' INDEX', ' CURNCY', ' COMDTY')):
    ticker = ticker + ' Equity'
```

#### 문제 4: 한글/이모지 출력 오류 (Windows 콘솔)
**증상**:
```
UnicodeEncodeError: 'cp949' codec can't encode character
```

**원인**: Windows 콘솔 기본 인코딩이 cp949

**해결**:
1. 배치 파일에 `chcp 65001` 추가
2. Python 코드에서 stdout 인코딩 강제 변경:
```python
if sys.platform == 'win32':
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
```

#### 문제 5: StockAnalyzer 초기화 오류
**증상**:
```
TypeError: StockAnalyzer.__init__() got an unexpected keyword argument 'screener'
```

**원인**: StockAnalyzer는 내부적으로 screener를 생성하므로 외부에서 전달할 수 없음

**해결**: `start.py`와 `start_bloomberg.py`에서 초기화 코드 수정
```python
# 수정 전
screener = ExitSignalScreener()
analyzer = StockAnalyzer(screener=screener)
result = analyzer.analyze_latest(ticker, df)

# 수정 후
analyzer = StockAnalyzer()
result = analyzer.analyze_latest(df, ticker)
```

#### 문제 6: 하회일 날짜 포맷 오류
**증상**:
```
하회일이 233.0, 235.0, 242.0 같은 숫자로 표시됨
```

**원인**: `_track_ma_crossover` 함수에서 DataFrame의 숫자 인덱스(0, 1, 2, ...)를 하회일로 저장

**해결**: `src/screener.py`의 `_track_ma_crossover` 함수 수정
```python
# 각 행 처리 시 Date 컬럼에서 실제 날짜를 가져와 사용
for idx, row in df.iterrows():
    # 현재 행의 날짜를 가져옴 (Date 컬럼이 있으면 사용, 없으면 인덱스 사용)
    current_date = row.get('Date', idx)
    if pd.notna(current_date) and hasattr(current_date, 'strftime'):
        current_date = current_date.strftime('%Y-%m-%d')

    # 하회 감지 시 current_date 저장 (기존: idx)
    if prev_position == 'above' and current_position == 'below':
        last_break_below = current_date  # idx 대신 current_date 사용
```

**결과**: 하회일이 "2024-10-18", "2024-10-20" 같은 표준 날짜 형식으로 표시됨

### Python 32비트 → 64비트 마이그레이션
**문제**: pandas 2.x가 32비트 Python 미지원
```
ERROR: Could not parse vswhere.exe output
```

**해결**:
1. Python 3.12 32비트 제거
2. Python 3.12 64비트 설치 (winget)
3. 패키지 재설치

### PowerShell 실행 오류
**문제**: 따옴표 없이 경로 실행 시 파싱 오류

**해결**:
```powershell
# 잘못된 방법
"C:\...\python.exe" start.py  # 오류!

# 올바른 방법
& "C:\...\python.exe" start.py  # OK
```

또는 `run.bat` 사용

### Bloomberg Terminal 연결 확인
**테스트 명령어**:
```bash
py -3.12 -c "from xbbg import blp; print(blp.bdp('AAPL US Equity', 'PX_LAST'))"
```

**예상 출력**:
```
                px_last
AAPL US Equity   262.36
```

**연결 실패 시 확인사항**:
1. Bloomberg Terminal 실행 여부
2. Bloomberg 로그인 상태
3. 네트워크 연결

### 16. Bloomberg 티커 → 한글 종목명 자동 변환 (2026-01-08)

**요구사항**:
> "000990 KS, 007340 KS 이런 티커들을 한글 종목명으로 바꾸는 거거든 혹시 이걸 csv 없이 바로 블룸버그 티커에서 종목명으로 바꿀 수 있는 방법이 있을까?"

**구현 내용**:

#### 1. Bloomberg API 종목명 조회 함수 추가 (`src/bloomberg.py`)

```python
def get_security_name(ticker: str) -> str:
    """
    Bloomberg 티커에서 종목명 조회

    Parameters:
    -----------
    ticker : str
        Bloomberg 티커 (예: "005930 KS", "AAPL US")

    Returns:
    --------
    str : 종목명 (예: "Samsung Electronics Co Ltd", "삼성전자")
    """
    from xbbg import blp

    # 티커 형식 조정 (Equity 접미사 자동 추가)
    if not ticker.upper().endswith((' EQUITY', ' INDEX', ' CURNCY', ' COMDTY')):
        ticker = ticker + ' Equity'

    # NAME 필드 조회
    result = blp.bdp(tickers=ticker, flds='NAME')

    if result is not None and not result.empty:
        name = result.iloc[0]['NAME']
        if pd.notna(name):
            return str(name)

    return ticker.replace(' Equity', '')

def get_multiple_security_names(tickers: List[str]) -> dict:
    """
    여러 티커의 종목명을 한번에 조회

    Parameters:
    -----------
    tickers : List[str]
        Bloomberg 티커 리스트

    Returns:
    --------
    dict : {티커: 종목명} 딕셔너리
    """
    from xbbg import blp

    # 티커 형식 조정
    formatted_tickers = []
    for ticker in tickers:
        if not ticker.upper().endswith((' EQUITY', ' INDEX', ' CURNCY', ' COMDTY')):
            formatted_tickers.append(ticker + ' Equity')
        else:
            formatted_tickers.append(ticker)

    # 한번에 조회 (효율적)
    result = blp.bdp(tickers=formatted_tickers, flds='NAME')

    # 티커별 매핑
    result_dict = {}
    for i, ticker in enumerate(tickers):
        formatted_ticker = formatted_tickers[i]
        if formatted_ticker in result.index:
            name = result.loc[formatted_ticker, 'NAME']
            if pd.notna(name):
                result_dict[ticker] = str(name)
            else:
                result_dict[ticker] = ticker
        else:
            result_dict[ticker] = ticker

    return result_dict
```

#### 2. start_bloomberg.py 통합

**티커 입력 후 종목명 조회**:
```python
# 티커 리스트 파싱
tickers = [t.strip() for t in user_input.split(',')]
print(f"\n입력된 티커: {tickers}")

# 종목명 조회 (Bloomberg API)
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
```

**요약 테이블에 종목명 표시**:
```python
# 종목명 가져오기
ticker = row['ticker']
security_name = ticker_names.get(ticker, ticker)

summary_data.append({
    '종목': security_name,  # 티커 대신 종목명 표시
    '현재가': f"{row['close_price']:.0f}",
    '전일비': price_change_str,
    '10일선': f"{row['ma10']:.0f}",
    '괴리율': f"{row['ma_distance_percent']:+.1f}%",
    '추세': row.get('trend_direction', '-'),
    'RVOL': f"{row['rvol']:.1f}배",
    '신호': row['signal']
})
```

**추세별/조건별 분류에 종목명 표시**:
```python
# 하락세 종목 출력
for _, stock in falling_stocks.iterrows():
    ticker = stock['ticker']
    security_name = ticker_names.get(ticker, ticker)
    trend_info = stock['trend_detail']
    rvol_info = f"RVOL {stock['rvol']:.1f}배"
    if stock['condition_2_volume_confirmation']:
        rvol_info += " [거래량 폭증!]"
    print(f"  - {security_name} ({ticker}): {trend_info}, {rvol_info}")
```

**장점**:
- ✅ **CSV 파일 불필요**: Bloomberg API에서 직접 조회
- ✅ **한번에 조회**: `get_multiple_security_names()`로 배치 처리
- ✅ **에러 처리**: 조회 실패 시 티커 그대로 표시
- ✅ **한글 종목명 지원**: Bloomberg NAME 필드에서 자동 획득
- ✅ **가독성 향상**: 티커 대신 실제 종목명 표시

**예시 출력**:
```
[종목명 조회 중...]
✓ 종목명 조회 완료

종목 정보:
  - 000990 KS: DB하이텍
  - 007340 KS: DN솔루션즈
  - 005930 KS: Samsung Electronics Co Ltd

================================================================================
분석 결과 요약
================================================================================

종목              현재가    전일비   10일선    괴리율   추세    RVOL    신호
--------------  -------  -------  -------  -------  -----  ------  ------
DB하이텍          48500    +2.1%    47800    +1.5%   상승세  2.3배   HOLD
DN솔루션즈        62000    -0.8%    63200    -1.9%   하락세  2.8배   SELL
Samsung...        75600    +1.2%    74300    +1.8%   상승세  1.5배   HOLD
```

## 참고 문서

- [README.md](README.md) - 사용자 가이드
- [MA_CALCULATION_VERIFICATION.md](MA_CALCULATION_VERIFICATION.md) - MA 계산 검증
- [src/screener.py](src/screener.py) - 핵심 계산 로직
- [src/analyzer.py](src/analyzer.py) - 분석 및 리포트 생성
- [src/bloomberg.py](src/bloomberg.py) - Bloomberg API 연동

## 연락처

프로젝트 관련 문의:
- 이지스자산운용
- 문서 경로: `C:\Users\10845\OneDrive - 이지스자산운용\문서`
