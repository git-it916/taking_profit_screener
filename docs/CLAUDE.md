# CLAUDE.md

이 문서는 Taking Profit Screener 프로젝트의 개발 과정과 사용자 요구사항을 기록합니다.

## 프로젝트 개요

**Taking Profit Screener**는 XLSX/CSV 주가 데이터를 분석하여 20일 이동평균선과 상대 거래량(RVOL)을 기반으로 매도 시점을 포착하는 스크리닝 도구입니다.

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

## 개발 히스토리

### 2026-01-07
- ✅ 윗꼬리/아래꼬리 분석 완전 제거
- ✅ 3개 조건 → 2개 조건으로 전략 단순화
- ✅ 20일선 가격 명시적 표시
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

## 참고 문서

- [README.md](README.md) - 사용자 가이드
- [MA_CALCULATION_VERIFICATION.md](MA_CALCULATION_VERIFICATION.md) - MA 계산 검증
- [src/screener.py](src/screener.py) - 핵심 계산 로직
- [src/analyzer.py](src/analyzer.py) - 분석 및 리포트 생성

## 연락처

프로젝트 관련 문의:
- 이지스자산운용
- 문서 경로: `C:\Users\10845\OneDrive - 이지스자산운용\문서`
