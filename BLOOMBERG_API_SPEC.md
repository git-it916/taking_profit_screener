# Bloomberg API 데이터 크롤링 스펙 (Claude Code용 프롬프트)

아래 스펙을 기반으로 Bloomberg API 데이터 크롤링 코드를 작성해주세요.

---

## 1. 환경 설정

### 필수 조건
- Bloomberg Terminal이 실행 중이어야 함
- Bloomberg에 로그인 상태여야 함
- Python 3.12 환경

### 필수 라이브러리
```python
pip install xbbg pandas openpyxl
```

### 기본 import
```python
from xbbg import blp
import pandas as pd
from datetime import datetime, timedelta
```

---

## 2. Bloomberg API 함수

### 2.1 과거 데이터 조회 (Historical Data) - `blp.bdh()`

일봉/주봉/월봉 등 시계열 데이터 조회에 사용

```python
df = blp.bdh(
    tickers='005930 KS Equity',      # 티커 (Equity 접미사 필수)
    flds=['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME'],  # 필드
    start_date='2024-01-01',         # 시작일 (YYYY-MM-DD)
    end_date='2024-12-31'            # 종료일 (YYYY-MM-DD)
)
```

**주요 OHLCV 필드:**
| 필드명 | 설명 |
|--------|------|
| PX_OPEN | 시가 |
| PX_HIGH | 고가 |
| PX_LOW | 저가 |
| PX_LAST | 종가 |
| PX_VOLUME | 거래량 |

**반환값:** MultiIndex DataFrame (ticker, field)
- 단일 티커일 경우 평탄화 필요:
```python
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(1)
```

---

### 2.2 현재 시점 데이터 조회 (Point-in-Time) - `blp.bdp()`

현재가, 종목명, 시가총액 등 단일 시점 데이터 조회에 사용

```python
result = blp.bdp(
    tickers='005930 KS Equity',    # 티커 (단일 또는 리스트)
    flds='CUR_MKT_CAP'             # 필드 (단일 또는 리스트)
)
```

**주요 필드:**
| 필드명 | 설명 |
|--------|------|
| NAME | 종목명 (영문) |
| CUR_MKT_CAP | 시가총액 |
| PX_LAST | 현재가/최근 종가 |
| VOLUME_AVG_10D | 10일 평균 거래량 |
| EQY_SH_OUT | 발행주식수 |
| DVD_YIELD | 배당수익률 |

---

## 3. 티커 형식

### 3.1 기본 형식
```
[종목코드] [거래소코드] Equity
```

### 3.2 국가별 거래소 코드
| 국가 | 거래소 | 코드 | 예시 |
|------|--------|------|------|
| 한국 | 유가증권 (KOSPI) | KS | 005930 KS |
| 한국 | 코스닥 (KOSDAQ) | KQ | 035720 KQ |
| 미국 | NYSE/NASDAQ | US | AAPL US |
| 일본 | 도쿄 | JP | 7203 JP |
| 중국 | 상해 | CH | 600519 CH |
| 홍콩 | 홍콩 | HK | 700 HK |

### 3.3 ETF 티커
- 한국 ETF도 동일한 형식 사용
- 예시: `069500 KS Equity` (KODEX 200)

### 3.4 티커 변환 함수
```python
def format_ticker(ticker: str) -> str:
    """티커에 Equity 접미사 추가"""
    if not ticker.upper().endswith((' EQUITY', ' INDEX', ' CURNCY', ' COMDTY')):
        return ticker + ' Equity'
    return ticker
```

---

## 4. 데이터 크롤링 코드 템플릿

### 4.1 단일 종목 OHLCV 다운로드
```python
from xbbg import blp
import pandas as pd
from datetime import datetime, timedelta

def download_ohlcv(ticker: str, period_days: int = 365) -> pd.DataFrame:
    """
    Bloomberg에서 OHLCV 데이터 다운로드

    Parameters:
    -----------
    ticker : str
        Bloomberg 티커 (예: "005930 KS")
    period_days : int
        조회 기간 (일수)

    Returns:
    --------
    pd.DataFrame : OHLCV 데이터
    """
    # 날짜 설정
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')

    # 티커 형식 조정
    if not ticker.upper().endswith((' EQUITY', ' INDEX', ' CURNCY', ' COMDTY')):
        ticker = ticker + ' Equity'

    # 데이터 다운로드
    df = blp.bdh(
        tickers=ticker,
        flds=['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME'],
        start_date=start_date,
        end_date=end_date
    )

    # 컬럼 평탄화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)

    # 인덱스를 Date 컬럼으로
    df = df.reset_index()

    # 컬럼명 변경
    df = df.rename(columns={
        'index': 'Date',
        'PX_OPEN': 'Open',
        'PX_HIGH': 'High',
        'PX_LOW': 'Low',
        'PX_LAST': 'Close',
        'PX_VOLUME': 'Volume'
    })

    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
```

### 4.2 여러 종목 한번에 조회
```python
def download_multiple_ohlcv(tickers: list, period_days: int = 90) -> dict:
    """여러 종목 OHLCV 다운로드"""
    results = {}

    for ticker in tickers:
        try:
            df = download_ohlcv(ticker, period_days)
            results[ticker] = df
            print(f"✓ {ticker}: {len(df)}개 데이터")
        except Exception as e:
            print(f"✗ {ticker}: 실패 - {e}")
            results[ticker] = None

    return results
```

### 4.3 시가총액 조회
```python
def get_market_caps(tickers: list) -> dict:
    """여러 종목 시가총액 조회"""
    formatted = [t + ' Equity' if not t.upper().endswith(' EQUITY') else t
                 for t in tickers]

    result = blp.bdp(tickers=formatted, flds='CUR_MKT_CAP')

    caps = {}
    for orig, fmt in zip(tickers, formatted):
        try:
            caps[orig] = result.loc[fmt, 'cur_mkt_cap']
        except:
            caps[orig] = None

    return caps
```

### 4.4 종목명 조회
```python
def get_names(tickers: list) -> dict:
    """여러 종목명 조회"""
    formatted = [t + ' Equity' if not t.upper().endswith(' EQUITY') else t
                 for t in tickers]

    result = blp.bdp(tickers=formatted, flds='NAME')

    names = {}
    for orig, fmt in zip(tickers, formatted):
        try:
            names[orig] = result.loc[fmt, 'name']
        except:
            names[orig] = orig

    return names
```

---

## 5. 엑셀 저장

```python
def save_to_excel(data_dict: dict, filename: str):
    """여러 종목 데이터를 엑셀로 저장 (시트별)"""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for ticker, df in data_dict.items():
            if df is not None and not df.empty:
                sheet_name = ticker.replace(' ', '_')[:31]  # 시트명 31자 제한
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✓ 저장 완료: {filename}")
```

---

## 6. 사용 예시

### 6.1 기본 사용
```python
# 삼성전자 1년 데이터
df = download_ohlcv("005930 KS", period_days=365)
print(df)

# 엑셀로 저장
df.to_excel("삼성전자_데이터.xlsx", index=False)
```

### 6.2 여러 종목 크롤링
```python
tickers = ["005930 KS", "000660 KS", "373220 KS", "035420 KS"]

# OHLCV 다운로드
data = download_multiple_ohlcv(tickers, period_days=90)

# 시가총액 조회
caps = get_market_caps(tickers)

# 종목명 조회
names = get_names(tickers)

# 엑셀 저장
save_to_excel(data, "종목_데이터.xlsx")
```

### 6.3 엑셀에서 티커 로드 후 크롤링
```python
# 엑셀에서 티커 리스트 로드
ticker_df = pd.read_excel("ticker_list.xlsx")
tickers = ticker_df['ticker'].tolist()

# 크롤링 실행
data = download_multiple_ohlcv(tickers, period_days=90)
save_to_excel(data, "크롤링_결과.xlsx")
```

---

## 7. 주의사항

1. **Bloomberg Terminal 필수**: API는 터미널 실행 중에만 작동
2. **속도 제한**: 대량 요청 시 딜레이 권장 (`time.sleep(0.5)`)
3. **티커 형식**: 반드시 `Equity` 접미사 포함
4. **에러 처리**: 네트워크/데이터 오류에 대비한 try-except 필수
5. **데이터 검증**: NaN, 빈 데이터 체크 필요

---

## 8. 요청 예시 프롬프트

아래와 같이 Claude Code에 요청하면 됩니다:

```
Bloomberg API를 사용하여 다음 작업을 수행하는 Python 코드를 작성해주세요:

1. 입력: 티커 리스트 (예: ["005930 KS", "000660 KS"])
2. 수집 데이터: OHLCV (시가, 고가, 저가, 종가, 거래량)
3. 기간: 최근 3개월
4. 출력: 엑셀 파일 (종목별 시트)

위 BLOOMBERG_API_SPEC.md의 코드 템플릿을 참고하여 작성해주세요.
라이브러리: xbbg, pandas, openpyxl
```
