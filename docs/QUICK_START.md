# Taking Profit Screener - 빠른 시작 가이드

## 1. 설치

```bash
# Python 라이브러리 설치
pip install -r requirements.txt

# Bloomberg 사용시 추가 설치
pip install xbbg
```

## 2. 실행 방법

### 방법 1: 통합 실행 (권장)

```bash
python start.py
```

**전체 워크플로우**:
```
[1단계] 데이터 소스 선택
  1: Bloomberg Terminal (실시간)
  2: 로컬 파일

[2단계] 입력 방법 선택 (Bloomberg 선택시)
  1: 한글 종목명 입력
  2: Bloomberg 티커 직접 입력
  3: 파일에서 읽기

[3단계] 데이터 기간 선택
  1: 1년
  2: 6개월
  3: 3개월

[4단계] 분석 및 결과 출력
```

### 방법 2: Bloomberg 전용

```bash
python start_bloomberg.py
```

티커를 직접 입력하는 간소화된 버전입니다.

## 3. 사용 시나리오

### 시나리오 A: 한글 종목명으로 300개 분석

```bash
python start.py

# 1단계
1 입력  # Bloomberg 선택

# 2단계
3 입력  # 파일에서 읽기

# 파일 경로
종목리스트.xlsx 입력

# 기간
엔터 (1년)

# → 자동 변환 및 분석 시작
```

**종목리스트.xlsx 예시**:
```
종목명
삼성전자
SK하이닉스
LG에너지솔루션
네이버
카카오
...
(300개)
```

### 시나리오 B: 소수 종목 빠른 분석

```bash
python start.py

# 1단계
1 입력  # Bloomberg 선택

# 2단계
1 입력  # 한글 종목명 입력

# 종목명
삼성전자, SK하이닉스, LG에너지솔루션, 네이버, 카카오

# 기간
엔터

# → 즉시 분석
```

### 시나리오 C: 로컬 파일 분석

```bash
python start.py

# 1단계
2 입력  # 로컬 파일

# 파일명
LGENSOL.xlsx

# → 단일 종목 상세 분석
```

## 4. 출력 결과

### 요약 테이블

```
종목          현재가  전일비  20일선  괴리율    하회일      RVOL  신호
005930 KS     71500   -2.4%   73200   -2.3%  2025-12-20  2.3배  SELL
000660 KS     85000   +0.3%   86500   -1.7%  2025-12-20  1.1배  HOLD
373220 KS    370000   +1.0%  375000   -1.3%  없음        1.0배  HOLD
```

### 조건별 분류

```
[강력 매도 신호] 15개 종목:
  - 005930 KS: 20일선 하회 + 거래량 2.3배

[주의 필요] 23개 종목 (20일선 하회, 거래량 부족):
  - 000660 KS: 20일선 대비 -1.73%, 하회일: 2025-12-20, RVOL 1.1배

[거래량 폭증만] 8개 종목 (20일선 위):
  - 035420 KS: RVOL 2.5배 (강함)
```

### CSV 저장

```
전체_분석결과.csv 생성
→ 엑셀에서 열어서 정렬/필터링 가능
```

## 5. 추가 종목 등록

### 방법 1: ticker_mapping.xlsx 사용

1. `ticker_mapping_template.xlsx` 복사
2. 종목명과 티커 추가:
   ```
   종목명          티커
   에코프로       086520 KS
   포스코퓨처엠   003670 KS
   ```
3. `ticker_mapping.xlsx`로 저장
4. start.py 실행시 매핑 파일 경로 입력

### 방법 2: 코드에 직접 추가

`src/ticker_converter.py` 파일의 `KOREAN_STOCK_MAP` 딕셔너리에 추가:

```python
KOREAN_STOCK_MAP = {
    '삼성전자': '005930 KS',
    'SK하이닉스': '000660 KS',
    # 여기에 추가
    '에코프로': '086520 KS',
    '포스코퓨처엠': '003670 KS',
}
```

## 6. 등록된 종목 확인

```bash
cd utils
python convert_tickers.py

# 옵션 3 선택
3

# → 등록된 50개 종목 목록 출력
```

## 7. Troubleshooting

### Bloomberg 연결 안됨

**증상**: `No data available` 또는 connection error

**해결**:
1. Bloomberg Terminal이 실행 중인지 확인
2. Bloomberg에 로그인되어 있는지 확인
3. xbbg 설치 확인: `pip install xbbg`

### 한글 종목명 변환 실패

**증상**: `변환 실패 종목 (N개)`

**해결**:
1. 종목명 철자 확인 (띄어쓰기 주의)
2. ticker_mapping.xlsx에 추가
3. 또는 Bloomberg 티커로 직접 입력

### 시간봉 데이터 문제

**증상**: Chart와 분석 결과가 다름

**해결**:
- 자동으로 일봉 변환됨 (convert_to_daily 함수)
- 변환 확인: 파일 읽은 후 행 수 확인

## 8. 파일 구조

```
taking_profit_screener/
├── start.py              ← 통합 실행 파일 (여기 실행!)
├── start_bloomberg.py    ← Bloomberg 전용
├── run.bat              ← Windows 더블클릭 실행
│
├── src/                 ← 핵심 로직 (수정 금지)
│   ├── bloomberg.py
│   ├── ticker_converter.py
│   ├── screener.py
│   └── analyzer.py
│
├── utils/               ← 유틸리티
│   └── convert_tickers.py  ← 종목명 변환 전용 도구
│
├── docs/                ← 문서
│   ├── README.md
│   ├── BLOOMBERG_GUIDE.md
│   └── TICKER_CONVERSION_GUIDE.md
│
└── ticker_mapping_template.xlsx  ← 추가 종목 템플릿
```

## 9. 자주 묻는 질문

**Q: 300개 종목을 한 번에 분석하는데 얼마나 걸리나요?**

A: Bloomberg API 속도에 따라 다르지만, 보통 5-10분 정도 소요됩니다.

**Q: 로컬 파일은 어떤 형식이어야 하나요?**

A: XLSX 또는 CSV 파일, 컬럼: Date, Open, High, Low, Close, Volume

**Q: 20일선 하회일이 12월 30일로 나오는데 정확한가요?**

A: 맞습니다. 시간봉 데이터가 자동으로 일봉으로 변환되어 계산됩니다. 검증은 `utils/debug_crossover.py` 참고.

**Q: SELL 신호가 너무 많이 나오는데요?**

A: 조건 (20일선 하회 + RVOL ≥ 2.0)이 모두 충족된 종목만 SELL입니다. "주의 필요" 섹션과 구분하여 확인하세요.

**Q: 한글 종목명이 등록되어 있지 않으면?**

A: 3가지 방법:
1. ticker_mapping.xlsx에 추가
2. src/ticker_converter.py의 KOREAN_STOCK_MAP에 추가
3. Bloomberg 티커로 직접 입력

## 10. 다음 단계

- 상세 사용법: [docs/README.md](README.md)
- Bloomberg 가이드: [docs/BLOOMBERG_GUIDE.md](BLOOMBERG_GUIDE.md)
- 티커 변환 가이드: [docs/TICKER_CONVERSION_GUIDE.md](TICKER_CONVERSION_GUIDE.md)
