#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Whole Stock 2 - 전종목 수급 스코어링

전종목_수급.xlsx의 Sheet1에 들어있는 종가, 거래대금, 기관/외국인 수급
데이터를 읽어 스코어링 보고서를 생성합니다.

실행:
    py -3.12 whole-stock-2.py
"""
from score_all_supply import main


if __name__ == "__main__":
    main()
