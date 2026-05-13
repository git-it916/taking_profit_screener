@echo off
chcp 65001 >nul
cd /d "C:\Users\Bloomberg\Documents\ssh_project\taking_profit_screener"

echo [%date% %time%] whole-stock.py 시작
(echo 1& echo A) | py -3.12 whole-stock.py

echo [%date% %time%] under_20w.py 시작
echo 1 | py -3.12 under_20w.py

echo [%date% %time%] 완료
