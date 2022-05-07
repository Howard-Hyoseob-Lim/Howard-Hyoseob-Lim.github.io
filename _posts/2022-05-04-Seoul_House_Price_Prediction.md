# 서울 아파트 가격 예측 모델링 Project

## (1) 필요한 라이브러리 Import 


```python
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

import os
import datetime
from datetime import datetime, date

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', None)

import matplotlib.pyplot as plt
%matplotlib inline

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/SeoulNamsanB.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
```

    2.1.0
    

## (2) CSV 데이터 READ 및 데이터 전처리


```python
#Macro 거시데이터 리드
korea_gdp_rate = pd.read_csv("./data/Marco/01_korea_gdp_series.csv",sep=",",encoding="UTF-8")
korea_interest_rate = pd.read_csv("./data/Marco/02_korea_interest_rate.csv",sep=",",encoding="UTF-8")
korea_personal_loan = pd.read_csv("./data/Marco/03_korea_personal_loan.csv",sep=",",encoding="UTF-8")
korea_loan_for_house = pd.read_csv("./data/Marco/04_korea_loan_for_house.csv",sep=",",encoding="UTF-8")
korea_personal_GDP = pd.read_csv("./data/Marco/05_korea_personal_GDP.csv",sep=",",encoding="UTF-8")
seoul_gdp_series = pd.read_csv("./data/Marco/06_seoul_gdp_series.csv",sep=",",encoding="UTF-8")
housing_count_yearly = pd.read_csv("./data/Marco/07_housing_count_yearly.csv",sep=",",encoding="UTF-8")
seoul_population= pd.read_csv("./data/Marco/08_seoul_population.csv",sep=",",encoding="UTF-8")
constructure_confirm= pd.read_csv("./data/Marco/09_constructure_confirm.csv",sep=",",encoding="UTF-8")

#2011~2020년의 필요한 연도 데이터 추출
korea_gdp_rate = korea_gdp_rate.loc[41:]
korea_interest_rate = korea_interest_rate.loc[12:]
korea_personal_loan = korea_personal_loan.loc[9:]
korea_loan_for_house = korea_loan_for_house.loc[:9]
korea_personal_GDP = korea_personal_GDP[1:]
seoul_gdp_series
housing_count_yearly
seoul_population
constructure_confirm
```




       year      합계     수도권      서울
    0  2011  549594  272156   88060
    1  2012  586884  269290   86123
    2  2013  440116  192610   77621
    3  2014  515251  241889   65249
    4  2015  765328  408773  101235
    5  2016  726048  341162   74739
    6  2017  653441  321402  113131
    7  2018  554136  280097   65751
    8  2019  487975  272226   62272
    9  2020  457514  252301   58181



(File) korea_gdp_series
1. 대한민국 명목 GDP 데이터
통계표명:	국내총생산 및 경제성장률	
단위:	십억원, 전년동기비 %	
출처:	한국은행「국민소득」												
주석:	* 국민총생산(명목, 시장가격)												
	* 실질GDP, 실질성장률은 발표시기(한국은행, GDP속보치 발표)와 명목GDP, 명목소득 증감률의 발표시기(한국은행, GDP잠정치 발표)가 차이가 있어 국내총생산(명목GDP)과 경제성장률(실질성장률) 업데이트 시기가 다름												

(File) korea_interest_rate
2. 대한민국 금리 추이 데이터
통계표명:	시장금리 추이
단위:	기간중 평균금리, %
출처:	한국은행 경제통계시스템 &gt; 4. 금리
주석:	* 콜금리 목표는 월말 기준이며,국고채10년은 00.11월부터, 콜금리목표는 99.4월부터임.

(File) korea_personal_loan
3. 가계대출정보데이터
통계표명:	가계신용 동향
단위:	조원, %
출처:	한국은행「가계신용동향」

(File) korea_loan_for_house
4. 주택금융신용보증기금
통계표명 : 주택금융신용보증기금 동향
단위 :     억원, %
출처 : 주택금융통계시스템

한국주택금융공사법 제56조와 시행규칙 제3조에 의거하여 주택금융신용보증기금에 출연하는 금융기관의 주택자금대출금 
규모이며, 한국주택금융공사법 시행규칙 개정(제3조, 2007.7.1 시행)으로 출연대상 주택자금대출의 정의 및 범위가 변경 적용됨

※ 주택금융신용보증기금에 출연한 금융기관 (전분기말 기준)
  - 시중은행 : 국민, 신한, 우리, KEB하나, 한국씨티, 한국스탠다드차타드은행, 카카오뱅크, 케이뱅크
  - 지방은행 : 경남, 광주, 대구, 부산, 전북, 제주은행
  - 특수은행 : 농협, 수협, 기업, 산업은행
  - 외은지점 : 미쓰비시 도쿄UFJ, 중국, 중국농업, 중국광대, 중국공상, 파키스탄 국립, 홍콩상하이

(File) seoul_personal_gdp
5. 1인당 지역내 개인소득 데이터
통계표명 : 시도별 1인당 지역내 개인소득
단위 : 천원
출처 : 통계청(KOSIS)

(File) seoul_gdp_series
6. 서울시 지역 내 총생산 데이터
통계표명 : 서울시 지역 내 총생산 데이터
단위 : 백만원
출처 : 서울특별시

(File) housing_count_yearly
7. 서울시 공동주택 현황 통계
통계표명 : 서울시 공동주택 현황
단위 : 건
출처 : 서울특별시

(File) seoul_population
8. 서울시 인구 통계 정보
통계표명 : 서울시 인구 통계 정보
단위 : 명
출처 : 서울특별시

(File) constructure_confirm
9. 주택건설인허가실적
통계표명 : 주택건설인허가실적
단위 : 호
출처 : 한국주택협회




```python
#아파트 거래 데이터 추출
tr_2011 = pd.read_csv("./data\/Transaction/transaction_data_2011.csv",sep=",",encoding="UTF-8")
tr_2012 = pd.read_csv("./data\/Transaction/transaction_data_2012.csv",sep=",",encoding="UTF-8")
tr_2013 = pd.read_csv("./data\/Transaction/transaction_data_2013.csv",sep=",",encoding="UTF-8")
tr_2014 = pd.read_csv("./data\/Transaction/transaction_data_2014.csv",sep=",",encoding="UTF-8")
tr_2015 = pd.read_csv("./data\/Transaction/transaction_data_2015.csv",sep=",",encoding="UTF-8")
tr_2016 = pd.read_csv("./data\/Transaction/transaction_data_2016.csv",sep=",",encoding="UTF-8")
tr_2017 = pd.read_csv("./data\/Transaction/transaction_data_2017.csv",sep=",",encoding="UTF-8")
tr_2018 = pd.read_csv("./data\/Transaction/transaction_data_2018.csv",sep=",",encoding="UTF-8")
tr_2019 = pd.read_csv("./data\/Transaction/transaction_data_2019.csv",sep=",",encoding="UTF-8")
tr_2020 = pd.read_csv("./data\/Transaction/transaction_data_2020.csv",sep=",",encoding="UTF-8")

```

□ 본 서비스에서 제공하는 정보는 법적인 효력이 없으므로 참고용으로만 활용하시기 바랍니다.	
□ 신고정보가 실시간 변경, 해제되어 제공시점에 따라 공개건수 및 내용이 상이할 수 있는 점 참고하시기 바랍니다.	
□ 본 자료는 계약일 기준입니다. (※ 7월 계약, 8월 신고건 → 7월 거래건으로  제공)	
□ 통계자료 활용시에는 수치가 왜곡될 수 있으니 참고자료로만 활용하시기  바라며,  외부 공개시에는 반드시 신고일 기준으로 집계되는 공식통계를 이용하여 주시기 바랍니다.	
	
* 국토교통부 실거래가 공개시스템의 궁금하신 점이나 문의사항은 콜센터 1588-0149로 연락 주시기 바랍니다.	
□ 검색조건	
계약일자 : 20110101 ~ 20211031	
실거래구분 : 아파트(매매)	
주소구분 : 지번주소	
시도 : 서울특별시	
시군구 : 전체	
읍면동 : 전체	
면적 : 전체	
금액선택 : 전체	


# (3) Data Preprocessing


```python
#2011~2020년의 아파트 거래 데이터 통합
tr_concat = pd.concat([tr_2011,tr_2012,tr_2013,tr_2014,tr_2015,tr_2016,tr_2017,tr_2018,tr_2019,tr_2020], ignore_index=True)
```


```python
tr_concat.columns
```




    Index(['시군구', '번지', '본번', '부번', '단지명', '전용면적(㎡)', '계약년월', '계약일', '거래금액(만원)',
           '층', '건축년도', '도로명', '해제사유발생일', '거래유형', '중개사소재지'],
          dtype='object')




```python
#날짜 형식 변경
year = tr_concat["계약년월"].astype(str).str[:4]
month = tr_concat["계약년월"].astype(str).str[4:6]
day = tr_concat["계약일"].astype(str)

#날짜형식 맞추기 위해 1자리수 일자에 0넣기
for i in range(len(day)):
    if len(day.iloc[i])==1:
        day.iloc[i] = "0"+ day.iloc[i]
        
year_month = year.str.cat(month,sep="-")
year_month_day = year_month.str.cat(day,sep="-")

year_month_day
```




    0         2011-07-09
    1         2011-07-28
    2         2011-01-19
    3         2011-09-02
    4         2011-12-17
                 ...    
    826104    2020-08-07
    826105    2020-07-10
    826106    2020-12-03
    826107    2020-09-28
    826108    2020-09-28
    Name: 계약년월, Length: 826109, dtype: object




```python
from datetime import datetime

format = "%Y-%m-%d"
dt_datetime=[]
for i in range(826109):
    dt_datetime.append(datetime.strptime(year_month_day[i],format))
    
dt_datetime
```




    [datetime.datetime(2011, 7, 9, 0, 0),
     datetime.datetime(2011, 7, 28, 0, 0),
     datetime.datetime(2011, 1, 19, 0, 0),
     datetime.datetime(2011, 9, 2, 0, 0),
     datetime.datetime(2011, 12, 17, 0, 0),
     datetime.datetime(2011, 3, 28, 0, 0),
     datetime.datetime(2011, 5, 30, 0, 0),
     datetime.datetime(2011, 7, 28, 0, 0),
     datetime.datetime(2011, 9, 16, 0, 0),
     datetime.datetime(2011, 2, 15, 0, 0),
     datetime.datetime(2011, 9, 27, 0, 0),
     datetime.datetime(2011, 1, 5, 0, 0),
     datetime.datetime(2011, 1, 12, 0, 0),
     datetime.datetime(2011, 1, 12, 0, 0),
     datetime.datetime(2011, 1, 14, 0, 0),
     datetime.datetime(2011, 1, 18, 0, 0),
     datetime.datetime(2011, 1, 19, 0, 0),
     datetime.datetime(2011, 1, 19, 0, 0),
     datetime.datetime(2011, 1, 20, 0, 0),
     datetime.datetime(2011, 1, 22, 0, 0),
     datetime.datetime(2011, 1, 22, 0, 0),
     datetime.datetime(2011, 1, 24, 0, 0),
     datetime.datetime(2011, 1, 24, 0, 0),
     datetime.datetime(2011, 1, 24, 0, 0),
     datetime.datetime(2011, 1, 25, 0, 0),
     datetime.datetime(2011, 1, 26, 0, 0),
     datetime.datetime(2011, 1, 26, 0, 0),
     datetime.datetime(2011, 1, 26, 0, 0),
     datetime.datetime(2011, 1, 27, 0, 0),
     datetime.datetime(2011, 1, 27, 0, 0),
     datetime.datetime(2011, 1, 27, 0, 0),
     datetime.datetime(2011, 1, 28, 0, 0),
     datetime.datetime(2011, 1, 29, 0, 0),
     datetime.datetime(2011, 1, 31, 0, 0),
     datetime.datetime(2011, 2, 1, 0, 0),
     datetime.datetime(2011, 2, 7, 0, 0),
     datetime.datetime(2011, 2, 10, 0, 0),
     datetime.datetime(2011, 2, 11, 0, 0),
     datetime.datetime(2011, 2, 12, 0, 0),
     datetime.datetime(2011, 2, 12, 0, 0),
     datetime.datetime(2011, 2, 12, 0, 0),
     datetime.datetime(2011, 2, 19, 0, 0),
     datetime.datetime(2011, 3, 2, 0, 0),
     datetime.datetime(2011, 3, 3, 0, 0),
     datetime.datetime(2011, 3, 8, 0, 0),
     datetime.datetime(2011, 3, 16, 0, 0),
     datetime.datetime(2011, 3, 21, 0, 0),
     datetime.datetime(2011, 3, 22, 0, 0),
     datetime.datetime(2011, 3, 24, 0, 0),
     datetime.datetime(2011, 3, 25, 0, 0),
     datetime.datetime(2011, 3, 25, 0, 0),
     datetime.datetime(2011, 3, 26, 0, 0),
     datetime.datetime(2011, 3, 26, 0, 0),
     datetime.datetime(2011, 3, 28, 0, 0),
     datetime.datetime(2011, 3, 29, 0, 0),
     datetime.datetime(2011, 3, 29, 0, 0),
     datetime.datetime(2011, 3, 29, 0, 0),
     datetime.datetime(2011, 4, 7, 0, 0),
     datetime.datetime(2011, 4, 9, 0, 0),
     datetime.datetime(2011, 4, 11, 0, 0),
     datetime.datetime(2011, 4, 12, 0, 0),
     datetime.datetime(2011, 4, 13, 0, 0),
     datetime.datetime(2011, 4, 13, 0, 0),
     datetime.datetime(2011, 4, 15, 0, 0),
     datetime.datetime(2011, 4, 19, 0, 0),
     datetime.datetime(2011, 4, 22, 0, 0),
     datetime.datetime(2011, 4, 25, 0, 0),
     datetime.datetime(2011, 4, 26, 0, 0),
     datetime.datetime(2011, 4, 28, 0, 0),
     datetime.datetime(2011, 5, 5, 0, 0),
     datetime.datetime(2011, 5, 9, 0, 0),
     datetime.datetime(2011, 5, 10, 0, 0),
     datetime.datetime(2011, 5, 16, 0, 0),
     datetime.datetime(2011, 5, 20, 0, 0),
     datetime.datetime(2011, 5, 20, 0, 0),
     datetime.datetime(2011, 5, 23, 0, 0),
     datetime.datetime(2011, 5, 23, 0, 0),
     datetime.datetime(2011, 6, 1, 0, 0),
     datetime.datetime(2011, 6, 3, 0, 0),
     datetime.datetime(2011, 6, 7, 0, 0),
     datetime.datetime(2011, 6, 13, 0, 0),
     datetime.datetime(2011, 6, 14, 0, 0),
     datetime.datetime(2011, 6, 14, 0, 0),
     datetime.datetime(2011, 6, 14, 0, 0),
     datetime.datetime(2011, 6, 20, 0, 0),
     datetime.datetime(2011, 6, 20, 0, 0),
     datetime.datetime(2011, 6, 20, 0, 0),
     datetime.datetime(2011, 6, 20, 0, 0),
     datetime.datetime(2011, 6, 21, 0, 0),
     datetime.datetime(2011, 6, 23, 0, 0),
     datetime.datetime(2011, 7, 4, 0, 0),
     datetime.datetime(2011, 7, 8, 0, 0),
     datetime.datetime(2011, 7, 9, 0, 0),
     datetime.datetime(2011, 7, 12, 0, 0),
     datetime.datetime(2011, 7, 13, 0, 0),
     datetime.datetime(2011, 7, 13, 0, 0),
     datetime.datetime(2011, 7, 13, 0, 0),
     datetime.datetime(2011, 7, 16, 0, 0),
     datetime.datetime(2011, 7, 19, 0, 0),
     datetime.datetime(2011, 7, 20, 0, 0),
     datetime.datetime(2011, 7, 20, 0, 0),
     datetime.datetime(2011, 7, 21, 0, 0),
     datetime.datetime(2011, 7, 22, 0, 0),
     datetime.datetime(2011, 7, 22, 0, 0),
     datetime.datetime(2011, 7, 25, 0, 0),
     datetime.datetime(2011, 7, 25, 0, 0),
     datetime.datetime(2011, 7, 25, 0, 0),
     datetime.datetime(2011, 7, 27, 0, 0),
     datetime.datetime(2011, 7, 28, 0, 0),
     datetime.datetime(2011, 7, 29, 0, 0),
     datetime.datetime(2011, 7, 30, 0, 0),
     datetime.datetime(2011, 7, 30, 0, 0),
     datetime.datetime(2011, 7, 30, 0, 0),
     datetime.datetime(2011, 7, 30, 0, 0),
     datetime.datetime(2011, 8, 9, 0, 0),
     datetime.datetime(2011, 8, 15, 0, 0),
     datetime.datetime(2011, 8, 18, 0, 0),
     datetime.datetime(2011, 8, 22, 0, 0),
     datetime.datetime(2011, 8, 23, 0, 0),
     datetime.datetime(2011, 8, 25, 0, 0),
     datetime.datetime(2011, 8, 27, 0, 0),
     datetime.datetime(2011, 8, 29, 0, 0),
     datetime.datetime(2011, 8, 30, 0, 0),
     datetime.datetime(2011, 9, 1, 0, 0),
     datetime.datetime(2011, 9, 1, 0, 0),
     datetime.datetime(2011, 9, 3, 0, 0),
     datetime.datetime(2011, 9, 3, 0, 0),
     datetime.datetime(2011, 9, 7, 0, 0),
     datetime.datetime(2011, 9, 7, 0, 0),
     datetime.datetime(2011, 9, 15, 0, 0),
     datetime.datetime(2011, 9, 16, 0, 0),
     datetime.datetime(2011, 9, 16, 0, 0),
     datetime.datetime(2011, 9, 16, 0, 0),
     datetime.datetime(2011, 9, 17, 0, 0),
     datetime.datetime(2011, 9, 20, 0, 0),
     datetime.datetime(2011, 9, 21, 0, 0),
     datetime.datetime(2011, 9, 21, 0, 0),
     datetime.datetime(2011, 9, 22, 0, 0),
     datetime.datetime(2011, 9, 24, 0, 0),
     datetime.datetime(2011, 9, 27, 0, 0),
     datetime.datetime(2011, 9, 27, 0, 0),
     datetime.datetime(2011, 9, 27, 0, 0),
     datetime.datetime(2011, 9, 28, 0, 0),
     datetime.datetime(2011, 10, 5, 0, 0),
     datetime.datetime(2011, 10, 5, 0, 0),
     datetime.datetime(2011, 10, 6, 0, 0),
     datetime.datetime(2011, 10, 7, 0, 0),
     datetime.datetime(2011, 10, 7, 0, 0),
     datetime.datetime(2011, 10, 7, 0, 0),
     datetime.datetime(2011, 10, 10, 0, 0),
     datetime.datetime(2011, 10, 11, 0, 0),
     datetime.datetime(2011, 10, 11, 0, 0),
     datetime.datetime(2011, 10, 11, 0, 0),
     datetime.datetime(2011, 10, 12, 0, 0),
     datetime.datetime(2011, 10, 12, 0, 0),
     datetime.datetime(2011, 10, 12, 0, 0),
     datetime.datetime(2011, 10, 12, 0, 0),
     datetime.datetime(2011, 10, 13, 0, 0),
     datetime.datetime(2011, 10, 14, 0, 0),
     datetime.datetime(2011, 10, 15, 0, 0),
     datetime.datetime(2011, 10, 15, 0, 0),
     datetime.datetime(2011, 10, 17, 0, 0),
     datetime.datetime(2011, 10, 17, 0, 0),
     datetime.datetime(2011, 10, 21, 0, 0),
     datetime.datetime(2011, 10, 21, 0, 0),
     datetime.datetime(2011, 10, 21, 0, 0),
     datetime.datetime(2011, 10, 22, 0, 0),
     datetime.datetime(2011, 10, 22, 0, 0),
     datetime.datetime(2011, 10, 28, 0, 0),
     datetime.datetime(2011, 11, 2, 0, 0),
     datetime.datetime(2011, 11, 5, 0, 0),
     datetime.datetime(2011, 11, 10, 0, 0),
     datetime.datetime(2011, 11, 11, 0, 0),
     datetime.datetime(2011, 11, 16, 0, 0),
     datetime.datetime(2011, 11, 17, 0, 0),
     datetime.datetime(2011, 11, 17, 0, 0),
     datetime.datetime(2011, 11, 18, 0, 0),
     datetime.datetime(2011, 11, 18, 0, 0),
     datetime.datetime(2011, 11, 18, 0, 0),
     datetime.datetime(2011, 11, 21, 0, 0),
     datetime.datetime(2011, 11, 22, 0, 0),
     datetime.datetime(2011, 11, 25, 0, 0),
     datetime.datetime(2011, 11, 25, 0, 0),
     datetime.datetime(2011, 12, 1, 0, 0),
     datetime.datetime(2011, 12, 1, 0, 0),
     datetime.datetime(2011, 12, 1, 0, 0),
     datetime.datetime(2011, 12, 1, 0, 0),
     datetime.datetime(2011, 12, 3, 0, 0),
     datetime.datetime(2011, 12, 3, 0, 0),
     datetime.datetime(2011, 12, 5, 0, 0),
     datetime.datetime(2011, 12, 6, 0, 0),
     datetime.datetime(2011, 12, 6, 0, 0),
     datetime.datetime(2011, 12, 7, 0, 0),
     datetime.datetime(2011, 12, 8, 0, 0),
     datetime.datetime(2011, 12, 8, 0, 0),
     datetime.datetime(2011, 12, 13, 0, 0),
     datetime.datetime(2011, 12, 13, 0, 0),
     datetime.datetime(2011, 12, 15, 0, 0),
     datetime.datetime(2011, 12, 19, 0, 0),
     datetime.datetime(2011, 12, 22, 0, 0),
     datetime.datetime(2011, 12, 23, 0, 0),
     datetime.datetime(2011, 12, 23, 0, 0),
     datetime.datetime(2011, 12, 24, 0, 0),
     datetime.datetime(2011, 12, 24, 0, 0),
     datetime.datetime(2011, 12, 27, 0, 0),
     datetime.datetime(2011, 12, 28, 0, 0),
     datetime.datetime(2011, 12, 29, 0, 0),
     datetime.datetime(2011, 1, 5, 0, 0),
     datetime.datetime(2011, 1, 14, 0, 0),
     datetime.datetime(2011, 1, 15, 0, 0),
     datetime.datetime(2011, 1, 21, 0, 0),
     datetime.datetime(2011, 2, 17, 0, 0),
     datetime.datetime(2011, 2, 19, 0, 0),
     datetime.datetime(2011, 3, 4, 0, 0),
     datetime.datetime(2011, 3, 24, 0, 0),
     datetime.datetime(2011, 3, 25, 0, 0),
     datetime.datetime(2011, 3, 25, 0, 0),
     datetime.datetime(2011, 4, 19, 0, 0),
     datetime.datetime(2011, 4, 26, 0, 0),
     datetime.datetime(2011, 5, 6, 0, 0),
     datetime.datetime(2011, 5, 10, 0, 0),
     datetime.datetime(2011, 6, 15, 0, 0),
     datetime.datetime(2011, 6, 17, 0, 0),
     datetime.datetime(2011, 7, 5, 0, 0),
     datetime.datetime(2011, 7, 6, 0, 0),
     datetime.datetime(2011, 7, 8, 0, 0),
     datetime.datetime(2011, 7, 21, 0, 0),
     datetime.datetime(2011, 7, 22, 0, 0),
     datetime.datetime(2011, 7, 26, 0, 0),
     datetime.datetime(2011, 7, 27, 0, 0),
     datetime.datetime(2011, 8, 13, 0, 0),
     datetime.datetime(2011, 8, 17, 0, 0),
     datetime.datetime(2011, 8, 23, 0, 0),
     datetime.datetime(2011, 8, 25, 0, 0),
     datetime.datetime(2011, 8, 30, 0, 0),
     datetime.datetime(2011, 9, 6, 0, 0),
     datetime.datetime(2011, 9, 23, 0, 0),
     datetime.datetime(2011, 10, 15, 0, 0),
     datetime.datetime(2011, 10, 17, 0, 0),
     datetime.datetime(2011, 10, 20, 0, 0),
     datetime.datetime(2011, 11, 7, 0, 0),
     datetime.datetime(2011, 11, 9, 0, 0),
     datetime.datetime(2011, 11, 23, 0, 0),
     datetime.datetime(2011, 11, 29, 0, 0),
     datetime.datetime(2011, 12, 3, 0, 0),
     datetime.datetime(2011, 12, 6, 0, 0),
     datetime.datetime(2011, 12, 9, 0, 0),
     datetime.datetime(2011, 12, 9, 0, 0),
     datetime.datetime(2011, 12, 9, 0, 0),
     datetime.datetime(2011, 12, 21, 0, 0),
     datetime.datetime(2011, 1, 10, 0, 0),
     datetime.datetime(2011, 1, 13, 0, 0),
     datetime.datetime(2011, 1, 24, 0, 0),
     datetime.datetime(2011, 1, 25, 0, 0),
     datetime.datetime(2011, 1, 25, 0, 0),
     datetime.datetime(2011, 1, 26, 0, 0),
     datetime.datetime(2011, 1, 27, 0, 0),
     datetime.datetime(2011, 1, 29, 0, 0),
     datetime.datetime(2011, 1, 31, 0, 0),
     datetime.datetime(2011, 2, 23, 0, 0),
     datetime.datetime(2011, 3, 7, 0, 0),
     datetime.datetime(2011, 3, 10, 0, 0),
     datetime.datetime(2011, 3, 12, 0, 0),
     datetime.datetime(2011, 3, 21, 0, 0),
     datetime.datetime(2011, 3, 21, 0, 0),
     datetime.datetime(2011, 3, 21, 0, 0),
     datetime.datetime(2011, 3, 24, 0, 0),
     datetime.datetime(2011, 3, 24, 0, 0),
     datetime.datetime(2011, 3, 24, 0, 0),
     datetime.datetime(2011, 3, 25, 0, 0),
     datetime.datetime(2011, 3, 30, 0, 0),
     datetime.datetime(2011, 4, 7, 0, 0),
     datetime.datetime(2011, 4, 21, 0, 0),
     datetime.datetime(2011, 4, 25, 0, 0),
     datetime.datetime(2011, 4, 26, 0, 0),
     datetime.datetime(2011, 4, 27, 0, 0),
     datetime.datetime(2011, 5, 2, 0, 0),
     datetime.datetime(2011, 5, 2, 0, 0),
     datetime.datetime(2011, 5, 3, 0, 0),
     datetime.datetime(2011, 5, 16, 0, 0),
     datetime.datetime(2011, 5, 20, 0, 0),
     datetime.datetime(2011, 5, 23, 0, 0),
     datetime.datetime(2011, 5, 26, 0, 0),
     datetime.datetime(2011, 6, 2, 0, 0),
     datetime.datetime(2011, 6, 3, 0, 0),
     datetime.datetime(2011, 6, 7, 0, 0),
     datetime.datetime(2011, 6, 8, 0, 0),
     datetime.datetime(2011, 6, 9, 0, 0),
     datetime.datetime(2011, 6, 10, 0, 0),
     datetime.datetime(2011, 6, 16, 0, 0),
     datetime.datetime(2011, 6, 21, 0, 0),
     datetime.datetime(2011, 7, 1, 0, 0),
     datetime.datetime(2011, 7, 2, 0, 0),
     datetime.datetime(2011, 7, 5, 0, 0),
     datetime.datetime(2011, 7, 5, 0, 0),
     datetime.datetime(2011, 7, 7, 0, 0),
     datetime.datetime(2011, 7, 7, 0, 0),
     datetime.datetime(2011, 7, 15, 0, 0),
     datetime.datetime(2011, 7, 18, 0, 0),
     datetime.datetime(2011, 7, 20, 0, 0),
     datetime.datetime(2011, 7, 21, 0, 0),
     datetime.datetime(2011, 7, 22, 0, 0),
     datetime.datetime(2011, 7, 22, 0, 0),
     datetime.datetime(2011, 7, 23, 0, 0),
     datetime.datetime(2011, 7, 26, 0, 0),
     datetime.datetime(2011, 7, 26, 0, 0),
     datetime.datetime(2011, 7, 29, 0, 0),
     datetime.datetime(2011, 8, 11, 0, 0),
     datetime.datetime(2011, 8, 15, 0, 0),
     datetime.datetime(2011, 8, 20, 0, 0),
     datetime.datetime(2011, 9, 17, 0, 0),
     datetime.datetime(2011, 9, 21, 0, 0),
     datetime.datetime(2011, 9, 21, 0, 0),
     datetime.datetime(2011, 9, 23, 0, 0),
     datetime.datetime(2011, 9, 27, 0, 0),
     datetime.datetime(2011, 9, 29, 0, 0),
     datetime.datetime(2011, 9, 30, 0, 0),
     datetime.datetime(2011, 10, 3, 0, 0),
     datetime.datetime(2011, 10, 7, 0, 0),
     datetime.datetime(2011, 10, 10, 0, 0),
     datetime.datetime(2011, 10, 10, 0, 0),
     datetime.datetime(2011, 10, 12, 0, 0),
     datetime.datetime(2011, 10, 12, 0, 0),
     datetime.datetime(2011, 10, 13, 0, 0),
     datetime.datetime(2011, 10, 13, 0, 0),
     datetime.datetime(2011, 10, 13, 0, 0),
     datetime.datetime(2011, 10, 13, 0, 0),
     datetime.datetime(2011, 10, 14, 0, 0),
     datetime.datetime(2011, 10, 19, 0, 0),
     datetime.datetime(2011, 10, 20, 0, 0),
     datetime.datetime(2011, 10, 21, 0, 0),
     datetime.datetime(2011, 10, 21, 0, 0),
     datetime.datetime(2011, 10, 21, 0, 0),
     datetime.datetime(2011, 10, 21, 0, 0),
     datetime.datetime(2011, 11, 3, 0, 0),
     datetime.datetime(2011, 11, 4, 0, 0),
     datetime.datetime(2011, 11, 7, 0, 0),
     datetime.datetime(2011, 11, 7, 0, 0),
     datetime.datetime(2011, 11, 9, 0, 0),
     datetime.datetime(2011, 11, 9, 0, 0),
     datetime.datetime(2011, 11, 10, 0, 0),
     datetime.datetime(2011, 11, 13, 0, 0),
     datetime.datetime(2011, 11, 14, 0, 0),
     datetime.datetime(2011, 11, 16, 0, 0),
     datetime.datetime(2011, 11, 22, 0, 0),
     datetime.datetime(2011, 11, 22, 0, 0),
     datetime.datetime(2011, 11, 24, 0, 0),
     datetime.datetime(2011, 11, 30, 0, 0),
     datetime.datetime(2011, 11, 30, 0, 0),
     datetime.datetime(2011, 12, 1, 0, 0),
     datetime.datetime(2011, 12, 1, 0, 0),
     datetime.datetime(2011, 12, 1, 0, 0),
     datetime.datetime(2011, 12, 1, 0, 0),
     datetime.datetime(2011, 12, 3, 0, 0),
     datetime.datetime(2011, 12, 5, 0, 0),
     datetime.datetime(2011, 12, 6, 0, 0),
     datetime.datetime(2011, 12, 6, 0, 0),
     datetime.datetime(2011, 12, 6, 0, 0),
     datetime.datetime(2011, 12, 7, 0, 0),
     datetime.datetime(2011, 12, 16, 0, 0),
     datetime.datetime(2011, 12, 16, 0, 0),
     datetime.datetime(2011, 12, 20, 0, 0),
     datetime.datetime(2011, 12, 21, 0, 0),
     datetime.datetime(2011, 12, 23, 0, 0),
     datetime.datetime(2011, 12, 26, 0, 0),
     datetime.datetime(2011, 12, 27, 0, 0),
     datetime.datetime(2011, 12, 28, 0, 0),
     datetime.datetime(2011, 12, 30, 0, 0),
     datetime.datetime(2011, 1, 10, 0, 0),
     datetime.datetime(2011, 1, 17, 0, 0),
     datetime.datetime(2011, 2, 8, 0, 0),
     datetime.datetime(2011, 2, 21, 0, 0),
     datetime.datetime(2011, 2, 27, 0, 0),
     datetime.datetime(2011, 5, 10, 0, 0),
     datetime.datetime(2011, 5, 25, 0, 0),
     datetime.datetime(2011, 6, 13, 0, 0),
     datetime.datetime(2011, 7, 2, 0, 0),
     datetime.datetime(2011, 7, 6, 0, 0),
     datetime.datetime(2011, 7, 22, 0, 0),
     datetime.datetime(2011, 7, 23, 0, 0),
     datetime.datetime(2011, 7, 26, 0, 0),
     datetime.datetime(2011, 8, 15, 0, 0),
     datetime.datetime(2011, 10, 1, 0, 0),
     datetime.datetime(2011, 10, 18, 0, 0),
     datetime.datetime(2011, 10, 25, 0, 0),
     datetime.datetime(2011, 10, 28, 0, 0),
     datetime.datetime(2011, 11, 7, 0, 0),
     datetime.datetime(2011, 12, 8, 0, 0),
     datetime.datetime(2011, 12, 8, 0, 0),
     datetime.datetime(2011, 12, 12, 0, 0),
     datetime.datetime(2011, 12, 15, 0, 0),
     datetime.datetime(2011, 12, 19, 0, 0),
     datetime.datetime(2011, 1, 5, 0, 0),
     datetime.datetime(2011, 1, 7, 0, 0),
     datetime.datetime(2011, 1, 10, 0, 0),
     datetime.datetime(2011, 1, 12, 0, 0),
     datetime.datetime(2011, 1, 18, 0, 0),
     datetime.datetime(2011, 2, 14, 0, 0),
     datetime.datetime(2011, 3, 2, 0, 0),
     datetime.datetime(2011, 5, 4, 0, 0),
     datetime.datetime(2011, 5, 4, 0, 0),
     datetime.datetime(2011, 5, 17, 0, 0),
     datetime.datetime(2011, 5, 21, 0, 0),
     datetime.datetime(2011, 6, 10, 0, 0),
     datetime.datetime(2011, 6, 12, 0, 0),
     datetime.datetime(2011, 7, 2, 0, 0),
     datetime.datetime(2011, 7, 8, 0, 0),
     datetime.datetime(2011, 7, 9, 0, 0),
     datetime.datetime(2011, 8, 1, 0, 0),
     datetime.datetime(2011, 8, 17, 0, 0),
     datetime.datetime(2011, 8, 20, 0, 0),
     datetime.datetime(2011, 9, 8, 0, 0),
     datetime.datetime(2011, 9, 24, 0, 0),
     datetime.datetime(2011, 9, 28, 0, 0),
     datetime.datetime(2011, 10, 17, 0, 0),
     datetime.datetime(2011, 10, 20, 0, 0),
     datetime.datetime(2011, 11, 3, 0, 0),
     datetime.datetime(2011, 11, 3, 0, 0),
     datetime.datetime(2011, 11, 3, 0, 0),
     datetime.datetime(2011, 11, 22, 0, 0),
     datetime.datetime(2011, 12, 5, 0, 0),
     datetime.datetime(2011, 12, 8, 0, 0),
     datetime.datetime(2011, 12, 14, 0, 0),
     datetime.datetime(2011, 12, 14, 0, 0),
     datetime.datetime(2011, 12, 26, 0, 0),
     datetime.datetime(2011, 12, 29, 0, 0),
     datetime.datetime(2011, 12, 29, 0, 0),
     datetime.datetime(2011, 1, 13, 0, 0),
     datetime.datetime(2011, 1, 18, 0, 0),
     datetime.datetime(2011, 3, 4, 0, 0),
     datetime.datetime(2011, 5, 21, 0, 0),
     datetime.datetime(2011, 7, 23, 0, 0),
     datetime.datetime(2011, 8, 19, 0, 0),
     datetime.datetime(2011, 8, 30, 0, 0),
     datetime.datetime(2011, 9, 2, 0, 0),
     datetime.datetime(2011, 9, 8, 0, 0),
     datetime.datetime(2011, 9, 22, 0, 0),
     datetime.datetime(2011, 9, 24, 0, 0),
     datetime.datetime(2011, 10, 19, 0, 0),
     datetime.datetime(2011, 10, 31, 0, 0),
     datetime.datetime(2011, 10, 31, 0, 0),
     datetime.datetime(2011, 11, 7, 0, 0),
     datetime.datetime(2011, 11, 10, 0, 0),
     datetime.datetime(2011, 11, 10, 0, 0),
     datetime.datetime(2011, 11, 10, 0, 0),
     datetime.datetime(2011, 11, 22, 0, 0),
     datetime.datetime(2011, 12, 24, 0, 0),
     datetime.datetime(2011, 12, 24, 0, 0),
     datetime.datetime(2011, 4, 23, 0, 0),
     datetime.datetime(2011, 1, 17, 0, 0),
     datetime.datetime(2011, 2, 10, 0, 0),
     datetime.datetime(2011, 4, 8, 0, 0),
     datetime.datetime(2011, 6, 5, 0, 0),
     datetime.datetime(2011, 7, 22, 0, 0),
     datetime.datetime(2011, 10, 17, 0, 0),
     datetime.datetime(2011, 10, 21, 0, 0),
     datetime.datetime(2011, 11, 7, 0, 0),
     datetime.datetime(2011, 11, 24, 0, 0),
     datetime.datetime(2011, 3, 8, 0, 0),
     datetime.datetime(2011, 7, 7, 0, 0),
     datetime.datetime(2011, 8, 17, 0, 0),
     datetime.datetime(2011, 3, 7, 0, 0),
     datetime.datetime(2011, 4, 15, 0, 0),
     datetime.datetime(2011, 1, 5, 0, 0),
     datetime.datetime(2011, 2, 9, 0, 0),
     datetime.datetime(2011, 2, 12, 0, 0),
     datetime.datetime(2011, 2, 16, 0, 0),
     datetime.datetime(2011, 2, 18, 0, 0),
     datetime.datetime(2011, 2, 18, 0, 0),
     datetime.datetime(2011, 2, 21, 0, 0),
     datetime.datetime(2011, 2, 25, 0, 0),
     datetime.datetime(2011, 3, 9, 0, 0),
     datetime.datetime(2011, 3, 12, 0, 0),
     datetime.datetime(2011, 3, 30, 0, 0),
     datetime.datetime(2011, 4, 9, 0, 0),
     datetime.datetime(2011, 4, 13, 0, 0),
     datetime.datetime(2011, 4, 15, 0, 0),
     datetime.datetime(2011, 4, 29, 0, 0),
     datetime.datetime(2011, 4, 29, 0, 0),
     datetime.datetime(2011, 5, 2, 0, 0),
     datetime.datetime(2011, 5, 12, 0, 0),
     datetime.datetime(2011, 6, 4, 0, 0),
     datetime.datetime(2011, 7, 5, 0, 0),
     datetime.datetime(2011, 7, 22, 0, 0),
     datetime.datetime(2011, 8, 6, 0, 0),
     datetime.datetime(2011, 8, 17, 0, 0),
     datetime.datetime(2011, 8, 22, 0, 0),
     datetime.datetime(2011, 8, 26, 0, 0),
     datetime.datetime(2011, 9, 29, 0, 0),
     datetime.datetime(2011, 10, 6, 0, 0),
     datetime.datetime(2011, 10, 22, 0, 0),
     datetime.datetime(2011, 10, 26, 0, 0),
     datetime.datetime(2011, 10, 29, 0, 0),
     datetime.datetime(2011, 11, 3, 0, 0),
     datetime.datetime(2011, 11, 3, 0, 0),
     datetime.datetime(2011, 12, 13, 0, 0),
     datetime.datetime(2011, 1, 4, 0, 0),
     datetime.datetime(2011, 1, 6, 0, 0),
     datetime.datetime(2011, 1, 6, 0, 0),
     datetime.datetime(2011, 1, 7, 0, 0),
     datetime.datetime(2011, 1, 11, 0, 0),
     datetime.datetime(2011, 1, 12, 0, 0),
     datetime.datetime(2011, 1, 13, 0, 0),
     datetime.datetime(2011, 1, 18, 0, 0),
     datetime.datetime(2011, 1, 18, 0, 0),
     datetime.datetime(2011, 1, 19, 0, 0),
     datetime.datetime(2011, 1, 20, 0, 0),
     datetime.datetime(2011, 1, 27, 0, 0),
     datetime.datetime(2011, 1, 28, 0, 0),
     datetime.datetime(2011, 1, 28, 0, 0),
     datetime.datetime(2011, 1, 29, 0, 0),
     datetime.datetime(2011, 2, 13, 0, 0),
     datetime.datetime(2011, 2, 16, 0, 0),
     datetime.datetime(2011, 2, 17, 0, 0),
     datetime.datetime(2011, 2, 19, 0, 0),
     datetime.datetime(2011, 2, 21, 0, 0),
     datetime.datetime(2011, 2, 23, 0, 0),
     datetime.datetime(2011, 2, 24, 0, 0),
     datetime.datetime(2011, 2, 24, 0, 0),
     datetime.datetime(2011, 2, 26, 0, 0),
     datetime.datetime(2011, 3, 1, 0, 0),
     datetime.datetime(2011, 3, 24, 0, 0),
     datetime.datetime(2011, 3, 25, 0, 0),
     datetime.datetime(2011, 3, 28, 0, 0),
     datetime.datetime(2011, 3, 30, 0, 0),
     datetime.datetime(2011, 4, 5, 0, 0),
     datetime.datetime(2011, 4, 12, 0, 0),
     datetime.datetime(2011, 4, 12, 0, 0),
     datetime.datetime(2011, 4, 12, 0, 0),
     datetime.datetime(2011, 4, 14, 0, 0),
     datetime.datetime(2011, 4, 21, 0, 0),
     datetime.datetime(2011, 4, 22, 0, 0),
     datetime.datetime(2011, 4, 27, 0, 0),
     datetime.datetime(2011, 5, 5, 0, 0),
     datetime.datetime(2011, 5, 5, 0, 0),
     datetime.datetime(2011, 5, 13, 0, 0),
     datetime.datetime(2011, 5, 16, 0, 0),
     datetime.datetime(2011, 5, 28, 0, 0),
     datetime.datetime(2011, 6, 1, 0, 0),
     datetime.datetime(2011, 6, 9, 0, 0),
     datetime.datetime(2011, 6, 13, 0, 0),
     datetime.datetime(2011, 6, 15, 0, 0),
     datetime.datetime(2011, 6, 20, 0, 0),
     datetime.datetime(2011, 6, 23, 0, 0),
     datetime.datetime(2011, 6, 27, 0, 0),
     datetime.datetime(2011, 7, 1, 0, 0),
     datetime.datetime(2011, 7, 23, 0, 0),
     datetime.datetime(2011, 8, 5, 0, 0),
     datetime.datetime(2011, 8, 8, 0, 0),
     datetime.datetime(2011, 8, 11, 0, 0),
     datetime.datetime(2011, 8, 30, 0, 0),
     datetime.datetime(2011, 9, 1, 0, 0),
     datetime.datetime(2011, 9, 1, 0, 0),
     datetime.datetime(2011, 9, 6, 0, 0),
     datetime.datetime(2011, 9, 6, 0, 0),
     datetime.datetime(2011, 9, 16, 0, 0),
     datetime.datetime(2011, 9, 24, 0, 0),
     datetime.datetime(2011, 10, 1, 0, 0),
     datetime.datetime(2011, 10, 3, 0, 0),
     datetime.datetime(2011, 10, 4, 0, 0),
     datetime.datetime(2011, 10, 15, 0, 0),
     datetime.datetime(2011, 11, 12, 0, 0),
     datetime.datetime(2011, 11, 14, 0, 0),
     datetime.datetime(2011, 12, 8, 0, 0),
     datetime.datetime(2011, 12, 8, 0, 0),
     datetime.datetime(2011, 12, 12, 0, 0),
     datetime.datetime(2011, 12, 14, 0, 0),
     datetime.datetime(2011, 12, 21, 0, 0),
     datetime.datetime(2011, 12, 24, 0, 0),
     datetime.datetime(2011, 12, 27, 0, 0),
     datetime.datetime(2011, 12, 28, 0, 0),
     datetime.datetime(2011, 12, 30, 0, 0),
     datetime.datetime(2011, 1, 7, 0, 0),
     datetime.datetime(2011, 1, 7, 0, 0),
     datetime.datetime(2011, 1, 25, 0, 0),
     datetime.datetime(2011, 1, 25, 0, 0),
     datetime.datetime(2011, 1, 27, 0, 0),
     datetime.datetime(2011, 1, 28, 0, 0),
     datetime.datetime(2011, 1, 28, 0, 0),
     datetime.datetime(2011, 2, 7, 0, 0),
     datetime.datetime(2011, 2, 7, 0, 0),
     datetime.datetime(2011, 2, 9, 0, 0),
     datetime.datetime(2011, 2, 9, 0, 0),
     datetime.datetime(2011, 2, 10, 0, 0),
     datetime.datetime(2011, 2, 21, 0, 0),
     datetime.datetime(2011, 2, 22, 0, 0),
     datetime.datetime(2011, 2, 23, 0, 0),
     datetime.datetime(2011, 2, 26, 0, 0),
     datetime.datetime(2011, 3, 5, 0, 0),
     datetime.datetime(2011, 3, 9, 0, 0),
     datetime.datetime(2011, 3, 26, 0, 0),
     datetime.datetime(2011, 3, 26, 0, 0),
     datetime.datetime(2011, 3, 31, 0, 0),
     datetime.datetime(2011, 4, 1, 0, 0),
     datetime.datetime(2011, 4, 2, 0, 0),
     datetime.datetime(2011, 4, 5, 0, 0),
     datetime.datetime(2011, 4, 17, 0, 0),
     datetime.datetime(2011, 4, 22, 0, 0),
     datetime.datetime(2011, 4, 28, 0, 0),
     datetime.datetime(2011, 5, 3, 0, 0),
     datetime.datetime(2011, 5, 13, 0, 0),
     datetime.datetime(2011, 5, 14, 0, 0),
     datetime.datetime(2011, 5, 20, 0, 0),
     datetime.datetime(2011, 5, 20, 0, 0),
     datetime.datetime(2011, 5, 26, 0, 0),
     datetime.datetime(2011, 5, 27, 0, 0),
     datetime.datetime(2011, 5, 30, 0, 0),
     datetime.datetime(2011, 6, 3, 0, 0),
     datetime.datetime(2011, 6, 8, 0, 0),
     datetime.datetime(2011, 7, 15, 0, 0),
     datetime.datetime(2011, 7, 20, 0, 0),
     datetime.datetime(2011, 7, 20, 0, 0),
     datetime.datetime(2011, 7, 22, 0, 0),
     datetime.datetime(2011, 7, 23, 0, 0),
     datetime.datetime(2011, 7, 25, 0, 0),
     datetime.datetime(2011, 7, 27, 0, 0),
     datetime.datetime(2011, 7, 31, 0, 0),
     datetime.datetime(2011, 8, 31, 0, 0),
     datetime.datetime(2011, 9, 5, 0, 0),
     datetime.datetime(2011, 9, 6, 0, 0),
     datetime.datetime(2011, 9, 7, 0, 0),
     datetime.datetime(2011, 10, 8, 0, 0),
     datetime.datetime(2011, 10, 19, 0, 0),
     datetime.datetime(2011, 10, 25, 0, 0),
     datetime.datetime(2011, 10, 28, 0, 0),
     datetime.datetime(2011, 10, 28, 0, 0),
     datetime.datetime(2011, 11, 1, 0, 0),
     datetime.datetime(2011, 11, 7, 0, 0),
     datetime.datetime(2011, 11, 11, 0, 0),
     datetime.datetime(2011, 12, 5, 0, 0),
     datetime.datetime(2011, 12, 5, 0, 0),
     datetime.datetime(2011, 12, 7, 0, 0),
     datetime.datetime(2011, 12, 8, 0, 0),
     datetime.datetime(2011, 12, 19, 0, 0),
     datetime.datetime(2011, 12, 21, 0, 0),
     datetime.datetime(2011, 12, 23, 0, 0),
     datetime.datetime(2011, 12, 23, 0, 0),
     datetime.datetime(2011, 3, 8, 0, 0),
     datetime.datetime(2011, 2, 22, 0, 0),
     datetime.datetime(2011, 6, 7, 0, 0),
     datetime.datetime(2011, 7, 3, 0, 0),
     datetime.datetime(2011, 7, 9, 0, 0),
     datetime.datetime(2011, 9, 15, 0, 0),
     datetime.datetime(2011, 9, 20, 0, 0),
     datetime.datetime(2011, 10, 1, 0, 0),
     datetime.datetime(2011, 12, 5, 0, 0),
     datetime.datetime(2011, 1, 17, 0, 0),
     datetime.datetime(2011, 2, 26, 0, 0),
     datetime.datetime(2011, 4, 14, 0, 0),
     datetime.datetime(2011, 4, 19, 0, 0),
     datetime.datetime(2011, 5, 20, 0, 0),
     datetime.datetime(2011, 9, 9, 0, 0),
     datetime.datetime(2011, 1, 28, 0, 0),
     datetime.datetime(2011, 2, 10, 0, 0),
     datetime.datetime(2011, 9, 28, 0, 0),
     datetime.datetime(2011, 1, 17, 0, 0),
     datetime.datetime(2011, 1, 9, 0, 0),
     datetime.datetime(2011, 1, 10, 0, 0),
     datetime.datetime(2011, 1, 11, 0, 0),
     datetime.datetime(2011, 1, 12, 0, 0),
     datetime.datetime(2011, 1, 18, 0, 0),
     datetime.datetime(2011, 1, 21, 0, 0),
     datetime.datetime(2011, 1, 22, 0, 0),
     datetime.datetime(2011, 2, 1, 0, 0),
     datetime.datetime(2011, 2, 9, 0, 0),
     datetime.datetime(2011, 2, 10, 0, 0),
     datetime.datetime(2011, 2, 11, 0, 0),
     datetime.datetime(2011, 2, 15, 0, 0),
     datetime.datetime(2011, 2, 28, 0, 0),
     datetime.datetime(2011, 3, 25, 0, 0),
     datetime.datetime(2011, 3, 25, 0, 0),
     datetime.datetime(2011, 3, 31, 0, 0),
     datetime.datetime(2011, 4, 5, 0, 0),
     datetime.datetime(2011, 4, 11, 0, 0),
     datetime.datetime(2011, 4, 27, 0, 0),
     datetime.datetime(2011, 4, 29, 0, 0),
     datetime.datetime(2011, 5, 2, 0, 0),
     datetime.datetime(2011, 5, 4, 0, 0),
     datetime.datetime(2011, 5, 13, 0, 0),
     datetime.datetime(2011, 5, 17, 0, 0),
     datetime.datetime(2011, 6, 14, 0, 0),
     datetime.datetime(2011, 6, 14, 0, 0),
     datetime.datetime(2011, 6, 14, 0, 0),
     datetime.datetime(2011, 7, 5, 0, 0),
     datetime.datetime(2011, 7, 5, 0, 0),
     datetime.datetime(2011, 7, 7, 0, 0),
     datetime.datetime(2011, 7, 20, 0, 0),
     datetime.datetime(2011, 7, 20, 0, 0),
     datetime.datetime(2011, 7, 22, 0, 0),
     datetime.datetime(2011, 7, 25, 0, 0),
     datetime.datetime(2011, 7, 25, 0, 0),
     datetime.datetime(2011, 7, 26, 0, 0),
     datetime.datetime(2011, 7, 28, 0, 0),
     datetime.datetime(2011, 7, 29, 0, 0),
     datetime.datetime(2011, 8, 8, 0, 0),
     datetime.datetime(2011, 8, 13, 0, 0),
     datetime.datetime(2011, 8, 13, 0, 0),
     datetime.datetime(2011, 8, 25, 0, 0),
     datetime.datetime(2011, 8, 26, 0, 0),
     datetime.datetime(2011, 8, 27, 0, 0),
     datetime.datetime(2011, 8, 30, 0, 0),
     datetime.datetime(2011, 9, 1, 0, 0),
     datetime.datetime(2011, 9, 8, 0, 0),
     datetime.datetime(2011, 9, 22, 0, 0),
     datetime.datetime(2011, 10, 14, 0, 0),
     datetime.datetime(2011, 10, 15, 0, 0),
     datetime.datetime(2011, 10, 20, 0, 0),
     datetime.datetime(2011, 10, 20, 0, 0),
     datetime.datetime(2011, 10, 24, 0, 0),
     datetime.datetime(2011, 10, 25, 0, 0),
     datetime.datetime(2011, 10, 25, 0, 0),
     datetime.datetime(2011, 11, 16, 0, 0),
     datetime.datetime(2011, 11, 21, 0, 0),
     datetime.datetime(2011, 12, 5, 0, 0),
     datetime.datetime(2011, 12, 7, 0, 0),
     datetime.datetime(2011, 12, 7, 0, 0),
     datetime.datetime(2011, 12, 8, 0, 0),
     datetime.datetime(2011, 12, 8, 0, 0),
     datetime.datetime(2011, 12, 8, 0, 0),
     datetime.datetime(2011, 12, 10, 0, 0),
     datetime.datetime(2011, 12, 19, 0, 0),
     datetime.datetime(2011, 12, 21, 0, 0),
     datetime.datetime(2011, 12, 22, 0, 0),
     datetime.datetime(2011, 11, 17, 0, 0),
     datetime.datetime(2011, 1, 5, 0, 0),
     datetime.datetime(2011, 1, 7, 0, 0),
     datetime.datetime(2011, 1, 29, 0, 0),
     datetime.datetime(2011, 3, 28, 0, 0),
     datetime.datetime(2011, 5, 27, 0, 0),
     datetime.datetime(2011, 6, 15, 0, 0),
     datetime.datetime(2011, 9, 2, 0, 0),
     datetime.datetime(2011, 10, 11, 0, 0),
     datetime.datetime(2011, 1, 10, 0, 0),
     datetime.datetime(2011, 1, 18, 0, 0),
     datetime.datetime(2011, 1, 26, 0, 0),
     datetime.datetime(2011, 5, 25, 0, 0),
     datetime.datetime(2011, 6, 2, 0, 0),
     datetime.datetime(2011, 6, 10, 0, 0),
     datetime.datetime(2011, 10, 18, 0, 0),
     datetime.datetime(2011, 2, 15, 0, 0),
     datetime.datetime(2011, 3, 3, 0, 0),
     datetime.datetime(2011, 12, 7, 0, 0),
     datetime.datetime(2011, 1, 7, 0, 0),
     datetime.datetime(2011, 2, 24, 0, 0),
     datetime.datetime(2011, 3, 5, 0, 0),
     datetime.datetime(2011, 3, 8, 0, 0),
     datetime.datetime(2011, 6, 21, 0, 0),
     datetime.datetime(2011, 10, 22, 0, 0),
     datetime.datetime(2011, 1, 31, 0, 0),
     datetime.datetime(2011, 1, 31, 0, 0),
     datetime.datetime(2011, 2, 9, 0, 0),
     datetime.datetime(2011, 2, 21, 0, 0),
     datetime.datetime(2011, 4, 6, 0, 0),
     datetime.datetime(2011, 4, 9, 0, 0),
     datetime.datetime(2011, 4, 13, 0, 0),
     datetime.datetime(2011, 4, 15, 0, 0),
     datetime.datetime(2011, 5, 11, 0, 0),
     datetime.datetime(2011, 6, 29, 0, 0),
     datetime.datetime(2011, 8, 19, 0, 0),
     datetime.datetime(2011, 10, 17, 0, 0),
     datetime.datetime(2011, 11, 16, 0, 0),
     datetime.datetime(2011, 1, 24, 0, 0),
     datetime.datetime(2011, 2, 1, 0, 0),
     datetime.datetime(2011, 2, 21, 0, 0),
     datetime.datetime(2011, 2, 24, 0, 0),
     datetime.datetime(2011, 3, 2, 0, 0),
     datetime.datetime(2011, 3, 2, 0, 0),
     datetime.datetime(2011, 6, 28, 0, 0),
     datetime.datetime(2011, 6, 28, 0, 0),
     datetime.datetime(2011, 6, 28, 0, 0),
     datetime.datetime(2011, 7, 15, 0, 0),
     datetime.datetime(2011, 7, 21, 0, 0),
     datetime.datetime(2011, 7, 25, 0, 0),
     datetime.datetime(2011, 7, 28, 0, 0),
     datetime.datetime(2011, 7, 28, 0, 0),
     datetime.datetime(2011, 8, 11, 0, 0),
     datetime.datetime(2011, 8, 29, 0, 0),
     datetime.datetime(2011, 9, 28, 0, 0),
     datetime.datetime(2011, 10, 4, 0, 0),
     datetime.datetime(2011, 10, 12, 0, 0),
     datetime.datetime(2011, 10, 20, 0, 0),
     datetime.datetime(2011, 10, 28, 0, 0),
     datetime.datetime(2011, 11, 28, 0, 0),
     datetime.datetime(2011, 11, 28, 0, 0),
     datetime.datetime(2011, 12, 6, 0, 0),
     datetime.datetime(2011, 12, 28, 0, 0),
     datetime.datetime(2011, 12, 30, 0, 0),
     datetime.datetime(2011, 12, 20, 0, 0),
     datetime.datetime(2011, 2, 25, 0, 0),
     datetime.datetime(2011, 3, 2, 0, 0),
     datetime.datetime(2011, 3, 5, 0, 0),
     datetime.datetime(2011, 3, 16, 0, 0),
     datetime.datetime(2011, 3, 21, 0, 0),
     datetime.datetime(2011, 4, 7, 0, 0),
     datetime.datetime(2011, 2, 17, 0, 0),
     datetime.datetime(2011, 2, 25, 0, 0),
     datetime.datetime(2011, 2, 25, 0, 0),
     datetime.datetime(2011, 2, 28, 0, 0),
     datetime.datetime(2011, 6, 16, 0, 0),
     datetime.datetime(2011, 11, 23, 0, 0),
     datetime.datetime(2011, 1, 31, 0, 0),
     datetime.datetime(2011, 2, 9, 0, 0),
     datetime.datetime(2011, 9, 30, 0, 0),
     datetime.datetime(2011, 11, 2, 0, 0),
     datetime.datetime(2011, 1, 19, 0, 0),
     datetime.datetime(2011, 3, 16, 0, 0),
     datetime.datetime(2011, 1, 4, 0, 0),
     datetime.datetime(2011, 2, 9, 0, 0),
     datetime.datetime(2011, 2, 20, 0, 0),
     datetime.datetime(2011, 5, 26, 0, 0),
     datetime.datetime(2011, 5, 27, 0, 0),
     datetime.datetime(2011, 8, 2, 0, 0),
     datetime.datetime(2011, 9, 15, 0, 0),
     datetime.datetime(2011, 10, 24, 0, 0),
     datetime.datetime(2011, 10, 11, 0, 0),
     datetime.datetime(2011, 10, 1, 0, 0),
     datetime.datetime(2011, 1, 26, 0, 0),
     datetime.datetime(2011, 2, 1, 0, 0),
     datetime.datetime(2011, 2, 17, 0, 0),
     datetime.datetime(2011, 3, 2, 0, 0),
     datetime.datetime(2011, 3, 19, 0, 0),
     datetime.datetime(2011, 5, 3, 0, 0),
     datetime.datetime(2011, 5, 24, 0, 0),
     datetime.datetime(2011, 9, 23, 0, 0),
     datetime.datetime(2011, 2, 23, 0, 0),
     datetime.datetime(2011, 3, 3, 0, 0),
     datetime.datetime(2011, 4, 19, 0, 0),
     datetime.datetime(2011, 8, 23, 0, 0),
     datetime.datetime(2011, 1, 7, 0, 0),
     datetime.datetime(2011, 1, 17, 0, 0),
     datetime.datetime(2011, 1, 17, 0, 0),
     datetime.datetime(2011, 1, 29, 0, 0),
     datetime.datetime(2011, 2, 12, 0, 0),
     datetime.datetime(2011, 2, 23, 0, 0),
     datetime.datetime(2011, 3, 5, 0, 0),
     datetime.datetime(2011, 3, 9, 0, 0),
     datetime.datetime(2011, 4, 7, 0, 0),
     datetime.datetime(2011, 5, 17, 0, 0),
     datetime.datetime(2011, 5, 23, 0, 0),
     datetime.datetime(2011, 7, 4, 0, 0),
     datetime.datetime(2011, 7, 15, 0, 0),
     datetime.datetime(2011, 7, 25, 0, 0),
     datetime.datetime(2011, 8, 5, 0, 0),
     datetime.datetime(2011, 8, 11, 0, 0),
     datetime.datetime(2011, 8, 23, 0, 0),
     datetime.datetime(2011, 10, 1, 0, 0),
     datetime.datetime(2011, 10, 3, 0, 0),
     datetime.datetime(2011, 12, 5, 0, 0),
     datetime.datetime(2011, 12, 28, 0, 0),
     datetime.datetime(2011, 12, 28, 0, 0),
     datetime.datetime(2011, 12, 29, 0, 0),
     datetime.datetime(2011, 12, 31, 0, 0),
     datetime.datetime(2011, 2, 17, 0, 0),
     datetime.datetime(2011, 6, 13, 0, 0),
     datetime.datetime(2011, 7, 6, 0, 0),
     datetime.datetime(2011, 6, 28, 0, 0),
     datetime.datetime(2011, 8, 22, 0, 0),
     datetime.datetime(2011, 9, 30, 0, 0),
     datetime.datetime(2011, 10, 4, 0, 0),
     datetime.datetime(2011, 11, 30, 0, 0),
     datetime.datetime(2011, 8, 5, 0, 0),
     datetime.datetime(2011, 5, 4, 0, 0),
     datetime.datetime(2011, 7, 28, 0, 0),
     datetime.datetime(2011, 9, 16, 0, 0),
     datetime.datetime(2011, 1, 6, 0, 0),
     datetime.datetime(2011, 1, 10, 0, 0),
     datetime.datetime(2011, 2, 9, 0, 0),
     datetime.datetime(2011, 2, 16, 0, 0),
     datetime.datetime(2011, 3, 24, 0, 0),
     datetime.datetime(2011, 4, 18, 0, 0),
     datetime.datetime(2011, 5, 11, 0, 0),
     datetime.datetime(2011, 5, 23, 0, 0),
     datetime.datetime(2011, 6, 1, 0, 0),
     datetime.datetime(2011, 6, 15, 0, 0),
     datetime.datetime(2011, 6, 24, 0, 0),
     datetime.datetime(2011, 7, 4, 0, 0),
     datetime.datetime(2011, 7, 12, 0, 0),
     datetime.datetime(2011, 8, 26, 0, 0),
     datetime.datetime(2011, 10, 1, 0, 0),
     datetime.datetime(2011, 11, 23, 0, 0),
     datetime.datetime(2011, 12, 30, 0, 0),
     datetime.datetime(2011, 5, 28, 0, 0),
     datetime.datetime(2011, 4, 1, 0, 0),
     datetime.datetime(2011, 7, 12, 0, 0),
     datetime.datetime(2011, 10, 17, 0, 0),
     datetime.datetime(2011, 6, 10, 0, 0),
     datetime.datetime(2011, 1, 8, 0, 0),
     datetime.datetime(2011, 1, 14, 0, 0),
     datetime.datetime(2011, 1, 15, 0, 0),
     datetime.datetime(2011, 1, 19, 0, 0),
     datetime.datetime(2011, 1, 22, 0, 0),
     datetime.datetime(2011, 1, 24, 0, 0),
     datetime.datetime(2011, 1, 24, 0, 0),
     datetime.datetime(2011, 1, 27, 0, 0),
     datetime.datetime(2011, 1, 29, 0, 0),
     datetime.datetime(2011, 1, 31, 0, 0),
     datetime.datetime(2011, 2, 12, 0, 0),
     datetime.datetime(2011, 2, 14, 0, 0),
     datetime.datetime(2011, 2, 18, 0, 0),
     datetime.datetime(2011, 2, 19, 0, 0),
     datetime.datetime(2011, 2, 19, 0, 0),
     datetime.datetime(2011, 3, 11, 0, 0),
     datetime.datetime(2011, 3, 16, 0, 0),
     datetime.datetime(2011, 4, 14, 0, 0),
     datetime.datetime(2011, 4, 14, 0, 0),
     datetime.datetime(2011, 4, 20, 0, 0),
     datetime.datetime(2011, 4, 21, 0, 0),
     datetime.datetime(2011, 5, 10, 0, 0),
     datetime.datetime(2011, 5, 13, 0, 0),
     datetime.datetime(2011, 5, 21, 0, 0),
     datetime.datetime(2011, 5, 31, 0, 0),
     datetime.datetime(2011, 6, 11, 0, 0),
     datetime.datetime(2011, 6, 18, 0, 0),
     datetime.datetime(2011, 7, 15, 0, 0),
     datetime.datetime(2011, 7, 29, 0, 0),
     datetime.datetime(2011, 8, 6, 0, 0),
     datetime.datetime(2011, 8, 17, 0, 0),
     datetime.datetime(2011, 8, 18, 0, 0),
     datetime.datetime(2011, 8, 27, 0, 0),
     datetime.datetime(2011, 9, 1, 0, 0),
     datetime.datetime(2011, 10, 21, 0, 0),
     datetime.datetime(2011, 10, 27, 0, 0),
     datetime.datetime(2011, 11, 10, 0, 0),
     datetime.datetime(2011, 12, 17, 0, 0),
     datetime.datetime(2011, 1, 6, 0, 0),
     datetime.datetime(2011, 3, 29, 0, 0),
     datetime.datetime(2011, 5, 21, 0, 0),
     datetime.datetime(2011, 5, 23, 0, 0),
     datetime.datetime(2011, 6, 6, 0, 0),
     datetime.datetime(2011, 9, 17, 0, 0),
     datetime.datetime(2011, 9, 21, 0, 0),
     datetime.datetime(2011, 12, 22, 0, 0),
     datetime.datetime(2011, 11, 5, 0, 0),
     datetime.datetime(2011, 1, 13, 0, 0),
     datetime.datetime(2011, 1, 15, 0, 0),
     datetime.datetime(2011, 1, 31, 0, 0),
     datetime.datetime(2011, 2, 7, 0, 0),
     datetime.datetime(2011, 6, 6, 0, 0),
     datetime.datetime(2011, 6, 17, 0, 0),
     datetime.datetime(2011, 8, 8, 0, 0),
     datetime.datetime(2011, 12, 8, 0, 0),
     datetime.datetime(2011, 3, 14, 0, 0),
     datetime.datetime(2011, 8, 1, 0, 0),
     datetime.datetime(2011, 12, 27, 0, 0),
     datetime.datetime(2011, 1, 17, 0, 0),
     datetime.datetime(2011, 2, 14, 0, 0),
     datetime.datetime(2011, 2, 15, 0, 0),
     datetime.datetime(2011, 2, 12, 0, 0),
     datetime.datetime(2011, 4, 1, 0, 0),
     datetime.datetime(2011, 1, 4, 0, 0),
     datetime.datetime(2011, 7, 26, 0, 0),
     datetime.datetime(2011, 7, 1, 0, 0),
     datetime.datetime(2011, 9, 29, 0, 0),
     datetime.datetime(2011, 6, 10, 0, 0),
     datetime.datetime(2011, 4, 18, 0, 0),
     datetime.datetime(2011, 1, 14, 0, 0),
     datetime.datetime(2011, 3, 4, 0, 0),
     datetime.datetime(2011, 4, 18, 0, 0),
     datetime.datetime(2011, 5, 16, 0, 0),
     datetime.datetime(2011, 5, 24, 0, 0),
     datetime.datetime(2011, 7, 9, 0, 0),
     datetime.datetime(2011, 8, 31, 0, 0),
     datetime.datetime(2011, 10, 17, 0, 0),
     datetime.datetime(2011, 4, 25, 0, 0),
     datetime.datetime(2011, 6, 15, 0, 0),
     datetime.datetime(2011, 12, 12, 0, 0),
     datetime.datetime(2011, 5, 27, 0, 0),
     datetime.datetime(2011, 11, 25, 0, 0),
     datetime.datetime(2011, 6, 15, 0, 0),
     datetime.datetime(2011, 6, 25, 0, 0),
     datetime.datetime(2011, 1, 19, 0, 0),
     datetime.datetime(2011, 10, 19, 0, 0),
     datetime.datetime(2011, 11, 7, 0, 0),
     datetime.datetime(2011, 12, 16, 0, 0),
     datetime.datetime(2011, 12, 17, 0, 0),
     datetime.datetime(2011, 6, 24, 0, 0),
     datetime.datetime(2011, 6, 27, 0, 0),
     datetime.datetime(2011, 8, 25, 0, 0),
     datetime.datetime(2011, 9, 2, 0, 0),
     datetime.datetime(2011, 9, 22, 0, 0),
     datetime.datetime(2011, 9, 30, 0, 0),
     datetime.datetime(2011, 12, 17, 0, 0),
     datetime.datetime(2011, 1, 26, 0, 0),
     datetime.datetime(2011, 7, 26, 0, 0),
     datetime.datetime(2011, 8, 27, 0, 0),
     datetime.datetime(2011, 9, 16, 0, 0),
     datetime.datetime(2011, 2, 14, 0, 0),
     datetime.datetime(2011, 3, 16, 0, 0),
     datetime.datetime(2011, 3, 21, 0, 0),
     datetime.datetime(2011, 4, 25, 0, 0),
     datetime.datetime(2011, 5, 24, 0, 0),
     datetime.datetime(2011, 5, 27, 0, 0),
     datetime.datetime(2011, 6, 3, 0, 0),
     datetime.datetime(2011, 7, 12, 0, 0),
     datetime.datetime(2011, 7, 20, 0, 0),
     datetime.datetime(2011, 11, 23, 0, 0),
     datetime.datetime(2011, 8, 10, 0, 0),
     datetime.datetime(2011, 1, 28, 0, 0),
     datetime.datetime(2011, 2, 8, 0, 0),
     datetime.datetime(2011, 7, 3, 0, 0),
     ...]




```python
year_month_day = pd.DataFrame(dt_datetime, columns=["계약년월일"])
year.name = "년도"
tr_concat = pd.concat([tr_concat, year_month_day, year], axis=1)
```


```python
#데이터분석에 필요한 컬럼만 추출
tr_data = tr_concat[["계약년월일","시군구","단지명","전용면적(㎡)","거래금액(만원)","년도"]]
```


```python
tr_data["area"] = tr_data.시군구.str.split(" ").str[1]
```

    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    


```python
tr_data["area"]
```




    0         강남구
    1         강남구
    2         강남구
    3         강남구
    4         강남구
             ... 
    826104    중랑구
    826105    중랑구
    826106    중랑구
    826107    중랑구
    826108    중랑구
    Name: area, Length: 826109, dtype: object




```python
type(tr_data["거래금액(만원)"])
```




    pandas.core.series.Series




```python
#문자형 > 숫자형 데이터로 변환
tr_data["거래금액(만원)"] = tr_data["거래금액(만원)"].str.replace(',','')
tr_data["거래금액(만원)"] = tr_data["거래금액(만원)"].apply(pd.to_numeric)
#평당가격 도출
tr_data["평당가"] = tr_data["거래금액(만원)"] / tr_data["전용면적(㎡)"]
tr_data
```

    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    




                계약년월일            시군구             단지명  전용면적(㎡)  거래금액(만원)    년도  \
    0      2011-07-09  서울특별시 강남구 개포동  개포2차현대아파트(220)    77.75     64000  2011   
    1      2011-07-28  서울특별시 강남구 개포동  개포2차현대아파트(220)    77.75     65500  2011   
    2      2011-01-19  서울특별시 강남구 개포동  개포6차우성아파트1동~8동    67.28     70500  2011   
    3      2011-09-02  서울특별시 강남구 개포동  개포6차우성아파트1동~8동    79.97     85000  2011   
    4      2011-12-17  서울특별시 강남구 개포동  개포6차우성아파트1동~8동    67.28     68000  2011   
    ...           ...            ...             ...      ...       ...   ...   
    826104 2020-08-07  서울특별시 중랑구 중화동         한영(104)    67.57     26000  2020   
    826105 2020-07-10  서울특별시 중랑구 중화동           현대휴앤미    95.94     44000  2020   
    826106 2020-12-03  서울특별시 중랑구 중화동           현대휴앤미   100.17     54800  2020   
    826107 2020-09-28  서울특별시 중랑구 중화동     현대휴앤미(102동)    77.71     40000  2020   
    826108 2020-09-28  서울특별시 중랑구 중화동     현대휴앤미(102동)    77.71     40000  2020   
    
           area          평당가  
    0       강남구   823.151125  
    1       강남구   842.443730  
    2       강남구  1047.859691  
    3       강남구  1062.898587  
    4       강남구  1010.701546  
    ...     ...          ...  
    826104  중랑구   384.786148  
    826105  중랑구   458.619971  
    826106  중랑구   547.069981  
    826107  중랑구   514.734268  
    826108  중랑구   514.734268  
    
    [826109 rows x 8 columns]




```python
#일자별 해당 지역 평당가 체크
tr_data_series = tr_data.groupby(['계약년월일','area'])['평당가'].agg(**{"일자별평당가":"mean"}).reset_index()
```


```python
# 구별 / 일자별 평당가 전처리 데이터
tr_data_series
```




               계약년월일 area       일자별평당가
    0     2011-01-01  강남구   663.166621
    1     2011-01-01  강동구   637.018752
    2     2011-01-01  강북구   421.807269
    3     2011-01-01  강서구   345.009218
    4     2011-01-01  관악구   565.387404
    ...          ...  ...          ...
    85226 2020-12-31  용산구  2056.587814
    85227 2020-12-31  은평구   933.400030
    85228 2020-12-31  종로구  1791.733836
    85229 2020-12-31   중구  2062.139194
    85230 2020-12-31  중랑구   928.100564
    
    [85231 rows x 3 columns]




```python
year = tr_data_series["계약년월일"].astype(str).str[:4]
tr_data_series["year"] = year
#연도(year)정보를 바탕으로 Macro 데이터 merge
korea_gdp_rate["year"] = korea_gdp_rate["year"].astype(str)
korea_personal_loan["year"] = korea_personal_loan["year"].astype(str)
korea_interest_rate["year"] = korea_interest_rate["year"].astype(str)
korea_loan_for_house["year"] = korea_loan_for_house["year"].astype(str)
korea_personal_GDP["year"] = korea_personal_GDP["year"].astype(str)
seoul_gdp_series["year"] = seoul_gdp_series["year"].astype(str)
housing_count_yearly["year"] = housing_count_yearly["year"].astype(str)
seoul_population["year"] = seoul_population["year"].astype(str)
constructure_confirm["year"] = constructure_confirm["year"].astype(str)
```


```python
merge1 = pd.merge(left=tr_data_series, right=korea_gdp_rate, how="left", on="year")
merge2 = pd.merge(left=merge1, right=korea_personal_loan, how="left", on="year")
merge3 = pd.merge(left=merge2, right=korea_interest_rate, how="left", on="year")
merge4 = pd.merge(left=merge3, right=korea_loan_for_house, how="left", on="year")
merge5 = pd.merge(left=merge4, right=korea_personal_GDP, how="left", on="year")
merge6 = pd.merge(left=merge5, right=seoul_gdp_series, how="left", on="year")
merge7 = pd.merge(left=merge6, right=housing_count_yearly, how="left", on="year")
merge8 = pd.merge(left=merge7, right=seoul_population, how="left", on="year")
merge9 = pd.merge(left=merge8, right=constructure_confirm, how="left", on="year")
```


```python
#SET INDEX (날짜 데이터를 인덱스로)
all_df_prepro = tr_data_series.set_index("계약년월일")
```


```python
all_df = merge9.set_index("계약년월일")
```


```python
#DataSet 1-1 
all_df
```




               area       일자별평당가  year  KOREA_GDP GDP_Growth_rate  loan  기준금리  \
    계약년월일                                                                       
    2011-01-01  강남구   663.166621  2011    1388937              4    916     3   
    2011-01-01  강동구   637.018752  2011    1388937              4    916     3   
    2011-01-01  강북구   421.807269  2011    1388937              4    916     3   
    2011-01-01  강서구   345.009218  2011    1388937              4    916     3   
    2011-01-01  관악구   565.387404  2011    1388937              4    916     3   
    ...         ...          ...   ...        ...             ...   ...   ...   
    2020-12-31  용산구  2056.587814  2020    1933152             (1)  1726     1   
    2020-12-31  은평구   933.400030  2020    1933152             (1)  1726     1   
    2020-12-31  종로구  1791.733836  2020    1933152             (1)  1726     1   
    2020-12-31   중구  2062.139194  2020    1933152             (1)  1726     1   
    2020-12-31  중랑구   928.100564  2020    1933152             (1)  1726     1   
    
                housing_loan  gdp_per_person        gdp  Viliages  Buildings  \
    계약년월일                                                                      
    2011-01-01       2511440           27901  326415107      4081      19022   
    2011-01-01       2511440           27901  326415107      4081      19022   
    2011-01-01       2511440           27901  326415107      4081      19022   
    2011-01-01       2511440           27901  326415107      4081      19022   
    2011-01-01       2511440           27901  326415107      4081      19022   
    ...                  ...             ...        ...       ...        ...   
    2020-12-31       5914223           37568  435102998      4134      20179   
    2020-12-31       5914223           37568  435102998      4134      20179   
    2020-12-31       5914223           37568  435102998      4134      20179   
    2020-12-31       5914223           37568  435102998      4134      20179   
    2020-12-31       5914223           37568  435102998      4134      20179   
    
                  House       세대        인구      합계     수도권     서울  
    계약년월일                                                          
    2011-01-01  1459112  4192752  10528774  549594  272156  88060  
    2011-01-01  1459112  4192752  10528774  549594  272156  88060  
    2011-01-01  1459112  4192752  10528774  549594  272156  88060  
    2011-01-01  1459112  4192752  10528774  549594  272156  88060  
    2011-01-01  1459112  4192752  10528774  549594  272156  88060  
    ...             ...      ...       ...     ...     ...    ...  
    2020-12-31  1544424  4417954   9911088  457514  252301  58181  
    2020-12-31  1544424  4417954   9911088  457514  252301  58181  
    2020-12-31  1544424  4417954   9911088  457514  252301  58181  
    2020-12-31  1544424  4417954   9911088  457514  252301  58181  
    2020-12-31  1544424  4417954   9911088  457514  252301  58181  
    
    [85231 rows x 18 columns]



# (4) 서울시 아파트 거래 평당가격 시계열 데이터 확인


```python
seoul_int = all_df[["일자별평당가","KOREA_GDP","GDP_Growth_rate","loan","기준금리","housing_loan","gdp_per_person","gdp",'Viliages',"Buildings",'House','세대',"인구","수도권"]]
```


```python
seoul_int.loc[(seoul_int.GDP_Growth_rate == "(1)"),"GDP_Growth_rate"] = 1
```

    C:\Users\MSI\Anaconda3\lib\site-packages\pandas\core\indexing.py:1720: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._setitem_single_column(loc, value, pi)
    


```python
seoul_int["KOREA_GDP"] = seoul_int["KOREA_GDP"].apply(pd.to_numeric)
seoul_int["GDP_Growth_rate"] = seoul_int["GDP_Growth_rate"].apply(pd.to_numeric)
seoul_int["loan"] = seoul_int["loan"].apply(pd.to_numeric)
seoul_int["기준금리"] = seoul_int["기준금리"].apply(pd.to_numeric)
seoul_int["housing_loan"] = seoul_int["housing_loan"].apply(pd.to_numeric)
seoul_int["gdp_per_person"] = seoul_int["gdp_per_person"].apply(pd.to_numeric)
seoul_int["Viliages"] = seoul_int["Viliages"].apply(pd.to_numeric)
seoul_int["Buildings"] = seoul_int["Buildings"].apply(pd.to_numeric)
seoul_int["House"] = seoul_int["House"].apply(pd.to_numeric)
```

    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      import sys
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if __name__ == '__main__':
    


```python
from sklearn.preprocessing import MinMaxScaler
```


```python
features_name = seoul_int.columns

scaler = MinMaxScaler()
scaler.fit(seoul_int)
seoul_scaled = scaler.transform(seoul_int)

seoul_df_scaled = pd.DataFrame(data=seoul_int, columns=features_name)
```


```python
seoul_df_scaled["일자별평당가"].plot(title="서울시 전체 일자별 평당가 추이")
```




    <AxesSubplot:title={'center':'서울시 전체 일자별 평당가 추이'}, xlabel='계약년월일'>




    
![png](output_32_1.png)
    


# (5) 서울 아파트 거래가격 Map 데이터로의 시각화


```python
import folium
import json
import re
```


```python
tr_data_series
```




               계약년월일 area       일자별평당가  year
    0     2011-01-01  강남구   663.166621  2011
    1     2011-01-01  강동구   637.018752  2011
    2     2011-01-01  강북구   421.807269  2011
    3     2011-01-01  강서구   345.009218  2011
    4     2011-01-01  관악구   565.387404  2011
    ...          ...  ...          ...   ...
    85226 2020-12-31  용산구  2056.587814  2020
    85227 2020-12-31  은평구   933.400030  2020
    85228 2020-12-31  종로구  1791.733836  2020
    85229 2020-12-31   중구  2062.139194  2020
    85230 2020-12-31  중랑구   928.100564  2020
    
    [85231 rows x 4 columns]




```python
data_for_map = tr_data_series.groupby(['year','area'])['일자별평당가'].agg(**{"연도별평당가":"mean"}).reset_index()
```


```python
#연도별 / 구별 평당가 체크
data_for_map_2011 = data_for_map[data_for_map["year"]=="2011"]
```


```python
geo_json = "https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json"
```


```python
df2 = data_for_map_2011[["area","연도별평당가"]]
df2.columns=["name","values"]

df2=df2.sort_values(by="name")
df2["name"] = df2["name"].apply(lambda x : re.compile('[가-힣]+').findall(x)[0])

n = folium.Map(
    location=[37.566345,126.977893],
    tiles ="Stamen Terrain"
)
```


```python
folium.Choropleth(
    geo_data=geo_json,
    name='choropleth',
    data=df2,
    columns=['name','values'],
    key_on='feature.properties.name',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2
).add_to(n)
```




    <folium.features.Choropleth at 0x13d31d9fa88>




```python
n
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Cscript%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20L_NO_TOUCH%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20L_DISABLE_3D%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%3C/script%3E%0A%20%20%20%20%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//code.jquery.com/jquery-1.12.4.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css%22/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cmeta%20name%3D%22viewport%22%20content%3D%22width%3Ddevice-width%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20initial-scale%3D1.0%2C%20maximum-scale%3D1.0%2C%20user-scalable%3Dno%22%20/%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%23map_34be2092659d453298ac36df9dcfb85a%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js%22%3E%3C/script%3E%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_34be2092659d453298ac36df9dcfb85a%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_34be2092659d453298ac36df9dcfb85a%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22map_34be2092659d453298ac36df9dcfb85a%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20center%3A%20%5B37.566345%2C%20126.977893%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2010%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoomControl%3A%20true%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20preferCanvas%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_305be28c26bc4990b86aadb17e316991%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22https%3A//stamen-tiles-%7Bs%7D.a.ssl.fastly.net/terrain/%7Bz%7D/%7Bx%7D/%7By%7D.jpg%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22attribution%22%3A%20%22Map%20tiles%20by%20%5Cu003ca%20href%3D%5C%22http%3A//stamen.com%5C%22%5Cu003eStamen%20Design%5Cu003c/a%5Cu003e%2C%20under%20%5Cu003ca%20href%3D%5C%22http%3A//creativecommons.org/licenses/by/3.0%5C%22%5Cu003eCC%20BY%203.0%5Cu003c/a%5Cu003e.%20Data%20by%20%5Cu0026copy%3B%20%5Cu003ca%20href%3D%5C%22http%3A//openstreetmap.org%5C%22%5Cu003eOpenStreetMap%5Cu003c/a%5Cu003e%2C%20under%20%5Cu003ca%20href%3D%5C%22http%3A//creativecommons.org/licenses/by-sa/3.0%5C%22%5Cu003eCC%20BY%20SA%5Cu003c/a%5Cu003e.%22%2C%20%22detectRetina%22%3A%20false%2C%20%22maxNativeZoom%22%3A%2018%2C%20%22maxZoom%22%3A%2018%2C%20%22minZoom%22%3A%200%2C%20%22noWrap%22%3A%20false%2C%20%22opacity%22%3A%201%2C%20%22subdomains%22%3A%20%22abc%22%2C%20%22tms%22%3A%20false%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_34be2092659d453298ac36df9dcfb85a%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20choropleth_f1ee5d3651ba45339daa75d470610bef%20%3D%20L.featureGroup%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_34be2092659d453298ac36df9dcfb85a%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20function%20geo_json_1b28918968244685a90882d3b2ac7954_styler%28feature%29%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20switch%28feature.properties.code%29%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20case%20%2211250%22%3A%20case%20%2211200%22%3A%20case%20%2211190%22%3A%20case%20%2211150%22%3A%20case%20%2211040%22%3A%20case%20%2211020%22%3A%20case%20%2211010%22%3A%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20%7B%22color%22%3A%20%22black%22%2C%20%22fillColor%22%3A%20%22%23d9f0a3%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22opacity%22%3A%200.2%2C%20%22weight%22%3A%201%7D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20case%20%2211240%22%3A%20case%20%2211220%22%3A%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20%7B%22color%22%3A%20%22black%22%2C%20%22fillColor%22%3A%20%22%2331a354%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22opacity%22%3A%200.2%2C%20%22weight%22%3A%201%7D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20case%20%2211230%22%3A%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20%7B%22color%22%3A%20%22black%22%2C%20%22fillColor%22%3A%20%22%23006837%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22opacity%22%3A%200.2%2C%20%22weight%22%3A%201%7D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20case%20%2211140%22%3A%20case%20%2211050%22%3A%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20%7B%22color%22%3A%20%22black%22%2C%20%22fillColor%22%3A%20%22%23addd8e%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22opacity%22%3A%200.2%2C%20%22weight%22%3A%201%7D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20case%20%2211030%22%3A%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20%7B%22color%22%3A%20%22black%22%2C%20%22fillColor%22%3A%20%22%2378c679%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22opacity%22%3A%200.2%2C%20%22weight%22%3A%201%7D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20default%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20%7B%22color%22%3A%20%22black%22%2C%20%22fillColor%22%3A%20%22%23ffffcc%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22opacity%22%3A%200.2%2C%20%22weight%22%3A%201%7D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%7D%0A%0A%20%20%20%20%20%20%20%20function%20geo_json_1b28918968244685a90882d3b2ac7954_onEachFeature%28feature%2C%20layer%29%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20layer.on%28%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%7D%29%3B%0A%20%20%20%20%20%20%20%20%7D%3B%0A%20%20%20%20%20%20%20%20var%20geo_json_1b28918968244685a90882d3b2ac7954%20%3D%20L.geoJson%28null%2C%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20onEachFeature%3A%20geo_json_1b28918968244685a90882d3b2ac7954_onEachFeature%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20style%3A%20geo_json_1b28918968244685a90882d3b2ac7954_styler%2C%0A%20%20%20%20%20%20%20%20%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20function%20geo_json_1b28918968244685a90882d3b2ac7954_add%20%28data%29%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20geo_json_1b28918968244685a90882d3b2ac7954%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20.addData%28data%29%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20.addTo%28choropleth_f1ee5d3651ba45339daa75d470610bef%29%3B%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20geo_json_1b28918968244685a90882d3b2ac7954_add%28%7B%22features%22%3A%20%5B%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.11519584981606%2C%2037.557533180704915%5D%2C%20%5B127.11879551821994%2C%2037.557222485451305%5D%2C%20%5B127.12146867175024%2C%2037.55986003393365%5D%2C%20%5B127.12435254630417%2C%2037.56144246249796%5D%2C%20%5B127.13593925898998%2C%2037.56564793048277%5D%2C%20%5B127.14930548011061%2C%2037.56892250303897%5D%2C%20%5B127.15511020940411%2C%2037.57093642128295%5D%2C%20%5B127.16683184366129%2C%2037.57672487388627%5D%2C%20%5B127.17038810813094%2C%2037.576465605301046%5D%2C%20%5B127.17607118428914%2C%2037.57678573961056%5D%2C%20%5B127.17905504160184%2C%2037.57791388161732%5D%2C%20%5B127.17747787800164%2C%2037.57448983055031%5D%2C%20%5B127.1781775408844%2C%2037.571481967974336%5D%2C%20%5B127.17995281860672%2C%2037.569309661290504%5D%2C%20%5B127.18122821955262%2C%2037.56636089217979%5D%2C%20%5B127.18169407550688%2C%2037.56286338914073%5D%2C%20%5B127.18408792330152%2C%2037.55814280369575%5D%2C%20%5B127.18350810324185%2C%2037.550053002101485%5D%2C%20%5B127.1852644795464%2C%2037.54888592026534%5D%2C%20%5B127.18480906237207%2C%2037.545296888806796%5D%2C%20%5B127.18543378919821%2C%2037.54260756512178%5D%2C%20%5B127.18364810569703%2C%2037.54241347907019%5D%2C%20%5B127.18116465939269%2C%2037.54384126582126%5D%2C%20%5B127.17770860504257%2C%2037.542414255164374%5D%2C%20%5B127.1744373170213%2C%2037.54277723796397%5D%2C%20%5B127.16830424484573%2C%2037.54145405702079%5D%2C%20%5B127.16530984307447%2C%2037.54221851258693%5D%2C%20%5B127.15566835118616%2C%2037.53119520531309%5D%2C%20%5B127.15538075046105%2C%2037.52652930087977%5D%2C%20%5B127.15154315998161%2C%2037.522828709496416%5D%2C%20%5B127.14981542759394%2C%2037.51926843453025%5D%2C%20%5B127.14791518058246%2C%2037.51918714979303%5D%2C%20%5B127.14684644251928%2C%2037.51661384818575%5D%2C%20%5B127.14672806823502%2C%2037.51415680680291%5D%2C%20%5B127.14532023498624%2C%2037.51464060108829%5D%2C%20%5B127.12123165719615%2C%2037.52528270089%5D%2C%20%5B127.12251496040881%2C%2037.52751810228347%5D%2C%20%5B127.12532464331997%2C%2037.53572787912298%5D%2C%20%5B127.12061313033807%2C%2037.538129867839416%5D%2C%20%5B127.1116764203608%2C%2037.540669955324965%5D%2C%20%5B127.11418412219375%2C%2037.54474592090681%5D%2C%20%5B127.11600200349189%2C%2037.55053147511706%5D%2C%20%5B127.11600943681239%2C%2037.55580061507081%5D%2C%20%5B127.11519584981606%2C%2037.557533180704915%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211250%22%2C%20%22name%22%3A%20%22%5Cuac15%5Cub3d9%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Gangdong-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.0690698130372%2C%2037.522279423505026%5D%2C%20%5B127.07496309841329%2C%2037.52091052765938%5D%2C%20%5B127.07968915919895%2C%2037.52077294752823%5D%2C%20%5B127.08639455667742%2C%2037.52161824624356%5D%2C%20%5B127.0943611414465%2C%2037.523984206117525%5D%2C%20%5B127.10087519791962%2C%2037.524841220167055%5D%2C%20%5B127.10484130265957%2C%2037.53120327509912%5D%2C%20%5B127.1116764203608%2C%2037.540669955324965%5D%2C%20%5B127.12061313033807%2C%2037.538129867839416%5D%2C%20%5B127.12532464331997%2C%2037.53572787912298%5D%2C%20%5B127.12251496040881%2C%2037.52751810228347%5D%2C%20%5B127.12123165719615%2C%2037.52528270089%5D%2C%20%5B127.14532023498624%2C%2037.51464060108829%5D%2C%20%5B127.14672806823502%2C%2037.51415680680291%5D%2C%20%5B127.14670263739373%2C%2037.512786602955565%5D%2C%20%5B127.14462782318448%2C%2037.511529542030715%5D%2C%20%5B127.14323992504048%2C%2037.50951977457089%5D%2C%20%5B127.1420864475393%2C%2037.50578973782813%5D%2C%20%5B127.14324986168657%2C%2037.502649431479774%5D%2C%20%5B127.1473517108062%2C%2037.50069754405746%5D%2C%20%5B127.14980119646964%2C%2037.50046502392898%5D%2C%20%5B127.15223804785649%2C%2037.50170492532197%5D%2C%20%5B127.15401160147654%2C%2037.500347919909956%5D%2C%20%5B127.16086308579277%2C%2037.49886565522751%5D%2C%20%5B127.1634944215765%2C%2037.497445406097484%5D%2C%20%5B127.16199885180917%2C%2037.49402577547199%5D%2C%20%5B127.16216448592424%2C%2037.491639601211624%5D%2C%20%5B127.16040295326431%2C%2037.4877818619403%5D%2C%20%5B127.15892216655034%2C%2037.486126922469445%5D%2C%20%5B127.15393282790794%2C%2037.48483891408459%5D%2C%20%5B127.15147990997852%2C%2037.47745324805034%5D%2C%20%5B127.1515017465549%2C%2037.475633269417585%5D%2C%20%5B127.14857580353349%2C%2037.47381386382568%5D%2C%20%5B127.14415938171436%2C%2037.473692508393505%5D%2C%20%5B127.14112111404233%2C%2037.470600239054825%5D%2C%20%5B127.13631568648837%2C%2037.47214721764681%5D%2C%20%5B127.13281577200672%2C%2037.47257463763244%5D%2C%20%5B127.13307493070646%2C%2037.468907694139894%5D%2C%20%5B127.13750907701846%2C%2037.46647058226059%5D%2C%20%5B127.13478085797742%2C%2037.46509524639883%5D%2C%20%5B127.1308437061496%2C%2037.46509985661207%5D%2C%20%5B127.12728991002369%2C%2037.46673043118672%5D%2C%20%5B127.12729757787379%2C%2037.46421548908766%5D%2C%20%5B127.12440571080893%2C%2037.46240445587048%5D%2C%20%5B127.12441393026374%2C%2037.46442715236855%5D%2C%20%5B127.12265007208167%2C%2037.46756987490939%5D%2C%20%5B127.11380709617507%2C%2037.479633334849325%5D%2C%20%5B127.1143875173445%2C%2037.48073157362458%5D%2C%20%5B127.11117085201238%2C%2037.485708381512445%5D%2C%20%5B127.1077937689776%2C%2037.48860875954992%5D%2C%20%5B127.10433125798602%2C%2037.490728250649646%5D%2C%20%5B127.0988509639092%2C%2037.49302529254068%5D%2C%20%5B127.08050206733888%2C%2037.49783151325589%5D%2C%20%5B127.0764808967127%2C%2037.498612695580306%5D%2C%20%5B127.0719146000724%2C%2037.50224013587669%5D%2C%20%5B127.06926628842805%2C%2037.51717796437217%5D%2C%20%5B127.06860425556381%2C%2037.51812758676938%5D%2C%20%5B127.0690698130372%2C%2037.522279423505026%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211240%22%2C%20%22name%22%3A%20%22%5Cuc1a1%5Cud30c%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Songpa-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.05867359288398%2C%2037.52629974922568%5D%2C%20%5B127.0690698130372%2C%2037.522279423505026%5D%2C%20%5B127.06860425556381%2C%2037.51812758676938%5D%2C%20%5B127.06926628842805%2C%2037.51717796437217%5D%2C%20%5B127.0719146000724%2C%2037.50224013587669%5D%2C%20%5B127.0764808967127%2C%2037.498612695580306%5D%2C%20%5B127.08050206733888%2C%2037.49783151325589%5D%2C%20%5B127.0988509639092%2C%2037.49302529254068%5D%2C%20%5B127.10433125798602%2C%2037.490728250649646%5D%2C%20%5B127.1077937689776%2C%2037.48860875954992%5D%2C%20%5B127.11117085201238%2C%2037.485708381512445%5D%2C%20%5B127.1143875173445%2C%2037.48073157362458%5D%2C%20%5B127.11380709617507%2C%2037.479633334849325%5D%2C%20%5B127.12265007208167%2C%2037.46756987490939%5D%2C%20%5B127.12441393026374%2C%2037.46442715236855%5D%2C%20%5B127.12440571080893%2C%2037.46240445587048%5D%2C%20%5B127.11957248720776%2C%2037.45936217377656%5D%2C%20%5B127.11885903757606%2C%2037.45578434878651%5D%2C%20%5B127.11535741803938%2C%2037.45722556454321%5D%2C%20%5B127.11413179478714%2C%2037.45875072431525%5D%2C%20%5B127.10841788934951%2C%2037.45972888008147%5D%2C%20%5B127.10561257180657%2C%2037.456815702518746%5D%2C%20%5B127.10032466845217%2C%2037.45598440195682%5D%2C%20%5B127.09842759318751%2C%2037.45862253857461%5D%2C%20%5B127.09712653145507%2C%2037.460848194480654%5D%2C%20%5B127.09039613625872%2C%2037.465520545397716%5D%2C%20%5B127.0866005634691%2C%2037.47006403057779%5D%2C%20%5B127.08640440578156%2C%2037.472697935184655%5D%2C%20%5B127.0802737559454%2C%2037.471973057552624%5D%2C%20%5B127.07602132306535%2C%2037.47005021331707%5D%2C%20%5B127.07476117209941%2C%2037.47199174520626%5D%2C%20%5B127.07231320371885%2C%2037.47234914588019%5D%2C%20%5B127.07135137525977%2C%2037.47107802023145%5D%2C%20%5B127.06463901956462%2C%2037.47003474490574%5D%2C%20%5B127.06371868919344%2C%2037.4661503234869%5D%2C%20%5B127.0588551029968%2C%2037.465611780743174%5D%2C%20%5B127.0559170481904%2C%2037.4659228914077%5D%2C%20%5B127.04713549385288%2C%2037.474479419244865%5D%2C%20%5B127.04345123620755%2C%2037.48276415595109%5D%2C%20%5B127.03621915098798%2C%2037.48175802427603%5D%2C%20%5B127.03372275812187%2C%2037.48674434662411%5D%2C%20%5B127.02265609299096%2C%2037.509970106251416%5D%2C%20%5B127.02038705349842%2C%2037.51771683027875%5D%2C%20%5B127.01917707838057%2C%2037.520085205855196%5D%2C%20%5B127.01397119667513%2C%2037.52503988289669%5D%2C%20%5B127.02302831890559%2C%2037.53231899582663%5D%2C%20%5B127.0269608080842%2C%2037.53484752757724%5D%2C%20%5B127.0319617044248%2C%2037.536064291470424%5D%2C%20%5B127.04806779588436%2C%2037.52970198575087%5D%2C%20%5B127.04903802830752%2C%2037.53140496708317%5D%2C%20%5B127.05116490008963%2C%2037.52975116557232%5D%2C%20%5B127.05867359288398%2C%2037.52629974922568%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211230%22%2C%20%22name%22%3A%20%22%5Cuac15%5Cub0a8%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Gangnam-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.01397119667513%2C%2037.52503988289669%5D%2C%20%5B127.01917707838057%2C%2037.520085205855196%5D%2C%20%5B127.02038705349842%2C%2037.51771683027875%5D%2C%20%5B127.02265609299096%2C%2037.509970106251416%5D%2C%20%5B127.03372275812187%2C%2037.48674434662411%5D%2C%20%5B127.03621915098798%2C%2037.48175802427603%5D%2C%20%5B127.04345123620755%2C%2037.48276415595109%5D%2C%20%5B127.04713549385288%2C%2037.474479419244865%5D%2C%20%5B127.0559170481904%2C%2037.4659228914077%5D%2C%20%5B127.0588551029968%2C%2037.465611780743174%5D%2C%20%5B127.06371868919344%2C%2037.4661503234869%5D%2C%20%5B127.06463901956462%2C%2037.47003474490574%5D%2C%20%5B127.07135137525977%2C%2037.47107802023145%5D%2C%20%5B127.07231320371885%2C%2037.47234914588019%5D%2C%20%5B127.07476117209941%2C%2037.47199174520626%5D%2C%20%5B127.07602132306535%2C%2037.47005021331707%5D%2C%20%5B127.0802737559454%2C%2037.471973057552624%5D%2C%20%5B127.08640440578156%2C%2037.472697935184655%5D%2C%20%5B127.0866005634691%2C%2037.47006403057779%5D%2C%20%5B127.09039613625872%2C%2037.465520545397716%5D%2C%20%5B127.09712653145507%2C%2037.460848194480654%5D%2C%20%5B127.09842759318751%2C%2037.45862253857461%5D%2C%20%5B127.09673714758375%2C%2037.45597209899094%5D%2C%20%5B127.09722129576434%2C%2037.45374822681991%5D%2C%20%5B127.09575982122928%2C%2037.45332980525459%5D%2C%20%5B127.09472136159357%2C%2037.450897902539175%5D%2C%20%5B127.09293250684935%2C%2037.450020696864506%5D%2C%20%5B127.09047890749349%2C%2037.44637473407341%5D%2C%20%5B127.09046928565951%2C%2037.44296826114185%5D%2C%20%5B127.0862358725955%2C%2037.44118543250345%5D%2C%20%5B127.08441983692467%2C%2037.4383879031398%5D%2C%20%5B127.07686576585408%2C%2037.43960712011444%5D%2C%20%5B127.07375875606847%2C%2037.43898415920535%5D%2C%20%5B127.07407631675713%2C%2037.43719357187124%5D%2C%20%5B127.07666569012467%2C%2037.43600054505559%5D%2C%20%5B127.07603719210388%2C%2037.43429107517633%5D%2C%20%5B127.07361291761038%2C%2037.43318474533595%5D%2C%20%5B127.07271473569163%2C%2037.42939553659177%5D%2C%20%5B127.0733788318578%2C%2037.42814484786288%5D%2C%20%5B127.06885354151605%2C%2037.42731815367302%5D%2C%20%5B127.06778107605433%2C%2037.426197424057314%5D%2C%20%5B127.06317558623768%2C%2037.4272916178182%5D%2C%20%5B127.05998777565219%2C%2037.4273224867045%5D%2C%20%5B127.05424556064274%2C%2037.42574929824175%5D%2C%20%5B127.05197080928994%2C%2037.42749842502397%5D%2C%20%5B127.04960937636815%2C%2037.42801020057224%5D%2C%20%5B127.04849622718511%2C%2037.430672016902065%5D%2C%20%5B127.04191594772718%2C%2037.43568906449929%5D%2C%20%5B127.0379686253535%2C%2037.43634417139204%5D%2C%20%5B127.03751805596916%2C%2037.438362795245276%5D%2C%20%5B127.04031700689708%2C%2037.44191429311459%5D%2C%20%5B127.03959875976469%2C%2037.443582700519194%5D%2C%20%5B127.0398984887873%2C%2037.44656106007936%5D%2C%20%5B127.03825522385397%2C%2037.448766467898395%5D%2C%20%5B127.03916301678915%2C%2037.45180237055558%5D%2C%20%5B127.03881782597922%2C%2037.45382039851715%5D%2C%20%5B127.03695436044305%2C%2037.45537592726508%5D%2C%20%5B127.03573307034355%2C%2037.4586703897792%5D%2C%20%5B127.03683946894893%2C%2037.46103886642786%5D%2C%20%5B127.03337331972266%2C%2037.462966775127626%5D%2C%20%5B127.02820831539744%2C%2037.455700834295826%5D%2C%20%5B127.02263694708293%2C%2037.45335816711404%5D%2C%20%5B127.01827371395349%2C%2037.4525593623189%5D%2C%20%5B127.01316256500736%2C%2037.45257906566242%5D%2C%20%5B127.01110931353561%2C%2037.45456166745922%5D%2C%20%5B127.00836380369604%2C%2037.45936868039916%5D%2C%20%5B127.00738548779366%2C%2037.459815333664274%5D%2C%20%5B127.00552362663117%2C%2037.46445102893571%5D%2C%20%5B127.00008523087483%2C%2037.46455774995882%5D%2C%20%5B126.99837609897334%2C%2037.46390918086617%5D%2C%20%5B126.99932142462428%2C%2037.46113351815481%5D%2C%20%5B126.99893310307874%2C%2037.459376062410314%5D%2C%20%5B126.9953054179472%2C%2037.45860121328987%5D%2C%20%5B126.99072073195462%2C%2037.455326143310025%5D%2C%20%5B126.98956736277059%2C%2037.457600756400446%5D%2C%20%5B126.99026416700147%2C%2037.46271603227842%5D%2C%20%5B126.98896316546526%2C%2037.465041871263544%5D%2C%20%5B126.98662755598336%2C%2037.466937278295305%5D%2C%20%5B126.9846374349825%2C%2037.46996301876212%5D%2C%20%5B126.98367668291802%2C%2037.473856492692086%5D%2C%20%5B126.98500224966135%2C%2037.49356837311327%5D%2C%20%5B126.9871787157338%2C%2037.49719505997539%5D%2C%20%5B126.9832495184969%2C%2037.49948552591205%5D%2C%20%5B126.98241580381733%2C%2037.50120029501884%5D%2C%20%5B126.98223807916081%2C%2037.509314966770326%5D%2C%20%5B126.98458580602838%2C%2037.51070333105394%5D%2C%20%5B126.98948242685965%2C%2037.5108780134613%5D%2C%20%5B126.99148001917875%2C%2037.50990503427709%5D%2C%20%5B127.00011962020382%2C%2037.513901653034374%5D%2C%20%5B127.00583392114271%2C%2037.516905128452926%5D%2C%20%5B127.00818058911564%2C%2037.51877313923874%5D%2C%20%5B127.01022186960886%2C%2037.522020085671926%5D%2C%20%5B127.01397119667513%2C%2037.52503988289669%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211220%22%2C%20%22name%22%3A%20%22%5Cuc11c%5Cucd08%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Seocho-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B126.98367668291802%2C%2037.473856492692086%5D%2C%20%5B126.9846374349825%2C%2037.46996301876212%5D%2C%20%5B126.98662755598336%2C%2037.466937278295305%5D%2C%20%5B126.98896316546526%2C%2037.465041871263544%5D%2C%20%5B126.99026416700147%2C%2037.46271603227842%5D%2C%20%5B126.98956736277059%2C%2037.457600756400446%5D%2C%20%5B126.99072073195462%2C%2037.455326143310025%5D%2C%20%5B126.98484249930785%2C%2037.45391909788938%5D%2C%20%5B126.9829408096241%2C%2037.450206782833206%5D%2C%20%5B126.97835022660695%2C%2037.447659155806164%5D%2C%20%5B126.97608193440507%2C%2037.44478918862847%5D%2C%20%5B126.9731300196836%2C%2037.444722870088114%5D%2C%20%5B126.96650852936277%2C%2037.44276983031553%5D%2C%20%5B126.96618702895445%2C%2037.439376482995094%5D%2C%20%5B126.96520439085143%2C%2037.438249784006246%5D%2C%20%5B126.9614877541633%2C%2037.437956805629675%5D%2C%20%5B126.96054904645496%2C%2037.43673997185797%5D%2C%20%5B126.95527369898224%2C%2037.43673711968809%5D%2C%20%5B126.9473688393239%2C%2037.4347689647565%5D%2C%20%5B126.94440352544498%2C%2037.43476162120059%5D%2C%20%5B126.9415292183489%2C%2037.43315139671158%5D%2C%20%5B126.94037501670272%2C%2037.43462213966344%5D%2C%20%5B126.9405640311191%2C%2037.437501011208845%5D%2C%20%5B126.9376981355065%2C%2037.44041709605302%5D%2C%20%5B126.93312955918624%2C%2037.44290014710262%5D%2C%20%5B126.93309127096236%2C%2037.44533734785938%5D%2C%20%5B126.93084408056525%2C%2037.447382928333994%5D%2C%20%5B126.92527839995981%2C%2037.45161884570837%5D%2C%20%5B126.9245243450059%2C%2037.45392293573877%5D%2C%20%5B126.91887928082078%2C%2037.45495082787016%5D%2C%20%5B126.9167728146601%2C%2037.45490566423789%5D%2C%20%5B126.91641538472182%2C%2037.45870245071989%5D%2C%20%5B126.91495285904284%2C%2037.461166184511065%5D%2C%20%5B126.91584245173756%2C%2037.462474576247985%5D%2C%20%5B126.91374656127704%2C%2037.46375990852858%5D%2C%20%5B126.91032166997253%2C%2037.469818629944285%5D%2C%20%5B126.91280966667205%2C%2037.47083063715413%5D%2C%20%5B126.91405961426707%2C%2037.47416764846582%5D%2C%20%5B126.9115784808617%2C%2037.4753960485947%5D%2C%20%5B126.91181700249076%2C%2037.47814319736339%5D%2C%20%5B126.90276666415615%2C%2037.47652007992712%5D%2C%20%5B126.90156094129895%2C%2037.47753842789901%5D%2C%20%5B126.90531975801812%2C%2037.48218087575429%5D%2C%20%5B126.90805655355825%2C%2037.48218338568103%5D%2C%20%5B126.91533979779165%2C%2037.484392208242134%5D%2C%20%5B126.91916807529428%2C%2037.48660606817164%5D%2C%20%5B126.92639563063156%2C%2037.48715979752876%5D%2C%20%5B126.92869559665061%2C%2037.49132126714011%5D%2C%20%5B126.92981699800066%2C%2037.49218420958284%5D%2C%20%5B126.93346386636452%2C%2037.49043826776755%5D%2C%20%5B126.93669800083833%2C%2037.49026778789087%5D%2C%20%5B126.93844070234584%2C%2037.4893532861132%5D%2C%20%5B126.94373156012337%2C%2037.48938843727846%5D%2C%20%5B126.94922661389508%2C%2037.49125437495649%5D%2C%20%5B126.95396955055433%2C%2037.48955250290043%5D%2C%20%5B126.9559655046206%2C%2037.48820165625994%5D%2C%20%5B126.95881175306481%2C%2037.48874989165474%5D%2C%20%5B126.96329694970828%2C%2037.4905835370787%5D%2C%20%5B126.96291787066104%2C%2037.48803272157808%5D%2C%20%5B126.96443983219191%2C%2037.48442261322104%5D%2C%20%5B126.9634428120456%2C%2037.48067931902171%5D%2C%20%5B126.9725891850662%2C%2037.472561363278125%5D%2C%20%5B126.97901795539295%2C%2037.47376525108475%5D%2C%20%5B126.98367668291802%2C%2037.473856492692086%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211210%22%2C%20%22name%22%3A%20%22%5Cuad00%5Cuc545%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Gwanak-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B126.98223807916081%2C%2037.509314966770326%5D%2C%20%5B126.98241580381733%2C%2037.50120029501884%5D%2C%20%5B126.9832495184969%2C%2037.49948552591205%5D%2C%20%5B126.9871787157338%2C%2037.49719505997539%5D%2C%20%5B126.98500224966135%2C%2037.49356837311327%5D%2C%20%5B126.98367668291802%2C%2037.473856492692086%5D%2C%20%5B126.97901795539295%2C%2037.47376525108475%5D%2C%20%5B126.9725891850662%2C%2037.472561363278125%5D%2C%20%5B126.9634428120456%2C%2037.48067931902171%5D%2C%20%5B126.96443983219191%2C%2037.48442261322104%5D%2C%20%5B126.96291787066104%2C%2037.48803272157808%5D%2C%20%5B126.96329694970828%2C%2037.4905835370787%5D%2C%20%5B126.95881175306481%2C%2037.48874989165474%5D%2C%20%5B126.9559655046206%2C%2037.48820165625994%5D%2C%20%5B126.95396955055433%2C%2037.48955250290043%5D%2C%20%5B126.94922661389508%2C%2037.49125437495649%5D%2C%20%5B126.94373156012337%2C%2037.48938843727846%5D%2C%20%5B126.93844070234584%2C%2037.4893532861132%5D%2C%20%5B126.93669800083833%2C%2037.49026778789087%5D%2C%20%5B126.93346386636452%2C%2037.49043826776755%5D%2C%20%5B126.92981699800066%2C%2037.49218420958284%5D%2C%20%5B126.92869559665061%2C%2037.49132126714011%5D%2C%20%5B126.92639563063156%2C%2037.48715979752876%5D%2C%20%5B126.91916807529428%2C%2037.48660606817164%5D%2C%20%5B126.91533979779165%2C%2037.484392208242134%5D%2C%20%5B126.90805655355825%2C%2037.48218338568103%5D%2C%20%5B126.90531975801812%2C%2037.48218087575429%5D%2C%20%5B126.91461888105147%2C%2037.493581242537296%5D%2C%20%5B126.92177893174825%2C%2037.494889877415176%5D%2C%20%5B126.9232469824303%2C%2037.49928149943772%5D%2C%20%5B126.92749463764046%2C%2037.50985955934051%5D%2C%20%5B126.92919938332032%2C%2037.51019685838638%5D%2C%20%5B126.92810628828279%2C%2037.51329595732015%5D%2C%20%5B126.93453120783802%2C%2037.5128512712934%5D%2C%20%5B126.94407346439685%2C%2037.51463101265907%5D%2C%20%5B126.95249990298159%2C%2037.51722500741813%5D%2C%20%5B126.95551848909955%2C%2037.514736123015844%5D%2C%20%5B126.95950268374823%2C%2037.51249532165974%5D%2C%20%5B126.96670111119346%2C%2037.50997579058433%5D%2C%20%5B126.98223807916081%2C%2037.509314966770326%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211200%22%2C%20%22name%22%3A%20%22%5Cub3d9%5Cuc791%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Dongjak-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B126.89184663862764%2C%2037.547373974997114%5D%2C%20%5B126.90281125423546%2C%2037.54133630026083%5D%2C%20%5B126.90829204147958%2C%2037.539206065016785%5D%2C%20%5B126.93132557924062%2C%2037.53415416375281%5D%2C%20%5B126.93680342222562%2C%2037.53344577095931%5D%2C%20%5B126.94566733083212%2C%2037.526617542453366%5D%2C%20%5B126.9488066464266%2C%2037.52424913252661%5D%2C%20%5B126.95003825019774%2C%2037.520781022055274%5D%2C%20%5B126.95249990298159%2C%2037.51722500741813%5D%2C%20%5B126.94407346439685%2C%2037.51463101265907%5D%2C%20%5B126.93453120783802%2C%2037.5128512712934%5D%2C%20%5B126.92810628828279%2C%2037.51329595732015%5D%2C%20%5B126.92919938332032%2C%2037.51019685838638%5D%2C%20%5B126.92749463764046%2C%2037.50985955934051%5D%2C%20%5B126.9232469824303%2C%2037.49928149943772%5D%2C%20%5B126.92177893174825%2C%2037.494889877415176%5D%2C%20%5B126.91461888105147%2C%2037.493581242537296%5D%2C%20%5B126.90531975801812%2C%2037.48218087575429%5D%2C%20%5B126.90260188508027%2C%2037.48282626920736%5D%2C%20%5B126.89861362258316%2C%2037.48625405368759%5D%2C%20%5B126.89581061458084%2C%2037.49391346191318%5D%2C%20%5B126.89549571721683%2C%2037.50033127915717%5D%2C%20%5B126.89594776782485%2C%2037.504675281309176%5D%2C%20%5B126.89253696873205%2C%2037.50875582175844%5D%2C%20%5B126.88156402353862%2C%2037.513970034765684%5D%2C%20%5B126.88191372979959%2C%2037.51939416754389%5D%2C%20%5B126.88260109180834%2C%2037.52242565920786%5D%2C%20%5B126.88382776477316%2C%2037.52352483439659%5D%2C%20%5B126.88904768965743%2C%2037.525856504359034%5D%2C%20%5B126.89057378109133%2C%2037.52792091672938%5D%2C%20%5B126.89213569003026%2C%2037.52757969298779%5D%2C%20%5B126.89361739665432%2C%2037.53033899535983%5D%2C%20%5B126.89339176028666%2C%2037.533030814524004%5D%2C%20%5B126.88938421776182%2C%2037.54060159145325%5D%2C%20%5B126.88825757860099%2C%2037.54079733630232%5D%2C%20%5B126.88736718003831%2C%2037.54350482420959%5D%2C%20%5B126.89184663862764%2C%2037.547373974997114%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211190%22%2C%20%22name%22%3A%20%22%5Cuc601%5Cub4f1%5Cud3ec%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Yeongdeungpo-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B126.90156094129895%2C%2037.47753842789901%5D%2C%20%5B126.90276666415615%2C%2037.47652007992712%5D%2C%20%5B126.91181700249076%2C%2037.47814319736339%5D%2C%20%5B126.9115784808617%2C%2037.4753960485947%5D%2C%20%5B126.91405961426707%2C%2037.47416764846582%5D%2C%20%5B126.91280966667205%2C%2037.47083063715413%5D%2C%20%5B126.91032166997253%2C%2037.469818629944285%5D%2C%20%5B126.91374656127704%2C%2037.46375990852858%5D%2C%20%5B126.91584245173756%2C%2037.462474576247985%5D%2C%20%5B126.91495285904284%2C%2037.461166184511065%5D%2C%20%5B126.91641538472182%2C%2037.45870245071989%5D%2C%20%5B126.9167728146601%2C%2037.45490566423789%5D%2C%20%5B126.91887928082078%2C%2037.45495082787016%5D%2C%20%5B126.9245243450059%2C%2037.45392293573877%5D%2C%20%5B126.92527839995981%2C%2037.45161884570837%5D%2C%20%5B126.93084408056525%2C%2037.447382928333994%5D%2C%20%5B126.9255681646224%2C%2037.44377627841776%5D%2C%20%5B126.92318732232543%2C%2037.44131494528283%5D%2C%20%5B126.92199241717724%2C%2037.43848070111403%5D%2C%20%5B126.92004664118903%2C%2037.43708741729147%5D%2C%20%5B126.91641802826501%2C%2037.43722730676683%5D%2C%20%5B126.91344497343947%2C%2037.43474365720405%5D%2C%20%5B126.91082677485002%2C%2037.43100963341445%5D%2C%20%5B126.90487628022693%2C%2037.43129996372531%5D%2C%20%5B126.90480610062333%2C%2037.433123813599884%5D%2C%20%5B126.90147608259903%2C%2037.4353323892334%5D%2C%20%5B126.90081465102077%2C%2037.436498759868456%5D%2C%20%5B126.90105333433885%2C%2037.44031094924801%5D%2C%20%5B126.90031753853916%2C%2037.441844173154756%5D%2C%20%5B126.89768314223053%2C%2037.442906858137974%5D%2C%20%5B126.89812451590424%2C%2037.44551117837958%5D%2C%20%5B126.89616541233094%2C%2037.44983024861048%5D%2C%20%5B126.8947204038491%2C%2037.4491097366517%5D%2C%20%5B126.89157226377172%2C%2037.449944282396714%5D%2C%20%5B126.89113632562855%2C%2037.45236898205529%5D%2C%20%5B126.88831381741582%2C%2037.45357939777933%5D%2C%20%5B126.88826482749008%2C%2037.45663270152334%5D%2C%20%5B126.89074130059865%2C%2037.45966561796733%5D%2C%20%5B126.88198617469523%2C%2037.469975509557976%5D%2C%20%5B126.87874781843654%2C%2037.47475533620029%5D%2C%20%5B126.87553760781829%2C%2037.48186220368496%5D%2C%20%5B126.87683271502428%2C%2037.482576591607305%5D%2C%20%5B126.88079109105627%2C%2037.48378287831426%5D%2C%20%5B126.8827497570056%2C%2037.48316340563878%5D%2C%20%5B126.88803217321346%2C%2037.47975290808737%5D%2C%20%5B126.89116882970154%2C%2037.47681803032367%5D%2C%20%5B126.89689977603885%2C%2037.47570593888643%5D%2C%20%5B126.90104536043339%2C%2037.47614746588584%5D%2C%20%5B126.90156094129895%2C%2037.47753842789901%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211180%22%2C%20%22name%22%3A%20%22%5Cuae08%5Cucc9c%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Geumcheon-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B126.82688081517314%2C%2037.50548972232896%5D%2C%20%5B126.8312211095328%2C%2037.50541191299514%5D%2C%20%5B126.8341914436862%2C%2037.50238060850881%5D%2C%20%5B126.8385046623421%2C%2037.49965438083505%5D%2C%20%5B126.84270526111828%2C%2037.501190052842375%5D%2C%20%5B126.8421330711049%2C%2037.50273297478404%5D%2C%20%5B126.84689618668257%2C%2037.50287086505149%5D%2C%20%5B126.84730715497963%2C%2037.50522159123912%5D%2C%20%5B126.85079659934003%2C%2037.50601938589165%5D%2C%20%5B126.8521985385098%2C%2037.507310601432195%5D%2C%20%5B126.8549910115563%2C%2037.507774255244875%5D%2C%20%5B126.85767018319959%2C%2037.50643578404816%5D%2C%20%5B126.8602448049945%2C%2037.50714262450898%5D%2C%20%5B126.86219488732715%2C%2037.50388438562706%5D%2C%20%5B126.86454914535695%2C%2037.50388350542007%5D%2C%20%5B126.86525729660478%2C%2037.502448483868896%5D%2C%20%5B126.86795186545288%2C%2037.502755995885714%5D%2C%20%5B126.87109220473953%2C%2037.50203922322315%5D%2C%20%5B126.87432642792102%2C%2037.50260287829134%5D%2C%20%5B126.87556919864359%2C%2037.505720626918%5D%2C%20%5B126.88054908630636%2C%2037.51148026214697%5D%2C%20%5B126.88156402353862%2C%2037.513970034765684%5D%2C%20%5B126.89253696873205%2C%2037.50875582175844%5D%2C%20%5B126.89594776782485%2C%2037.504675281309176%5D%2C%20%5B126.89549571721683%2C%2037.50033127915717%5D%2C%20%5B126.89581061458084%2C%2037.49391346191318%5D%2C%20%5B126.89861362258316%2C%2037.48625405368759%5D%2C%20%5B126.90260188508027%2C%2037.48282626920736%5D%2C%20%5B126.90531975801812%2C%2037.48218087575429%5D%2C%20%5B126.90156094129895%2C%2037.47753842789901%5D%2C%20%5B126.90104536043339%2C%2037.47614746588584%5D%2C%20%5B126.89689977603885%2C%2037.47570593888643%5D%2C%20%5B126.89116882970154%2C%2037.47681803032367%5D%2C%20%5B126.88803217321346%2C%2037.47975290808737%5D%2C%20%5B126.8827497570056%2C%2037.48316340563878%5D%2C%20%5B126.88079109105627%2C%2037.48378287831426%5D%2C%20%5B126.87683271502428%2C%2037.482576591607305%5D%2C%20%5B126.87926901338844%2C%2037.4851363312754%5D%2C%20%5B126.87807822721697%2C%2037.486247661404484%5D%2C%20%5B126.87500855887376%2C%2037.485529408954044%5D%2C%20%5B126.87499999632084%2C%2037.48723558386031%5D%2C%20%5B126.86985088086946%2C%2037.490972856926746%5D%2C%20%5B126.86690708512153%2C%2037.48850048185492%5D%2C%20%5B126.86334463261252%2C%2037.48702105213313%5D%2C%20%5B126.85979281993241%2C%2037.48309390333688%5D%2C%20%5B126.8571926758503%2C%2037.482400254369296%5D%2C%20%5B126.85582775745682%2C%2037.48008159809108%5D%2C%20%5B126.85397991619827%2C%2037.47882533996402%5D%2C%20%5B126.84914329670241%2C%2037.4792573077648%5D%2C%20%5B126.84804505350411%2C%2037.478160467930344%5D%2C%20%5B126.84762676054953%2C%2037.47146723936323%5D%2C%20%5B126.84154264465728%2C%2037.4728980419%5D%2C%20%5B126.83754691879544%2C%2037.472514053936045%5D%2C%20%5B126.83388005989259%2C%2037.4747683882548%5D%2C%20%5B126.83136668931549%2C%2037.47344154955525%5D%2C%20%5B126.82660025197819%2C%2037.47364544953152%5D%2C%20%5B126.82419365698964%2C%2037.472951080902234%5D%2C%20%5B126.82179895415682%2C%2037.47518076838956%5D%2C%20%5B126.82206789884786%2C%2037.47889514031285%5D%2C%20%5B126.82139445214092%2C%2037.48136482338644%5D%2C%20%5B126.82208805042494%2C%2037.48299688518288%5D%2C%20%5B126.82558489219227%2C%2037.48497306755705%5D%2C%20%5B126.82482114643597%2C%2037.48717399070965%5D%2C%20%5B126.81993148808915%2C%2037.48877864988337%5D%2C%20%5B126.81649745505314%2C%2037.490455810141476%5D%2C%20%5B126.81480709048222%2C%2037.493362284349615%5D%2C%20%5B126.81518179823208%2C%2037.495233793642186%5D%2C%20%5B126.81771493003457%2C%2037.494748098657496%5D%2C%20%5B126.8208295567048%2C%2037.49594689979241%5D%2C%20%5B126.82204657426578%2C%2037.49851634597747%5D%2C%20%5B126.82367963750009%2C%2037.49925830108059%5D%2C%20%5B126.82504736331406%2C%2037.50302612640443%5D%2C%20%5B126.82469248121312%2C%2037.50496239513798%5D%2C%20%5B126.82688081517314%2C%2037.50548972232896%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211170%22%2C%20%22name%22%3A%20%22%5Cuad6c%5Cub85c%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Guro-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B126.85984199399667%2C%2037.571847855292745%5D%2C%20%5B126.85950389772532%2C%2037.5682771531189%5D%2C%20%5B126.8604929702619%2C%2037.566825621733905%5D%2C%20%5B126.86837407967381%2C%2037.56309561411933%5D%2C%20%5B126.87997924964915%2C%2037.55510740490174%5D%2C%20%5B126.89184663862764%2C%2037.547373974997114%5D%2C%20%5B126.88736718003831%2C%2037.54350482420959%5D%2C%20%5B126.88825757860099%2C%2037.54079733630232%5D%2C%20%5B126.8872486543506%2C%2037.54079100234749%5D%2C%20%5B126.88280520161935%2C%2037.545072121233225%5D%2C%20%5B126.8761093656006%2C%2037.54412882794802%5D%2C%20%5B126.8727924099192%2C%2037.544853699294116%5D%2C%20%5B126.86637464321238%2C%2037.54859191094823%5D%2C%20%5B126.86426635332332%2C%2037.54172958759955%5D%2C%20%5B126.86582129720519%2C%2037.53817151116851%5D%2C%20%5B126.8655037497872%2C%2037.53382926555605%5D%2C%20%5B126.86610073476395%2C%2037.52699964144669%5D%2C%20%5B126.85098397861944%2C%2037.525098716169985%5D%2C%20%5B126.84257291943153%2C%2037.52373707805596%5D%2C%20%5B126.836555914069%2C%2037.53367208325903%5D%2C%20%5B126.83716591765655%2C%2037.534935320492906%5D%2C%20%5B126.83522688458329%2C%2037.539042988809484%5D%2C%20%5B126.8325348945036%2C%2037.538953250433295%5D%2C%20%5B126.83191667070415%2C%2037.54146500511403%5D%2C%20%5B126.83017074557299%2C%2037.542611079015344%5D%2C%20%5B126.82889818288362%2C%2037.53909381305992%5D%2C%20%5B126.8242331426722%2C%2037.53788078753248%5D%2C%20%5B126.81674221631081%2C%2037.5378396500627%5D%2C%20%5B126.81246052552456%2C%2037.538810793377344%5D%2C%20%5B126.81143604908785%2C%2037.54033621168525%5D%2C%20%5B126.80542840499083%2C%2037.54008921830378%5D%2C%20%5B126.80185404828612%2C%2037.537645443377826%5D%2C%20%5B126.80090914554204%2C%2037.53503009075454%5D%2C%20%5B126.79688612254975%2C%2037.53302974862096%5D%2C%20%5B126.79582133969424%2C%2037.536641561833754%5D%2C%20%5B126.79693641290046%2C%2037.53865858611534%5D%2C%20%5B126.79388711477147%2C%2037.53902211451394%5D%2C%20%5B126.7938616681597%2C%2037.54104361952839%5D%2C%20%5B126.79075533377627%2C%2037.54165251407983%5D%2C%20%5B126.78880225462409%2C%2037.54353706379955%5D%2C%20%5B126.78198339188025%2C%2037.543449601019624%5D%2C%20%5B126.77756215424237%2C%2037.54611355396897%5D%2C%20%5B126.77324417717703%2C%2037.5459123450554%5D%2C%20%5B126.76977011413412%2C%2037.55052082471595%5D%2C%20%5B126.76700465024426%2C%2037.552821566629916%5D%2C%20%5B126.77074629769308%2C%2037.55296836994276%5D%2C%20%5B126.77145103135192%2C%2037.55434307460708%5D%2C%20%5B126.77879087345151%2C%2037.55919525318415%5D%2C%20%5B126.77889121370164%2C%2037.5613614424496%5D%2C%20%5B126.77671213061004%2C%2037.5645429268672%5D%2C%20%5B126.77986476402239%2C%2037.564245932540665%5D%2C%20%5B126.78252024622797%2C%2037.565367145342954%5D%2C%20%5B126.78471963959866%2C%2037.56745669198498%5D%2C%20%5B126.78398460461828%2C%2037.56905814599349%5D%2C%20%5B126.78496836516075%2C%2037.57090748246567%5D%2C%20%5B126.79172648531066%2C%2037.57472630536462%5D%2C%20%5B126.79539987549317%2C%2037.57451148875729%5D%2C%20%5B126.79523455851671%2C%2037.57760277954844%5D%2C%20%5B126.7958606942207%2C%2037.58019957877273%5D%2C%20%5B126.79799133400897%2C%2037.58036436587069%5D%2C%20%5B126.80091228188235%2C%2037.5854309825683%5D%2C%20%5B126.80289369340177%2C%2037.58621464221784%5D%2C%20%5B126.80131704756816%2C%2037.58839794302751%5D%2C%20%5B126.80149834313248%2C%2037.59012749570681%5D%2C%20%5B126.7996634054858%2C%2037.59296530943065%5D%2C%20%5B126.79910601240701%2C%2037.59569886491464%5D%2C%20%5B126.80046544382346%2C%2037.59827267924192%5D%2C%20%5B126.80198459129242%2C%2037.598541940075755%5D%2C%20%5B126.80268446118524%2C%2037.601312560472834%5D%2C%20%5B126.80393696882469%2C%2037.601857300987895%5D%2C%20%5B126.80759006979085%2C%2037.60089755124775%5D%2C%20%5B126.81814502537962%2C%2037.591566052513244%5D%2C%20%5B126.82251438477105%2C%2037.5880430810082%5D%2C%20%5B126.82891304761237%2C%2037.5855611764797%5D%2C%20%5B126.85302823436479%2C%2037.57282468882299%5D%2C%20%5B126.85984199399667%2C%2037.571847855292745%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211160%22%2C%20%22name%22%3A%20%22%5Cuac15%5Cuc11c%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Gangseo-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B126.8242331426722%2C%2037.53788078753248%5D%2C%20%5B126.82889818288362%2C%2037.53909381305992%5D%2C%20%5B126.83017074557299%2C%2037.542611079015344%5D%2C%20%5B126.83191667070415%2C%2037.54146500511403%5D%2C%20%5B126.8325348945036%2C%2037.538953250433295%5D%2C%20%5B126.83522688458329%2C%2037.539042988809484%5D%2C%20%5B126.83716591765655%2C%2037.534935320492906%5D%2C%20%5B126.836555914069%2C%2037.53367208325903%5D%2C%20%5B126.84257291943153%2C%2037.52373707805596%5D%2C%20%5B126.85098397861944%2C%2037.525098716169985%5D%2C%20%5B126.86610073476395%2C%2037.52699964144669%5D%2C%20%5B126.8655037497872%2C%2037.53382926555605%5D%2C%20%5B126.86582129720519%2C%2037.53817151116851%5D%2C%20%5B126.86426635332332%2C%2037.54172958759955%5D%2C%20%5B126.86637464321238%2C%2037.54859191094823%5D%2C%20%5B126.8727924099192%2C%2037.544853699294116%5D%2C%20%5B126.8761093656006%2C%2037.54412882794802%5D%2C%20%5B126.88280520161935%2C%2037.545072121233225%5D%2C%20%5B126.8872486543506%2C%2037.54079100234749%5D%2C%20%5B126.88825757860099%2C%2037.54079733630232%5D%2C%20%5B126.88938421776182%2C%2037.54060159145325%5D%2C%20%5B126.89339176028666%2C%2037.533030814524004%5D%2C%20%5B126.89361739665432%2C%2037.53033899535983%5D%2C%20%5B126.89213569003026%2C%2037.52757969298779%5D%2C%20%5B126.89057378109133%2C%2037.52792091672938%5D%2C%20%5B126.88904768965743%2C%2037.525856504359034%5D%2C%20%5B126.88382776477316%2C%2037.52352483439659%5D%2C%20%5B126.88260109180834%2C%2037.52242565920786%5D%2C%20%5B126.88191372979959%2C%2037.51939416754389%5D%2C%20%5B126.88156402353862%2C%2037.513970034765684%5D%2C%20%5B126.88054908630636%2C%2037.51148026214697%5D%2C%20%5B126.87556919864359%2C%2037.505720626918%5D%2C%20%5B126.87432642792102%2C%2037.50260287829134%5D%2C%20%5B126.87109220473953%2C%2037.50203922322315%5D%2C%20%5B126.86795186545288%2C%2037.502755995885714%5D%2C%20%5B126.86525729660478%2C%2037.502448483868896%5D%2C%20%5B126.86454914535695%2C%2037.50388350542007%5D%2C%20%5B126.86219488732715%2C%2037.50388438562706%5D%2C%20%5B126.8602448049945%2C%2037.50714262450898%5D%2C%20%5B126.85767018319959%2C%2037.50643578404816%5D%2C%20%5B126.8549910115563%2C%2037.507774255244875%5D%2C%20%5B126.8521985385098%2C%2037.507310601432195%5D%2C%20%5B126.85079659934003%2C%2037.50601938589165%5D%2C%20%5B126.84730715497963%2C%2037.50522159123912%5D%2C%20%5B126.84689618668257%2C%2037.50287086505149%5D%2C%20%5B126.8421330711049%2C%2037.50273297478404%5D%2C%20%5B126.84270526111828%2C%2037.501190052842375%5D%2C%20%5B126.8385046623421%2C%2037.49965438083505%5D%2C%20%5B126.8341914436862%2C%2037.50238060850881%5D%2C%20%5B126.8312211095328%2C%2037.50541191299514%5D%2C%20%5B126.82688081517314%2C%2037.50548972232896%5D%2C%20%5B126.82609821744505%2C%2037.507816771867255%5D%2C%20%5B126.82665326173496%2C%2037.510416148524136%5D%2C%20%5B126.82529622550616%2C%2037.513385210403136%5D%2C%20%5B126.82763384465879%2C%2037.516923263281946%5D%2C%20%5B126.8273575421771%2C%2037.52002629298419%5D%2C%20%5B126.83054711509516%2C%2037.52390261457357%5D%2C%20%5B126.82938708105253%2C%2037.5268052663749%5D%2C%20%5B126.8276955169658%2C%2037.52706155314193%5D%2C%20%5B126.82389942108053%2C%2037.53199443525418%5D%2C%20%5B126.8242331426722%2C%2037.53788078753248%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211150%22%2C%20%22name%22%3A%20%22%5Cuc591%5Cucc9c%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Yangcheon-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B126.90522065831053%2C%2037.57409700522574%5D%2C%20%5B126.90370105002282%2C%2037.57266722738834%5D%2C%20%5B126.90687243065778%2C%2037.57059762097416%5D%2C%20%5B126.92189004506%2C%2037.56391798973296%5D%2C%20%5B126.92778174854314%2C%2037.562495624023775%5D%2C%20%5B126.93034243306369%2C%2037.56054720372433%5D%2C%20%5B126.92881397392811%2C%2037.558202848902%5D%2C%20%5B126.92872097190046%2C%2037.556034533941734%5D%2C%20%5B126.93898161798973%2C%2037.552310003728124%5D%2C%20%5B126.94314477022111%2C%2037.5536460848349%5D%2C%20%5B126.95916768398142%2C%2037.55468176051932%5D%2C%20%5B126.96080686210321%2C%2037.55386236039188%5D%2C%20%5B126.96358226710812%2C%2037.55605635475154%5D%2C%20%5B126.96519694864509%2C%2037.55362533505407%5D%2C%20%5B126.96380145704283%2C%2037.55254525759954%5D%2C%20%5B126.96448570553055%2C%2037.548705692021635%5D%2C%20%5B126.96604189284825%2C%2037.546894141748815%5D%2C%20%5B126.96401856825223%2C%2037.54584596959762%5D%2C%20%5B126.96231305253527%2C%2037.543511558047456%5D%2C%20%5B126.9605977865388%2C%2037.542661954880806%5D%2C%20%5B126.95926437828754%2C%2037.53897908363236%5D%2C%20%5B126.95340780191557%2C%2037.533494726370755%5D%2C%20%5B126.94717864071288%2C%2037.53213495568077%5D%2C%20%5B126.94566733083212%2C%2037.526617542453366%5D%2C%20%5B126.93680342222562%2C%2037.53344577095931%5D%2C%20%5B126.93132557924062%2C%2037.53415416375281%5D%2C%20%5B126.90829204147958%2C%2037.539206065016785%5D%2C%20%5B126.90281125423546%2C%2037.54133630026083%5D%2C%20%5B126.89184663862764%2C%2037.547373974997114%5D%2C%20%5B126.87997924964915%2C%2037.55510740490174%5D%2C%20%5B126.86837407967381%2C%2037.56309561411933%5D%2C%20%5B126.8604929702619%2C%2037.566825621733905%5D%2C%20%5B126.85950389772532%2C%2037.5682771531189%5D%2C%20%5B126.85984199399667%2C%2037.571847855292745%5D%2C%20%5B126.85993476176495%2C%2037.5728262143511%5D%2C%20%5B126.8638132887273%2C%2037.57306147014704%5D%2C%20%5B126.86560520354786%2C%2037.57385540098251%5D%2C%20%5B126.86766286078968%2C%2037.57269227137124%5D%2C%20%5B126.87008117117851%2C%2037.574598289168996%5D%2C%20%5B126.87282267062741%2C%2037.574956427500126%5D%2C%20%5B126.8779661566318%2C%2037.57680133323819%5D%2C%20%5B126.87918874599603%2C%2037.5796889248137%5D%2C%20%5B126.87876320682938%2C%2037.581327335058546%5D%2C%20%5B126.88107183862735%2C%2037.583788024645344%5D%2C%20%5B126.88237824849728%2C%2037.586847436468204%5D%2C%20%5B126.88433284773288%2C%2037.588143322880526%5D%2C%20%5B126.89150044994719%2C%2037.58202374305761%5D%2C%20%5B126.89532313269488%2C%2037.579420322822145%5D%2C%20%5B126.89738573904876%2C%2037.578668647687564%5D%2C%20%5B126.90023584510952%2C%2037.575506080437606%5D%2C%20%5B126.90373193212757%2C%2037.573123712282076%5D%2C%20%5B126.90522065831053%2C%2037.57409700522574%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211140%22%2C%20%22name%22%3A%20%22%5Cub9c8%5Cud3ec%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Mapo-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B126.9524752030572%2C%2037.60508692737045%5D%2C%20%5B126.95480868778256%2C%2037.60381765067903%5D%2C%20%5B126.95564117002459%2C%2037.601827148276776%5D%2C%20%5B126.95484771718951%2C%2037.59761368186209%5D%2C%20%5B126.95619178283661%2C%2037.59575826218021%5D%2C%20%5B126.95924434840931%2C%2037.59545024466215%5D%2C%20%5B126.95842779914786%2C%2037.592356673207824%5D%2C%20%5B126.960424902266%2C%2037.58868913291801%5D%2C%20%5B126.96048802632431%2C%2037.587202077053746%5D%2C%20%5B126.95879970726017%2C%2037.58189826471162%5D%2C%20%5B126.96177754536156%2C%2037.57970124256911%5D%2C%20%5B126.95789326971087%2C%2037.57793453336025%5D%2C%20%5B126.95565425846463%2C%2037.576080790881456%5D%2C%20%5B126.96873633279075%2C%2037.56313604690827%5D%2C%20%5B126.97169209525231%2C%2037.55921654641677%5D%2C%20%5B126.96900073076728%2C%2037.55850929094393%5D%2C%20%5B126.96570855677983%2C%2037.556512377492325%5D%2C%20%5B126.96358226710812%2C%2037.55605635475154%5D%2C%20%5B126.96080686210321%2C%2037.55386236039188%5D%2C%20%5B126.95916768398142%2C%2037.55468176051932%5D%2C%20%5B126.94314477022111%2C%2037.5536460848349%5D%2C%20%5B126.93898161798973%2C%2037.552310003728124%5D%2C%20%5B126.92872097190046%2C%2037.556034533941734%5D%2C%20%5B126.92881397392811%2C%2037.558202848902%5D%2C%20%5B126.93034243306369%2C%2037.56054720372433%5D%2C%20%5B126.92778174854314%2C%2037.562495624023775%5D%2C%20%5B126.92189004506%2C%2037.56391798973296%5D%2C%20%5B126.90687243065778%2C%2037.57059762097416%5D%2C%20%5B126.90370105002282%2C%2037.57266722738834%5D%2C%20%5B126.90522065831053%2C%2037.57409700522574%5D%2C%20%5B126.91464724464083%2C%2037.583228529985455%5D%2C%20%5B126.91827498278953%2C%2037.58276881451649%5D%2C%20%5B126.91792000144513%2C%2037.58034997804668%5D%2C%20%5B126.92444169370404%2C%2037.581233899739914%5D%2C%20%5B126.92596484570709%2C%2037.58441659356971%5D%2C%20%5B126.9281697003186%2C%2037.584376616028365%5D%2C%20%5B126.9302171100533%2C%2037.58559126746845%5D%2C%20%5B126.92996402426377%2C%2037.58767009299767%5D%2C%20%5B126.93104958440722%2C%2037.58996937993664%5D%2C%20%5B126.93575441237547%2C%2037.593605788596975%5D%2C%20%5B126.94277481511082%2C%2037.59587806191211%5D%2C%20%5B126.94359054641505%2C%2037.60034871312552%5D%2C%20%5B126.94451473597087%2C%2037.60203195380752%5D%2C%20%5B126.94701525068315%2C%2037.602028095480975%5D%2C%20%5B126.94919787550161%2C%2037.60506379949065%5D%2C%20%5B126.9524752030572%2C%2037.60508692737045%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211130%22%2C%20%22name%22%3A%20%22%5Cuc11c%5Cub300%5Cubb38%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Seodaemun-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B126.9738864128702%2C%2037.62949634786888%5D%2C%20%5B126.97135221665513%2C%2037.62743418897386%5D%2C%20%5B126.96164158910712%2C%2037.62569322976388%5D%2C%20%5B126.95885198650147%2C%2037.6225761621008%5D%2C%20%5B126.95427017006129%2C%2037.622033431339425%5D%2C%20%5B126.95393421039743%2C%2037.61877981567335%5D%2C%20%5B126.9528332649861%2C%2037.6161065117682%5D%2C%20%5B126.95145384404022%2C%2037.61493028446385%5D%2C%20%5B126.95249458941419%2C%2037.61333750249896%5D%2C%20%5B126.95308190738618%2C%2037.60926681659027%5D%2C%20%5B126.95187994741721%2C%2037.60600915874973%5D%2C%20%5B126.9524752030572%2C%2037.60508692737045%5D%2C%20%5B126.94919787550161%2C%2037.60506379949065%5D%2C%20%5B126.94701525068315%2C%2037.602028095480975%5D%2C%20%5B126.94451473597087%2C%2037.60203195380752%5D%2C%20%5B126.94359054641505%2C%2037.60034871312552%5D%2C%20%5B126.94277481511082%2C%2037.59587806191211%5D%2C%20%5B126.93575441237547%2C%2037.593605788596975%5D%2C%20%5B126.93104958440722%2C%2037.58996937993664%5D%2C%20%5B126.92996402426377%2C%2037.58767009299767%5D%2C%20%5B126.9302171100533%2C%2037.58559126746845%5D%2C%20%5B126.9281697003186%2C%2037.584376616028365%5D%2C%20%5B126.92596484570709%2C%2037.58441659356971%5D%2C%20%5B126.92444169370404%2C%2037.581233899739914%5D%2C%20%5B126.91792000144513%2C%2037.58034997804668%5D%2C%20%5B126.91827498278953%2C%2037.58276881451649%5D%2C%20%5B126.91464724464083%2C%2037.583228529985455%5D%2C%20%5B126.90522065831053%2C%2037.57409700522574%5D%2C%20%5B126.90373193212757%2C%2037.573123712282076%5D%2C%20%5B126.90023584510952%2C%2037.575506080437606%5D%2C%20%5B126.89738573904876%2C%2037.578668647687564%5D%2C%20%5B126.89532313269488%2C%2037.579420322822145%5D%2C%20%5B126.89150044994719%2C%2037.58202374305761%5D%2C%20%5B126.88433284773288%2C%2037.588143322880526%5D%2C%20%5B126.88715278104091%2C%2037.59100341655796%5D%2C%20%5B126.88936046370014%2C%2037.59099007316069%5D%2C%20%5B126.88753401663872%2C%2037.58829545592628%5D%2C%20%5B126.88935419108029%2C%2037.58580092310326%5D%2C%20%5B126.89349057571656%2C%2037.585662344444785%5D%2C%20%5B126.89532781702978%2C%2037.58637054299599%5D%2C%20%5B126.89905466698256%2C%2037.58582731337662%5D%2C%20%5B126.9019347597747%2C%2037.58707359614439%5D%2C%20%5B126.9010825805882%2C%2037.58986595756633%5D%2C%20%5B126.90396681003595%2C%2037.59227403419942%5D%2C%20%5B126.90321089756087%2C%2037.594542731935476%5D%2C%20%5B126.90358350094938%2C%2037.59657528019595%5D%2C%20%5B126.90235425214276%2C%2037.60036302950128%5D%2C%20%5B126.90417801431465%2C%2037.60102912046836%5D%2C%20%5B126.90396561274416%2C%2037.60721180273151%5D%2C%20%5B126.90303066177668%2C%2037.609977911401344%5D%2C%20%5B126.90548675623195%2C%2037.61601216482774%5D%2C%20%5B126.90721633741286%2C%2037.6164590533837%5D%2C%20%5B126.90935539569412%2C%2037.61912234588074%5D%2C%20%5B126.90852982623917%2C%2037.62123670148755%5D%2C%20%5B126.91081768678396%2C%2037.62338674837964%5D%2C%20%5B126.91117179533428%2C%2037.62562846875194%5D%2C%20%5B126.90862530043799%2C%2037.62973539928895%5D%2C%20%5B126.91295124779275%2C%2037.633086577991136%5D%2C%20%5B126.91232131977827%2C%2037.635879167956396%5D%2C%20%5B126.91380622398978%2C%2037.63822817856539%5D%2C%20%5B126.91455481429648%2C%2037.64150050996935%5D%2C%20%5B126.9097405206299%2C%2037.643548736416925%5D%2C%20%5B126.9104285840059%2C%2037.64469477594629%5D%2C%20%5B126.91598194662826%2C%2037.64192000919822%5D%2C%20%5B126.92335892549015%2C%2037.642768849393626%5D%2C%20%5B126.92751618877016%2C%2037.644656244616336%5D%2C%20%5B126.93099150738166%2C%2037.64717805899008%5D%2C%20%5B126.93658892420821%2C%2037.64777520545395%5D%2C%20%5B126.93914008753828%2C%2037.64914400014214%5D%2C%20%5B126.94248904852749%2C%2037.65384179343651%5D%2C%20%5B126.94571884127255%2C%2037.655369042036206%5D%2C%20%5B126.94940373004215%2C%2037.656145979585894%5D%2C%20%5B126.94982400027399%2C%2037.65461320918016%5D%2C%20%5B126.95334970767568%2C%2037.65216408903825%5D%2C%20%5B126.956473797387%2C%2037.652480737339445%5D%2C%20%5B126.9588647426878%2C%2037.6498628918019%5D%2C%20%5B126.95972824076269%2C%2037.64638504253487%5D%2C%20%5B126.96223689995006%2C%2037.64553441285544%5D%2C%20%5B126.96424221141602%2C%2037.64071627640099%5D%2C%20%5B126.96652025322855%2C%2037.640285044901944%5D%2C%20%5B126.97092373342322%2C%2037.63589354812246%5D%2C%20%5B126.97086605626453%2C%2037.633738820281394%5D%2C%20%5B126.9738864128702%2C%2037.62949634786888%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211120%22%2C%20%22name%22%3A%20%22%5Cuc740%5Cud3c9%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Eunpyeong-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.10782277688129%2C%2037.61804244241069%5D%2C%20%5B127.10361952102048%2C%2037.61701182935077%5D%2C%20%5B127.10191523948123%2C%2037.61531698025357%5D%2C%20%5B127.09825969127313%2C%2037.61431077622918%5D%2C%20%5B127.09125855705382%2C%2037.61703046232404%5D%2C%20%5B127.08796048322492%2C%2037.617471883010936%5D%2C%20%5B127.08325130652017%2C%2037.61626773063051%5D%2C%20%5B127.07351243825278%2C%2037.61283660342313%5D%2C%20%5B127.07011594002495%2C%2037.6127769191002%5D%2C%20%5B127.06726783142085%2C%2037.61136014256744%5D%2C%20%5B127.06412260483201%2C%2037.6115839902766%5D%2C%20%5B127.05631594723272%2C%2037.61738373018764%5D%2C%20%5B127.05209373568619%2C%2037.62164065487782%5D%2C%20%5B127.04999984182327%2C%2037.62412061598568%5D%2C%20%5B127.04358800895609%2C%2037.62848931298715%5D%2C%20%5B127.046042857549%2C%2037.630647660209426%5D%2C%20%5B127.0471214260595%2C%2037.63407841321815%5D%2C%20%5B127.05227148282157%2C%2037.642016305290156%5D%2C%20%5B127.05704472959141%2C%2037.63797342537787%5D%2C%20%5B127.05800075220091%2C%2037.64318263878276%5D%2C%20%5B127.05640030258566%2C%2037.648142414223344%5D%2C%20%5B127.05620364700786%2C%2037.65295201497795%5D%2C%20%5B127.05366382763933%2C%2037.65780685245659%5D%2C%20%5B127.05343284205686%2C%2037.660866393548005%5D%2C%20%5B127.05093085431419%2C%2037.666308257661356%5D%2C%20%5B127.05124324520023%2C%2037.6703356616521%5D%2C%20%5B127.05259811896667%2C%2037.67463041579101%5D%2C%20%5B127.05402878877452%2C%2037.682018621183914%5D%2C%20%5B127.05288479710485%2C%2037.68423857084347%5D%2C%20%5B127.05673771202906%2C%2037.68650223136443%5D%2C%20%5B127.05893512852644%2C%2037.68678532745334%5D%2C%20%5B127.06366923747898%2C%2037.68598834805241%5D%2C%20%5B127.06722106517597%2C%2037.68706409219909%5D%2C%20%5B127.07150256171397%2C%2037.69158365999211%5D%2C%20%5B127.07502582169255%2C%2037.691675325399515%5D%2C%20%5B127.07945174893383%2C%2037.693602239076704%5D%2C%20%5B127.0838752703195%2C%2037.69359534202034%5D%2C%20%5B127.08640047239444%2C%2037.69122798275615%5D%2C%20%5B127.08834747450551%2C%2037.68753106067129%5D%2C%20%5B127.09706391309695%2C%2037.686383719372294%5D%2C%20%5B127.09839746601683%2C%2037.682954904948026%5D%2C%20%5B127.09481040538887%2C%2037.678799295731295%5D%2C%20%5B127.0939394572467%2C%2037.676403111386776%5D%2C%20%5B127.09572273018789%2C%2037.673837879820674%5D%2C%20%5B127.09657954289648%2C%2037.670746511824845%5D%2C%20%5B127.09767362583241%2C%2037.67002252402865%5D%2C%20%5B127.0980220884148%2C%2037.66742967749355%5D%2C%20%5B127.09621282296045%2C%2037.66347990019021%5D%2C%20%5B127.09845931549125%2C%2037.659077182069595%5D%2C%20%5B127.0978618070161%2C%2037.656734861475485%5D%2C%20%5B127.08845241642267%2C%2037.65271564775536%5D%2C%20%5B127.09496093471334%2C%2037.65209266377008%5D%2C%20%5B127.09618835483448%2C%2037.65001330395237%5D%2C%20%5B127.09440766298717%2C%2037.64713490473045%5D%2C%20%5B127.09686381732382%2C%2037.642772590812655%5D%2C%20%5B127.10024317327698%2C%2037.642337895238754%5D%2C%20%5B127.10266382494892%2C%2037.64291284604662%5D%2C%20%5B127.11014084969742%2C%2037.641850613893396%5D%2C%20%5B127.11406637789241%2C%2037.63742154695932%5D%2C%20%5B127.1144974746579%2C%2037.632439003890255%5D%2C%20%5B127.11309628774731%2C%2037.627803407908374%5D%2C%20%5B127.10737639048635%2C%2037.62412346388371%5D%2C%20%5B127.10736111050278%2C%2037.62243873312504%5D%2C%20%5B127.10561964188106%2C%2037.620112931875134%5D%2C%20%5B127.10782277688129%2C%2037.61804244241069%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211110%22%2C%20%22name%22%3A%20%22%5Cub178%5Cuc6d0%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Nowon-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.05288479710485%2C%2037.68423857084347%5D%2C%20%5B127.05402878877452%2C%2037.682018621183914%5D%2C%20%5B127.05259811896667%2C%2037.67463041579101%5D%2C%20%5B127.05124324520023%2C%2037.6703356616521%5D%2C%20%5B127.05093085431419%2C%2037.666308257661356%5D%2C%20%5B127.05343284205686%2C%2037.660866393548005%5D%2C%20%5B127.05366382763933%2C%2037.65780685245659%5D%2C%20%5B127.05620364700786%2C%2037.65295201497795%5D%2C%20%5B127.05640030258566%2C%2037.648142414223344%5D%2C%20%5B127.05800075220091%2C%2037.64318263878276%5D%2C%20%5B127.05704472959141%2C%2037.63797342537787%5D%2C%20%5B127.05227148282157%2C%2037.642016305290156%5D%2C%20%5B127.0471214260595%2C%2037.63407841321815%5D%2C%20%5B127.046042857549%2C%2037.630647660209426%5D%2C%20%5B127.04358800895609%2C%2037.62848931298715%5D%2C%20%5B127.04058571489718%2C%2037.6311245692538%5D%2C%20%5B127.03905574421407%2C%2037.63395360172076%5D%2C%20%5B127.03660759925259%2C%2037.63506293914692%5D%2C%20%5B127.03477664182829%2C%2037.63884292238491%5D%2C%20%5B127.02950136469548%2C%2037.64228467033725%5D%2C%20%5B127.0265542519312%2C%2037.644688649276304%5D%2C%20%5B127.02233660280599%2C%2037.64628067142158%5D%2C%20%5B127.01772305897457%2C%2037.64631597694605%5D%2C%20%5B127.01534913067235%2C%2037.64765266597785%5D%2C%20%5B127.01465935892466%2C%2037.64943687496812%5D%2C%20%5B127.01599412132472%2C%2037.65623256113309%5D%2C%20%5B127.01726703299362%2C%2037.65880555518138%5D%2C%20%5B127.01776269066357%2C%2037.66391319760785%5D%2C%20%5B127.02062116141389%2C%2037.667173575971205%5D%2C%20%5B127.01873683359163%2C%2037.66988242479123%5D%2C%20%5B127.0159718440919%2C%2037.67258796420281%5D%2C%20%5B127.01401600772708%2C%2037.67653037200686%5D%2C%20%5B127.01073060671976%2C%2037.67694372299732%5D%2C%20%5B127.01039666042071%2C%2037.681894589603594%5D%2C%20%5B127.01103947380624%2C%2037.684405447624954%5D%2C%20%5B127.01017954927539%2C%2037.686333239654594%5D%2C%20%5B127.0109997247323%2C%2037.691942369792514%5D%2C%20%5B127.01405303313638%2C%2037.69546278140397%5D%2C%20%5B127.01645586375106%2C%2037.69548055092485%5D%2C%20%5B127.01802453368332%2C%2037.698275937456124%5D%2C%20%5B127.02143522814708%2C%2037.698589417759045%5D%2C%20%5B127.02419558273166%2C%2037.69682256792619%5D%2C%20%5B127.02700292435075%2C%2037.69665589205863%5D%2C%20%5B127.03018866626446%2C%2037.69776745888886%5D%2C%20%5B127.03132724235057%2C%2037.69621818994332%5D%2C%20%5B127.03183574218306%2C%2037.69340418484943%5D%2C%20%5B127.03328657751797%2C%2037.69010729962784%5D%2C%20%5B127.0345286215439%2C%2037.688986544260594%5D%2C%20%5B127.0379422140845%2C%2037.68943836856245%5D%2C%20%5B127.04383030416193%2C%2037.692565730249534%5D%2C%20%5B127.04510703173885%2C%2037.690951514278055%5D%2C%20%5B127.04772506731972%2C%2037.68962174626068%5D%2C%20%5B127.05108189294938%2C%2037.69116379209157%5D%2C%20%5B127.05288479710485%2C%2037.68423857084347%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211100%22%2C%20%22name%22%3A%20%22%5Cub3c4%5Cubd09%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Dobong-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.01039666042071%2C%2037.681894589603594%5D%2C%20%5B127.01073060671976%2C%2037.67694372299732%5D%2C%20%5B127.01401600772708%2C%2037.67653037200686%5D%2C%20%5B127.0159718440919%2C%2037.67258796420281%5D%2C%20%5B127.01873683359163%2C%2037.66988242479123%5D%2C%20%5B127.02062116141389%2C%2037.667173575971205%5D%2C%20%5B127.01776269066357%2C%2037.66391319760785%5D%2C%20%5B127.01726703299362%2C%2037.65880555518138%5D%2C%20%5B127.01599412132472%2C%2037.65623256113309%5D%2C%20%5B127.01465935892466%2C%2037.64943687496812%5D%2C%20%5B127.01534913067235%2C%2037.64765266597785%5D%2C%20%5B127.01772305897457%2C%2037.64631597694605%5D%2C%20%5B127.02233660280599%2C%2037.64628067142158%5D%2C%20%5B127.0265542519312%2C%2037.644688649276304%5D%2C%20%5B127.02950136469548%2C%2037.64228467033725%5D%2C%20%5B127.03477664182829%2C%2037.63884292238491%5D%2C%20%5B127.03660759925259%2C%2037.63506293914692%5D%2C%20%5B127.03905574421407%2C%2037.63395360172076%5D%2C%20%5B127.04058571489718%2C%2037.6311245692538%5D%2C%20%5B127.04358800895609%2C%2037.62848931298715%5D%2C%20%5B127.04999984182327%2C%2037.62412061598568%5D%2C%20%5B127.05209373568619%2C%2037.62164065487782%5D%2C%20%5B127.04887981022048%2C%2037.61973841113549%5D%2C%20%5B127.0460676006384%2C%2037.615885647801704%5D%2C%20%5B127.0419720518426%2C%2037.612838591864076%5D%2C%20%5B127.03892400992301%2C%2037.609715611023816%5D%2C%20%5B127.03251659844592%2C%2037.60634705009134%5D%2C%20%5B127.03229982090541%2C%2037.609536104167034%5D%2C%20%5B127.02851994524015%2C%2037.60987827182396%5D%2C%20%5B127.02433990415497%2C%2037.60847763545628%5D%2C%20%5B127.02136282940177%2C%2037.610961475137174%5D%2C%20%5B127.01685658093551%2C%2037.61280115359516%5D%2C%20%5B127.0128154749523%2C%2037.613652243470256%5D%2C%20%5B127.01060611893628%2C%2037.615741236385354%5D%2C%20%5B127.00960949401902%2C%2037.61822603840364%5D%2C%20%5B127.00997935126598%2C%2037.62111906051553%5D%2C%20%5B127.00210238658002%2C%2037.622984514557714%5D%2C%20%5B126.99867431516041%2C%2037.626297613391166%5D%2C%20%5B126.99614706382866%2C%2037.62719717129899%5D%2C%20%5B126.99529660787617%2C%2037.62923863577322%5D%2C%20%5B126.9933714170822%2C%2037.62922019292486%5D%2C%20%5B126.9877052521691%2C%2037.63265610340949%5D%2C%20%5B126.98672705513869%2C%2037.63377641288196%5D%2C%20%5B126.98826481934299%2C%2037.637416984207924%5D%2C%20%5B126.98537069512379%2C%2037.64080296617396%5D%2C%20%5B126.987065858581%2C%2037.64320008084445%5D%2C%20%5B126.98580395043626%2C%2037.64699767714015%5D%2C%20%5B126.9832621473545%2C%2037.64947328029498%5D%2C%20%5B126.9817452676551%2C%2037.65209769387776%5D%2C%20%5B126.98205504313285%2C%2037.653797128044495%5D%2C%20%5B126.98709809733522%2C%2037.65651691642121%5D%2C%20%5B126.99031779354014%2C%2037.661013121145366%5D%2C%20%5B126.99494735642686%2C%2037.66223870806347%5D%2C%20%5B126.99581225370686%2C%2037.6651171428027%5D%2C%20%5B126.99622950422442%2C%2037.669428820661516%5D%2C%20%5B126.99506026831365%2C%2037.674827885285765%5D%2C%20%5B126.99384134064161%2C%2037.67665247641944%5D%2C%20%5B126.9959894573757%2C%2037.67754299691771%5D%2C%20%5B127.00000021675876%2C%2037.68100582640454%5D%2C%20%5B127.00566931208934%2C%2037.68228507374621%5D%2C%20%5B127.01039666042071%2C%2037.681894589603594%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211090%22%2C%20%22name%22%3A%20%22%5Cuac15%5Cubd81%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Gangbuk-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B126.977175406416%2C%2037.62859715400388%5D%2C%20%5B126.9794090069433%2C%2037.63071544568365%5D%2C%20%5B126.98360012419735%2C%2037.631915771386076%5D%2C%20%5B126.98672705513869%2C%2037.63377641288196%5D%2C%20%5B126.9877052521691%2C%2037.63265610340949%5D%2C%20%5B126.9933714170822%2C%2037.62922019292486%5D%2C%20%5B126.99529660787617%2C%2037.62923863577322%5D%2C%20%5B126.99614706382866%2C%2037.62719717129899%5D%2C%20%5B126.99867431516041%2C%2037.626297613391166%5D%2C%20%5B127.00210238658002%2C%2037.622984514557714%5D%2C%20%5B127.00997935126598%2C%2037.62111906051553%5D%2C%20%5B127.00960949401902%2C%2037.61822603840364%5D%2C%20%5B127.01060611893628%2C%2037.615741236385354%5D%2C%20%5B127.0128154749523%2C%2037.613652243470256%5D%2C%20%5B127.01685658093551%2C%2037.61280115359516%5D%2C%20%5B127.02136282940177%2C%2037.610961475137174%5D%2C%20%5B127.02433990415497%2C%2037.60847763545628%5D%2C%20%5B127.02851994524015%2C%2037.60987827182396%5D%2C%20%5B127.03229982090541%2C%2037.609536104167034%5D%2C%20%5B127.03251659844592%2C%2037.60634705009134%5D%2C%20%5B127.03892400992301%2C%2037.609715611023816%5D%2C%20%5B127.0419720518426%2C%2037.612838591864076%5D%2C%20%5B127.0460676006384%2C%2037.615885647801704%5D%2C%20%5B127.04887981022048%2C%2037.61973841113549%5D%2C%20%5B127.05209373568619%2C%2037.62164065487782%5D%2C%20%5B127.05631594723272%2C%2037.61738373018764%5D%2C%20%5B127.06412260483201%2C%2037.6115839902766%5D%2C%20%5B127.06726783142085%2C%2037.61136014256744%5D%2C%20%5B127.07011594002495%2C%2037.6127769191002%5D%2C%20%5B127.07351243825278%2C%2037.61283660342313%5D%2C%20%5B127.07382707099227%2C%2037.60401928986419%5D%2C%20%5B127.07257736686556%2C%2037.60654335765868%5D%2C%20%5B127.07069716820665%2C%2037.60653037341939%5D%2C%20%5B127.07084342033339%2C%2037.60407877132597%5D%2C%20%5B127.06753185518703%2C%2037.602724214598744%5D%2C%20%5B127.06424828533608%2C%2037.60234356864383%5D%2C%20%5B127.0612685122857%2C%2037.59823077263369%5D%2C%20%5B127.059485363799%2C%2037.598743571420485%5D%2C%20%5B127.05413734593897%2C%2037.5971595374718%5D%2C%20%5B127.05238061017225%2C%2037.598312271275574%5D%2C%20%5B127.04975439355248%2C%2037.59349421284317%5D%2C%20%5B127.0461318196879%2C%2037.593514180513594%5D%2C%20%5B127.042705222094%2C%2037.59239437593391%5D%2C%20%5B127.04116895171082%2C%2037.58847599306138%5D%2C%20%5B127.0384945038446%2C%2037.587129138658426%5D%2C%20%5B127.03121630347839%2C%2037.57957997701485%5D%2C%20%5B127.02527254528003%2C%2037.57524616245249%5D%2C%20%5B127.02395698453867%2C%2037.576068086896726%5D%2C%20%5B127.02039107536422%2C%2037.575771872553595%5D%2C%20%5B127.01849412471284%2C%2037.57904760334465%5D%2C%20%5B127.01678966486051%2C%2037.57943162455397%5D%2C%20%5B127.01094467951529%2C%2037.57765758282494%5D%2C%20%5B127.00914513999258%2C%2037.57928387879304%5D%2C%20%5B127.00896672237498%2C%2037.582512606964876%5D%2C%20%5B127.00803641804285%2C%2037.5841154647404%5D%2C%20%5B127.00453322588274%2C%2037.58626325611708%5D%2C%20%5B127.00304474231643%2C%2037.58959898679736%5D%2C%20%5B126.99774058571116%2C%2037.58944568945197%5D%2C%20%5B126.99647930158565%2C%2037.58845217708608%5D%2C%20%5B126.99348293358314%2C%2037.588565457216156%5D%2C%20%5B126.98858114244759%2C%2037.58971272682123%5D%2C%20%5B126.98596926877026%2C%2037.59105697466976%5D%2C%20%5B126.98411250331745%2C%2037.59305007059415%5D%2C%20%5B126.98366752610544%2C%2037.596350797276386%5D%2C%20%5B126.98700989661556%2C%2037.59664611897289%5D%2C%20%5B126.98984276495587%2C%2037.59842301677683%5D%2C%20%5B126.9880021251863%2C%2037.60434406739665%5D%2C%20%5B126.98906118629816%2C%2037.60778324866486%5D%2C%20%5B126.98879865992384%2C%2037.6118927319756%5D%2C%20%5B126.9849070918475%2C%2037.61390303132951%5D%2C%20%5B126.98297129719916%2C%2037.61996223346171%5D%2C%20%5B126.98130955822086%2C%2037.621781567952816%5D%2C%20%5B126.98162508963613%2C%2037.626412913357804%5D%2C%20%5B126.97877284074367%2C%2037.62605559220399%5D%2C%20%5B126.977175406416%2C%2037.62859715400388%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211080%22%2C%20%22name%22%3A%20%22%5Cuc131%5Cubd81%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Seongbuk-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.07351243825278%2C%2037.61283660342313%5D%2C%20%5B127.08325130652017%2C%2037.61626773063051%5D%2C%20%5B127.08796048322492%2C%2037.617471883010936%5D%2C%20%5B127.09125855705382%2C%2037.61703046232404%5D%2C%20%5B127.09825969127313%2C%2037.61431077622918%5D%2C%20%5B127.10191523948123%2C%2037.61531698025357%5D%2C%20%5B127.10361952102048%2C%2037.61701182935077%5D%2C%20%5B127.10782277688129%2C%2037.61804244241069%5D%2C%20%5B127.11345331993296%2C%2037.617863842655304%5D%2C%20%5B127.11916337651404%2C%2037.615000227833356%5D%2C%20%5B127.1187755615964%2C%2037.61339189160851%5D%2C%20%5B127.11970006573152%2C%2037.60891483985411%5D%2C%20%5B127.11876125411283%2C%2037.606109446640644%5D%2C%20%5B127.12043057261968%2C%2037.60482121408794%5D%2C%20%5B127.1201246020114%2C%2037.60178457598188%5D%2C%20%5B127.11594480655864%2C%2037.59717030173888%5D%2C%20%5B127.1161657841268%2C%2037.5962121213837%5D%2C%20%5B127.12048134936907%2C%2037.59221917979044%5D%2C%20%5B127.11993709114493%2C%2037.5912110652886%5D%2C%20%5B127.11547153661459%2C%2037.590528329194775%5D%2C%20%5B127.11264015185725%2C%2037.58640964166709%5D%2C%20%5B127.11144053497418%2C%2037.58186897928644%5D%2C%20%5B127.10355419751173%2C%2037.58101570870827%5D%2C%20%5B127.10488038804347%2C%2037.57896712496416%5D%2C%20%5B127.10478332632734%2C%2037.575507303239235%5D%2C%20%5B127.10316405057817%2C%2037.57293156986855%5D%2C%20%5B127.10304174249214%2C%2037.57076342290955%5D%2C%20%5B127.1015990771266%2C%2037.56973288819573%5D%2C%20%5B127.09327554832984%2C%2037.566762290300666%5D%2C%20%5B127.08553261581505%2C%2037.56856310839328%5D%2C%20%5B127.08068541280403%2C%2037.56906425519017%5D%2C%20%5B127.08029626481297%2C%2037.57521980955321%5D%2C%20%5B127.07912345005859%2C%2037.57855657914261%5D%2C%20%5B127.07327401376529%2C%2037.585954498442064%5D%2C%20%5B127.07152840437725%2C%2037.593413161750675%5D%2C%20%5B127.07216156147413%2C%2037.59537631888819%5D%2C%20%5B127.07457336676376%2C%2037.5983180561341%5D%2C%20%5B127.07481016030349%2C%2037.60000012932336%5D%2C%20%5B127.07382707099227%2C%2037.60401928986419%5D%2C%20%5B127.07351243825278%2C%2037.61283660342313%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211070%22%2C%20%22name%22%3A%20%22%5Cuc911%5Cub791%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Jungnang-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.02527254528003%2C%2037.57524616245249%5D%2C%20%5B127.03121630347839%2C%2037.57957997701485%5D%2C%20%5B127.0384945038446%2C%2037.587129138658426%5D%2C%20%5B127.04116895171082%2C%2037.58847599306138%5D%2C%20%5B127.042705222094%2C%2037.59239437593391%5D%2C%20%5B127.0461318196879%2C%2037.593514180513594%5D%2C%20%5B127.04975439355248%2C%2037.59349421284317%5D%2C%20%5B127.05238061017225%2C%2037.598312271275574%5D%2C%20%5B127.05413734593897%2C%2037.5971595374718%5D%2C%20%5B127.059485363799%2C%2037.598743571420485%5D%2C%20%5B127.0612685122857%2C%2037.59823077263369%5D%2C%20%5B127.06424828533608%2C%2037.60234356864383%5D%2C%20%5B127.06753185518703%2C%2037.602724214598744%5D%2C%20%5B127.07084342033339%2C%2037.60407877132597%5D%2C%20%5B127.07069716820665%2C%2037.60653037341939%5D%2C%20%5B127.07257736686556%2C%2037.60654335765868%5D%2C%20%5B127.07382707099227%2C%2037.60401928986419%5D%2C%20%5B127.07481016030349%2C%2037.60000012932336%5D%2C%20%5B127.07457336676376%2C%2037.5983180561341%5D%2C%20%5B127.07216156147413%2C%2037.59537631888819%5D%2C%20%5B127.07152840437725%2C%2037.593413161750675%5D%2C%20%5B127.07327401376529%2C%2037.585954498442064%5D%2C%20%5B127.07912345005859%2C%2037.57855657914261%5D%2C%20%5B127.08029626481297%2C%2037.57521980955321%5D%2C%20%5B127.08068541280403%2C%2037.56906425519017%5D%2C%20%5B127.07421053024362%2C%2037.55724769712085%5D%2C%20%5B127.07287485628252%2C%2037.55777591771644%5D%2C%20%5B127.06151678590773%2C%2037.55942885203987%5D%2C%20%5B127.06031059899311%2C%2037.55992251180729%5D%2C%20%5B127.05005601081567%2C%2037.567577612590846%5D%2C%20%5B127.0442866611438%2C%2037.57022476304866%5D%2C%20%5B127.04003329296518%2C%2037.57010227772625%5D%2C%20%5B127.03483042272745%2C%2037.567549767306716%5D%2C%20%5B127.03182413083377%2C%2037.56712900013391%5D%2C%20%5B127.02547266349976%2C%2037.568943552237734%5D%2C%20%5B127.02527254528003%2C%2037.57524616245249%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211060%22%2C%20%22name%22%3A%20%22%5Cub3d9%5Cub300%5Cubb38%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Dongdaemun-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.08068541280403%2C%2037.56906425519017%5D%2C%20%5B127.08553261581505%2C%2037.56856310839328%5D%2C%20%5B127.09327554832984%2C%2037.566762290300666%5D%2C%20%5B127.1015990771266%2C%2037.56973288819573%5D%2C%20%5B127.10304174249214%2C%2037.57076342290955%5D%2C%20%5B127.10627148043552%2C%2037.568124945986824%5D%2C%20%5B127.10545359063936%2C%2037.56685230388649%5D%2C%20%5B127.10407152037101%2C%2037.55958871940823%5D%2C%20%5B127.10325742736646%2C%2037.5572251707506%5D%2C%20%5B127.11270952006532%2C%2037.55702358575743%5D%2C%20%5B127.11519584981606%2C%2037.557533180704915%5D%2C%20%5B127.11600943681239%2C%2037.55580061507081%5D%2C%20%5B127.11600200349189%2C%2037.55053147511706%5D%2C%20%5B127.11418412219375%2C%2037.54474592090681%5D%2C%20%5B127.1116764203608%2C%2037.540669955324965%5D%2C%20%5B127.10484130265957%2C%2037.53120327509912%5D%2C%20%5B127.10087519791962%2C%2037.524841220167055%5D%2C%20%5B127.0943611414465%2C%2037.523984206117525%5D%2C%20%5B127.08639455667742%2C%2037.52161824624356%5D%2C%20%5B127.07968915919895%2C%2037.52077294752823%5D%2C%20%5B127.07496309841329%2C%2037.52091052765938%5D%2C%20%5B127.0690698130372%2C%2037.522279423505026%5D%2C%20%5B127.05867359288398%2C%2037.52629974922568%5D%2C%20%5B127.06896218881212%2C%2037.544361436565524%5D%2C%20%5B127.07580697427795%2C%2037.556641581290656%5D%2C%20%5B127.07421053024362%2C%2037.55724769712085%5D%2C%20%5B127.08068541280403%2C%2037.56906425519017%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211050%22%2C%20%22name%22%3A%20%22%5Cuad11%5Cuc9c4%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Gwangjin-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.02547266349976%2C%2037.568943552237734%5D%2C%20%5B127.03182413083377%2C%2037.56712900013391%5D%2C%20%5B127.03483042272745%2C%2037.567549767306716%5D%2C%20%5B127.04003329296518%2C%2037.57010227772625%5D%2C%20%5B127.0442866611438%2C%2037.57022476304866%5D%2C%20%5B127.05005601081567%2C%2037.567577612590846%5D%2C%20%5B127.06031059899311%2C%2037.55992251180729%5D%2C%20%5B127.06151678590773%2C%2037.55942885203987%5D%2C%20%5B127.07287485628252%2C%2037.55777591771644%5D%2C%20%5B127.07421053024362%2C%2037.55724769712085%5D%2C%20%5B127.07580697427795%2C%2037.556641581290656%5D%2C%20%5B127.06896218881212%2C%2037.544361436565524%5D%2C%20%5B127.05867359288398%2C%2037.52629974922568%5D%2C%20%5B127.05116490008963%2C%2037.52975116557232%5D%2C%20%5B127.04903802830752%2C%2037.53140496708317%5D%2C%20%5B127.04806779588436%2C%2037.52970198575087%5D%2C%20%5B127.0319617044248%2C%2037.536064291470424%5D%2C%20%5B127.0269608080842%2C%2037.53484752757724%5D%2C%20%5B127.02302831890559%2C%2037.53231899582663%5D%2C%20%5B127.01689265453608%2C%2037.536101393926174%5D%2C%20%5B127.01157414590769%2C%2037.53677688273679%5D%2C%20%5B127.01043978345277%2C%2037.53905983303592%5D%2C%20%5B127.01070894177482%2C%2037.54118048964762%5D%2C%20%5B127.01172101406588%2C%2037.545252245650516%5D%2C%20%5B127.01376082027429%2C%2037.54571276061997%5D%2C%20%5B127.01889368846282%2C%2037.55057696424215%5D%2C%20%5B127.01951516360089%2C%2037.55318470254581%5D%2C%20%5B127.02174792168286%2C%2037.55473509405241%5D%2C%20%5B127.02496143707425%2C%2037.555070476260596%5D%2C%20%5B127.0257913546443%2C%2037.558352834264504%5D%2C%20%5B127.02836991434461%2C%2037.56019645010606%5D%2C%20%5B127.02881029425372%2C%2037.56219283885279%5D%2C%20%5B127.02571971403893%2C%2037.56237200595601%5D%2C%20%5B127.02547266349976%2C%2037.568943552237734%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211040%22%2C%20%22name%22%3A%20%22%5Cuc131%5Cub3d9%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Seongdong-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.01070894177482%2C%2037.54118048964762%5D%2C%20%5B127.01043978345277%2C%2037.53905983303592%5D%2C%20%5B127.01157414590769%2C%2037.53677688273679%5D%2C%20%5B127.01689265453608%2C%2037.536101393926174%5D%2C%20%5B127.02302831890559%2C%2037.53231899582663%5D%2C%20%5B127.01397119667513%2C%2037.52503988289669%5D%2C%20%5B127.01022186960886%2C%2037.522020085671926%5D%2C%20%5B127.00818058911564%2C%2037.51877313923874%5D%2C%20%5B127.00583392114271%2C%2037.516905128452926%5D%2C%20%5B127.00011962020382%2C%2037.513901653034374%5D%2C%20%5B126.99148001917875%2C%2037.50990503427709%5D%2C%20%5B126.98948242685965%2C%2037.5108780134613%5D%2C%20%5B126.98458580602838%2C%2037.51070333105394%5D%2C%20%5B126.98223807916081%2C%2037.509314966770326%5D%2C%20%5B126.96670111119346%2C%2037.50997579058433%5D%2C%20%5B126.95950268374823%2C%2037.51249532165974%5D%2C%20%5B126.95551848909955%2C%2037.514736123015844%5D%2C%20%5B126.95249990298159%2C%2037.51722500741813%5D%2C%20%5B126.95003825019774%2C%2037.520781022055274%5D%2C%20%5B126.9488066464266%2C%2037.52424913252661%5D%2C%20%5B126.94566733083212%2C%2037.526617542453366%5D%2C%20%5B126.94717864071288%2C%2037.53213495568077%5D%2C%20%5B126.95340780191557%2C%2037.533494726370755%5D%2C%20%5B126.95926437828754%2C%2037.53897908363236%5D%2C%20%5B126.9605977865388%2C%2037.542661954880806%5D%2C%20%5B126.96231305253527%2C%2037.543511558047456%5D%2C%20%5B126.96401856825223%2C%2037.54584596959762%5D%2C%20%5B126.96604189284825%2C%2037.546894141748815%5D%2C%20%5B126.96448570553055%2C%2037.548705692021635%5D%2C%20%5B126.96782902931233%2C%2037.55132047039716%5D%2C%20%5B126.97427174983227%2C%2037.55109017579016%5D%2C%20%5B126.97859017732588%2C%2037.550336476582174%5D%2C%20%5B126.97925452152829%2C%2037.552184137181925%5D%2C%20%5B126.98262900956787%2C%2037.5506055959842%5D%2C%20%5B126.98584427779701%2C%2037.55023778139842%5D%2C%20%5B126.98752996903328%2C%2037.55094818807139%5D%2C%20%5B126.9899124474417%2C%2037.54869376545355%5D%2C%20%5B126.99238536723166%2C%2037.54862980831976%5D%2C%20%5B126.99742220893982%2C%2037.544438365587226%5D%2C%20%5B127.00062378484931%2C%2037.54713274618077%5D%2C%20%5B127.00478682371764%2C%2037.54680216333233%5D%2C%20%5B127.00632779182564%2C%2037.54757707053058%5D%2C%20%5B127.00694507580798%2C%2037.5433832956489%5D%2C%20%5B127.00936066823724%2C%2037.54101133407434%5D%2C%20%5B127.01070894177482%2C%2037.54118048964762%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211030%22%2C%20%22name%22%3A%20%22%5Cuc6a9%5Cuc0b0%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Yongsan-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B127.02547266349976%2C%2037.568943552237734%5D%2C%20%5B127.02571971403893%2C%2037.56237200595601%5D%2C%20%5B127.02881029425372%2C%2037.56219283885279%5D%2C%20%5B127.02836991434461%2C%2037.56019645010606%5D%2C%20%5B127.0257913546443%2C%2037.558352834264504%5D%2C%20%5B127.02496143707425%2C%2037.555070476260596%5D%2C%20%5B127.02174792168286%2C%2037.55473509405241%5D%2C%20%5B127.01951516360089%2C%2037.55318470254581%5D%2C%20%5B127.01889368846282%2C%2037.55057696424215%5D%2C%20%5B127.01376082027429%2C%2037.54571276061997%5D%2C%20%5B127.01172101406588%2C%2037.545252245650516%5D%2C%20%5B127.01070894177482%2C%2037.54118048964762%5D%2C%20%5B127.00936066823724%2C%2037.54101133407434%5D%2C%20%5B127.00694507580798%2C%2037.5433832956489%5D%2C%20%5B127.00632779182564%2C%2037.54757707053058%5D%2C%20%5B127.00478682371764%2C%2037.54680216333233%5D%2C%20%5B127.00062378484931%2C%2037.54713274618077%5D%2C%20%5B126.99742220893982%2C%2037.544438365587226%5D%2C%20%5B126.99238536723166%2C%2037.54862980831976%5D%2C%20%5B126.9899124474417%2C%2037.54869376545355%5D%2C%20%5B126.98752996903328%2C%2037.55094818807139%5D%2C%20%5B126.98584427779701%2C%2037.55023778139842%5D%2C%20%5B126.98262900956787%2C%2037.5506055959842%5D%2C%20%5B126.97925452152829%2C%2037.552184137181925%5D%2C%20%5B126.97859017732588%2C%2037.550336476582174%5D%2C%20%5B126.97427174983227%2C%2037.55109017579016%5D%2C%20%5B126.96782902931233%2C%2037.55132047039716%5D%2C%20%5B126.96448570553055%2C%2037.548705692021635%5D%2C%20%5B126.96380145704283%2C%2037.55254525759954%5D%2C%20%5B126.96519694864509%2C%2037.55362533505407%5D%2C%20%5B126.96358226710812%2C%2037.55605635475154%5D%2C%20%5B126.96570855677983%2C%2037.556512377492325%5D%2C%20%5B126.96900073076728%2C%2037.55850929094393%5D%2C%20%5B126.97169209525231%2C%2037.55921654641677%5D%2C%20%5B126.96873633279075%2C%2037.56313604690827%5D%2C%20%5B126.97114791678374%2C%2037.56539818101368%5D%2C%20%5B126.97500684322326%2C%2037.566406971064836%5D%2C%20%5B126.97990305661519%2C%2037.5664536437083%5D%2C%20%5B126.9910070921652%2C%2037.565312022428806%5D%2C%20%5B126.99879870609924%2C%2037.56591346564579%5D%2C%20%5B127.00372480409301%2C%2037.56679519621814%5D%2C%20%5B127.01786686709805%2C%2037.56701276414023%5D%2C%20%5B127.02250839667563%2C%2037.56892943928301%5D%2C%20%5B127.02547266349976%2C%2037.568943552237734%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211020%22%2C%20%22name%22%3A%20%22%5Cuc911%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Jung-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%2C%20%7B%22geometry%22%3A%20%7B%22coordinates%22%3A%20%5B%5B%5B126.9738864128702%2C%2037.62949634786888%5D%2C%20%5B126.977175406416%2C%2037.62859715400388%5D%2C%20%5B126.97877284074367%2C%2037.62605559220399%5D%2C%20%5B126.98162508963613%2C%2037.626412913357804%5D%2C%20%5B126.98130955822086%2C%2037.621781567952816%5D%2C%20%5B126.98297129719916%2C%2037.61996223346171%5D%2C%20%5B126.9849070918475%2C%2037.61390303132951%5D%2C%20%5B126.98879865992384%2C%2037.6118927319756%5D%2C%20%5B126.98906118629816%2C%2037.60778324866486%5D%2C%20%5B126.9880021251863%2C%2037.60434406739665%5D%2C%20%5B126.98984276495587%2C%2037.59842301677683%5D%2C%20%5B126.98700989661556%2C%2037.59664611897289%5D%2C%20%5B126.98366752610544%2C%2037.596350797276386%5D%2C%20%5B126.98411250331745%2C%2037.59305007059415%5D%2C%20%5B126.98596926877026%2C%2037.59105697466976%5D%2C%20%5B126.98858114244759%2C%2037.58971272682123%5D%2C%20%5B126.99348293358314%2C%2037.588565457216156%5D%2C%20%5B126.99647930158565%2C%2037.58845217708608%5D%2C%20%5B126.99774058571116%2C%2037.58944568945197%5D%2C%20%5B127.00304474231643%2C%2037.58959898679736%5D%2C%20%5B127.00453322588274%2C%2037.58626325611708%5D%2C%20%5B127.00803641804285%2C%2037.5841154647404%5D%2C%20%5B127.00896672237498%2C%2037.582512606964876%5D%2C%20%5B127.00914513999258%2C%2037.57928387879304%5D%2C%20%5B127.01094467951529%2C%2037.57765758282494%5D%2C%20%5B127.01678966486051%2C%2037.57943162455397%5D%2C%20%5B127.01849412471284%2C%2037.57904760334465%5D%2C%20%5B127.02039107536422%2C%2037.575771872553595%5D%2C%20%5B127.02395698453867%2C%2037.576068086896726%5D%2C%20%5B127.02527254528003%2C%2037.57524616245249%5D%2C%20%5B127.02547266349976%2C%2037.568943552237734%5D%2C%20%5B127.02250839667563%2C%2037.56892943928301%5D%2C%20%5B127.01786686709805%2C%2037.56701276414023%5D%2C%20%5B127.00372480409301%2C%2037.56679519621814%5D%2C%20%5B126.99879870609924%2C%2037.56591346564579%5D%2C%20%5B126.9910070921652%2C%2037.565312022428806%5D%2C%20%5B126.97990305661519%2C%2037.5664536437083%5D%2C%20%5B126.97500684322326%2C%2037.566406971064836%5D%2C%20%5B126.97114791678374%2C%2037.56539818101368%5D%2C%20%5B126.96873633279075%2C%2037.56313604690827%5D%2C%20%5B126.95565425846463%2C%2037.576080790881456%5D%2C%20%5B126.95789326971087%2C%2037.57793453336025%5D%2C%20%5B126.96177754536156%2C%2037.57970124256911%5D%2C%20%5B126.95879970726017%2C%2037.58189826471162%5D%2C%20%5B126.96048802632431%2C%2037.587202077053746%5D%2C%20%5B126.960424902266%2C%2037.58868913291801%5D%2C%20%5B126.95842779914786%2C%2037.592356673207824%5D%2C%20%5B126.95924434840931%2C%2037.59545024466215%5D%2C%20%5B126.95619178283661%2C%2037.59575826218021%5D%2C%20%5B126.95484771718951%2C%2037.59761368186209%5D%2C%20%5B126.95564117002459%2C%2037.601827148276776%5D%2C%20%5B126.95480868778256%2C%2037.60381765067903%5D%2C%20%5B126.9524752030572%2C%2037.60508692737045%5D%2C%20%5B126.95187994741721%2C%2037.60600915874973%5D%2C%20%5B126.95308190738618%2C%2037.60926681659027%5D%2C%20%5B126.95249458941419%2C%2037.61333750249896%5D%2C%20%5B126.95145384404022%2C%2037.61493028446385%5D%2C%20%5B126.9528332649861%2C%2037.6161065117682%5D%2C%20%5B126.95393421039743%2C%2037.61877981567335%5D%2C%20%5B126.95427017006129%2C%2037.622033431339425%5D%2C%20%5B126.95885198650147%2C%2037.6225761621008%5D%2C%20%5B126.96164158910712%2C%2037.62569322976388%5D%2C%20%5B126.97135221665513%2C%2037.62743418897386%5D%2C%20%5B126.9738864128702%2C%2037.62949634786888%5D%5D%5D%2C%20%22type%22%3A%20%22Polygon%22%7D%2C%20%22properties%22%3A%20%7B%22base_year%22%3A%20%222013%22%2C%20%22code%22%3A%20%2211010%22%2C%20%22name%22%3A%20%22%5Cuc885%5Cub85c%5Cuad6c%22%2C%20%22name_eng%22%3A%20%22Jongno-gu%22%7D%2C%20%22type%22%3A%20%22Feature%22%7D%5D%2C%20%22type%22%3A%20%22FeatureCollection%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20var%20color_map_3fabb853964c4a20a4d6eefd24969d34%20%3D%20%7B%7D%3B%0A%0A%20%20%20%20%0A%20%20%20%20color_map_3fabb853964c4a20a4d6eefd24969d34.color%20%3D%20d3.scale.threshold%28%29%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20.domain%28%5B369.2655902479585%2C%20370.657019582932%2C%20372.0484489179055%2C%20373.43987825287894%2C%20374.83130758785245%2C%20376.2227369228259%2C%20377.61416625779935%2C%20379.00559559277286%2C%20380.3970249277463%2C%20381.7884542627198%2C%20383.1798835976933%2C%20384.5713129326668%2C%20385.96274226764024%2C%20387.3541716026137%2C%20388.7456009375872%2C%20390.13703027256065%2C%20391.52845960753416%2C%20392.9198889425076%2C%20394.31131827748106%2C%20395.7027476124546%2C%20397.094176947428%2C%20398.48560628240153%2C%20399.877035617375%2C%20401.2684649523485%2C%20402.65989428732195%2C%20404.0513236222954%2C%20405.4427529572689%2C%20406.83418229224236%2C%20408.22561162721587%2C%20409.6170409621893%2C%20411.0084702971628%2C%20412.3998996321363%2C%20413.79132896710973%2C%20415.18275830208324%2C%20416.5741876370567%2C%20417.96561697203015%2C%20419.35704630700366%2C%20420.7484756419771%2C%20422.1399049769506%2C%20423.53133431192407%2C%20424.9227636468976%2C%20426.31419298187103%2C%20427.7056223168445%2C%20429.097051651818%2C%20430.48848098679144%2C%20431.87991032176495%2C%20433.2713396567384%2C%20434.6627689917119%2C%20436.05419832668537%2C%20437.4456276616588%2C%20438.8370569966323%2C%20440.2284863316058%2C%20441.61991566657923%2C%20443.01134500155274%2C%20444.40277433652625%2C%20445.7942036714997%2C%20447.18563300647315%2C%20448.57706234144666%2C%20449.9684916764201%2C%20451.35992101139357%2C%20452.7513503463671%2C%20454.1427796813405%2C%20455.53420901631404%2C%20456.9256383512875%2C%20458.317067686261%2C%20459.70849702123445%2C%20461.0999263562079%2C%20462.4913556911814%2C%20463.88278502615486%2C%20465.2742143611284%2C%20466.6656436961018%2C%20468.05707303107533%2C%20469.4485023660488%2C%20470.83993170102224%2C%20472.23136103599575%2C%20473.6227903709692%2C%20475.01421970594265%2C%20476.40564904091616%2C%20477.7970783758896%2C%20479.1885077108631%2C%20480.5799370458366%2C%20481.9713663808101%2C%20483.36279571578353%2C%20484.754225050757%2C%20486.1456543857305%2C%20487.53708372070395%2C%20488.92851305567746%2C%20490.3199423906509%2C%20491.7113717256244%2C%20493.10280106059787%2C%20494.4942303955713%2C%20495.88565973054483%2C%20497.2770890655183%2C%20498.66851840049173%2C%20500.05994773546524%2C%20501.45137707043875%2C%20502.84280640541215%2C%20504.23423574038566%2C%20505.62566507535917%2C%20507.0170944103327%2C%20508.40852374530607%2C%20509.7999530802796%2C%20511.1913824152531%2C%20512.5828117502265%2C%20513.9742410852%2C%20515.3656704201735%2C%20516.757099755147%2C%20518.1485290901204%2C%20519.5399584250939%2C%20520.9313877600673%2C%20522.3228170950408%2C%20523.7142464300143%2C%20525.1056757649878%2C%20526.4971050999613%2C%20527.8885344349347%2C%20529.2799637699082%2C%20530.6713931048818%2C%20532.0628224398552%2C%20533.4542517748287%2C%20534.8456811098022%2C%20536.2371104447756%2C%20537.6285397797491%2C%20539.0199691147226%2C%20540.411398449696%2C%20541.8028277846695%2C%20543.194257119643%2C%20544.5856864546165%2C%20545.97711578959%2C%20547.3685451245634%2C%20548.7599744595369%2C%20550.1514037945103%2C%20551.5428331294838%2C%20552.9342624644573%2C%20554.3256917994308%2C%20555.7171211344042%2C%20557.1085504693777%2C%20558.4999798043513%2C%20559.8914091393247%2C%20561.2828384742982%2C%20562.6742678092717%2C%20564.0656971442452%2C%20565.4571264792186%2C%20566.8485558141921%2C%20568.2399851491655%2C%20569.631414484139%2C%20571.0228438191125%2C%20572.414273154086%2C%20573.8057024890595%2C%20575.1971318240329%2C%20576.5885611590064%2C%20577.9799904939799%2C%20579.3714198289533%2C%20580.7628491639268%2C%20582.1542784989003%2C%20583.5457078338737%2C%20584.9371371688472%2C%20586.3285665038208%2C%20587.7199958387941%2C%20589.1114251737677%2C%20590.5028545087412%2C%20591.8942838437147%2C%20593.2857131786882%2C%20594.6771425136616%2C%20596.0685718486351%2C%20597.4600011836085%2C%20598.851430518582%2C%20600.2428598535555%2C%20601.634289188529%2C%20603.0257185235024%2C%20604.4171478584759%2C%20605.8085771934494%2C%20607.2000065284228%2C%20608.5914358633963%2C%20609.9828651983698%2C%20611.3742945333433%2C%20612.7657238683167%2C%20614.1571532032902%2C%20615.5485825382638%2C%20616.9400118732372%2C%20618.3314412082107%2C%20619.7228705431842%2C%20621.1142998781577%2C%20622.5057292131311%2C%20623.8971585481046%2C%20625.288587883078%2C%20626.6800172180515%2C%20628.071446553025%2C%20629.4628758879985%2C%20630.854305222972%2C%20632.2457345579454%2C%20633.6371638929189%2C%20635.0285932278923%2C%20636.4200225628658%2C%20637.8114518978393%2C%20639.2028812328128%2C%20640.5943105677864%2C%20641.9857399027597%2C%20643.3771692377331%2C%20644.7685985727068%2C%20646.1600279076802%2C%20647.5514572426537%2C%20648.9428865776272%2C%20650.3343159126006%2C%20651.7257452475741%2C%20653.1171745825476%2C%20654.508603917521%2C%20655.9000332524945%2C%20657.291462587468%2C%20658.6828919224415%2C%20660.074321257415%2C%20661.4657505923884%2C%20662.8571799273619%2C%20664.2486092623353%2C%20665.6400385973088%2C%20667.0314679322823%2C%20668.4228972672558%2C%20669.8143266022294%2C%20671.2057559372028%2C%20672.5971852721761%2C%20673.9886146071497%2C%20675.3800439421232%2C%20676.7714732770967%2C%20678.1629026120702%2C%20679.5543319470436%2C%20680.9457612820171%2C%20682.3371906169905%2C%20683.728619951964%2C%20685.1200492869375%2C%20686.511478621911%2C%20687.9029079568845%2C%20689.2943372918579%2C%20690.6857666268314%2C%20692.0771959618049%2C%20693.4686252967783%2C%20694.8600546317518%2C%20696.2514839667253%2C%20697.6429133016987%2C%20699.0343426366724%2C%20700.4257719716458%2C%20701.8172013066192%2C%20703.2086306415927%2C%20704.6000599765662%2C%20705.9914893115397%2C%20707.3829186465132%2C%20708.7743479814866%2C%20710.1657773164601%2C%20711.5572066514335%2C%20712.948635986407%2C%20714.3400653213805%2C%20715.731494656354%2C%20717.1229239913275%2C%20718.5143533263009%2C%20719.9057826612743%2C%20721.2972119962478%2C%20722.6886413312213%2C%20724.0800706661948%2C%20725.4715000011684%2C%20726.8629293361417%2C%20728.2543586711153%2C%20729.6457880060888%2C%20731.0372173410622%2C%20732.4286466760357%2C%20733.8200760110092%2C%20735.2115053459827%2C%20736.6029346809561%2C%20737.9943640159296%2C%20739.3857933509031%2C%20740.7772226858765%2C%20742.16865202085%2C%20743.5600813558235%2C%20744.951510690797%2C%20746.3429400257705%2C%20747.7343693607439%2C%20749.1257986957173%2C%20750.5172280306908%2C%20751.9086573656643%2C%20753.3000867006378%2C%20754.6915160356114%2C%20756.0829453705848%2C%20757.4743747055583%2C%20758.8658040405317%2C%20760.2572333755052%2C%20761.6486627104787%2C%20763.0400920454522%2C%20764.4315213804257%2C%20765.8229507153991%2C%20767.2143800503725%2C%20768.605809385346%2C%20769.9972387203195%2C%20771.388668055293%2C%20772.7800973902665%2C%20774.1715267252399%2C%20775.5629560602134%2C%20776.9543853951869%2C%20778.3458147301603%2C%20779.7372440651338%2C%20781.1286734001073%2C%20782.5201027350809%2C%20783.9115320700544%2C%20785.3029614050278%2C%20786.6943907400013%2C%20788.0858200749747%2C%20789.4772494099482%2C%20790.8686787449217%2C%20792.2601080798952%2C%20793.6515374148687%2C%20795.0429667498421%2C%20796.4343960848155%2C%20797.825825419789%2C%20799.2172547547625%2C%20800.608684089736%2C%20802.0001134247095%2C%20803.3915427596829%2C%20804.7829720946564%2C%20806.1744014296298%2C%20807.5658307646033%2C%20808.9572600995768%2C%20810.3486894345504%2C%20811.7401187695239%2C%20813.1315481044973%2C%20814.5229774394708%2C%20815.9144067744442%2C%20817.3058361094177%2C%20818.6972654443912%2C%20820.0886947793647%2C%20821.4801241143381%2C%20822.8715534493117%2C%20824.2629827842851%2C%20825.6544121192585%2C%20827.045841454232%2C%20828.4372707892055%2C%20829.828700124179%2C%20831.2201294591525%2C%20832.6115587941259%2C%20834.0029881290994%2C%20835.3944174640728%2C%20836.7858467990463%2C%20838.1772761340198%2C%20839.5687054689934%2C%20840.9601348039669%2C%20842.3515641389403%2C%20843.7429934739137%2C%20845.1344228088872%2C%20846.5258521438607%2C%20847.9172814788342%2C%20849.3087108138077%2C%20850.7001401487811%2C%20852.0915694837546%2C%20853.4829988187281%2C%20854.8744281537015%2C%20856.265857488675%2C%20857.6572868236485%2C%20859.048716158622%2C%20860.4401454935954%2C%20861.8315748285689%2C%20863.2230041635423%2C%20864.6144334985158%2C%20866.0058628334893%2C%20867.3972921684629%2C%20868.7887215034364%2C%20870.1801508384099%2C%20871.5715801733833%2C%20872.9630095083567%2C%20874.3544388433302%2C%20875.7458681783037%2C%20877.1372975132772%2C%20878.5287268482507%2C%20879.9201561832241%2C%20881.3115855181975%2C%20882.703014853171%2C%20884.0944441881445%2C%20885.485873523118%2C%20886.8773028580915%2C%20888.268732193065%2C%20889.6601615280383%2C%20891.0515908630118%2C%20892.4430201979853%2C%20893.8344495329588%2C%20895.2258788679324%2C%20896.6173082029059%2C%20898.0087375378794%2C%20899.4001668728529%2C%20900.7915962078262%2C%20902.1830255427997%2C%20903.5744548777732%2C%20904.9658842127467%2C%20906.3573135477202%2C%20907.7487428826935%2C%20909.1401722176672%2C%20910.5316015526405%2C%20911.923030887614%2C%20913.3144602225875%2C%20914.705889557561%2C%20916.0973188925345%2C%20917.4887482275078%2C%20918.8801775624813%2C%20920.2716068974551%2C%20921.6630362324283%2C%20923.0544655674018%2C%20924.4458949023754%2C%20925.8373242373489%2C%20927.2287535723224%2C%20928.6201829072957%2C%20930.0116122422692%2C%20931.4030415772427%2C%20932.7944709122162%2C%20934.1859002471897%2C%20935.5773295821632%2C%20936.9687589171367%2C%20938.36018825211%2C%20939.7516175870835%2C%20941.143046922057%2C%20942.5344762570305%2C%20943.925905592004%2C%20945.3173349269775%2C%20946.708764261951%2C%20948.1001935969243%2C%20949.4916229318978%2C%20950.8830522668713%2C%20952.2744816018449%2C%20953.6659109368184%2C%20955.0573402717916%2C%20956.4487696067654%2C%20957.8401989417389%2C%20959.2316282767122%2C%20960.6230576116857%2C%20962.0144869466592%2C%20963.4059162816327%2C%20964.797345616606%2C%20966.1887749515795%2C%20967.5802042865532%2C%20968.9716336215265%2C%20970.3630629565%2C%20971.7544922914735%2C%20973.145921626447%2C%20974.5373509614205%2C%20975.9287802963938%2C%20977.3202096313673%2C%20978.7116389663408%2C%20980.1030683013144%2C%20981.4944976362879%2C%20982.8859269712614%2C%20984.2773563062349%2C%20985.6687856412082%2C%20987.0602149761817%2C%20988.4516443111552%2C%20989.8430736461287%2C%20991.2345029811022%2C%20992.6259323160757%2C%20994.0173616510492%2C%20995.4087909860225%2C%20996.800220320996%2C%20998.1916496559695%2C%20999.583078990943%2C%201000.9745083259165%2C%201002.3659376608898%2C%201003.7573669958635%2C%201005.1487963308371%2C%201006.5402256658103%2C%201007.9316550007838%2C%201009.3230843357574%2C%201010.7145136707309%2C%201012.1059430057044%2C%201013.4973723406777%2C%201014.8888016756514%2C%201016.2802310106247%2C%201017.6716603455982%2C%201019.0630896805717%2C%201020.4545190155452%2C%201021.8459483505187%2C%201023.237377685492%2C%201024.6288070204655%2C%201026.020236355439%2C%201027.4116656904125%2C%201028.803095025386%2C%201030.1945243603595%2C%201031.585953695333%2C%201032.9773830303063%2C%201034.3688123652798%2C%201035.7602417002533%2C%201037.1516710352269%2C%201038.5431003702004%2C%201039.9345297051739%2C%201041.3259590401474%2C%201042.7173883751207%2C%201044.1088177100942%2C%201045.5002470450677%2C%201046.8916763800412%2C%201048.2831057150147%2C%201049.674535049988%2C%201051.0659643849617%2C%201052.4573937199352%2C%201053.8488230549085%2C%201055.240252389882%2C%201056.6316817248555%2C%201058.023111059829%2C%201059.4145403948025%2C%201060.8059697297758%2C%201062.1973990647496%2C%201063.5888283997228%5D%29%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20.range%28%5B%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23ffffccff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23d9f0a3ff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%23addd8eff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2378c679ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%2331a354ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%2C%20%27%23006837ff%27%5D%29%3B%0A%20%20%20%20%0A%0A%20%20%20%20color_map_3fabb853964c4a20a4d6eefd24969d34.x%20%3D%20d3.scale.linear%28%29%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20.domain%28%5B369.2655902479585%2C%201063.5888283997228%5D%29%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20.range%28%5B0%2C%20400%5D%29%3B%0A%0A%20%20%20%20color_map_3fabb853964c4a20a4d6eefd24969d34.legend%20%3D%20L.control%28%7Bposition%3A%20%27topright%27%7D%29%3B%0A%20%20%20%20color_map_3fabb853964c4a20a4d6eefd24969d34.legend.onAdd%20%3D%20function%20%28map%29%20%7Bvar%20div%20%3D%20L.DomUtil.create%28%27div%27%2C%20%27legend%27%29%3B%20return%20div%7D%3B%0A%20%20%20%20color_map_3fabb853964c4a20a4d6eefd24969d34.legend.addTo%28map_34be2092659d453298ac36df9dcfb85a%29%3B%0A%0A%20%20%20%20color_map_3fabb853964c4a20a4d6eefd24969d34.xAxis%20%3D%20d3.svg.axis%28%29%0A%20%20%20%20%20%20%20%20.scale%28color_map_3fabb853964c4a20a4d6eefd24969d34.x%29%0A%20%20%20%20%20%20%20%20.orient%28%22top%22%29%0A%20%20%20%20%20%20%20%20.tickSize%281%29%0A%20%20%20%20%20%20%20%20.tickValues%28%5B369.2655902479585%2C%20484.98612993991924%2C%20600.70666963188%2C%20716.4272093238408%2C%20832.1477490158014%2C%20947.8682887077621%2C%201063.5888283997228%5D%29%3B%0A%0A%20%20%20%20color_map_3fabb853964c4a20a4d6eefd24969d34.svg%20%3D%20d3.select%28%22.legend.leaflet-control%22%29.append%28%22svg%22%29%0A%20%20%20%20%20%20%20%20.attr%28%22id%22%2C%20%27legend%27%29%0A%20%20%20%20%20%20%20%20.attr%28%22width%22%2C%20450%29%0A%20%20%20%20%20%20%20%20.attr%28%22height%22%2C%2040%29%3B%0A%0A%20%20%20%20color_map_3fabb853964c4a20a4d6eefd24969d34.g%20%3D%20color_map_3fabb853964c4a20a4d6eefd24969d34.svg.append%28%22g%22%29%0A%20%20%20%20%20%20%20%20.attr%28%22class%22%2C%20%22key%22%29%0A%20%20%20%20%20%20%20%20.attr%28%22transform%22%2C%20%22translate%2825%2C16%29%22%29%3B%0A%0A%20%20%20%20color_map_3fabb853964c4a20a4d6eefd24969d34.g.selectAll%28%22rect%22%29%0A%20%20%20%20%20%20%20%20.data%28color_map_3fabb853964c4a20a4d6eefd24969d34.color.range%28%29.map%28function%28d%2C%20i%29%20%7B%0A%20%20%20%20%20%20%20%20%20%20return%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20x0%3A%20i%20%3F%20color_map_3fabb853964c4a20a4d6eefd24969d34.x%28color_map_3fabb853964c4a20a4d6eefd24969d34.color.domain%28%29%5Bi%20-%201%5D%29%20%3A%20color_map_3fabb853964c4a20a4d6eefd24969d34.x.range%28%29%5B0%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20x1%3A%20i%20%3C%20color_map_3fabb853964c4a20a4d6eefd24969d34.color.domain%28%29.length%20%3F%20color_map_3fabb853964c4a20a4d6eefd24969d34.x%28color_map_3fabb853964c4a20a4d6eefd24969d34.color.domain%28%29%5Bi%5D%29%20%3A%20color_map_3fabb853964c4a20a4d6eefd24969d34.x.range%28%29%5B1%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20z%3A%20d%0A%20%20%20%20%20%20%20%20%20%20%7D%3B%0A%20%20%20%20%20%20%20%20%7D%29%29%0A%20%20%20%20%20%20.enter%28%29.append%28%22rect%22%29%0A%20%20%20%20%20%20%20%20.attr%28%22height%22%2C%2010%29%0A%20%20%20%20%20%20%20%20.attr%28%22x%22%2C%20function%28d%29%20%7B%20return%20d.x0%3B%20%7D%29%0A%20%20%20%20%20%20%20%20.attr%28%22width%22%2C%20function%28d%29%20%7B%20return%20d.x1%20-%20d.x0%3B%20%7D%29%0A%20%20%20%20%20%20%20%20.style%28%22fill%22%2C%20function%28d%29%20%7B%20return%20d.z%3B%20%7D%29%3B%0A%0A%20%20%20%20color_map_3fabb853964c4a20a4d6eefd24969d34.g.call%28color_map_3fabb853964c4a20a4d6eefd24969d34.xAxis%29.append%28%22text%22%29%0A%20%20%20%20%20%20%20%20.attr%28%22class%22%2C%20%22caption%22%29%0A%20%20%20%20%20%20%20%20.attr%28%22y%22%2C%2021%29%0A%20%20%20%20%20%20%20%20.text%28%27%27%29%3B%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



# (6) 모델링 1번 시도 : 시계열 LSTM 모델링


```python
#강남구에 한정하여 시계열 모델링 시도
gangnam_df = all_df[all_df["area"]=="강남구"]
```


```python
gangnam_df.columns
```




    Index(['area', '일자별평당가', 'year', 'KOREA_GDP', 'GDP_Growth_rate', 'loan',
           '기준금리', 'housing_loan', 'gdp_per_person', 'gdp', 'Viliages',
           'Buildings', 'House', '세대', '인구', '합계', '수도권', '서울'],
          dtype='object')




```python
split_date = pd.Timestamp('2018-01-01')

train = gangnam_df.loc[:split_date, ['일자별평당가']]
test = gangnam_df.loc[split_date:, ['일자별평당가']]

ax = train.plot()
test.plot(ax=ax)
plt.legend(['train','test'])
```




    <matplotlib.legend.Legend at 0x13d28a19808>




    
![png](output_45_1.png)
    



```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(train)
train_sc = scaler.transform(train)

scaler.fit(test)
test_sc = scaler.transform(test)
```


```python
train_sc_df = pd.DataFrame(train_sc, columns=["일자별평당가"], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=["일자별평당가"], index=test.index)

train_sc_df.head()
```




                  일자별평당가
    계약년월일               
    2011-01-01 -2.405885
    2011-01-03 -1.610236
    2011-01-04 -1.410823
    2011-01-05 -0.270497
    2011-01-06 -0.258758




```python
#sliding window 구성하기
for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['일자별평당가'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['일자별평당가'].shift(s)
    
train_sc_df.head(13)
```




                  일자별평당가   shift_1   shift_2   shift_3   shift_4   shift_5  \
    계약년월일                                                                    
    2011-01-01 -2.405885       NaN       NaN       NaN       NaN       NaN   
    2011-01-03 -1.610236 -2.405885       NaN       NaN       NaN       NaN   
    2011-01-04 -1.410823 -1.610236 -2.405885       NaN       NaN       NaN   
    2011-01-05 -0.270497 -1.410823 -1.610236 -2.405885       NaN       NaN   
    2011-01-06 -0.258758 -0.270497 -1.410823 -1.610236 -2.405885       NaN   
    2011-01-07 -0.346819 -0.258758 -0.270497 -1.410823 -1.610236 -2.405885   
    2011-01-08 -0.805440 -0.346819 -0.258758 -0.270497 -1.410823 -1.610236   
    2011-01-09 -0.447909 -0.805440 -0.346819 -0.258758 -0.270497 -1.410823   
    2011-01-10 -0.275092 -0.447909 -0.805440 -0.346819 -0.258758 -0.270497   
    2011-01-11 -0.406449 -0.275092 -0.447909 -0.805440 -0.346819 -0.258758   
    2011-01-12 -0.143209 -0.406449 -0.275092 -0.447909 -0.805440 -0.346819   
    2011-01-13 -0.963023 -0.143209 -0.406449 -0.275092 -0.447909 -0.805440   
    2011-01-14 -0.844905 -0.963023 -0.143209 -0.406449 -0.275092 -0.447909   
    
                 shift_6   shift_7   shift_8   shift_9  shift_10  shift_11  \
    계약년월일                                                                    
    2011-01-01       NaN       NaN       NaN       NaN       NaN       NaN   
    2011-01-03       NaN       NaN       NaN       NaN       NaN       NaN   
    2011-01-04       NaN       NaN       NaN       NaN       NaN       NaN   
    2011-01-05       NaN       NaN       NaN       NaN       NaN       NaN   
    2011-01-06       NaN       NaN       NaN       NaN       NaN       NaN   
    2011-01-07       NaN       NaN       NaN       NaN       NaN       NaN   
    2011-01-08 -2.405885       NaN       NaN       NaN       NaN       NaN   
    2011-01-09 -1.610236 -2.405885       NaN       NaN       NaN       NaN   
    2011-01-10 -1.410823 -1.610236 -2.405885       NaN       NaN       NaN   
    2011-01-11 -0.270497 -1.410823 -1.610236 -2.405885       NaN       NaN   
    2011-01-12 -0.258758 -0.270497 -1.410823 -1.610236 -2.405885       NaN   
    2011-01-13 -0.346819 -0.258758 -0.270497 -1.410823 -1.610236 -2.405885   
    2011-01-14 -0.805440 -0.346819 -0.258758 -0.270497 -1.410823 -1.610236   
    
                shift_12  
    계약년월일                 
    2011-01-01       NaN  
    2011-01-03       NaN  
    2011-01-04       NaN  
    2011-01-05       NaN  
    2011-01-06       NaN  
    2011-01-07       NaN  
    2011-01-08       NaN  
    2011-01-09       NaN  
    2011-01-10       NaN  
    2011-01-11       NaN  
    2011-01-12       NaN  
    2011-01-13       NaN  
    2011-01-14 -2.405885  




```python
#dropNA Train set
X_train = train_sc_df.dropna().drop('일자별평당가', axis=1)
y_train = train_sc_df.dropna()[['일자별평당가']]
```


```python
#dropNA Test set
X_test = test_sc_df.dropna().drop('일자별평당가', axis=1)
y_test = test_sc_df.dropna()[['일자별평당가']]
```


```python
X_train.head()
```




                 shift_1   shift_2   shift_3   shift_4   shift_5   shift_6  \
    계약년월일                                                                    
    2011-01-14 -0.963023 -0.143209 -0.406449 -0.275092 -0.447909 -0.805440   
    2011-01-15 -0.844905 -0.963023 -0.143209 -0.406449 -0.275092 -0.447909   
    2011-01-16 -0.530647 -0.844905 -0.963023 -0.143209 -0.406449 -0.275092   
    2011-01-17 -0.336816 -0.530647 -0.844905 -0.963023 -0.143209 -0.406449   
    2011-01-18 -1.084970 -0.336816 -0.530647 -0.844905 -0.963023 -0.143209   
    
                 shift_7   shift_8   shift_9  shift_10  shift_11  shift_12  
    계약년월일                                                                   
    2011-01-14 -0.346819 -0.258758 -0.270497 -1.410823 -1.610236 -2.405885  
    2011-01-15 -0.805440 -0.346819 -0.258758 -0.270497 -1.410823 -1.610236  
    2011-01-16 -0.447909 -0.805440 -0.346819 -0.258758 -0.270497 -1.410823  
    2011-01-17 -0.275092 -0.447909 -0.805440 -0.346819 -0.258758 -0.270497  
    2011-01-18 -0.406449 -0.275092 -0.447909 -0.805440 -0.346819 -0.258758  




```python
print(type(X_train))
X_train = X_train.values
print(type(X_test))
X_test = X_test.values

y_train = y_train.values
y_test = y_test.values

print(X_train.shape)
print(y_train.shape)
```

    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.frame.DataFrame'>
    (2464, 12)
    (2464, 1)
    


```python
X_train_t = X_train.reshape(X_train.shape[0], 12, 1)
X_test_t = X_test.reshape(X_test.shape[0], 12, 1)


print("최종 DATA SET")
print(X_train_t.shape)
print(X_train_t)
print(y_train)
```

    최종 DATA SET
    (2464, 12, 1)
    [[[-0.96302271]
      [-0.14320865]
      [-0.4064492 ]
      ...
      [-1.41082274]
      [-1.61023636]
      [-2.40588522]]
    
     [[-0.84490503]
      [-0.96302271]
      [-0.14320865]
      ...
      [-0.27049713]
      [-1.41082274]
      [-1.61023636]]
    
     [[-0.53064669]
      [-0.84490503]
      [-0.96302271]
      ...
      [-0.25875777]
      [-0.27049713]
      [-1.41082274]]
    
     ...
    
     [[ 2.24085583]
      [ 2.62587908]
      [ 2.89125758]
      ...
      [ 2.98609925]
      [ 2.24576779]
      [ 2.08951722]]
    
     [[ 1.816919  ]
      [ 2.24085583]
      [ 2.62587908]
      ...
      [ 1.68106014]
      [ 2.98609925]
      [ 2.24576779]]
    
     [[ 2.74317732]
      [ 1.816919  ]
      [ 2.24085583]
      ...
      [ 1.66028695]
      [ 1.68106014]
      [ 2.98609925]]]
    [[-0.84490503]
     [-0.53064669]
     [-0.33681626]
     ...
     [ 1.816919  ]
     [ 2.74317732]
     [ 3.6066599 ]]
    


```python
#시계열 모델링 LSTM
from keras.layers import LSTM 
from keras.models import Sequential 
from keras.layers import Dense 
import keras.backend as K 
from keras.callbacks import EarlyStopping

K.clear_session()
    
model = Sequential() # Sequeatial Model 
model.add(LSTM(20, input_shape=(12, 1))) # (timestep, feature) 
model.add(Dense(1)) # output = 1 
model.compile(loss='mean_squared_error', optimizer='adam') 
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 20)                1760      
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 21        
    =================================================================
    Total params: 1,781
    Trainable params: 1,781
    Non-trainable params: 0
    _________________________________________________________________
    

    Using TensorFlow backend.
    


```python
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model.fit(X_train_t, y_train, epochs=100,
          batch_size=30, verbose=1, callbacks=[early_stop])
```

    Epoch 1/100
    2464/2464 [==============================] - 1s 431us/step - loss: 0.5652
    Epoch 2/100
    2464/2464 [==============================] - 0s 178us/step - loss: 0.4182
    Epoch 3/100
    2464/2464 [==============================] - 0s 198us/step - loss: 0.4152
    Epoch 4/100
    2464/2464 [==============================] - 0s 184us/step - loss: 0.4152
    Epoch 5/100
    2464/2464 [==============================] - 0s 199us/step - loss: 0.4135
    Epoch 6/100
    2464/2464 [==============================] - 0s 184us/step - loss: 0.4135
    Epoch 7/100
    2464/2464 [==============================] - 0s 183us/step - loss: 0.4126
    Epoch 8/100
    2464/2464 [==============================] - 0s 192us/step - loss: 0.4121
    Epoch 9/100
    2464/2464 [==============================] - 0s 176us/step - loss: 0.4111
    Epoch 10/100
    2464/2464 [==============================] - 0s 182us/step - loss: 0.4093
    Epoch 11/100
    2464/2464 [==============================] - 0s 175us/step - loss: 0.4088
    Epoch 12/100
    2464/2464 [==============================] - 0s 170us/step - loss: 0.4084
    Epoch 13/100
    2464/2464 [==============================] - 0s 177us/step - loss: 0.4085
    Epoch 00013: early stopping
    




    <keras.callbacks.callbacks.History at 0x13d2babe5c8>




```python
score = model.evaluate(X_test_t, y_test, batch_size=30)
```

    1025/1025 [==============================] - 0s 122us/step
    


```python
print(score)
#mse 가 0.81 수준으로 나왔다
```

    0.8044285868726125
    


```python
test_sc_df.describe()
```




                 일자별평당가      shift_1      shift_2      shift_3      shift_4  \
    count  1.037000e+03  1036.000000  1035.000000  1034.000000  1033.000000   
    mean  -2.740763e-17    -0.001252    -0.001312    -0.001604    -0.003016   
    std    1.000483e+00     1.000152     1.000634     1.001074     1.000528   
    min   -3.474334e+00    -3.474334    -3.474334    -3.474334    -3.474334   
    25%   -5.708210e-01    -0.571740    -0.572659    -0.573578    -0.574497   
    50%    2.178348e-03     0.000132    -0.001914    -0.003474    -0.005035   
    75%    5.915490e-01     0.588946     0.589814     0.590681     0.588078   
    max    5.024884e+00     5.024884     5.024884     5.024884     5.024884   
    
               shift_5      shift_6      shift_7      shift_8      shift_9  \
    count  1032.000000  1031.000000  1030.000000  1029.000000  1028.000000   
    mean     -0.004648    -0.004920    -0.005242    -0.006563    -0.007535   
    std       0.999636     1.000083     1.000516     1.000103     1.000104   
    min      -3.474334    -3.474334    -3.474334    -3.474334    -3.474334   
    25%      -0.574833    -0.575170    -0.575507    -0.575843    -0.576000   
    50%      -0.005327    -0.005620    -0.005881    -0.006142    -0.007253   
    75%       0.587501     0.587694     0.587886     0.587309     0.580915   
    max       5.024884     5.024884     5.024884     5.024884     5.024884   
    
              shift_10     shift_11     shift_12  
    count  1027.000000  1026.000000  1025.000000  
    mean     -0.007700    -0.008266    -0.009181  
    std       1.000577     1.000901     1.000960  
    min      -3.474334    -3.474334    -3.474334  
    25%      -0.576157    -0.576313    -0.576470  
    50%      -0.008364    -0.008375    -0.008386  
    75%       0.583046     0.585178     0.578784  
    max       5.024884     5.024884     5.024884  




```python
y_pred = model.predict(X_test_t, batch_size=32)
plt.scatter(y_test, y_pred)
plt.xlabel("실제 일자별 평당가: $Y_i$")
plt.ylabel("예측한 일자별 평당가: $\hat{Y}_i$")
plt.title("Prices vs Predicted price Index: $Y_i$ vs $\hat{Y}_i$")
```




    Text(0.5, 1.0, 'Prices vs Predicted price Index: $Y_i$ vs $\\hat{Y}_i$')



    C:\Users\MSI\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py:240: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    C:\Users\MSI\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py:203: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0, flags=flags)
    


    
![png](output_59_2.png)
    


# (7) 다중 회귀 모델링 1차 시도 with Tensorflow 2.0 


```python
#강남구로 한정하여 분석 시도
absgangnam_df = all_df[all_df["area"]=="강남구"]
```


```python
gangnam_int = gangnam_df[["일자별평당가","KOREA_GDP","GDP_Growth_rate","loan","기준금리","housing_loan","gdp_per_person","gdp",'Viliages',"Buildings",'House','세대',"인구","수도권"]]
```


```python
gangnam_int.loc[(gangnam_int.GDP_Growth_rate == "(1)"),"GDP_Growth_rate"] = -1
```

    C:\Users\MSI\Anaconda3\lib\site-packages\pandas\core\indexing.py:1720: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._setitem_single_column(loc, value, pi)
    


```python
gangnam_int["KOREA_GDP"] = gangnam_int["KOREA_GDP"].apply(pd.to_numeric)
gangnam_int["GDP_Growth_rate"] = gangnam_int["GDP_Growth_rate"].apply(pd.to_numeric)
gangnam_int["loan"] = gangnam_int["loan"].apply(pd.to_numeric)
gangnam_int["기준금리"] = gangnam_int["기준금리"].apply(pd.to_numeric)
gangnam_int["housing_loan"] = gangnam_int["housing_loan"].apply(pd.to_numeric)
gangnam_int["gdp_per_person"] = gangnam_int["gdp_per_person"].apply(pd.to_numeric)
gangnam_int["Viliages"] = gangnam_int["Viliages"].apply(pd.to_numeric)
gangnam_int["Buildings"] = gangnam_int["Buildings"].apply(pd.to_numeric)
gangnam_int["House"] = gangnam_int["House"].apply(pd.to_numeric)
```

    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      import sys
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\MSI\Anaconda3\lib\site-packages\ipykernel_launcher.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if __name__ == '__main__':
    


```python
features_name = gangnam_int.columns
```


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(gangnam_int)
gangnam_scaled = scaler.transform(gangnam_int)

gangnam_df_scaled = pd.DataFrame(data=gangnam_scaled, columns=features_name)
```


```python
train_dataset = gangnam_df_scaled.sample(frac=0.8, random_state=0)
test_dataset = gangnam_df_scaled.drop(train_dataset.index)
```


```python
sns.pairplot(train_dataset[["일자별평당가","KOREA_GDP","기준금리","인구","housing_loan"]])
```

    C:\Users\MSI\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py:240: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    




    <seaborn.axisgrid.PairGrid at 0x13d2b133088>



    C:\Users\MSI\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py:203: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0, flags=flags)
    


    
![png](output_68_3.png)
    


# 데이터 컬럼간 상관관계 체크 


```python
corr = gangnam_df_scaled.corr(method='pearson')
corr_df = corr.apply(lambda x : round(x,2))
```


```python
print(corr_df.iloc[0,:])
```

    일자별평당가             1.00
    KOREA_GDP          0.78
    GDP_Growth_rate   -0.59
    loan               0.82
    기준금리              -0.65
    housing_loan       0.85
    gdp_per_person     0.79
    gdp                0.82
    Viliages           0.01
    Buildings          0.53
    House              0.42
    세대                 0.82
    인구                -0.82
    수도권               -0.02
    Name: 일자별평당가, dtype: float64
    


```python
#데이터셋과 데이러라벨 분리
train_labels = train_dataset.pop("일자별평당가")
test_labels = test_dataset.pop("일자별평당가")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_28600/3224697975.py in <module>
          1 #데이터셋과 데이러라벨 분리
    ----> 2 train_labels = train_dataset.pop("일자별평당가")
          3 test_labels = test_dataset.pop("일자별평당가")
    

    NameError: name 'train_dataset' is not defined



```python
#순차형 모델 설계

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
```


```python
model = build_model()
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 64)                896       
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 9,281
    Trainable params: 9,281
    Non-trainable params: 0
    _________________________________________________________________
    


```python
example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
example_result
```




    array([[ 0.02163886],
           [ 0.5113964 ],
           [ 0.48028725],
           [-0.36780232],
           [ 0.22917318],
           [ 0.22917318],
           [-0.09433231],
           [-0.36780232],
           [ 0.38817587],
           [ 0.38817587]], dtype=float32)




```python
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

```

    
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................


```python
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [일자별평당가]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0.25,0.5])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$일자별평당가^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0.2,0.5])
  plt.legend()
  plt.show()

plot_history(history)
```

    Font 'rm' does not have a glyph for '\uc77c' [U+c77c], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\uc790' [U+c790], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\ubcc4' [U+bcc4], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\ud3c9' [U+d3c9], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\ub2f9' [U+b2f9], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\uac00' [U+ac00], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\uc77c' [U+c77c], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\uc790' [U+c790], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\ubcc4' [U+bcc4], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\ud3c9' [U+d3c9], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\ub2f9' [U+b2f9], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\uac00' [U+ac00], substituting with a dummy symbol.
    


    
![png](output_78_1.png)
    



```python
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
print("테스트 세트의 평균 절대 오차 : {:5.2f} 일자별평당가".format(mae))
```

    702/702 - 0s - loss: 0.2435 - mae: 0.3191 - mse: 0.2435
    테스트 세트의 평균 절대 오차 :  0.32 일자별평당가
    


```python
test_predictions = model.predict(test_dataset).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('실제값 [일자별평당가]')
plt.ylabel('예측값 [일자별평당가]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
```


    
![png](output_80_0.png)
    



```python
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [일자별평당가]")
_ = plt.ylabel("Count")
```

    C:\Users\MSI\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py:240: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    C:\Users\MSI\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py:203: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0, flags=flags)
    


    
![png](output_81_1.png)
    


# (8) 다중 회귀 모델링 2차 시도 with Tensorflow 2.0 (interpolate)


```python
gangnam_df = all_df[all_df["area"]=="강남구"]
```


```python
#날짜 인덱스 길이에 따라 list 생성
date_index = gangnam_df.index
len(date_index)
list_test = []
for i in range(3512):
    list_test.append(np.nan)
```


```python
df_test = pd.DataFrame(list_test, columns = ["test"], index = date_index)
```


```python
#Macro 거시데이터 리드
t_korea_gdp_rate = pd.read_csv("./data/Marco/01_korea_gdp_series.csv",sep=",",encoding="UTF-8")
t_korea_interest_rate = pd.read_csv("./data/Marco/02_korea_interest_rate.csv",sep=",",encoding="UTF-8")
t_korea_personal_loan = pd.read_csv("./data/Marco/03_korea_personal_loan.csv",sep=",",encoding="UTF-8")
t_korea_loan_for_house = pd.read_csv("./data/Marco/04_korea_loan_for_house.csv",sep=",",encoding="UTF-8")
t_korea_personal_GDP = pd.read_csv("./data/Marco/05_korea_personal_GDP.csv",sep=",",encoding="UTF-8")
t_seoul_gdp_series = pd.read_csv("./data/Marco/06_seoul_gdp_series.csv",sep=",",encoding="UTF-8")
t_housing_count_yearly = pd.read_csv("./data/Marco/07_housing_count_yearly.csv",sep=",",encoding="UTF-8")
t_seoul_population= pd.read_csv("./data/Marco/08_seoul_population.csv",sep=",",encoding="UTF-8")
t_constructure_confirm= pd.read_csv("./data/Marco/09_constructure_confirm.csv",sep=",",encoding="UTF-8")

#2011~2020년의 필요한 연도 데이터 추출
t_korea_gdp_rate = t_korea_gdp_rate.loc[41:]
t_korea_interest_rate = t_korea_interest_rate.loc[12:]
t_korea_personal_loan = t_korea_personal_loan.loc[9:]
t_korea_loan_for_house = t_korea_loan_for_house.loc[:9]
t_korea_personal_GDP = t_korea_personal_GDP[1:]
t_seoul_gdp_series
t_housing_count_yearly
t_seoul_population
t_constructure_confirm
```




       year      합계     수도권      서울
    0  2011  549594  272156   88060
    1  2012  586884  269290   86123
    2  2013  440116  192610   77621
    3  2014  515251  241889   65249
    4  2015  765328  408773  101235
    5  2016  726048  341162   74739
    6  2017  653441  321402  113131
    7  2018  554136  280097   65751
    8  2019  487975  272226   62272
    9  2020  457514  252301   58181




```python
#날짜별 데이터 인덱스 대표값 입력
date_time_index = [datetime(2011,1,1),datetime(2012,12,31),datetime(2013,12,31),datetime(2014,12,31),datetime(2015,12,31),datetime(2016,12,31),datetime(2017,12,31),datetime(2018,12,31),datetime(2019,12,31),datetime(2020,12,31)]
```


```python
t_korea_gdp_rate["date_time_index"] = date_time_index
t_korea_gdp_rate = t_korea_gdp_rate.set_index("date_time_index")

t_korea_interest_rate["date_time_index"] = date_time_index
t_korea_interest_rate = t_korea_interest_rate.set_index("date_time_index")

t_korea_personal_loan["date_time_index"] = date_time_index
t_korea_personal_loan = t_korea_personal_loan.set_index("date_time_index")

t_korea_loan_for_house["date_time_index"] = date_time_index
t_korea_loan_for_house = t_korea_loan_for_house.set_index("date_time_index")

t_korea_personal_GDP["date_time_index"] = date_time_index
t_korea_personal_GDP = t_korea_personal_GDP.set_index("date_time_index")

t_seoul_gdp_series["date_time_index"] = date_time_index
t_seoul_gdp_series = t_seoul_gdp_series.set_index("date_time_index")

t_housing_count_yearly["date_time_index"] = date_time_index
t_housing_count_yearly = t_housing_count_yearly.set_index("date_time_index")

t_seoul_population["date_time_index"] = date_time_index
t_seoul_population = t_seoul_population.set_index("date_time_index")

t_constructure_confirm["date_time_index"] = date_time_index
t_constructure_confirm = t_constructure_confirm.set_index("date_time_index")
```


```python
#날짜 인덱스 입력 체크
t_korea_personal_loan
```




                     year  loan
    date_time_index            
    2011-01-01       2011   916
    2012-12-31       2012   964
    2013-12-31       2013  1019
    2014-12-31       2014  1085
    2015-12-31       2015  1203
    2016-12-31       2016  1343
    2017-12-31       2017  1451
    2018-12-31       2018  1537
    2019-12-31       2019  1600
    2020-12-31       2020  1726




```python
t_korea_gdp_rate.loc[(t_korea_gdp_rate.GDP_Growth_rate == "(1)"),"GDP_Growth_rate"] = -1
```


```python
t_korea_gdp_rate["KOREA_GDP"] = t_korea_gdp_rate["KOREA_GDP"].apply(pd.to_numeric)
t_korea_gdp_rate["GDP_Growth_rate"] = t_korea_gdp_rate["GDP_Growth_rate"].apply(pd.to_numeric)
t_korea_personal_loan["loan"] = t_korea_personal_loan["loan"].apply(pd.to_numeric)
t_korea_interest_rate["기준금리"] = t_korea_interest_rate["기준금리"].apply(pd.to_numeric)
t_korea_loan_for_house["housing_loan"] = t_korea_loan_for_house["housing_loan"].apply(pd.to_numeric)
t_korea_personal_GDP["gdp_per_person"] = t_korea_personal_GDP["gdp_per_person"].apply(pd.to_numeric)
t_housing_count_yearly["Viliages"] = t_housing_count_yearly["Viliages"].apply(pd.to_numeric)
t_housing_count_yearly["Buildings"] = t_housing_count_yearly["Buildings"].apply(pd.to_numeric)
t_housing_count_yearly["House"] = t_housing_count_yearly["House"].apply(pd.to_numeric)
t_constructure_confirm["수도권"] = t_constructure_confirm["수도권"].apply(pd.to_numeric)
```


```python
# Interpolate를 활용하여 데이터 선형화 / 연속화
interpol1 = pd.merge(left=df_test, right=t_korea_gdp_rate, how="left", left_index=True, right_index=True)
interpol2 = pd.merge(left=df_test, right=t_korea_personal_loan, how="left", left_index=True, right_index=True)
interpol3 = pd.merge(left=df_test, right=t_korea_interest_rate, how="left", left_index=True, right_index=True)
interpol4 = pd.merge(left=df_test, right=t_korea_loan_for_house, how="left", left_index=True, right_index=True)
interpol5 = pd.merge(left=df_test, right=t_korea_personal_GDP, how="left", left_index=True, right_index=True)
interpol6 = pd.merge(left=df_test, right=t_housing_count_yearly, how="left", left_index=True, right_index=True)
interpol7 = pd.merge(left=df_test, right=t_constructure_confirm, how="left", left_index=True, right_index=True)
interpol8 = pd.merge(left=df_test, right=t_seoul_gdp_series, how="left", left_index=True, right_index=True)
interpol9 = pd.merge(left=df_test, right=t_seoul_population, how="left", left_index=True, right_index=True)
```


```python
interpol1 = interpol1.interpolate()
interpol2 = interpol2.interpolate()
interpol3 = interpol3.interpolate()
interpol4 = interpol4.interpolate()
interpol5 = interpol5.interpolate()
interpol6 = interpol6.interpolate()
interpol7 = interpol7.interpolate()
interpol8 = interpol8.interpolate()
interpol9 = interpol9.interpolate()
```


```python
all_df_prepro = tr_data_series.set_index("계약년월일")
```


```python
all_df_prepro = all_df_prepro[all_df_prepro["area"]=="강남구"]
```


```python
merge1 = pd.merge(left=all_df_prepro, right=interpol1, how="left", left_index=True, right_index=True)
merge2 = pd.merge(left=merge1, right=interpol2, how="left", left_index=True, right_index=True)
merge3 = pd.merge(left=merge2, right=interpol3, how="left", left_index=True, right_index=True)
merge4 = pd.merge(left=merge3, right=interpol4, how="left", left_index=True, right_index=True)
merge5 = pd.merge(left=merge4, right=interpol5, how="left", left_index=True, right_index=True)
merge6 = pd.merge(left=merge5, right=interpol6, how="left", left_index=True, right_index=True)
merge7 = pd.merge(left=merge6, right=interpol7, how="left", left_index=True, right_index=True)
merge8 = pd.merge(left=merge7, right=interpol8, how="left", left_index=True, right_index=True)
merge9 = pd.merge(left=merge8, right=interpol9, how="left", left_index=True, right_index=True)
```


```python
target_df = merge9[["일자별평당가","KOREA_GDP","GDP_Growth_rate","loan","기준금리","housing_loan","gdp_per_person","gdp",'Viliages',"Buildings",'House','세대',"인구","수도권"]]
```


```python
target_df.plot()
```




    <AxesSubplot:xlabel='계약년월일'>




    
![png](output_98_1.png)
    



```python
na_df = target_df[target_df["KOREA_GDP"].isna()]
#중복된 데이터 개수 세기
na_df.index.drop_duplicates()
```




    DatetimeIndex([], dtype='datetime64[ns]', name='계약년월일', freq=None)




```python
target_df = target_df.dropna()
```


```python
target_df
```




                     일자별평당가     KOREA_GDP  GDP_Growth_rate         loan  기준금리  \
    계약년월일                                                                       
    2011-01-01   663.166621  1.388937e+06         4.000000   916.000000   3.0   
    2011-01-03   820.308356  1.389012e+06         3.997085   916.069971   3.0   
    2011-01-04   859.692819  1.389086e+06         3.994169   916.139942   3.0   
    2011-01-05  1084.908684  1.389161e+06         3.991254   916.209913   3.0   
    2011-01-06  1087.227224  1.389235e+06         3.988338   916.279883   3.0   
    ...                 ...           ...              ...          ...   ...   
    2020-12-27  2557.440399  1.933051e+06        -0.965015  1724.530612   1.0   
    2020-12-28  2471.905319  1.933076e+06        -0.973761  1724.897959   1.0   
    2020-12-29  2030.309822  1.933102e+06        -0.982507  1725.265306   1.0   
    2020-12-30  1938.892652  1.933127e+06        -0.991254  1725.632653   1.0   
    2020-12-31  2410.882176  1.933152e+06        -1.000000  1726.000000   1.0   
    
                housing_loan  gdp_per_person           gdp     Viliages  \
    계약년월일                                                                 
    2011-01-01  2.511440e+06    27901.000000  3.264151e+08  4081.000000   
    2011-01-03  2.511656e+06    27902.300292  3.264266e+08  4081.017493   
    2011-01-04  2.511871e+06    27903.600583  3.264382e+08  4081.034985   
    2011-01-05  2.512087e+06    27904.900875  3.264497e+08  4081.052478   
    2011-01-06  2.512303e+06    27906.201166  3.264612e+08  4081.069971   
    ...                  ...             ...           ...          ...   
    2020-12-27  5.907140e+06    37564.571429  4.351126e+08  4134.501458   
    2020-12-28  5.908911e+06    37565.428571  4.351102e+08  4134.376093   
    2020-12-29  5.910681e+06    37566.285714  4.351078e+08  4134.250729   
    2020-12-30  5.912452e+06    37567.142857  4.351054e+08  4134.125364   
    2020-12-31  5.914223e+06    37568.000000  4.351030e+08  4134.000000   
    
                   Buildings         House            세대            인구  \
    계약년월일                                                                
    2011-01-01  19022.000000  1.459112e+06  4.192752e+06  1.052877e+07   
    2011-01-03  19022.180758  1.459114e+06  4.192730e+06  1.052865e+07   
    2011-01-04  19022.361516  1.459117e+06  4.192709e+06  1.052852e+07   
    2011-01-05  19022.542274  1.459119e+06  4.192687e+06  1.052840e+07   
    2011-01-06  19022.723032  1.459121e+06  4.192666e+06  1.052827e+07   
    ...                  ...           ...           ...           ...   
    2020-12-27  20177.845481  1.544251e+06  4.416900e+06  9.912253e+06   
    2020-12-28  20178.134111  1.544295e+06  4.417164e+06  9.911962e+06   
    2020-12-29  20178.422741  1.544338e+06  4.417427e+06  9.911670e+06   
    2020-12-30  20178.711370  1.544381e+06  4.417691e+06  9.911379e+06   
    2020-12-31  20179.000000  1.544424e+06  4.417954e+06  9.911088e+06   
    
                          수도권  
    계약년월일                      
    2011-01-01  272156.000000  
    2011-01-03  272151.822157  
    2011-01-04  272147.644315  
    2011-01-05  272143.466472  
    2011-01-06  272139.288630  
    ...                   ...  
    2020-12-27  252533.361516  
    2020-12-28  252475.271137  
    2020-12-29  252417.180758  
    2020-12-30  252359.090379  
    2020-12-31  252301.000000  
    
    [3512 rows x 14 columns]




```python
features_name = target_df.columns
```


```python
#표준화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(target_df)
gangnam_scaled = scaler.transform(target_df)

gangnam_df_scaled = pd.DataFrame(data=gangnam_scaled, columns=features_name)
```


```python
target_df["일자별평당가"].mean()
```




    1367.6529994162436




```python
target_df["일자별평당가"].std()
```




    443.0142889012346




```python
train_dataset = gangnam_df_scaled.sample(frac=0.8, random_state=0)
test_dataset = gangnam_df_scaled.drop(train_dataset.index)
```


```python
train_dataset = train_dataset.dropna()
test_dataset = test_dataset.dropna()
```


```python
sns.pairplot(train_dataset[["일자별평당가","KOREA_GDP","기준금리","인구","housing_loan"]])
```

    C:\Users\MSI\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py:240: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    




    <seaborn.axisgrid.PairGrid at 0x13d460e7788>



    C:\Users\MSI\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py:203: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0, flags=flags)
    


    
![png](output_108_3.png)
    



```python
train_labels = train_dataset.pop("일자별평당가")
test_labels = test_dataset.pop("일자별평당가")
```


```python
test_dataset[:2]
```




       KOREA_GDP  GDP_Growth_rate      loan      기준금리  housing_loan  \
    0  -1.424635          1.60839 -1.279383  1.250434     -1.152464   
    3  -1.423473          1.59793 -1.278568  1.250434     -1.151826   
    
       gdp_per_person       gdp  Viliages  Buildings     House        세대  \
    0       -1.413319 -1.266843 -1.511219  -1.531661 -1.008147 -0.498564   
    3       -1.412172 -1.265952 -1.510096  -1.530136 -1.007907 -0.499625   
    
             인구       수도권  
    0  1.487077 -0.283590  
    3  1.484989 -0.283843  




```python
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
```


```python
model = build_model()
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_4 (Dense)              (None, 64)                896       
    _________________________________________________________________
    dense_5 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_6 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_7 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 9,281
    Trainable params: 9,281
    Non-trainable params: 0
    _________________________________________________________________
    


```python
example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
example_result
```




    array([[ 0.00797851],
           [-0.01199541],
           [-0.09142744],
           [ 0.05832197],
           [ 0.27348617],
           [ 0.18634151],
           [ 0.04727918],
           [ 0.07454206],
           [ 0.10136928],
           [ 0.0705907 ]], dtype=float32)




```python
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

```

    
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................


```python
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
```




             loss       mae       mse  val_loss   val_mae   val_mse  epoch
    495  0.211812  0.297824  0.211812  0.220288  0.304418  0.220288    495
    496  0.210872  0.299693  0.210872  0.217033  0.306330  0.217033    496
    497  0.211817  0.299451  0.211817  0.220548  0.306266  0.220548    497
    498  0.211599  0.299263  0.211599  0.219617  0.303908  0.219617    498
    499  0.212285  0.299121  0.212285  0.218657  0.304192  0.218657    499




```python
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [일자별평당가]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,1])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$일자별평당가^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,1])
  plt.legend()
  plt.show()

plot_history(history)
```

    Font 'rm' does not have a glyph for '\uc77c' [U+c77c], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\uc790' [U+c790], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\ubcc4' [U+bcc4], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\ud3c9' [U+d3c9], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\ub2f9' [U+b2f9], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\uac00' [U+ac00], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\uc77c' [U+c77c], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\uc790' [U+c790], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\ubcc4' [U+bcc4], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\ud3c9' [U+d3c9], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\ub2f9' [U+b2f9], substituting with a dummy symbol.
    Font 'rm' does not have a glyph for '\uac00' [U+ac00], substituting with a dummy symbol.
    


    
![png](output_117_1.png)
    



```python
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
print("테스트 세트의 평균 절대 오차 : {:5.2f} 일자별평당가".format(mae))
```

    702/702 - 0s - loss: 0.2247 - mae: 0.3053 - mse: 0.2247
    테스트 세트의 평균 절대 오차 :  0.31 일자별평당가
    


```python
test_predictions = model.predict(test_dataset).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('실제값 [일자별평당가]')
plt.ylabel('예측값 [일자별평당가]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
```


    
![png](output_119_0.png)
    


# 회귀 계수 확인


```python
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
mlr.fit(train_dataset,train_labels)
```




    LinearRegression()




```python
test_predict = mlr.predict(test_dataset)
```


```python
mlr.intercept_
```




    -0.0018036691811668545




```python
a = mlr.coef_
b = a.tolist()
```


```python
b
```




    [-1623274064068.07,
     481600679659.31866,
     -3101418178340.6064,
     65030817020.926315,
     5234723716722.214,
     29619458914.86346,
     -2526308842104.1704,
     -219004405806.8065,
     206194477323.95392,
     -87436112647.77512,
     -1268401596474.5413,
     -3336944064158.5127,
     125243097565.24297]




```python
plt.scatter(test_labels, test_predict, alpha = 0.4)
plt.show()
```

    C:\Users\MSI\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py:240: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    C:\Users\MSI\Anaconda3\lib\site-packages\matplotlib\backends\backend_agg.py:203: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0, flags=flags)
    


    
![png](output_126_1.png)
    


# (9) 모델을 활용한 예측 시나리오 


```python
#한국 GDP(십억원). GDP성장률(%), 가계대출(조원), 금리(%), 주택금융(억원), 1인당 GDP, 서울GDP, 아파트단지, 동수, 가구수, 세대수, 서울시 인구, 수도권 건축허가
prediction_2020 =  [1933152, -1.00, 1726, 1.0, 5914223, 37568, 435102998, 4134.0, 20179.0, 1544424.0, 4417954.0, 9911088.0, 252301.0]
```

아래 prediction_2023 리스트에 원하는 시나리오에 해당 하는 값으로 변경


```python
prediction_2023 =  [1933152, -1.00, 1500, 1.0, 5914223, 37568, 435102998, 4134.0, 20179.0, 1544424.0, 4417954.0, 9911088.0, 252301.0]
```


```python
p_t_korea_gdp_rate = t_korea_gdp_rate.iloc[:,1].values.tolist()
p_t_korea_gdp_rate.append(prediction_2023[0])
p_t_korea_gdp_rate_2 = t_korea_gdp_rate.iloc[:,2].values.tolist()
p_t_korea_gdp_rate_2.append(prediction_2023[1])
p_t_korea_personal_loan = t_korea_personal_loan.iloc[:,1].values.tolist()
p_t_korea_personal_loan.append(prediction_2023[2])
p_t_korea_interest_rate = t_korea_interest_rate.iloc[:,1].values.tolist()
p_t_korea_interest_rate.append(prediction_2023[3])
p_t_korea_loan_for_house = t_korea_loan_for_house.iloc[:,1].values.tolist()
p_t_korea_loan_for_house.append(prediction_2023[4])
p_t_korea_personal_GDP = t_korea_personal_GDP.iloc[:,1].values.tolist()
p_t_korea_personal_GDP.append(prediction_2023[5])
p_t_seoul_gdp_series = t_seoul_gdp_series.iloc[:,1].values.tolist()
p_t_seoul_gdp_series.append(prediction_2023[6])
p_t_housing_count_yearly = t_housing_count_yearly.iloc[:,1].values.tolist()
p_t_housing_count_yearly.append(prediction_2023[7])
p_t_housing_count_yearly2 = t_housing_count_yearly.iloc[:,2].values.tolist()
p_t_housing_count_yearly2.append(prediction_2023[8])
p_t_housing_count_yearly3 = t_housing_count_yearly.iloc[:,3].values.tolist()
p_t_housing_count_yearly3.append(prediction_2023[9])
p_t_seoul_population = t_seoul_population.iloc[:,1].values.tolist()
p_t_seoul_population.append(prediction_2023[10])
p_t_seoul_population2 = t_seoul_population.iloc[:,2].values.tolist()
p_t_seoul_population2.append(prediction_2023[11])
p_t_constructure_confirm = t_constructure_confirm.iloc[:,2].values.tolist()
p_t_constructure_confirm.append(prediction_2023[12])
```


```python
result = {"1":p_t_korea_gdp_rate,
          "2":p_t_korea_gdp_rate_2,
          "3":p_t_korea_personal_loan,
          "4":p_t_korea_interest_rate,
          "5":p_t_korea_loan_for_house,
          "6":p_t_korea_personal_GDP,
          "7":p_t_seoul_gdp_series,
          "8":p_t_housing_count_yearly,
          "9":p_t_housing_count_yearly2,
          "10":p_t_housing_count_yearly3,
          "11":p_t_seoul_population,
          "12":p_t_seoul_population2,
          "13":p_t_constructure_confirm}
```


```python
result_df = pd.DataFrame(result)
```


```python
features_name = result_df.columns
```


```python
scaler = StandardScaler()
scaler.fit(result_df)
result_scaled = scaler.transform(result_df)

result_df_scaled = pd.DataFrame(data=result_scaled, columns=features_name)
```


```python
prediction_2023 = result_df_scaled.iloc[10,:].tolist()
```


```python
prediction_2023
```




    [1.1106653210030881,
     -2.007387671367415,
     0.7327348334733893,
     -1.147078669352809,
     1.546410679260238,
     1.1285861640701929,
     1.1917846307057862,
     -0.34676527372731875,
     1.2473695929018436,
     1.334659438424521,
     1.8650400734240828,
     -1.412205733642385,
     -0.5471789150810322]




```python
prediction_2023_np = np.array(prediction_2023)
prediction_2023_np = np.reshape(prediction_2023_np, (13))
prediction_2023_np = np.expand_dims(prediction_2023_np, axis=0)


predictions = model.predict(prediction_2023_np)
```


```python
p = predictions[0][0]
```


```python
# 모델이 예측한 평당가
(p * target_df["일자별평당가"].std() + target_df["일자별평당가"].mean()) * 3.3
```




    6544.339630303525


