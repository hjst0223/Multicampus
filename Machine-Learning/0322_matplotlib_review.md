# matplotlib 😶

## 1. Line plot (선 그래프)


```python
import matplotlib.pyplot as plt

# plot의 제목 설정
plt.title('Line Plot')

# plot()이라는 메소드를 이용해서 line plot을 그릴 수 있음ㄴ
# x 축의 자료 위치(x 축 눈금) => tick은 기본적으로 0, 1, 2, 3
plt.plot([1, 5, 12, 25])
plt.show()  # => 화면에 rendering하고 마우스 이벤트를 기다리는 함수
# pycharm과 같은 일반 개발 툴을 이용하면 show()를 이용해야 그래프가 나옴
# jupyter notebook은 셀 단위로 plot을 rendering하기 때문에 show()를 할 필요가 없음
```


    
![png](/Machine-Learning/images/0322/output_3_0.png)
    



```python
# 만약 tick을 별도로 명시하는 경우
plt.title('Line Plot')
plt.plot([10, 20, 30, 40], [1, 5, 12, 25])
plt.show() 
```


    
![png](/Machine-Learning/images/0322/output_4_0.png)
    


## 데이터 추출하기


```python
import numpy as np
import pandas as pd

df = pd.read_excel('./data/lineplot_sample_data.xlsx')
df = df.fillna(method='ffill')  # 이전 행의 값으로 NaN 채우기
display(df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>전출지별</th>
      <th>전입지별</th>
      <th>1970</th>
      <th>1971</th>
      <th>1972</th>
      <th>1973</th>
      <th>1974</th>
      <th>1975</th>
      <th>1976</th>
      <th>1977</th>
      <th>...</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>전출지별</td>
      <td>전입지별</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>...</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
      <td>이동자수 (명)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>전국</td>
      <td>전국</td>
      <td>4046536</td>
      <td>4210164</td>
      <td>3687938</td>
      <td>4860418</td>
      <td>5297969</td>
      <td>9011440</td>
      <td>6773250</td>
      <td>7397623</td>
      <td>...</td>
      <td>8808256</td>
      <td>8487275</td>
      <td>8226594</td>
      <td>8127195</td>
      <td>7506691</td>
      <td>7411784</td>
      <td>7629098</td>
      <td>7755286</td>
      <td>7378430</td>
      <td>7154226</td>
    </tr>
    <tr>
      <th>2</th>
      <td>전국</td>
      <td>서울특별시</td>
      <td>1742813</td>
      <td>1671705</td>
      <td>1349333</td>
      <td>1831858</td>
      <td>2050392</td>
      <td>3396662</td>
      <td>2756510</td>
      <td>2893403</td>
      <td>...</td>
      <td>2025358</td>
      <td>1873188</td>
      <td>1733015</td>
      <td>1721748</td>
      <td>1555281</td>
      <td>1520090</td>
      <td>1573594</td>
      <td>1589431</td>
      <td>1515602</td>
      <td>1472937</td>
    </tr>
    <tr>
      <th>3</th>
      <td>전국</td>
      <td>부산광역시</td>
      <td>448577</td>
      <td>389797</td>
      <td>362202</td>
      <td>482061</td>
      <td>680984</td>
      <td>805979</td>
      <td>724664</td>
      <td>785117</td>
      <td>...</td>
      <td>514502</td>
      <td>519310</td>
      <td>519334</td>
      <td>508043</td>
      <td>461042</td>
      <td>478451</td>
      <td>485710</td>
      <td>507031</td>
      <td>459015</td>
      <td>439073</td>
    </tr>
    <tr>
      <th>4</th>
      <td>전국</td>
      <td>대구광역시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>409938</td>
      <td>398626</td>
      <td>370817</td>
      <td>370563</td>
      <td>348642</td>
      <td>351873</td>
      <td>350213</td>
      <td>351424</td>
      <td>328228</td>
      <td>321182</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>320</th>
      <td>제주특별자치도</td>
      <td>전라북도</td>
      <td>139</td>
      <td>202</td>
      <td>141</td>
      <td>210</td>
      <td>219</td>
      <td>415</td>
      <td>392</td>
      <td>408</td>
      <td>...</td>
      <td>668</td>
      <td>579</td>
      <td>672</td>
      <td>561</td>
      <td>551</td>
      <td>516</td>
      <td>609</td>
      <td>683</td>
      <td>733</td>
      <td>768</td>
    </tr>
    <tr>
      <th>321</th>
      <td>제주특별자치도</td>
      <td>전라남도</td>
      <td>631</td>
      <td>965</td>
      <td>857</td>
      <td>952</td>
      <td>1133</td>
      <td>2808</td>
      <td>2608</td>
      <td>2652</td>
      <td>...</td>
      <td>1143</td>
      <td>1123</td>
      <td>1002</td>
      <td>1026</td>
      <td>966</td>
      <td>1001</td>
      <td>928</td>
      <td>1062</td>
      <td>1127</td>
      <td>1102</td>
    </tr>
    <tr>
      <th>322</th>
      <td>제주특별자치도</td>
      <td>경상북도</td>
      <td>374</td>
      <td>619</td>
      <td>468</td>
      <td>576</td>
      <td>625</td>
      <td>1123</td>
      <td>1141</td>
      <td>1004</td>
      <td>...</td>
      <td>761</td>
      <td>704</td>
      <td>738</td>
      <td>756</td>
      <td>699</td>
      <td>781</td>
      <td>728</td>
      <td>903</td>
      <td>931</td>
      <td>994</td>
    </tr>
    <tr>
      <th>323</th>
      <td>제주특별자치도</td>
      <td>경상남도</td>
      <td>474</td>
      <td>479</td>
      <td>440</td>
      <td>571</td>
      <td>1208</td>
      <td>1517</td>
      <td>863</td>
      <td>1122</td>
      <td>...</td>
      <td>1517</td>
      <td>1474</td>
      <td>1324</td>
      <td>1367</td>
      <td>1227</td>
      <td>1278</td>
      <td>1223</td>
      <td>1500</td>
      <td>1448</td>
      <td>1501</td>
    </tr>
    <tr>
      <th>324</th>
      <td>제주특별자치도</td>
      <td>제주특별자치도</td>
      <td>9290</td>
      <td>12427</td>
      <td>12210</td>
      <td>16158</td>
      <td>19580</td>
      <td>34221</td>
      <td>23291</td>
      <td>31028</td>
      <td>...</td>
      <td>59564</td>
      <td>55673</td>
      <td>55507</td>
      <td>59846</td>
      <td>54280</td>
      <td>60607</td>
      <td>59673</td>
      <td>59036</td>
      <td>66444</td>
      <td>63275</td>
    </tr>
  </tbody>
</table>
<p>325 rows × 50 columns</p>
</div>


### 서울특별시에서 다른 지역으로 이동한 데이터만 추출하기
- boolean mask를 만든 후 indexing


```python
bool_mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul = df.loc[bool_mask,:]
display(df_seoul)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>전출지별</th>
      <th>전입지별</th>
      <th>1970</th>
      <th>1971</th>
      <th>1972</th>
      <th>1973</th>
      <th>1974</th>
      <th>1975</th>
      <th>1976</th>
      <th>1977</th>
      <th>...</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>서울특별시</td>
      <td>전국</td>
      <td>1448985</td>
      <td>1419016</td>
      <td>1210559</td>
      <td>1647268</td>
      <td>1819660</td>
      <td>2937093</td>
      <td>2495620</td>
      <td>2678007</td>
      <td>...</td>
      <td>2083352</td>
      <td>1925452</td>
      <td>1848038</td>
      <td>1834806</td>
      <td>1658928</td>
      <td>1620640</td>
      <td>1661425</td>
      <td>1726687</td>
      <td>1655859</td>
      <td>1571423</td>
    </tr>
    <tr>
      <th>21</th>
      <td>서울특별시</td>
      <td>부산광역시</td>
      <td>11568</td>
      <td>11130</td>
      <td>11768</td>
      <td>16307</td>
      <td>22220</td>
      <td>27515</td>
      <td>23732</td>
      <td>27213</td>
      <td>...</td>
      <td>17353</td>
      <td>17738</td>
      <td>17418</td>
      <td>18816</td>
      <td>16135</td>
      <td>16153</td>
      <td>17320</td>
      <td>17009</td>
      <td>15062</td>
      <td>14484</td>
    </tr>
    <tr>
      <th>22</th>
      <td>서울특별시</td>
      <td>대구광역시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>9720</td>
      <td>10464</td>
      <td>10277</td>
      <td>10397</td>
      <td>10135</td>
      <td>10631</td>
      <td>10062</td>
      <td>10191</td>
      <td>9623</td>
      <td>8891</td>
    </tr>
    <tr>
      <th>23</th>
      <td>서울특별시</td>
      <td>인천광역시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>50493</td>
      <td>45392</td>
      <td>46082</td>
      <td>51641</td>
      <td>49640</td>
      <td>47424</td>
      <td>43212</td>
      <td>44915</td>
      <td>43745</td>
      <td>40485</td>
    </tr>
    <tr>
      <th>24</th>
      <td>서울특별시</td>
      <td>광주광역시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>10846</td>
      <td>11725</td>
      <td>11095</td>
      <td>10587</td>
      <td>10154</td>
      <td>9129</td>
      <td>9759</td>
      <td>9216</td>
      <td>8354</td>
      <td>7932</td>
    </tr>
    <tr>
      <th>25</th>
      <td>서울특별시</td>
      <td>대전광역시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>13515</td>
      <td>13632</td>
      <td>13819</td>
      <td>13900</td>
      <td>14080</td>
      <td>13440</td>
      <td>13403</td>
      <td>13453</td>
      <td>12619</td>
      <td>11815</td>
    </tr>
    <tr>
      <th>26</th>
      <td>서울특별시</td>
      <td>울산광역시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>5057</td>
      <td>4845</td>
      <td>4742</td>
      <td>5188</td>
      <td>5691</td>
      <td>5542</td>
      <td>6047</td>
      <td>5950</td>
      <td>5102</td>
      <td>4260</td>
    </tr>
    <tr>
      <th>27</th>
      <td>서울특별시</td>
      <td>세종특별자치시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>2998</td>
      <td>2851</td>
      <td>6481</td>
      <td>7550</td>
      <td>5943</td>
      <td>5813</td>
    </tr>
    <tr>
      <th>28</th>
      <td>서울특별시</td>
      <td>경기도</td>
      <td>130149</td>
      <td>150313</td>
      <td>93333</td>
      <td>143234</td>
      <td>149045</td>
      <td>253705</td>
      <td>202276</td>
      <td>207722</td>
      <td>...</td>
      <td>412408</td>
      <td>398282</td>
      <td>410735</td>
      <td>373771</td>
      <td>354135</td>
      <td>340801</td>
      <td>332785</td>
      <td>359337</td>
      <td>370760</td>
      <td>342433</td>
    </tr>
    <tr>
      <th>29</th>
      <td>서울특별시</td>
      <td>강원도</td>
      <td>9352</td>
      <td>12885</td>
      <td>13561</td>
      <td>16481</td>
      <td>15479</td>
      <td>27837</td>
      <td>25927</td>
      <td>25415</td>
      <td>...</td>
      <td>23668</td>
      <td>23331</td>
      <td>22736</td>
      <td>23624</td>
      <td>22332</td>
      <td>20601</td>
      <td>21173</td>
      <td>22659</td>
      <td>21590</td>
      <td>21016</td>
    </tr>
    <tr>
      <th>30</th>
      <td>서울특별시</td>
      <td>충청북도</td>
      <td>6700</td>
      <td>9457</td>
      <td>10853</td>
      <td>12617</td>
      <td>11786</td>
      <td>21073</td>
      <td>18029</td>
      <td>17478</td>
      <td>...</td>
      <td>15294</td>
      <td>15295</td>
      <td>15461</td>
      <td>15318</td>
      <td>14555</td>
      <td>13783</td>
      <td>14244</td>
      <td>14379</td>
      <td>14087</td>
      <td>13302</td>
    </tr>
    <tr>
      <th>31</th>
      <td>서울특별시</td>
      <td>충청남도</td>
      <td>15954</td>
      <td>18943</td>
      <td>23406</td>
      <td>27139</td>
      <td>25509</td>
      <td>51205</td>
      <td>41447</td>
      <td>43993</td>
      <td>...</td>
      <td>27458</td>
      <td>24889</td>
      <td>24522</td>
      <td>24723</td>
      <td>22269</td>
      <td>21486</td>
      <td>21473</td>
      <td>22299</td>
      <td>21741</td>
      <td>21020</td>
    </tr>
    <tr>
      <th>32</th>
      <td>서울특별시</td>
      <td>전라북도</td>
      <td>10814</td>
      <td>13192</td>
      <td>16583</td>
      <td>18642</td>
      <td>16647</td>
      <td>34411</td>
      <td>29835</td>
      <td>28444</td>
      <td>...</td>
      <td>18390</td>
      <td>18332</td>
      <td>17569</td>
      <td>17755</td>
      <td>16120</td>
      <td>14909</td>
      <td>14566</td>
      <td>14835</td>
      <td>13835</td>
      <td>13179</td>
    </tr>
    <tr>
      <th>33</th>
      <td>서울특별시</td>
      <td>전라남도</td>
      <td>10513</td>
      <td>16755</td>
      <td>20157</td>
      <td>22160</td>
      <td>21314</td>
      <td>46610</td>
      <td>46251</td>
      <td>43430</td>
      <td>...</td>
      <td>16601</td>
      <td>17468</td>
      <td>16429</td>
      <td>15974</td>
      <td>14765</td>
      <td>14187</td>
      <td>14591</td>
      <td>14598</td>
      <td>13065</td>
      <td>12426</td>
    </tr>
    <tr>
      <th>34</th>
      <td>서울특별시</td>
      <td>경상북도</td>
      <td>11868</td>
      <td>16459</td>
      <td>22073</td>
      <td>27531</td>
      <td>26902</td>
      <td>46177</td>
      <td>40376</td>
      <td>41155</td>
      <td>...</td>
      <td>15425</td>
      <td>16569</td>
      <td>16042</td>
      <td>15818</td>
      <td>15191</td>
      <td>14420</td>
      <td>14456</td>
      <td>15113</td>
      <td>14236</td>
      <td>12464</td>
    </tr>
    <tr>
      <th>35</th>
      <td>서울특별시</td>
      <td>경상남도</td>
      <td>8409</td>
      <td>10001</td>
      <td>11263</td>
      <td>15193</td>
      <td>16771</td>
      <td>23150</td>
      <td>22400</td>
      <td>27393</td>
      <td>...</td>
      <td>15438</td>
      <td>15303</td>
      <td>15689</td>
      <td>16039</td>
      <td>14474</td>
      <td>14447</td>
      <td>14799</td>
      <td>15220</td>
      <td>13717</td>
      <td>12692</td>
    </tr>
    <tr>
      <th>36</th>
      <td>서울특별시</td>
      <td>제주특별자치도</td>
      <td>1039</td>
      <td>1325</td>
      <td>1617</td>
      <td>2456</td>
      <td>2261</td>
      <td>3440</td>
      <td>3623</td>
      <td>3551</td>
      <td>...</td>
      <td>5473</td>
      <td>5332</td>
      <td>5714</td>
      <td>6133</td>
      <td>6954</td>
      <td>7828</td>
      <td>9031</td>
      <td>10434</td>
      <td>10465</td>
      <td>10404</td>
    </tr>
  </tbody>
</table>
<p>17 rows × 50 columns</p>
</div>



```python
# '전출지별' column 삭제
df_seoul.drop('전출지별', axis=1, inplace=True)

# '전입지별' -> '전입지'로 column명 변경
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
display(df_seoul)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>전입지</th>
      <th>1970</th>
      <th>1971</th>
      <th>1972</th>
      <th>1973</th>
      <th>1974</th>
      <th>1975</th>
      <th>1976</th>
      <th>1977</th>
      <th>1978</th>
      <th>...</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>전국</td>
      <td>1448985</td>
      <td>1419016</td>
      <td>1210559</td>
      <td>1647268</td>
      <td>1819660</td>
      <td>2937093</td>
      <td>2495620</td>
      <td>2678007</td>
      <td>3028911</td>
      <td>...</td>
      <td>2083352</td>
      <td>1925452</td>
      <td>1848038</td>
      <td>1834806</td>
      <td>1658928</td>
      <td>1620640</td>
      <td>1661425</td>
      <td>1726687</td>
      <td>1655859</td>
      <td>1571423</td>
    </tr>
    <tr>
      <th>21</th>
      <td>부산광역시</td>
      <td>11568</td>
      <td>11130</td>
      <td>11768</td>
      <td>16307</td>
      <td>22220</td>
      <td>27515</td>
      <td>23732</td>
      <td>27213</td>
      <td>29856</td>
      <td>...</td>
      <td>17353</td>
      <td>17738</td>
      <td>17418</td>
      <td>18816</td>
      <td>16135</td>
      <td>16153</td>
      <td>17320</td>
      <td>17009</td>
      <td>15062</td>
      <td>14484</td>
    </tr>
    <tr>
      <th>22</th>
      <td>대구광역시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>9720</td>
      <td>10464</td>
      <td>10277</td>
      <td>10397</td>
      <td>10135</td>
      <td>10631</td>
      <td>10062</td>
      <td>10191</td>
      <td>9623</td>
      <td>8891</td>
    </tr>
    <tr>
      <th>23</th>
      <td>인천광역시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>50493</td>
      <td>45392</td>
      <td>46082</td>
      <td>51641</td>
      <td>49640</td>
      <td>47424</td>
      <td>43212</td>
      <td>44915</td>
      <td>43745</td>
      <td>40485</td>
    </tr>
    <tr>
      <th>24</th>
      <td>광주광역시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>10846</td>
      <td>11725</td>
      <td>11095</td>
      <td>10587</td>
      <td>10154</td>
      <td>9129</td>
      <td>9759</td>
      <td>9216</td>
      <td>8354</td>
      <td>7932</td>
    </tr>
    <tr>
      <th>25</th>
      <td>대전광역시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>13515</td>
      <td>13632</td>
      <td>13819</td>
      <td>13900</td>
      <td>14080</td>
      <td>13440</td>
      <td>13403</td>
      <td>13453</td>
      <td>12619</td>
      <td>11815</td>
    </tr>
    <tr>
      <th>26</th>
      <td>울산광역시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>5057</td>
      <td>4845</td>
      <td>4742</td>
      <td>5188</td>
      <td>5691</td>
      <td>5542</td>
      <td>6047</td>
      <td>5950</td>
      <td>5102</td>
      <td>4260</td>
    </tr>
    <tr>
      <th>27</th>
      <td>세종특별자치시</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>2998</td>
      <td>2851</td>
      <td>6481</td>
      <td>7550</td>
      <td>5943</td>
      <td>5813</td>
    </tr>
    <tr>
      <th>28</th>
      <td>경기도</td>
      <td>130149</td>
      <td>150313</td>
      <td>93333</td>
      <td>143234</td>
      <td>149045</td>
      <td>253705</td>
      <td>202276</td>
      <td>207722</td>
      <td>237684</td>
      <td>...</td>
      <td>412408</td>
      <td>398282</td>
      <td>410735</td>
      <td>373771</td>
      <td>354135</td>
      <td>340801</td>
      <td>332785</td>
      <td>359337</td>
      <td>370760</td>
      <td>342433</td>
    </tr>
    <tr>
      <th>29</th>
      <td>강원도</td>
      <td>9352</td>
      <td>12885</td>
      <td>13561</td>
      <td>16481</td>
      <td>15479</td>
      <td>27837</td>
      <td>25927</td>
      <td>25415</td>
      <td>26700</td>
      <td>...</td>
      <td>23668</td>
      <td>23331</td>
      <td>22736</td>
      <td>23624</td>
      <td>22332</td>
      <td>20601</td>
      <td>21173</td>
      <td>22659</td>
      <td>21590</td>
      <td>21016</td>
    </tr>
    <tr>
      <th>30</th>
      <td>충청북도</td>
      <td>6700</td>
      <td>9457</td>
      <td>10853</td>
      <td>12617</td>
      <td>11786</td>
      <td>21073</td>
      <td>18029</td>
      <td>17478</td>
      <td>18420</td>
      <td>...</td>
      <td>15294</td>
      <td>15295</td>
      <td>15461</td>
      <td>15318</td>
      <td>14555</td>
      <td>13783</td>
      <td>14244</td>
      <td>14379</td>
      <td>14087</td>
      <td>13302</td>
    </tr>
    <tr>
      <th>31</th>
      <td>충청남도</td>
      <td>15954</td>
      <td>18943</td>
      <td>23406</td>
      <td>27139</td>
      <td>25509</td>
      <td>51205</td>
      <td>41447</td>
      <td>43993</td>
      <td>48091</td>
      <td>...</td>
      <td>27458</td>
      <td>24889</td>
      <td>24522</td>
      <td>24723</td>
      <td>22269</td>
      <td>21486</td>
      <td>21473</td>
      <td>22299</td>
      <td>21741</td>
      <td>21020</td>
    </tr>
    <tr>
      <th>32</th>
      <td>전라북도</td>
      <td>10814</td>
      <td>13192</td>
      <td>16583</td>
      <td>18642</td>
      <td>16647</td>
      <td>34411</td>
      <td>29835</td>
      <td>28444</td>
      <td>29676</td>
      <td>...</td>
      <td>18390</td>
      <td>18332</td>
      <td>17569</td>
      <td>17755</td>
      <td>16120</td>
      <td>14909</td>
      <td>14566</td>
      <td>14835</td>
      <td>13835</td>
      <td>13179</td>
    </tr>
    <tr>
      <th>33</th>
      <td>전라남도</td>
      <td>10513</td>
      <td>16755</td>
      <td>20157</td>
      <td>22160</td>
      <td>21314</td>
      <td>46610</td>
      <td>46251</td>
      <td>43430</td>
      <td>44624</td>
      <td>...</td>
      <td>16601</td>
      <td>17468</td>
      <td>16429</td>
      <td>15974</td>
      <td>14765</td>
      <td>14187</td>
      <td>14591</td>
      <td>14598</td>
      <td>13065</td>
      <td>12426</td>
    </tr>
    <tr>
      <th>34</th>
      <td>경상북도</td>
      <td>11868</td>
      <td>16459</td>
      <td>22073</td>
      <td>27531</td>
      <td>26902</td>
      <td>46177</td>
      <td>40376</td>
      <td>41155</td>
      <td>42940</td>
      <td>...</td>
      <td>15425</td>
      <td>16569</td>
      <td>16042</td>
      <td>15818</td>
      <td>15191</td>
      <td>14420</td>
      <td>14456</td>
      <td>15113</td>
      <td>14236</td>
      <td>12464</td>
    </tr>
    <tr>
      <th>35</th>
      <td>경상남도</td>
      <td>8409</td>
      <td>10001</td>
      <td>11263</td>
      <td>15193</td>
      <td>16771</td>
      <td>23150</td>
      <td>22400</td>
      <td>27393</td>
      <td>28697</td>
      <td>...</td>
      <td>15438</td>
      <td>15303</td>
      <td>15689</td>
      <td>16039</td>
      <td>14474</td>
      <td>14447</td>
      <td>14799</td>
      <td>15220</td>
      <td>13717</td>
      <td>12692</td>
    </tr>
    <tr>
      <th>36</th>
      <td>제주특별자치도</td>
      <td>1039</td>
      <td>1325</td>
      <td>1617</td>
      <td>2456</td>
      <td>2261</td>
      <td>3440</td>
      <td>3623</td>
      <td>3551</td>
      <td>3937</td>
      <td>...</td>
      <td>5473</td>
      <td>5332</td>
      <td>5714</td>
      <td>6133</td>
      <td>6954</td>
      <td>7828</td>
      <td>9031</td>
      <td>10434</td>
      <td>10465</td>
      <td>10404</td>
    </tr>
  </tbody>
</table>
<p>17 rows × 49 columns</p>
</div>



```python
# '전입지' column을 행 index로 설정
df_seoul.set_index('전입지', inplace=True)
display(df_seoul)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1970</th>
      <th>1971</th>
      <th>1972</th>
      <th>1973</th>
      <th>1974</th>
      <th>1975</th>
      <th>1976</th>
      <th>1977</th>
      <th>1978</th>
      <th>1979</th>
      <th>...</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
    <tr>
      <th>전입지</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>전국</th>
      <td>1448985</td>
      <td>1419016</td>
      <td>1210559</td>
      <td>1647268</td>
      <td>1819660</td>
      <td>2937093</td>
      <td>2495620</td>
      <td>2678007</td>
      <td>3028911</td>
      <td>2441242</td>
      <td>...</td>
      <td>2083352</td>
      <td>1925452</td>
      <td>1848038</td>
      <td>1834806</td>
      <td>1658928</td>
      <td>1620640</td>
      <td>1661425</td>
      <td>1726687</td>
      <td>1655859</td>
      <td>1571423</td>
    </tr>
    <tr>
      <th>부산광역시</th>
      <td>11568</td>
      <td>11130</td>
      <td>11768</td>
      <td>16307</td>
      <td>22220</td>
      <td>27515</td>
      <td>23732</td>
      <td>27213</td>
      <td>29856</td>
      <td>28542</td>
      <td>...</td>
      <td>17353</td>
      <td>17738</td>
      <td>17418</td>
      <td>18816</td>
      <td>16135</td>
      <td>16153</td>
      <td>17320</td>
      <td>17009</td>
      <td>15062</td>
      <td>14484</td>
    </tr>
    <tr>
      <th>대구광역시</th>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>9720</td>
      <td>10464</td>
      <td>10277</td>
      <td>10397</td>
      <td>10135</td>
      <td>10631</td>
      <td>10062</td>
      <td>10191</td>
      <td>9623</td>
      <td>8891</td>
    </tr>
    <tr>
      <th>인천광역시</th>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>50493</td>
      <td>45392</td>
      <td>46082</td>
      <td>51641</td>
      <td>49640</td>
      <td>47424</td>
      <td>43212</td>
      <td>44915</td>
      <td>43745</td>
      <td>40485</td>
    </tr>
    <tr>
      <th>광주광역시</th>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>10846</td>
      <td>11725</td>
      <td>11095</td>
      <td>10587</td>
      <td>10154</td>
      <td>9129</td>
      <td>9759</td>
      <td>9216</td>
      <td>8354</td>
      <td>7932</td>
    </tr>
    <tr>
      <th>대전광역시</th>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>13515</td>
      <td>13632</td>
      <td>13819</td>
      <td>13900</td>
      <td>14080</td>
      <td>13440</td>
      <td>13403</td>
      <td>13453</td>
      <td>12619</td>
      <td>11815</td>
    </tr>
    <tr>
      <th>울산광역시</th>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>5057</td>
      <td>4845</td>
      <td>4742</td>
      <td>5188</td>
      <td>5691</td>
      <td>5542</td>
      <td>6047</td>
      <td>5950</td>
      <td>5102</td>
      <td>4260</td>
    </tr>
    <tr>
      <th>세종특별자치시</th>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>2998</td>
      <td>2851</td>
      <td>6481</td>
      <td>7550</td>
      <td>5943</td>
      <td>5813</td>
    </tr>
    <tr>
      <th>경기도</th>
      <td>130149</td>
      <td>150313</td>
      <td>93333</td>
      <td>143234</td>
      <td>149045</td>
      <td>253705</td>
      <td>202276</td>
      <td>207722</td>
      <td>237684</td>
      <td>278411</td>
      <td>...</td>
      <td>412408</td>
      <td>398282</td>
      <td>410735</td>
      <td>373771</td>
      <td>354135</td>
      <td>340801</td>
      <td>332785</td>
      <td>359337</td>
      <td>370760</td>
      <td>342433</td>
    </tr>
    <tr>
      <th>강원도</th>
      <td>9352</td>
      <td>12885</td>
      <td>13561</td>
      <td>16481</td>
      <td>15479</td>
      <td>27837</td>
      <td>25927</td>
      <td>25415</td>
      <td>26700</td>
      <td>27599</td>
      <td>...</td>
      <td>23668</td>
      <td>23331</td>
      <td>22736</td>
      <td>23624</td>
      <td>22332</td>
      <td>20601</td>
      <td>21173</td>
      <td>22659</td>
      <td>21590</td>
      <td>21016</td>
    </tr>
    <tr>
      <th>충청북도</th>
      <td>6700</td>
      <td>9457</td>
      <td>10853</td>
      <td>12617</td>
      <td>11786</td>
      <td>21073</td>
      <td>18029</td>
      <td>17478</td>
      <td>18420</td>
      <td>20047</td>
      <td>...</td>
      <td>15294</td>
      <td>15295</td>
      <td>15461</td>
      <td>15318</td>
      <td>14555</td>
      <td>13783</td>
      <td>14244</td>
      <td>14379</td>
      <td>14087</td>
      <td>13302</td>
    </tr>
    <tr>
      <th>충청남도</th>
      <td>15954</td>
      <td>18943</td>
      <td>23406</td>
      <td>27139</td>
      <td>25509</td>
      <td>51205</td>
      <td>41447</td>
      <td>43993</td>
      <td>48091</td>
      <td>45388</td>
      <td>...</td>
      <td>27458</td>
      <td>24889</td>
      <td>24522</td>
      <td>24723</td>
      <td>22269</td>
      <td>21486</td>
      <td>21473</td>
      <td>22299</td>
      <td>21741</td>
      <td>21020</td>
    </tr>
    <tr>
      <th>전라북도</th>
      <td>10814</td>
      <td>13192</td>
      <td>16583</td>
      <td>18642</td>
      <td>16647</td>
      <td>34411</td>
      <td>29835</td>
      <td>28444</td>
      <td>29676</td>
      <td>31570</td>
      <td>...</td>
      <td>18390</td>
      <td>18332</td>
      <td>17569</td>
      <td>17755</td>
      <td>16120</td>
      <td>14909</td>
      <td>14566</td>
      <td>14835</td>
      <td>13835</td>
      <td>13179</td>
    </tr>
    <tr>
      <th>전라남도</th>
      <td>10513</td>
      <td>16755</td>
      <td>20157</td>
      <td>22160</td>
      <td>21314</td>
      <td>46610</td>
      <td>46251</td>
      <td>43430</td>
      <td>44624</td>
      <td>47934</td>
      <td>...</td>
      <td>16601</td>
      <td>17468</td>
      <td>16429</td>
      <td>15974</td>
      <td>14765</td>
      <td>14187</td>
      <td>14591</td>
      <td>14598</td>
      <td>13065</td>
      <td>12426</td>
    </tr>
    <tr>
      <th>경상북도</th>
      <td>11868</td>
      <td>16459</td>
      <td>22073</td>
      <td>27531</td>
      <td>26902</td>
      <td>46177</td>
      <td>40376</td>
      <td>41155</td>
      <td>42940</td>
      <td>43565</td>
      <td>...</td>
      <td>15425</td>
      <td>16569</td>
      <td>16042</td>
      <td>15818</td>
      <td>15191</td>
      <td>14420</td>
      <td>14456</td>
      <td>15113</td>
      <td>14236</td>
      <td>12464</td>
    </tr>
    <tr>
      <th>경상남도</th>
      <td>8409</td>
      <td>10001</td>
      <td>11263</td>
      <td>15193</td>
      <td>16771</td>
      <td>23150</td>
      <td>22400</td>
      <td>27393</td>
      <td>28697</td>
      <td>30183</td>
      <td>...</td>
      <td>15438</td>
      <td>15303</td>
      <td>15689</td>
      <td>16039</td>
      <td>14474</td>
      <td>14447</td>
      <td>14799</td>
      <td>15220</td>
      <td>13717</td>
      <td>12692</td>
    </tr>
    <tr>
      <th>제주특별자치도</th>
      <td>1039</td>
      <td>1325</td>
      <td>1617</td>
      <td>2456</td>
      <td>2261</td>
      <td>3440</td>
      <td>3623</td>
      <td>3551</td>
      <td>3937</td>
      <td>4261</td>
      <td>...</td>
      <td>5473</td>
      <td>5332</td>
      <td>5714</td>
      <td>6133</td>
      <td>6954</td>
      <td>7828</td>
      <td>9031</td>
      <td>10434</td>
      <td>10465</td>
      <td>10404</td>
    </tr>
  </tbody>
</table>
<p>17 rows × 48 columns</p>
</div>



```python
# 서울특별시에서 경기도로 전입한 데이터만 가져오기
s = df_seoul.loc['경기도']

print(s)
```

    1970    130149
    1971    150313
    1972     93333
    1973    143234
    1974    149045
    1975    253705
    1976    202276
    1977    207722
    1978    237684
    1979    278411
    1980    297539
    1981    252073
    1982    320174
    1983    400875
    1984    352238
    1985    390265
    1986    412535
    1987    405220
    1988    415174
    1989    412933
    1990    473889
    1991    384714
    1992    428344
    1993    502584
    1994    542204
    1995    599411
    1996    520566
    1997    495454
    1998    407050
    1999    471841
    2000    435573
    2001    499575
    2002    516765
    2003    457656
    2004    400206
    2005    414621
    2006    449632
    2007    431637
    2008    412408
    2009    398282
    2010    410735
    2011    373771
    2012    354135
    2013    340801
    2014    332785
    2015    359337
    2016    370760
    2017    342433
    Name: 경기도, dtype: object
    


```python
plt.plot(s.index, s.values)

plt.title('서울에서 경기도로 전입한 사람 추이') # 한글이 깨짐

plt.show()
```


    
![png](/Machine-Learning/images/0322/output_12_0.png)
    


#### 한글이 깨지는 현상 해결하기
- 필요한 module import


```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings(action='ignore')  # 경고메시지 출력되지 않게 설정

# 그래프에서 '-' 기호때문에 문제가 발생할 여지 방지
mpl.rcParams['axes.unicode_minus'] = False

# 한글폰트 사용
font_path = './font/malgun.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()  # Malgun Gothic
rc('font', family=font_name)

# 이전 셀에서 만들어 둔 Series(연도별 경기도로 전입한 인구수)를 이용해서 line plot 그리기
plt.plot(s.index, s.values)

plt.title('서울에서 경기도로 전입한 인구 추이')

plt.xlabel('연도')
plt.ylabel('이동 인구수')

plt.show()
```


    
![png](/Machine-Learning/images/0322/output_14_0.png)
    


#### style 적용하기


```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings(action='ignore')  # 경고메시지 출력되지 않게 설정

# 그래프에서 '-' 기호때문에 문제가 발생할 여지 방지
mpl.rcParams['axes.unicode_minus'] = False

# 한글폰트 사용
font_path = './font/malgun.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()  # Malgun Gothic
rc('font', family=font_name)

# 그래프에 스타일 적용
plt.style.use('ggplot')

# 그래프의 크기 설정
plt.figure(figsize=(10,5))  # 단위는 인치. 가로, 세로순으로 크기를 설정

# x의 눈금이 읽기 어려우므로 label 회전
plt.xticks(rotation='vertical')

plt.plot(s.index, s.values,
         marker='o',    # o는 circle marker, s는 square marker
         markersize=8,
         markerfacecolor='r',
         color='g',
         linewidth=2)   

plt.title('서울에서 경기도로 전입한 인구 추이')
plt.xlabel('연도')
plt.ylabel('이동 인구수')

# 범례
plt.legend(labels=['서울 -> 경기'], loc='best')

plt.show()
```


    
![png](/Machine-Learning/images/0322/output_16_0.png)
    


## 2. sub_plot
- 그래프 여러 개를 한 번에 그리는 방법

### Figure, Axes, Axis
- Figure는 canvas를 나타내는 객체 (도화지 역할)
- jupyter notebook은 plot()을 사용하면 자동으로 Figure 객체를 하나 만들어 줌
- canvas(도화지)의 크기를 임의로 설정하려면 Figure 객체를 만들어야 함
- Figure 객체는 1개 이상의 Axes로 구성됨
- plot() 함수를 이용하면 자동으로 Axes도 하나 생성됨


```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings(action='ignore')  # 경고메시지 출력되지 않게 설정

# 그래프에서 '-' 기호때문에 문제가 발생할 여지 방지
mpl.rcParams['axes.unicode_minus'] = False

# 한글폰트 사용
font_path = './font/malgun.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()  # Malgun Gothic
rc('font', family=font_name)

# 그래프에 스타일 적용
plt.style.use('ggplot')

# 그래프의 크기 설정
fig = plt.figure(figsize=(10,10))  # 단위는 인치. 가로, 세로순으로 크기를 설정
ax1 = fig.add_subplot(1,2,1)  # 1행 2열의 첫번째 axes
ax2 = fig.add_subplot(1,2,2)  # 1행 2열의 두번째 axes

ax1.set_title('서울 -> 경기 인구 이동')
ax1.set_xlabel('연도')
ax1.set_ylabel('이동 인구수')

ax1.plot(s.index, s.values)

ax2.plot(s.index, s.values,
         marker='o',    # o는 circle marker, s는 square marker
         markersize=8,
         markerfacecolor='r',
         color='g',
         linewidth=2)   

plt.show()
```


    
![png](/Machine-Learning/images/0322/output_19_0.png)
    


## 3. Histogram
- 변수가 하나인 단변수 데이터의 빈도수를 그래프로 표현한 것
- x축 : 같은 크기의 여러 구간(bin)으로 나눔 
- y축 : 해당 구간 안에 포함된 데이터의 개수(빈도, mode) 표현
- x축의 구간(bin)을 조절하면 historgram의 모양은 달라지게 됨

### MPG data set 이용
- mpg : 연비(mile per gallon)
- cylinders : 실린더 개수
- displacement : 배기량
- horsepower : 마력(출력)
- weight : 중량
- acceleration : 가속능력
- year : 출시년도(70 => 1970년도)
- origin : 제조국 (1: USA, 2:EU, 3:JPN)
- name : 차량 이름


```python
import numpy as np
import pandas as pd

df = pd.read_csv('./data/auto-mpg.csv', header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower',
              'weight', 'acceleration', 'year', 'origin', 'name']

display(df.head(3))
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
  </tbody>
</table>
</div>



```python
# mpg column의 최대값과 최소값
print(df['mpg'].max(), df['mpg'].min())  
```

    46.6 9.0
    


```python
# matplotlib이 아닌 pandas의 기능을 이용해서 historgram 그리기
df['mpg'].plot(kind='hist',
               bins=10,       # 최소값과 최대값 사이를 10개의 구간으로 분할
               color='blue',
               figsize=(10,5))

plt.show()
```


    
![png](/Machine-Learning/images/0322/output_24_0.png)
    


## 4. 산점도(scatter)
- 서로 다른 두 변수 사이의 관계 표현
- 2개의 변수의 값을 각각 x축과 y에 하나씩 놓고 데이터 값이 위치하는 (x,y)좌표를 찾아 점으로 표현


```python
import numpy as np
import pandas as pd

df = pd.read_csv('./data/auto-mpg.csv', header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower',
              'weight', 'acceleration', 'year', 'origin', 'name']

display(df.head(3))
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
  </tbody>
</table>
</div>


### mpg(연비)와 weight(중량) 이용해서 scatter 그리기


```python
df.plot(kind='scatter',
        x='weight',
        y='mpg',
        color='red',
        s=10,
        figsize=(10,5))

plt.show()
```


    
![png](/Machine-Learning/images/0322/output_28_0.png)
    


## 5. Box plot
- 단변수의 데이터 분포를 살펴보기 위해 사용
- 이상치를 눈으로 쉽게 파악할 수 있음 => 동그라미가 이상치


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/auto-mpg.csv', header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower',
              'weight', 'acceleration', 'year', 'origin', 'name']

display(df.head(3))
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
  </tbody>
</table>
</div>


### 제조국별 연비분포를 boxplot으로 그리기
- 제조국(origin) => 1 : USA, 2: EU, 3: JPN


```python
plt.boxplot(x=[df.loc[df['origin']==1,'mpg'],
               df.loc[df['origin']==2,'mpg'],
               df.loc[df['origin']==3,'mpg']])
plt.show()
```


    
![png](/Machine-Learning/images/0322/output_32_0.png)
    

