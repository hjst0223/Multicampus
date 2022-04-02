외부 resource를 이용해서 DataFrame을 생성할 수 있다.
- CSV, MySQL Database, Open API, JSON

## CSV 파일을 이용해서 DataFrame 생성 😮


```python
import numpy as np
import pandas as pd

df = pd.read_csv('./data/student.csv')

display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>입학연도</th>
      <th>성적</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>아이유</td>
      <td>2015</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>김연아</td>
      <td>2016</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>홍길동</td>
      <td>2015</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>강감찬</td>
      <td>2017</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>이순신</td>
      <td>2016</td>
      <td>2.7</td>
    </tr>
  </tbody>
</table>
</div>


## MySQL안에 Database로 부터 SQL을 이용해 DataFrame 생성 😐
- 프로그램적으로 SQL을 직접 이용
- ORM방식을 이용해서 사용(Django)
- 외부 모듈 필요 => pymysql module (conda install pymysql)
- MySQL 안에 데이터베이스 생성 후 제공된 Script 파일을 이용해서 Table 생성


```python
import pymysql
import numpy as np
import pandas as pd

# Database연결
# 연결을 시도해보고 만약 성공하면 연결객체를 얻음
con = pymysql.connect(host='localhost',
                      user='root',
                      password='비밀번호',
                      db='lecture_0317',
                      charset='utf8')

# 연결을 성공하면 SQL 작성하기
sql = "SELECT bisbn, btitle, bauthor, bprice FROM book WHERE btitle LIKE '%java%'"

df = pd.read_sql(sql, con)
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bisbn</th>
      <th>btitle</th>
      <th>bauthor</th>
      <th>bprice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>89-7914-371-0</td>
      <td>Head First Java: 뇌 회로를 자극하는 자바 학습법(개정판)</td>
      <td>케이시 시에라,버트 베이츠</td>
      <td>28000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>89-7914-397-4</td>
      <td>뇌를 자극하는 Java 프로그래밍</td>
      <td>김윤명</td>
      <td>27000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>978-89-6848-042-3</td>
      <td>모던 웹을 위한 JavaScript + jQuery 입문(개정판) : 자바스크립트에...</td>
      <td>윤인성</td>
      <td>32000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>978-89-6848-132-1</td>
      <td>JavaScript+jQuery 정복 : 보고, 이해하고, 바로 쓰는 자바스크립트 공략집</td>
      <td>김상형</td>
      <td>28000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>978-89-6848-147-5</td>
      <td>이것이 자바다 : 신용권의 Java 프로그래밍 정복</td>
      <td>신용권</td>
      <td>30000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>978-89-6848-156-7</td>
      <td>Head First JavaScript Programming : 게임과 퍼즐로 배우...</td>
      <td>에릭 프리먼, 엘리자베스 롭슨</td>
      <td>36000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>978-89-7914-582-3</td>
      <td>Head First JavaScript : 대화형 웹 애플리케이션의 시작</td>
      <td>마이클 모리슨</td>
      <td>28000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>978-89-7914-659-2</td>
      <td>UML과 JAVA로 배우는 객체지향 CBD 실전 프로젝트 : 도서 관리 시스템</td>
      <td>채흥석</td>
      <td>40000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>978-89-7914-832-9</td>
      <td>IT CookBook, 웹 프로그래밍 입문 : XHTML, CSS2, JavaScript</td>
      <td>김형철, 안치현</td>
      <td>23000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>978-89-7914-855-8</td>
      <td>자바스크립트 성능 최적화: High Performance JavaScript</td>
      <td>니콜라스 자카스</td>
      <td>20000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>978-89-98756-79-6</td>
      <td>IT CookBook, Java for Beginner</td>
      <td>황희정, 강운구</td>
      <td>23000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>978-89-98756-89-5</td>
      <td>IT CookBook, 인터넷 프로그래밍 입문: HTML, CSS, JavaScript</td>
      <td>주성례(저자), 정선호(저자), 한민형(저자), 권원상</td>
      <td>22000</td>
    </tr>
  </tbody>
</table>
</div>


### DataFrame을 얻으면 그 DataFrame의 내용을 JSON으로 저장할 수 있다.
- 4가지 형태의 JSON으로 저장 가능
- with 구문 이용 (파일 open -> 파일 안에 내용 저장 -> 파일 close)
- 폴더는 사전에 만들어 놓아야 함


```python
with open('./data/json/books.json', 'w', encoding='utf-8') as file:
    df.to_json(file, force_ascii=False)
```

## JSON을 DataFrame으로 변형하기 🙄


```python
import numpy as np
import pandas as pd
import json    # 내장 module => 따로 설치하지 않아도 사용 가능

with open('./data/json/books.json', 'r', encoding='utf-8') as file:
    dict_books = json.load(file)
    
print(dict_books)    
```

    {'bisbn': {'0': '89-7914-371-0', '1': '89-7914-397-4', '2': '978-89-6848-042-3', '3': '978-89-6848-132-1', '4': '978-89-6848-147-5', '5': '978-89-6848-156-7', '6': '978-89-7914-582-3', '7': '978-89-7914-659-2', '8': '978-89-7914-832-9', '9': '978-89-7914-855-8', '10': '978-89-98756-79-6', '11': '978-89-98756-89-5'}, 'btitle': {'0': 'Head First Java: 뇌 회로를 자극하는 자바 학습법(개정판)', '1': '뇌를 자극하는 Java 프로그래밍', '2': '모던 웹을 위한 JavaScript + jQuery 입문(개정판) : 자바스크립트에서 제이쿼리, 제이쿼리 모바일까지 한 권으로 끝낸다', '3': 'JavaScript+jQuery 정복 : 보고, 이해하고, 바로 쓰는 자바스크립트 공략집', '4': '이것이 자바다 : 신용권의 Java 프로그래밍 정복', '5': 'Head First JavaScript Programming : 게임과 퍼즐로 배우는 자바스크립트 입문서', '6': 'Head First JavaScript : 대화형 웹 애플리케이션의 시작', '7': 'UML과 JAVA로 배우는 객체지향 CBD 실전 프로젝트 : 도서 관리 시스템', '8': 'IT CookBook, 웹 프로그래밍 입문 : XHTML, CSS2, JavaScript', '9': '자바스크립트 성능 최적화: High Performance JavaScript', '10': 'IT CookBook, Java for Beginner', '11': 'IT CookBook, 인터넷 프로그래밍 입문: HTML, CSS, JavaScript'}, 'bauthor': {'0': '케이시 시에라,버트 베이츠', '1': '김윤명', '2': '윤인성', '3': '김상형', '4': '신용권', '5': '에릭 프리먼, 엘리자베스 롭슨', '6': '마이클 모리슨', '7': '채흥석', '8': '김형철, 안치현', '9': '니콜라스 자카스', '10': '황희정, 강운구', '11': '주성례(저자), 정선호(저자), 한민형(저자), 권원상'}, 'bprice': {'0': 28000, '1': 27000, '2': 32000, '3': 28000, '4': 30000, '5': 36000, '6': 28000, '7': 40000, '8': 23000, '9': 20000, '10': 23000, '11': 22000}}
    


```python
df = pd.DataFrame.from_dict(dict_books)
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bisbn</th>
      <th>btitle</th>
      <th>bauthor</th>
      <th>bprice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>89-7914-371-0</td>
      <td>Head First Java: 뇌 회로를 자극하는 자바 학습법(개정판)</td>
      <td>케이시 시에라,버트 베이츠</td>
      <td>28000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>89-7914-397-4</td>
      <td>뇌를 자극하는 Java 프로그래밍</td>
      <td>김윤명</td>
      <td>27000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>978-89-6848-042-3</td>
      <td>모던 웹을 위한 JavaScript + jQuery 입문(개정판) : 자바스크립트에...</td>
      <td>윤인성</td>
      <td>32000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>978-89-6848-132-1</td>
      <td>JavaScript+jQuery 정복 : 보고, 이해하고, 바로 쓰는 자바스크립트 공략집</td>
      <td>김상형</td>
      <td>28000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>978-89-6848-147-5</td>
      <td>이것이 자바다 : 신용권의 Java 프로그래밍 정복</td>
      <td>신용권</td>
      <td>30000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>978-89-6848-156-7</td>
      <td>Head First JavaScript Programming : 게임과 퍼즐로 배우...</td>
      <td>에릭 프리먼, 엘리자베스 롭슨</td>
      <td>36000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>978-89-7914-582-3</td>
      <td>Head First JavaScript : 대화형 웹 애플리케이션의 시작</td>
      <td>마이클 모리슨</td>
      <td>28000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>978-89-7914-659-2</td>
      <td>UML과 JAVA로 배우는 객체지향 CBD 실전 프로젝트 : 도서 관리 시스템</td>
      <td>채흥석</td>
      <td>40000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>978-89-7914-832-9</td>
      <td>IT CookBook, 웹 프로그래밍 입문 : XHTML, CSS2, JavaScript</td>
      <td>김형철, 안치현</td>
      <td>23000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>978-89-7914-855-8</td>
      <td>자바스크립트 성능 최적화: High Performance JavaScript</td>
      <td>니콜라스 자카스</td>
      <td>20000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>978-89-98756-79-6</td>
      <td>IT CookBook, Java for Beginner</td>
      <td>황희정, 강운구</td>
      <td>23000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>978-89-98756-89-5</td>
      <td>IT CookBook, 인터넷 프로그래밍 입문: HTML, CSS, JavaScript</td>
      <td>주성례(저자), 정선호(저자), 한민형(저자), 권원상</td>
      <td>22000</td>
    </tr>
  </tbody>
</table>
</div>


## Open API를 이용해서 DataFrame 생성하기 👻
- 영화진흥위원회 OPEN API


```python
import numpy as np
import pandas as pd
import urllib      # open api 호출하기 위해서 필요한 module
import json

# 영화진흥위원회 OPEN API 호출에 대한 URL을 Query String을 이용해서 작성
url = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'
key = '발급받은 key'
targetDt = '20220301'
query_string = '?key=' + key + '&targetDt=' + targetDt
movie_url = url + query_string

load_page = urllib.request.urlopen(movie_url)
print(load_page)  # request에 대한 response 객체(이 안에 결과 JSON이 들어 있음)

# 결과 JSON 얻기
my_dict = json.loads(load_page.read())
print(my_dict)
```

    <http.client.HTTPResponse object at 0x000001374E24A880>
    {'boxOfficeResult': {'boxofficeType': '일별 박스오피스', 'showRange': '20220301~20220301', 'dailyBoxOfficeList': [{'rnum': '1', 'rank': '1', 'rankInten': '3', 'rankOldAndNew': 'OLD', 'movieCd': '20212973', 'movieNm': '더 배트맨', 'openDt': '2022-03-01', 'salesAmt': '1946514750', 'salesShare': '74.1', 'salesInten': '1873788030', 'salesChange': '2576.5', 'salesAcc': '2032351470', 'audiCnt': '192841', 'audiInten': '187395', 'audiChange': '3441', 'audiAcc': '199161', 'scrnCnt': '2370', 'showCnt': '7544'}, {'rnum': '2', 'rank': '2', 'rankInten': '-1', 'rankOldAndNew': 'OLD', 'movieCd': '20211200', 'movieNm': '언차티드', 'openDt': '2022-02-16', 'salesAmt': '272829540', 'salesShare': '10.4', 'salesInten': '26524900', 'salesChange': '10.8', 'salesAcc': '6269626510', 'audiCnt': '27212', 'audiInten': '331', 'audiChange': '1.2', 'audiAcc': '642054', 'scrnCnt': '800', 'showCnt': '1627'}, {'rnum': '3', 'rank': '3', 'rankInten': '-1', 'rankOldAndNew': 'OLD', 'movieCd': '20223278', 'movieNm': '극장판 주술회전 0', 'openDt': '2022-02-17', 'salesAmt': '139050330', 'salesShare': '5.3', 'salesInten': '-8409390', 'salesChange': '-5.7', 'salesAcc': '3205328130', 'audiCnt': '13868', 'audiInten': '-1446', 'audiChange': '-9.4', 'audiAcc': '320693', 'scrnCnt': '602', 'showCnt': '1113'}, {'rnum': '4', 'rank': '4', 'rankInten': '-1', 'rankOldAndNew': 'OLD', 'movieCd': '20212741', 'movieNm': '안테벨룸', 'openDt': '2022-02-23', 'salesAmt': '52002320', 'salesShare': '2.0', 'salesInten': '-12111280', 'salesChange': '-18.9', 'salesAcc': '645753360', 'audiCnt': '5084', 'audiInten': '-1839', 'audiChange': '-26.6', 'audiAcc': '68040', 'scrnCnt': '454', 'showCnt': '580'}, {'rnum': '5', 'rank': '5', 'rankInten': '0', 'rankOldAndNew': 'OLD', 'movieCd': '20208006', 'movieNm': '인민을 위해 복무하라', 'openDt': '2022-02-23', 'salesAmt': '37808320', 'salesShare': '1.4', 'salesInten': '-9779840', 'salesChange': '-20.6', 'salesAcc': '531670660', 'audiCnt': '3829', 'audiInten': '-1446', 'audiChange': '-27.4', 'audiAcc': '57588', 'scrnCnt': '391', 'showCnt': '496'}, {'rnum': '6', 'rank': '6', 'rankInten': '0', 'rankOldAndNew': 'OLD', 'movieCd': '20201965', 'movieNm': '해적: 도깨비 깃발', 'openDt': '2022-01-26', 'salesAmt': '15007600', 'salesShare': '0.6', 'salesInten': '-4912700', 'salesChange': '-24.7', 'salesAcc': '12408496590', 'audiCnt': '3617', 'audiInten': '-33', 'audiChange': '-0.9', 'audiAcc': '1322081', 'scrnCnt': '157', 'showCnt': '188'}, {'rnum': '7', 'rank': '7', 'rankInten': '0', 'rankOldAndNew': 'OLD', 'movieCd': '20223743', 'movieNm': '나이트메어 앨리', 'openDt': '2022-02-23', 'salesAmt': '22611190', 'salesShare': '0.9', 'salesInten': '-5057590', 'salesChange': '-18.3', 'salesAcc': '270229080', 'audiCnt': '2207', 'audiInten': '-785', 'audiChange': '-26.2', 'audiAcc': '27685', 'scrnCnt': '246', 'showCnt': '273'}, {'rnum': '8', 'rank': '8', 'rankInten': '5', 'rankOldAndNew': 'OLD', 'movieCd': '20223308', 'movieNm': '극장판 바다 탐험대 옥토넛 : 해저동굴 대탈출\t', 'openDt': '2022-02-17', 'salesAmt': '15387000', 'salesShare': '0.6', 'salesInten': '6082200', 'salesChange': '65.4', 'salesAcc': '170775200', 'audiCnt': '1842', 'audiInten': '670', 'audiChange': '57.2', 'audiAcc': '20598', 'scrnCnt': '130', 'showCnt': '159'}, {'rnum': '9', 'rank': '9', 'rankInten': '-1', 'rankOldAndNew': 'OLD', 'movieCd': '20210028', 'movieNm': '스파이더맨: 노 웨이 홈', 'openDt': '2021-12-15', 'salesAmt': '14061800', 'salesShare': '0.5', 'salesInten': '-7699950', 'salesChange': '-35.4', 'salesAcc': '75022736170', 'audiCnt': '1606', 'audiInten': '-958', 'audiChange': '-37.4', 'audiAcc': '7532426', 'scrnCnt': '99', 'showCnt': '105'}, {'rnum': '10', 'rank': '10', 'rankInten': '4', 'rankOldAndNew': 'OLD', 'movieCd': '20218764', 'movieNm': '씽2게더', 'openDt': '2022-01-05', 'salesAmt': '12091300', 'salesShare': '0.5', 'salesInten': '3687700', 'salesChange': '43.9', 'salesAcc': '8122747140', 'audiCnt': '1269', 'audiInten': '318', 'audiChange': '33.4', 'audiAcc': '877343', 'scrnCnt': '68', 'showCnt': '92'}]}}
    

### dictionary를 DataFrame으로 만들기 😼


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


### 하나의 column 추출하기


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)

print(df['이름'])  # 결과는 Series
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


    one      이채영
    two      박시은
    three    장예은
    four     윤세은
    Name: 이름, dtype: object
    

### 하나의 column을 추출하면 view로 추출 => 원본이 변경됨


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
my_name = df['이름']
print(my_name)

my_name['one'] = '심자윤'
print(my_name)
```

    one      이채영
    two      박시은
    three    장예은
    four     윤세은
    Name: 이름, dtype: object
    one      심자윤
    two      박시은
    three    장예은
    four     윤세은
    Name: 이름, dtype: object
    


```python
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>심자윤</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


### 비연속적인 두 개 이상의 column 추출하기 => fancy indexing


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)

display(df[['이름', '학년']])   # 결과는 DataFrame
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>이채영</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>박시은</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>장예은</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>윤세은</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


### 특정 column의 값 수정하기


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)

df['학년'] = 1   # broadcasting
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)

df['학년'] = [2, 3, 4, 4]
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


### 새로운 column 추가하기


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)

df['나이'] = [20, 22, 30, 25]
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
      <th>나이</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
      <td>20</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
      <td>22</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
      <td>30</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>



```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)

df['조정학점'] = df['학점'] * 1.2
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
      <th>조정학점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
      <td>1.68</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
      <td>3.24</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
      <td>4.20</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
      <td>3.48</td>
    </tr>
  </tbody>
</table>
</div>


### 원하는 row/column 삭제하기 - drop()


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)

new_df = df.drop('two', axis=0, inplace=False)  # row 삭제
display(new_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)

new_df = df.drop('학점', axis=1, inplace=False)  # column 삭제
display(new_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


## row indexing 🦊

### 숫자 index 이용하기


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
display(df[0:2])  # 슬라이싱 가능. slicing한 결과는 DataFrame. view
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



```python
# print(df[0])  # Error - 단일 indexing 안 됨
```


```python
# display(df[[0,2]])  # Error - Fancy indexing 안 됨
```

### 지정 index 이용하기


```python
display(df['one':'three'])  # 슬라이싱 가능. slicing한 결과는 DataFrame. view
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
# print(df['one'])  # Error - 단일 indexing 안 됨
```


```python
# display(df[['one', 'three']])  # Error - Fancy indexing 안 됨
```

## column indexing 🐰


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
print(df['이름'])  # Series
```

    one      이채영
    two      박시은
    three    장예은
    four     윤세은
    Name: 이름, dtype: object
    


```python
display(df[['학과', '이름', '학년']])  # 결과는 DataFrame
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
# print(df['이름':'학년'])  # Error - column의 slicing은 안 됨
```

### df[ ] : column indexing할 때만 사용하도록 한다.

## df.loc[ ] 😗
- 행과 열에 대한 indexing 가능
- 숫자 index 사용 불가, 지정 index만 사용 가능


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
print(df.loc['one'])  # 결과는 Series, 단일 row 추출 가능
```

    학과     철학
    이름    이채영
    학점    1.4
    학년      1
    Name: one, dtype: object
    


```python
display(df.loc['one':'three'])  # 결과는 DataFrame
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
display(df.loc[['one', 'three']]) # 결과는 DataFrame, Fancy indexing 가능
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
# print(df.loc[0]) # Error - loc는 숫자 index 사용 불가
```


```python
# print(df.loc['one':-1])  # Error - loc는 숫자 index 사용 불가
```

## df.iloc[ ] 😳
- 지정 index 사용 불가, 숫자 index만 사용 가능


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
print(df.iloc[0])
```

    학과     철학
    이름    이채영
    학점    1.4
    학년      1
    Name: one, dtype: object
    

## loc를 이용해서 row indexing과 column indexing을 같이 할 수 있다.


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
display(df.loc['two':'three'])  # row slicing
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.loc['two':'three','학과':'이름']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.loc['two':'three','이름']) # 결과는 Series
```

    two      박시은
    three    장예은
    Name: 이름, dtype: object
    


```python
display(df.loc['two':'three', ['이름', '학년']])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>two</th>
      <td>박시은</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>장예은</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


## boolean indexing


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


### 학점이 3.0 이상인 학생의 학과와 이름


```python
df.loc[df['학점'] >= 3.0, ['학과', '이름']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
    </tr>
  </tbody>
</table>
</div>



## 새로운 row 추가하기


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.loc['five',:] = ['영어영문', '배수민', 3.7, 1]
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>five</th>
      <td>영어영문</td>
      <td>배수민</td>
      <td>3.7</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.loc['five',['학과','이름']] = ['물리학과', '심자윤']
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>five</th>
      <td>물리학과</td>
      <td>심자윤</td>
      <td>3.7</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


NaN(Not a Number) : 결치값(값이 없는 것을 나타냄) - 실수로 간주

## row 삭제하기


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
new_df = df.drop('three', axis=0, inplace=False)
display(new_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


## column 삭제하기


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['이채영', '박시은', '장예은', '윤세은'],
           '학과': ['철학', '수학', '컴퓨터', '국어국문'],
           '학년': [1, 3, 2, 4],
           '학점': [1.4, 2.7, 3.5, 2.9]}

df = pd.DataFrame(my_dict,
                  columns=['학과', '이름', '학점', '학년'],
                  index=['one', 'two', 'three', 'four'])
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학과</th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>철학</td>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>수학</td>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>컴퓨터</td>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>국어국문</td>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
new_df = df.drop('학과', axis=1, inplace=False)
display(new_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>학점</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>이채영</td>
      <td>1.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>two</th>
      <td>박시은</td>
      <td>2.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>three</th>
      <td>장예은</td>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>four</th>
      <td>윤세은</td>
      <td>2.9</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


# DataFrame이 제공하는 함수 😌

## UCI Machine Learning Repository에서 제공하는 MPG Data set 이용
- MPG(Mile Per Gallon) Data set => 자동차 연비에 관련된 Data set
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

df = pd.read_csv('./data/auto-mpg.csv', header=None) # header=None을 뺄 경우 첫 줄을 header로 잡음


df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower',
              'weight', 'acceleration', 'year', 'origin', 'name']

display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
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
    </tr>
    <tr>
      <th>393</th>
      <td>27.0</td>
      <td>4</td>
      <td>140.0</td>
      <td>86.00</td>
      <td>2790.0</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford mustang gl</td>
    </tr>
    <tr>
      <th>394</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52.00</td>
      <td>2130.0</td>
      <td>24.6</td>
      <td>82</td>
      <td>2</td>
      <td>vw pickup</td>
    </tr>
    <tr>
      <th>395</th>
      <td>32.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84.00</td>
      <td>2295.0</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
      <td>dodge rampage</td>
    </tr>
    <tr>
      <th>396</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79.00</td>
      <td>2625.0</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford ranger</td>
    </tr>
    <tr>
      <th>397</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82.00</td>
      <td>2720.0</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
      <td>chevy s-10</td>
    </tr>
  </tbody>
</table>
<p>398 rows × 9 columns</p>
</div>


## 1. head(), tail() - DataFrame 안의 데이터 앞에서 5개(기본), 뒤에서 5개(기본) 추출


```python
display(df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>



```python
display(df.head(3))  # 상위 3개의 행만 확인
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
display(df.tail())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>393</th>
      <td>27.0</td>
      <td>4</td>
      <td>140.0</td>
      <td>86.00</td>
      <td>2790.0</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford mustang gl</td>
    </tr>
    <tr>
      <th>394</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52.00</td>
      <td>2130.0</td>
      <td>24.6</td>
      <td>82</td>
      <td>2</td>
      <td>vw pickup</td>
    </tr>
    <tr>
      <th>395</th>
      <td>32.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84.00</td>
      <td>2295.0</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
      <td>dodge rampage</td>
    </tr>
    <tr>
      <th>396</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79.00</td>
      <td>2625.0</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford ranger</td>
    </tr>
    <tr>
      <th>397</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82.00</td>
      <td>2720.0</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
      <td>chevy s-10</td>
    </tr>
  </tbody>
</table>
</div>



```python
display(df.tail(2)) # 하위 2개의 행만 확인
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>396</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79.00</td>
      <td>2625.0</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford ranger</td>
    </tr>
    <tr>
      <th>397</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82.00</td>
      <td>2720.0</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
      <td>chevy s-10</td>
    </tr>
  </tbody>
</table>
</div>


## 2. shape


```python
print(df.shape)  
```

    (398, 9)
    

## 3. info() - DataFrame의 기본 정보 추출


```python
print(df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 398 entries, 0 to 397
    Data columns (total 9 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   mpg           398 non-null    float64
     1   cylinders     398 non-null    int64  
     2   displacement  398 non-null    float64
     3   horsepower    398 non-null    object 
     4   weight        398 non-null    float64
     5   acceleration  398 non-null    float64
     6   year          398 non-null    int64  
     7   origin        398 non-null    int64  
     8   name          398 non-null    object 
    dtypes: float64(4), int64(3), object(2)
    memory usage: 28.1+ KB
    None
    

## 4. count() - 유효한 값의 개수(NaN이 아닌 값의 개수)


```python
print(df.count())  # 결과는 Series
```

    mpg             398
    cylinders       398
    displacement    398
    horsepower      398
    weight          398
    acceleration    398
    year            398
    origin          398
    name            398
    dtype: int64
    

## 5. value_counts() - Series에 대해서 unique value의 개수를 알려줌

- origin이라는 컬럼은 제조국을 나타내고 1,2,3 중 하나의 값을 가짐
- 1: USA, 2: EU, 3: JPN


```python
print(df['origin'].value_counts())  # 결과는 Series
```

    1    249
    3     79
    2     70
    Name: origin, dtype: int64
    

### 만약 NaN값이 있으면 value_counts()는 어떻게 동작할까?
### => 기본적으로 NaN을 포함해서 계산하며, 옵션을 줄 경우 NaN을 제외하고 수행 가능


```python
print(df['origin'].value_counts(dropna=True))
```

    1    249
    3     79
    2     70
    Name: origin, dtype: int64
    

## 6. unique() - Series에 대해서 중복을 제거해서 유일한 값이 어떤값이 있는지를 알려줌


```python
print(df['year'].unique())
```

    [70 71 72 73 74 75 76 77 78 79 80 81 82]
    

## 7. isin() - boolean mask를 만들기 위해 많이 사용


```python
df['origin'].isin([3])  # 제조국이 일본(3)인 mask
```




    0      False
    1      False
    2      False
    3      False
    4      False
           ...  
    393    False
    394    False
    395    False
    396    False
    397    False
    Name: origin, Length: 398, dtype: bool




```python
df.loc[df['origin'].isin([3]),:]  # boolean indexing
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>14</th>
      <td>24.0</td>
      <td>4</td>
      <td>113.0</td>
      <td>95.00</td>
      <td>2372.0</td>
      <td>15.0</td>
      <td>70</td>
      <td>3</td>
      <td>toyota corona mark ii</td>
    </tr>
    <tr>
      <th>18</th>
      <td>27.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>88.00</td>
      <td>2130.0</td>
      <td>14.5</td>
      <td>70</td>
      <td>3</td>
      <td>datsun pl510</td>
    </tr>
    <tr>
      <th>29</th>
      <td>27.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>88.00</td>
      <td>2130.0</td>
      <td>14.5</td>
      <td>71</td>
      <td>3</td>
      <td>datsun pl510</td>
    </tr>
    <tr>
      <th>31</th>
      <td>25.0</td>
      <td>4</td>
      <td>113.0</td>
      <td>95.00</td>
      <td>2228.0</td>
      <td>14.0</td>
      <td>71</td>
      <td>3</td>
      <td>toyota corona</td>
    </tr>
    <tr>
      <th>53</th>
      <td>31.0</td>
      <td>4</td>
      <td>71.0</td>
      <td>65.00</td>
      <td>1773.0</td>
      <td>19.0</td>
      <td>71</td>
      <td>3</td>
      <td>toyota corolla 1200</td>
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
    </tr>
    <tr>
      <th>382</th>
      <td>34.0</td>
      <td>4</td>
      <td>108.0</td>
      <td>70.00</td>
      <td>2245.0</td>
      <td>16.9</td>
      <td>82</td>
      <td>3</td>
      <td>toyota corolla</td>
    </tr>
    <tr>
      <th>383</th>
      <td>38.0</td>
      <td>4</td>
      <td>91.0</td>
      <td>67.00</td>
      <td>1965.0</td>
      <td>15.0</td>
      <td>82</td>
      <td>3</td>
      <td>honda civic</td>
    </tr>
    <tr>
      <th>384</th>
      <td>32.0</td>
      <td>4</td>
      <td>91.0</td>
      <td>67.00</td>
      <td>1965.0</td>
      <td>15.7</td>
      <td>82</td>
      <td>3</td>
      <td>honda civic (auto)</td>
    </tr>
    <tr>
      <th>385</th>
      <td>38.0</td>
      <td>4</td>
      <td>91.0</td>
      <td>67.00</td>
      <td>1995.0</td>
      <td>16.2</td>
      <td>82</td>
      <td>3</td>
      <td>datsun 310 gx</td>
    </tr>
    <tr>
      <th>390</th>
      <td>32.0</td>
      <td>4</td>
      <td>144.0</td>
      <td>96.00</td>
      <td>2665.0</td>
      <td>13.9</td>
      <td>82</td>
      <td>3</td>
      <td>toyota celica gt</td>
    </tr>
  </tbody>
</table>
<p>79 rows × 9 columns</p>
</div>



# DataFrame 안의 데이터 정렬하기 😎


```python
import numpy as np
import pandas as pd

# 난수의 재현성을 확보
np.random.seed(1)
df = pd.DataFrame(np.random.randint(0,10,(6,4)),
                  columns=['A', 'B', 'C', 'D'],
                  index=pd.date_range('20220101', periods=6))
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-01</th>
      <td>5</td>
      <td>8</td>
      <td>9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2022-01-02</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2022-01-03</th>
      <td>6</td>
      <td>9</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2022-01-04</th>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2022-01-05</th>
      <td>4</td>
      <td>7</td>
      <td>7</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2022-01-06</th>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


## index 랜덤하게 섞기
- np.random.shuffle(df.index) => shuffle()은 원본데이터를 변경함
- DataFrame의 index는 mutable operation을 지원하지 않기 때문에 index 자체를 변경시킬 수 없음
- np.random.permutation() => 섞어서 원본을 변경하지 않고 복사본을 만드는 함수


```python
random_index = np.random.permutation(df.index)
print(random_index)
```

    ['2022-01-03T00:00:00.000000000' '2022-01-01T00:00:00.000000000'
     '2022-01-04T00:00:00.000000000' '2022-01-05T00:00:00.000000000'
     '2022-01-02T00:00:00.000000000' '2022-01-06T00:00:00.000000000']
    

## 변경된 index로 DataFrame 재설정하기


```python
df2 = df.reindex(index=random_index,
                 columns=['B', 'A', 'D', 'C'])
display(df2)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>A</th>
      <th>D</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-03</th>
      <td>9</td>
      <td>6</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2022-01-01</th>
      <td>8</td>
      <td>5</td>
      <td>5</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2022-01-04</th>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2022-01-05</th>
      <td>7</td>
      <td>4</td>
      <td>9</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2022-01-02</th>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2022-01-06</th>
      <td>7</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


## index를 기준으로 정렬하기


```python
display(df2.sort_index(axis=1, ascending=True))  # column 정렬
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-03</th>
      <td>6</td>
      <td>9</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2022-01-01</th>
      <td>5</td>
      <td>8</td>
      <td>9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2022-01-04</th>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2022-01-05</th>
      <td>4</td>
      <td>7</td>
      <td>7</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2022-01-02</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2022-01-06</th>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


## value를 기준으로 정렬하기


```python
display(df2.sort_index(axis=0, ascending=True))  # row 정렬
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>A</th>
      <th>D</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-01</th>
      <td>8</td>
      <td>5</td>
      <td>5</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2022-01-02</th>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2022-01-03</th>
      <td>9</td>
      <td>6</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2022-01-04</th>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2022-01-05</th>
      <td>7</td>
      <td>4</td>
      <td>9</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2022-01-06</th>
      <td>7</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


## 특정 column의 값으로 row 정렬


```python
display(df2.sort_values(by='B', ascending=True))  
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>A</th>
      <th>D</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-02</th>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2022-01-04</th>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2022-01-05</th>
      <td>7</td>
      <td>4</td>
      <td>9</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2022-01-06</th>
      <td>7</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-01-01</th>
      <td>8</td>
      <td>5</td>
      <td>5</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2022-01-03</th>
      <td>9</td>
      <td>6</td>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

