# Ybigta-Team-project
Ybigta Team project

final data : final_train_merged.csv, final_test_merged.csv, pca_test_all_c_fraud.csv

# 목차
1. 데이터 전처리 과정
2. 활용 기법 및 개념
3. 한계와 반성

# 데이터 전처리 과정

## Identification File
이 파일은 ID01~38, Device Type, Device Info 등을 포함했다. 이 파일에 포함된 col에서 나타나는 값은 IP와 같은 고유값이기 때문에 값의 크기와 사기비율은 상관관계는 없다고 가정하고 분석을 진행하였다. 또한 NaN의 빈도가 절반이 넘는 경우가 빈번했다. 따라서 값의 종류가 다양해서 one hot encoding (OHE)이 불가능한 col의 경우, 값의 존재 유무와 사기의 상관관계라도 이용하고자 했다. 이는 컬럼이 매우 많았기에 자동화를 통해 진행했다.

### ID 01~13
Col 02,05,06,11,19,20은 NAN 값이 같은 row에서 동시에 등장했다. 값이 다양해 원래의 방식으로는 OHE가 불가했으므로 값의 존재 유무를 OHE으로 처리했고, 유의미한 상관관계를 발견할 수 있어 분석에 포함했다. Col 07, 08 등과 같은 경우 값의 존재 유무와 사기여부의 상관관계가 없었다. 다른 col들도 이와 같은 방식으로 처리하여 13개의 col들을 4개(01,03,10,11)로 압축했다.

### ID 14~38
OHE가 가능한 컬럼들(12,14,15,16,28,29,30,31,34,35,36,37,38)에 대해서 OHE처리 후, 만들어진 모든 col에 대해 사기여부와의 연관성을 조사했다. 이를 통해 12의 'found', 15의 'found', 16의 'found', 28의 'found', 35의 'T', 36의 'F', 37의 'T', 38의 'F'가 사기와 유의미한 상관관계가 있음을 밝혔고, 분석에 포함하였다(단 16의 경우 'found'값이 15와 동시에 출현하기 때문에 제외). Col 33의 경우 화면 크기가 '1000x700'과 같은 형식으로 표현되어 있었고, 전체 픽셀 개수로 변환하여 상관관계를 확인해보았으나 유의미한 관계를 찾지 못했다. 나머지는 존재 유무만을 OHE 처리하여 상관관계를 파악했으며, 상관관계가 없는 것들과 col 01~13에서 나온 것과 겹치는 열들을 제외하니 col 15, 18을 얻게 되었다.

### Device Type
NaN의 존재가 02의 nan의 존재와 동기화되어 있는 col이었다. 값이 두 종류라 OHE 처리를 시도해보았으니 사기와 상관관계는 없었다. 따라서 존재 유무만을 OHE처리하여 확인해보았고 상관관계를 발견하여 분석에 포함하였다.

### Device INFO
기기 명을 나타내는 col이었다. OHE를 시도하기 위해 샘플을 뽑아 확인해보았으나 종류가 방대하여 처리가 불가하다고 판단했다. window나 samsung 등의 대분류로 나타내보려는 시도도 해보았지만 대분류의 경우 사기 상관관계을 나타내지 않았다. NaN의 존재가 Device Type과 동기화되어 있어 이 col은 분석에서 배제하였다.

## TransactionAmt_log
기존 Amt 분포가 심하게 skewed 되어 있어서 outlier을 제거한 후 Log scale로 변환하였다.

## TransactionAmt_residue
Fraud 거래에서 TransactionAmt가 소숫점 이하 자릿수 15자리 이상인 경우가 많이 나타나는 점에 착안하여, 소숫점 이하 자릿수 15자리 이상의 여부에 대한 boolean 컬럼을 생성하였다.

## repeated
card1~6 정보가 모두 일치하는 경우를 동일 카드로 간주, 카드 정보와 TransactionAmt, ProductCD가 일치하는 거래를 동일 거래로 취급한다. 이를 바탕으로 각 거래의 반복 횟수를 값으로 갖는 새로운 컬럼 repeated를 생성하였다.

## TransactionDT
첫번째 시간이 86400으로 나왔다. 즉 86400=24*60*60 이므로 하루를 초로 나타낸 단위이다. 따라서 이것을 이용해 시간, 분, 날짜, 요일을 알아내었다.

## c_fraud
나라별로 시차가 있어 hour 변수가 다를 테니 나라별 시간별 사기비율을 만들었다.

## ProductCD
Dummy 컬럼들을 생성하여 카테고리형인 ProductCD를 one-hot encoding 하였다.

## card1_count, card3_count, card5_count, card6_count
수치형 데이터이지만 사실상 각 수치가 식별자 역할을 하는 카테고리형 변수이기 때문에, 각 수치값의 카운트 값을 value로 갖는 새로운 컬럼 card*n*\_count를 생성하였다.
card3와 card5는 카테고리형 변수를 보정하기 위해 dummify하여 one-hot encoding 처리하였다.
card3의 경우 국가 정보라는 가정을 하였다.

## card2_na
NA가 높은 비율로 나타나기 때문에 NA 여부에 대한 binary 컬럼을 생성하였다.

## card4
Random sampling으로 null value를 imputation하였다.

## addr1
지역 정보라는 가정 하에, NA 여부에 대한 binary 컬럼을 생성하였다. 

## addr2
국가 정보라는 가정 하에, random sampling으로 null value를 imputation하고 수치값의 카운트 값을 value로 갖는 새로운 컬럼 addr2_count를 생성하였다. 이후 dummify하여 one-hot encoding 처리하였다. 

## dist1, dist2
NA가 높은 비율로 나타나기 때문에 NA 여부에 대한 binary 컬럼을 생성하였다.

## P_emaildomain, R_emaildomain
P_emaildomain이 NA가 아닌 경우 P_emaildomain 값을 취하고, NA인 경우 R_emaildomain 값을 취함으로써 두 변수의 정보를 합쳤다. 이후 여전히 NA인 경우는 random sampling으로 imputation 처리하였다.
이후 카테고리형을 보정하기 위해 dummify하여 one-hot encoding 하고, 컬럼 정보를 축약하기 위해 PCA를 진행하였다.
PCA 처리를 할 때에는 train set과 test set을 concatenate하여 한번에 PCA 처리 한 뒤 다시 train set과 test set으로 분리하였다.

## C1~C14
C1부터 C14의 14개 컬럼 값을 전부 이어붙이 컬럼 C를 생성하여, 각 C 값을 일종의 식별자로 취급하고 수치값 카운트 값을 value로 갖는 새로운 컬럼 C_count를 생성하였다.
마찬가지로 C1~C14 각 컬럼에 대해 수치값의 카운트 값을 value로 갖는 새로운 컬럼 C*n*\_count를 생성하였다.

## D, M
NA가 높은 비율로 나타나기 때문에 NA 여부에 대한 binary 컬럼을 생성하였다.

## V
최대한의 정보를 추출하기 위해 NaN의 존재 여부와 사기의 상관관계, 값 자체와 사기의 상관관계를 둘 다 분석하였다. NaN의 존재 여부와 사기의 상관관계를 분석하기 위해 NaN이 존재하는 col들을 존재 여부에 따라 one hot encoding 한 후 분석한 결과 V138~278은 V169하나로 압축이 가능하며 V1~V11 또한 V1로 압축이 가능함을 발견했다. 나머지 col들은 유의미한 상관관계가 없었다. 다음으로 값 자체와 사기의 상관관계를 찾기 위해서 NaN을 imputation 해 준 후 상관관계를 분석했다. 구체적으로 NaN을 median, mean으로 각각 imputation 한 후 상관관계가 높게 나오는 방법을 col 별로 구분하여 채택했다. 이후 상관관계가 0.05가 넘는 col만을 남겼고 PCA를 통해 총합 V에서 도출된 134개의 행을 45개로 축소하였다

# 활용 기법 및 개념들

## Smote
데이터 분석 시 자주 마주하게 되는 문제인 **데이터의 불균형**을 해결하기 위한 방법이다.
이번 데이터도 사기비율이 전체 거래의 3.5% 내외로 극소수인 케이스였다. 이러한 비대칭 데이터셋에서는 정확도(accuracy)가 높아도 재현율(recall, 실제 부실을 부실이라고 예측할 확률)이 급격히 작아진다. 100개의 데이터 중 4개 사기일 때, 모두 정상이라고 예측해도 정확도가 96%가 나오기 때문이다.

이러한 데이터 불균형을 해결하는 방법을 크게 2가지다.

~~~
Undersampling
1. 무작위추출: 무작위로 정상데이터를 일부만 선택
2. 유의정보: 유의한 데이터만을 남기는 방식(알고리즘: EasyEnsemble, BalanceCascade
3. 유의점: Undersampling의 경우 데이터의 소실이 매우 크고, 때로는 중요한 정상데이터를 잃을 수가 있다.
~~~

~~~
Oversampling
1. 무작위추출: 무작위로 소수 데이터를 복제
2. 유의정보: 사전에 기준을 정해서 소수 데이터를 복제
3. 유의점: 정보가 손실되지 않지만 오버피팅(overfitting)을 초래할 수 있다. 이러한 경우 test set의 성능이 나빠질 수 있다.
~~~

### Smote는?
smote는 oversampling의 한 방식으로 부트스트랭이나 KNN(최근접이웃) 모델 기법을 활용한다.

#### 파이썬 코드
```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
```

##### Reference
https://mkjjo.github.io/python/2019/01/04/smote_duplicate.html


## Robust sampling
Robust sampling은 scaling의 한 종류이다. 
데이터를 모델링하기 전에는 반드시 스케일링 과정을 거쳐야 한다. 스케일링을 통해 다차원의 값들을 비교 분석하기 쉽게 만들어주며, 자료의 오버플로우(overflow)나 언더플로우(underflow)를 방지 하고, 독립 변수의 공분산 행렬의 조건수(condition number)를 감소시켜 최적화 과정에서의 안정성 및 수렴 속도를 향상 시킨다. 특히 k-means 등 거리 기반의 모델에서는 스케일링이 매우 중요하다.

|  | 종류 | 설명 |
|---|:---:|---:|
| 1 | 'StandardScaler' | 기본 스케일. 평균과 표준편차 사용 |
| 2 | 'MinMaxScaler' | 최대/최소값이 각각 1, 0이 되도록 스케일링 |
| 3 | 'axAbsScaler' | 최대/절대값과 0이 각각 1, 0이 되도록 스케일링 |
| 4 | 'RobustScaler' | 중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화 |

### 파이썬 코드
```python
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
robust_scaler.fit(X)
X_robust_scaled = robust_scaler.transform(X)
```

##### Reference
https://mkjjo.github.io/python/2019/01/10/scaler.html


## PCA(Principal component analysis)
#### 기법 개요
PCA는 데이터의 분산(variance)을 최대한 보존하면서 서로 직교하는 새 기저(축)를 찾아, 고차원 공간의 표본들을 선형 연관성이 없는 저차원 공간으로 변환하는 기법입니다. 이를 그림(출처)으로 나타내면 아래와 같습니다. 2차원 공간에 있는 데이터들이 하나의 주성분(PC1)을 새로운 기저로 선형변환된 걸 확인할 수 있습니다. 여기에서 핑크색 표시가 돼 있는 사선축이 원 데이터의 분산을 최대한 보존하는(=데이터가 가장 많이 흩뿌려져 있는) 새 기저입니다. PCA의 목적은 바로 이런 축을 찾는 데 있습니다.

#### PCA의 절차
기존 데이터 X의 공분산행렬 계산
공분산행렬의 고유값과 고유벡터 계산
고유값의 크기 순서대로 고유벡터를 나열
정렬된 고유벡터 가운데 일부 선택
해당 고유벡터와 X 내적
PCA는 서로 직교하는 새 기저로 데이터를 변환하기 때문에 변수 간 상관관계가 높은 데이터에 효과가 좋다고 합니다. 데이터 차원축소, 압축에 널리 쓰이고 있습니다.

#### python example

```python
>>> import numpy as np
>>> from sklearn.decomposition import PCA
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> pca = PCA(n_components=2)
>>> pca.fit(X)  
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
>>> print(pca.explained_variance_ratio_)  
[0.9924... 0.0075...]
>>> print(pca.singular_values_)  
[6.30061... 0.54980...]
```

```python
>>> pca = PCA(n_components=2, svd_solver='full')
>>> pca.fit(X)                 
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='full', tol=0.0, whiten=False)
>>> print(pca.explained_variance_ratio_)  
[0.9924... 0.00755...]
>>> print(pca.singular_values_)  
[6.30061... 0.54980...]
```

```python
>>> pca = PCA(n_components=1, svd_solver='arpack')
>>> pca.fit(X)  
PCA(copy=True, iterated_power='auto', n_components=1, random_state=None,
  svd_solver='arpack', tol=0.0, whiten=False)
>>> print(pca.explained_variance_ratio_)  
[0.99244...]
>>> print(pca.singular_values_)  
[6.30061...]
```

##### Reference
https://ratsgo.github.io/machine%20learning/2017/04/24/PCA/
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

## LGBM(Light GBM)
LGBM은 GBM의 한 종류.
GBM(Gradient Boosting Algorithm): 회귀분석 또는 분류 분석을 수행할 수 있는 예측모형. 예측모형의 앙상블 방법론 중 부스팅 계열에 속하는 알고리즘.
LightGBM, CatBoost, XGBoost : Gradient Boosting Algorithm을 구현한 패키지

![image](https://user-images.githubusercontent.com/46089347/67144313-20cd4480-f2b0-11e9-8e39-bc825e8f1366.png)

### Boosting의 개념은 약한 분류기를 결합하여 강한 분류기를 만드는 과정
분류기 A, B, C 가 있고, 각각이 0.2 정도의 accuracy를 보여준다고 하면 A, B, C를 결합하여 더 높은 정확도(약 0.7의 accuracy)를 얻는 게 앙상블 알고리즘의 기본 원리다. Boosting은 이 과정을 순차적으로 실행하는데 A 분류기를 만든 후, 그 정보를 바탕으로 B 분류기를 만들고, 다시 그 정보를 바탕으로 C 분류기를 만든다. 결과적으로는 만들어진 모든 분류기를 결합해 최종 모델을 만든다.

### XGBoost vs. LGBM
XGboost는 학습 시간이 오래 걸린다. Grid search를 통해 최적의 파라미터를 탐색한다면 학습 시간의 효율을 고려해야 한다. LGBM은 XGBoost보다 학습 시간이 덜 걸리며 메모리 사용도 작다는 장점이 있다. 둘 다 좋은 알고리즘이므로 특별히 어느 하나가 좋다고 할 수는 없다. 단, LGBM은 데이터셋의 과적합을 조심해야 하는데, 1만개 이하의 데이터셋은 과적합 위험이 있다. 

### LGBM의 leaf wise 방식
일반적인 GBM 패키지와 다르게 LGBM은 leaf wise방식을 사용한다. 기존의 트리기반 알고리즘은 트리의 깊이를 줄이기 위해 level wise(균형 트리 분할)를 사용한다면 LGBM은 leaf wise(리프 중심 트리 분할)를 이용한다. 앞의 level wise는 트리를 균형적으로 분할하는데 균형작업이 추가로 들어간다고 보면 된다. LGBM은 균형적으로 트리를 분할하지 않는 대신 최대 손실값(max delta loss)을 갖는 트리 노드를 계속 분할한다. 이 때문에 비대칭적으로 어느 트리는 깊이가 아주 깊어지게 된다. 이 방식은 균형 트리 분할보다 오류를 최소화할 수 있다. 

![image](https://user-images.githubusercontent.com/46089347/67144522-3479aa80-f2b2-11e9-9a17-1206005286f6.png)
xgboost와 lgbm의 차이

### 주요 hyper parameter
1. n_estimators : default=100, 반복하려는 트리의 갯수
2. learning)rate : 0~1사이 값 지정. gradient descent에서 얼마나 움직일 것인지 설정한다. 간단히 학습률이라고 생각하면 된다.
3. max_depth : default=-1, 최대 깊이를 조절
4. min_child_samples : default=20, leaf node로 분류되기 위한 최소 데이터 수
5. num_leaves : default=31, one tree가 가잘 수 있는 leaf 갯수
6. boost : default=gbdt, gbdt는 gradient descent를 의미. rt는 random forest
7. reg_lambda : L2 규제 적용
8. leg_alpha : L1 규제 

출처
* https://lsjsj92.tistory.com/525
* https://3months.tistory.com/368 [Deep Play]


## 한계와 반성

### 데이터 전처리 과정에서의 한계 
1. 데이터를 팀원들과 분할해서 전처리한 후 통합 과정에서 PCA를 전체 데이터에 대해서 진행하지 않았다. PCA는 각자 진행할 필요가 없다.
2. D, M 항목들을 더 자세히 처리할 수 있었다. 전처리 과정에서 사용한 기법들을 서로 공유하거나 한 사람이 총괄할 필요가 있다.
3. 전처리를 잘 해 놓고도 최종 데이터셋으로 통합하는 과정에서 발생했다고 추정되는 오류로 인해 사용하지 못한 col들이 있다. 전처리에 너무 많은 시간을 소모하여 오류를 수정할 시간이 없었던 것으로 판단된다.

### 분석 과정에서의 한계
1. 최종 데이터셋이 늦게 확정되어 최적의 파라미터를 찾는 효과적인 방법(grid search 등)을 
사용하지 못했다. 전처리 과정만큼이나 중요하고 오래 걸리는 것이 분석 과정이란 점을 명심해야 한다.
2. Smote를 사용해서 데이터의 불균형을 맞추는 것도 좋지만, 다른 방법들도 시도해 볼 수 있었다. 예를 들어 부족한 사기 데이터에 대해 undersampling을 진행한 후, 모든 비사기 데이터를 고려할 수 있도록 때까지 데이터셋을 여러개 만드는 방법 등이 있다. 
