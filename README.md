# Ybigta-Team-project
Ybigta Team project

final data : final_train_merged.csv, final_test_merged.csv, pca_test_all_c_fraud.csv

# 데이터 전처리 과정
##TransactionAmt_log
기존 Amt 분포가 심하게 skewed 되어 있어서 outlier을 제거한 후 Log scale로 변환 하였다.
##TransactionDT
첫번째 시간이 86400으로 나왔다. 즉 86400=24*60*60 이므로 하루를 초로 나타낸 단위이다. 따라서 이것을 이용해 시간, 분, 날짜, 요일을 알아내었다.
##c_fraud
나라별로 시차가 있어 hour 변수가 다를 테니 나라별 시간별 사기비율을 만들었다.

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
