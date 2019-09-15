# Ybigta-Team-project
Ybigta Team project

final data : final_train_merged.csv, final_test_merged.csv, pca_test_all_c_fraud.csv

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
~~~
**from** imblearn.over_sampling **import** SMOTE
**from** sklearn.model_selection **import** train_test_split

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
~~~
