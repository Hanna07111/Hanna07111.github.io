---
title: "Distilling the knowledge"
date: 2025-09-17
categories: [Capstone, 논문리뷰]
tags: [knowledge distillation]
---

URL: https://arxiv.org/abs/1503.02531

## Abstract

- 이 논문에서 제안한 두 가지
    - 지식 증류 방안
    - New type of ensemble

## Introduction

- 모델의 지식은 파라미터가 아닌 인풋, 아웃풋 벡터의 맵핑
- 그전까지의 문제 incorrect answer에도 확률값을 주지만, 이 확률값은 사실 거의 사용되지 않음
- Probability of incorrect answers
    
    → ’모델이 어떻게 지식을 일반화하는지‘를 보여줌 (가장 중요한 부분)
    
    → 이걸 이용하면 large model이 지식을 일반화하는 방식을 small model에 가르칠 수 있음
    
- 따라서 위 정보를 small model을 학습시키는 soft target으로 사용
    - soft target이 더 많은 정보를 가지고 있음
    - training case 당 gradient가 적은 분산 및 일정한 업데이트 방향을 가짐 (하드타겟보다 덜 유동적, 아무래도 모든 클래스에 대한 확률을 정답으로 주기 때문!)
    
    ⇒ 효율적이고 안정적인 학습 가능
    
- Caruna는 logit (softmax의 input) 간의 차이를 최소화 하는 방법을 사용
    
    ⇒ 여기서는 ‘temperature’라는 개념을 사용
    
- Transfer set은 unlabeled data나 original training set을 쓸 수 있지만, 후자가 좀 더 성능이 좋았음
    - Objective function을 둘 다 다 맞출 수 있도록 형식을 맞춰 줌

## 2 Distillation

- Softmax 함수에 temperature 을 적용한 soft target 사용
- 방법 두 가지
    - Soft target 만 사용
    - Soft target + hard target 함께 사용
        - 두 타겟을 모두 반영한 objective function 사용
        - 하드 타겟에 더 적은 가중치를 주는 게 훨씬 성능에 좋음
        - Soft target의 gradient에 1/T2 값이 곱해지므로, T2 을 다시 곱해줘야 이 둘의 비중을 유지해줄 수 있음

### 2.1 Matching logits is a special case of distillation

- 온도가 너무 높으면 값끼리의 스무딩 효과 높아짐
    
    → 모델의 gradient도 거의 그냥 로짓끼리의 제곱오차를 줄이는 수준
    
    → 모델이 정보를 배우기가 어렵다는 소리
    
    → 큰 모델에서는 괜찮겠지만, 작은 모델에서는 성능이 더더 안나올 수도
    
- 너무 낮으면 negative logit에서 정보를 배울 수가 없음..
    
    → 값들이 너무 극단적인 상태로 남아있다는 뜻
    
    → 하드 타겟과 크게 다를 게 없다는 뜻
    

⇒ 적당한 온도를 유지하는 게 중요

## 3 Preliminary experiments on MNIST

- MNIST 데이터에 적용 → soft target 만 사용
- Large model → small model 로
- Soft target 만 줬는데도 오류 급감
- 한 번도 본적 없는 데이터에 대해서도 작동을 잘함

## 4 Experiments on speech recognition

- Acoustic model에 적용 → soft target + hard target 함께 사용
- Ensmeble → single 모델로
- 성능 유지되는 것 확인할 수 있음

## 5 Training ensembles of specailists on very big datasets

- 거대한 모델로 거대한 데이터셋을 학습시키는 건 계산적으로 거의 불가능

→ specialist model 사용

→ but overfitting이라는 문제점 있긴 함 → soft target으로 prevent 가능

### 5.1 The JFT dataset

- 매우 큰 구글의 데이터셋 (라벨만 15000개)
- 병렬훈련의 두 가지 방식
    - 데이터 병렬 - 여러 코어에 같은 모델을 두고, 이 모델들에 서로 다른 배치를 줘 학습하게 함
    - 모델 병렬 - 여러 코어에 걸쳐서 하나의 모델이 있음
    
    ⇒ 이 두 개를 감싸는 방법이 앙상블 (여러 코어에 서로 다른 모델, 각각의 모델은 또 데이터 병렬이나 모델 병렬 등을 사용할 수 있음)
    
- 이 데이터셋을 가지고 위와 같은 앙상블 훈련을 하는건, 몇 년을 한다해도 이상할 게 없음
    
    → 간단화할 방법 필요
    ⇒ general model + specialist model 
    

### 5.2 Specialist Models

- 잘 분류 되지 않는 클래스에 한해 specialist model 만듦
- Specialist model
    - General model의 가중치로 초기화 (general model이 low-level의 feature를 배우는 방법은 배워야 하니까)
    - 학습에 사용되는 데이터셋 = 잘 분류되지 않는 클래스의 데이터+ 나머지 클래스에서 random하게 뽑은 데이터
    - 내가 분류하고자 하는 클래스가 아닌 경우 single dustin class로 취급(Softmax 출력층을 낮춤으로써 효율성 증가)
    - Training 후에 특정 클래스에 대해 oversampling 된 걸 보완하기 위해 bias 를 일부 조정해줄 수 있음

### 5.3 Assigning classes to specialists

- 그럼 전문가 모델이 판단하는 클래스는 어떻게 정함?
    
    ⇒ confusion matrix 사용 가능 → true label이 있어야 사용 가능 
    
    ⇒ 공분산 행렬에 clustering 적용
    
- 한 input에 대한 output → softmax 출력층을 거친 확률값들의 집합 → 하나의 확률 분포
    
    → 이 확률 분포를 하나의 벡터로 생각해, 각 벡터들의 공분산 행렬 계산
    
    → 각 공분산 값은 각 클래스의 예측 확률 끼리의 공분산 
    
    → 클수록 한 클래스의 예측 확률이 높을 때 다른 클래스의 예측 확률이 높음 ⇒ 비슷하게 예측될 확률이 높다는 것
    
    → 다시 이 공분산 행렬의 하나의 열을 데이터 포인트로 하여 k-means clustering 
    
    → 클래스 간의 관계가 비슷한 애들끼리 묶이겟조 (예를 들어 치와와는 요크셔와 비슷하고 자동차와 다름, 진돗개도 요크셔와 비슷하고 자동차와 다름 ⇒ 이러면 한 클래스)
    
- 다른 군집화 방법 사용해 봤지만 결과는 비슷했음

### 5.4 Performing inference with ensembles of specialists

- Specialist model 을 이용한 앙상블 잘 하는지 확인
- Test 방법
    1. Each test case 에 대해 general 모델에 넣어 가장 probable 한 클래스, 즉 높은 확률을 가지고 있는 classes k 고름 (여기서는 한 개만 고름)
    2. k 클래스들에 해당하는 specialist model들(k와 각 모델에 할당된 클래스들이 하나라도 겹치면 그 모델의 사용) 에 test data를 넣어서 결과를 얻음 (이 때 사용되는 specialist 모델 m들의 집합을 Ak 라고 함)
    
     ⇒ KL divergence 식 도출 
    
    ⇒ 이 식을 최소화할 수 있는 q를 찾아야 함 
    
    ⇒ 이 때 이 적절한 q=softmax(z)를 찾기 위해 KL 식에 대한 gradient descent 적용
    
    (모델의 가중치가 바뀌는 건 아니고, 최적화된 q를 찾기 위한 z를 찾는 과정)
    

### 5.5 Results

- Baseline 과 Baseline+special model ⇒ 4.4% 향상
- 클래스에 해당하는 specialist model이 많을수록, 성능이 향상됨 → ..?

## 6 Soft Targets as Regularizers

- hard target만은 쓰는 모델은 overfitting 문제가 심함
    
    ⇒ soft target 활용하면 문제 해결 가능
    
    ⇒ regularizer로 사용 가능
    

### 6.1 Using soft targets to prevent specialists from overfitting

- Specialist model → 학습 데이터셋 매우 적기 때문에, overfitting 가능성 매우 높음
- general 한 모델의 정보는 써야하니 모델 크기를 줄일 수는 없음
    
    ⇒ soft target으로 하면 이 문제를 줄일 수 있지 않을까
    
    ⇒ 이 방식을 지금 explore 중
    

## 7 Relationship to Mixtures of Experts

- Expert 를 mix 하는 방법이 이외에도 있음
    
    ⇒ 병렬화 어려워서 잘 쓰이지 않음 
    
    - Gating neetwork 사용

## 8 Discussion

- specialist model을 single large net으로 distill 하는 건 아직 보여주지 못함