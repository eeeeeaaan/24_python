# Python Final Project 

## 데이터셋
- **출처:** Kaggle의 Mushroom Dataset
- **전처리:** 결측값 처리, 범주형 데이터의 원핫 인코딩~

## 파일 구조
- `mushroom_classification.py`: 딥러닝 모델을 정의하고 학습 및 평가하는 메인 스크립트
- `mushroom_analysis.py`: 버섯 특징과 독성 간의 상관관계를 분석하고 시각화하는 스크립트
- `README.md`: 프로젝트 실행 방법 및 설명을 포함한 파일

## 필수 라이브러리
프로젝트를 실행하기 위해 필요한 라이브러리:
- torch
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- torchinfo

## 실행 방법

1. **환경 설정:**
    - Python 3.7 이상 설치
    - 필요한 라이브러리 설치:
    ```bash
    pip install torch numpy pandas matplotlib seaborn scikit-learn torchinfo
    ```

2. **데이터 다운로드:**
    - Kaggle에서 Mushroom Dataset을 다운로드 받아 `PythonFinalProject` 폴더에 저장
    - 데이터셋 파일 이름은 `mushrooms.csv`로 지정

3. **모델 학습 및 평가:**
    - `coding.py` 파일을 실행하여 모델을 학습시키고 평가
    ```bash
    python coding.py
    ```
    - 학습 도중 `Do you want to train? (y or n):`라는 질문에 `y`를 입력하면 모델이 학습을 시작 `n`을 입력하면 저장된 모델을 로드하여 테스트 데이터로 평가

4. **특징과 독성 간의 상관관계 분석:**
    - `mushroom_analysis.py` 파일을 실행하여 버섯 특징과 독성 간의 상관관계를 분석하고 시각화
    ```bash
    python graph.py
    ```

## 결과 시각화
모델 학습 및 평가 결과는 Matplotlib을 사용하여 시각화됩니다. 학습 과정에서의 손실 그래프와 특정 특징과 독성 간의 상관관계 그래프를 확인 가능함

## 기여자
- **안현정:** 코드 및 테스트 담당
- **이서희:** 코드, 보고서 담당

