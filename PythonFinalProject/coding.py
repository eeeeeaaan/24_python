#필요한 것들
import torch  # 딥러닝 프레임워크
import numpy as np  # 수치 연산 라이브러리
import pandas as pd  # 데이터 처리 라이브러리
import torch.nn as nn  # 신경망 모듈
import torch.optim as optim  # 최적화 알고리즘 모듈
import matplotlib.pyplot as plt  # 그래프 라이브러리
import torch.nn.functional as F  # 신경망 함수들
from torchinfo import summary  # 모델 요약 정보 출력
from torch.utils.data import Dataset, DataLoader  # 데이터셋과 데이터로더
from sklearn.model_selection import train_test_split  # 데이터셋 분할 도구


#커스텀 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        feature = self.features.iloc[index]
        target = self.targets.iloc[index]
        feature = torch.tensor(feature, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return feature, target

    def normalization(self):
        pass

#신경망 모델 정의    
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer1 = nn.Linear(in_features=input_size, out_features=input_size * 2)
        self.layer2 = nn.Linear(in_features=input_size * 2, out_features=input_size * 4)
        self.layer3 = nn.Linear(in_features=input_size * 4, out_features=input_size * 8)
        self.layer4 = nn.Linear(in_features=input_size * 8, out_features=input_size * 4)
        self.layer5 = nn.Linear(in_features=input_size * 4, out_features=input_size * 2)
        self.layer6 = nn.Linear(in_features=input_size * 2, out_features=input_size)
        self.layer7 = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = torch.sigmoid(self.layer7(x))
        return x
    
#데이터셋 준비하고 모델 학습 및 평가
#NAN 값이 있는지 확인하는 과정이 포함되어 실행시 칼럼별 NAN값 개수를 sum한 결과를 출력한다.
if __name__ == "__main__":
    data = pd.read_csv("PythonFinalProject\mushrooms.csv")

    features = data.drop(["class"], axis=1)
    labels = data.loc[:, "class"]

    print(features.isnull().sum())
    print(labels.isnull().sum())

    features = pd.get_dummies(features)
    feature_count = len(features.columns)

    print(labels.value_counts())

    labels = pd.get_dummies(labels)
    label_count = len(labels.columns)

    x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.4, stratify=labels, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5, stratify=y_valid, random_state=42)

    train_dataset = CustomDataset(x_train, y_train)
    valid_dataset = CustomDataset(x_valid, y_valid)
    test_dataset = CustomDataset(x_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = NeuralNetwork(input_size=feature_count, output_size=label_count)
    summary(model)
    print(f"### Model is ready!!! ###\n\n{model}")

    learning_rate = 0.00001
    num_epochs = 30
    model_save_path = "./model/mushroom_epoch30.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n### The device we are using is {device} ###")
    model = model.to(device)
    train_signal = input("Do you want to train? (y or n): ")

    if train_signal.lower() == "y":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = nn.BCELoss()
        loss_history = {"train": [], "val": []}

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in valid_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(valid_loader)
            loss_history["train"].append(avg_train_loss)
            loss_history["val"].append(avg_val_loss)

            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
            scheduler.step()

        epochs = np.arange(1, num_epochs + 1)
        plt.plot(epochs, loss_history["train"], 'r', label="Train Loss")
        plt.plot(epochs, loss_history["val"], 'b', label="Validation Loss")
        plt.legend()
        plt.title("Mushroom Deep Learning")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        torch.save(model.state_dict(), model_save_path)

    elif train_signal.lower() == "n":
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        results = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                correct = predicted.eq(targets.argmax(dim=1))
                results.extend(correct.cpu().numpy().tolist())

        accuracy = sum(results) / len(results) * 100
        print(f"Model Test Accuracy: {accuracy:.2f}%")
      
      
        
#각 특성이 독성 여부에 어떤 영향을 미치는지 
#seaborn 과 matplotlib.pyplot을 이용하여 분석 및 그래프로 나타내기
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 로드
file_path = "PythonFinalProject\mushrooms.csv"
data = pd.read_csv(file_path)

# 독성 여부를 수치로 변환 (edible: 0, poison: 1)
data['class'] = data['class'].map({'e': 0, 'p': 1})

# 피쳐 원핫인코딩
encoded_data = pd.get_dummies(data.drop('class', axis=1))
encoded_data['class'] = data['class']

# 그래프 설정
sns.set(style="ticks")

# 대표적으로 'cap-shape_x' 열을 사용하여 시각화
column = 'cap-shape_x'
plt.figure(figsize=(10, 6))

# 선형 회귀 모델 피팅
X = encoded_data[[column]].values
y = encoded_data['class'].values
model = LinearRegression()
model.fit(X, y)

# 회귀선 예측
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = model.predict(X_plot)

# 산점도 및 회귀선 그래프
sns.scatterplot(x=column, y='class', data=encoded_data, alpha=0.3)
plt.plot(X_plot, y_plot, color='red', linewidth=2)

plt.title(f'{column} vs Toxicity')
plt.xlabel(column)
plt.ylabel('Toxicity (0: Edible, 1: Poison)')

plt.show()
