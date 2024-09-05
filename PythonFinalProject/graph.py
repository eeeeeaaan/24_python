# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# data= pd.read_csv("PythonFinalProject\mushrooms.csv")
# # 'cap_shape'와 독성 유무의 관계 시각화
# plt.figure(figsize=(12, 6))
# sns.countplot(x='cap-shape', hue='class', data=data, palette={'e': 'green', 'p': 'red'})
# plt.title('Cap Shape and Toxicity Relationship')
# plt.xlabel('Cap Shape')
# plt.ylabel('Count')
# plt.legend(title='Class', labels=['Edible', 'Poisonous'])
# plt.show()

# # 'odor'와 독성 유무의 관계 시각화
# plt.figure(figsize=(12, 6))
# sns.countplot(x='odor', hue='class', data=data, palette={'e': 'green', 'p': 'red'})
# plt.title('Odor and Toxicity Relationship')
# plt.xlabel('Odor')
# plt.ylabel('Count')
# plt.legend(title='Class', labels=['Edible', 'Poisonous'])
# plt.show()



import unittest
import pandas as pd
import torch
from torch.utils.data import DataLoader
from coding import CustomDataset, NeuralNetwork
from sklearn.model_selection import train_test_split

class TestMushroomClassification(unittest.TestCase):

    def setUp(self):
        # 데이터 로드 및 전처리
        data = pd.read_csv("PythonFinalProject/mushrooms.csv")
        self.features = data.drop(["class"], axis=1)
        self.labels = data.loc[:, "class"]
        self.features = pd.get_dummies(self.features)
        self.labels = pd.get_dummies(self.labels)
        self.feature_count = len(self.features.columns)
        self.label_count = len(self.labels.columns)

        # 데이터셋 분할
        x_train, x_valid, y_train, y_valid = train_test_split(self.features, self.labels, test_size=0.4, stratify=self.labels, random_state=42)
        x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5, stratify=y_valid, random_state=42)

        self.train_dataset = CustomDataset(x_train, y_train)
        self.valid_dataset = CustomDataset(x_valid, y_valid)
        self.test_dataset = CustomDataset(x_test, y_test)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=1)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1)

        self.model = NeuralNetwork(input_size=self.feature_count, output_size=self.label_count)

    def test_dataset_length(self):
        # 데이터셋 크기 테스트
        self.assertEqual(len(self.train_dataset), len(self.train_dataset.targets))
        self.assertEqual(len(self.valid_dataset), len(self.valid_dataset.targets))
        self.assertEqual(len(self.test_dataset), len(self.test_dataset.targets))

    def test_model_forward_pass(self):
        # 모델의 순전파 테스트
        sample_data, _ = self.train_dataset[0]
        sample_data = sample_data.unsqueeze(0)  # 배치 차원 추가
        output = self.model(sample_data)
        self.assertEqual(output.shape, (1, self.label_count))

    def test_training_step(self):
        # 모델 학습 테스트
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        criterion = torch.nn.BCELoss()

        self.model.train()
        for batch in self.train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            break  # 첫 배치에 대해서만 테스트

        self.assertTrue(loss.item() > 0)

    def test_validation_step(self):
        # 모델 검증 테스트
        criterion = torch.nn.BCELoss()

        self.model.eval()
        with torch.no_grad():
            for batch in self.valid_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                break  # 첫 배치에 대해서만 테스트

        self.assertTrue(loss.item() > 0)

if __name__ == "__main__":
    unittest.main()