import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
import numpy as np

hf = pd.read_csv("heart_failure_clinical_records_dataset.csv")
hf["CPK"] = hf["creatinine_phosphokinase"]
hf = hf.drop("creatinine_phosphokinase", axis=1)

numerical_features = ["age", "CPK", "ejection_fraction", "serum_creatinine", "serum_sodium"]
categorical_features = ["anaemia", "sex"]

hf_norm = hf.copy()
scaler = StandardScaler()
hf_norm[numerical_features] = scaler.fit_transform(hf_norm[numerical_features])

all_features = ['anaemia', 'sex', 'age', 'CPK', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']

train_ratio = 0.75
val_ratio = 0.25

ho_train_df, ho_val_df = train_test_split(hf_norm, train_size=train_ratio, random_state=42)
unnorm_ho_train_df, unnorm_ho_val_df = train_test_split(hf, train_size=train_ratio, random_state=42)

n_to_sample = len(ho_train_df[ho_train_df.DEATH_EVENT == 0]) - len(ho_train_df[ho_train_df.DEATH_EVENT == 1])
new_samples = ho_train_df[ho_train_df.DEATH_EVENT == 1].sample(n_to_sample, replace=True, random_state=42)
ho_train_df_rs = pd.concat([ho_train_df, new_samples])

new_samples = unnorm_ho_train_df[unnorm_ho_train_df.DEATH_EVENT == 1].sample(n_to_sample, replace=True, random_state=42)
unnorm_ho_train_df_rs = pd.concat([unnorm_ho_train_df, new_samples])

target = 'DEATH_EVENT'

def create_edges(X, k=5):
    distances = euclidean_distances(X)
    knn_graph = np.argsort(distances, axis=1)[:, 1:k + 1]
    edge_index = []
    for i, neighbors in enumerate(knn_graph):
        for neighbor in neighbors:
            edge_index.append([i, neighbor])
            edge_index.append([neighbor, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

X_resampled = ho_train_df_rs[all_features].values
y_resampled = ho_train_df_rs[target].values
edge_index_rs = create_edges(X_resampled, k=5)

X_original = ho_train_df[all_features].values
y_original = ho_train_df[target].values
edge_index_original = create_edges(X_original, k=5)

X_val = ho_val_df[all_features].values
y_val = ho_val_df[target].values
edge_index_val = create_edges(X_val, k=5)

x_rs = torch.tensor(X_resampled, dtype=torch.float)
y_rs = torch.tensor(y_resampled, dtype=torch.long)
x_original = torch.tensor(X_original, dtype=torch.float)
y_original = torch.tensor(y_original, dtype=torch.long)
x_val = torch.tensor(X_val, dtype=torch.float)
y_val = torch.tensor(y_val, dtype=torch.long)

data_rs = Data(x=x_rs, edge_index=edge_index_rs, y=y_rs)
data_original = Data(x=x_original, edge_index=edge_index_original, y=y_original)
data_val = Data(x=x_val, edge_index=edge_index_val, y=y_val)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=16, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data_rs.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()

def train(loader):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.y.size(0)
    return correct / total

train_loader_rs = DataLoader([data_rs], batch_size=1, shuffle=True)
train_loader_original = DataLoader([data_original], batch_size=1, shuffle=True)
val_loader = DataLoader([data_val], batch_size=1, shuffle=False)

train_accs_rs, val_accs_rs = [], []
for epoch in range(1, 201):
    train(train_loader_rs)
    train_acc = test(train_loader_rs)
    val_acc = test(val_loader)
    train_accs_rs.append(train_acc)
    val_accs_rs.append(val_acc)

def evaluate(loader):
    model.eval()
    y_true = []
    y_pred = []
    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        y_true.extend(data.y.tolist())
        y_pred.extend(pred.tolist())
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, rec, pre, f1

rs_acc, rs_rec, rs_pre, rs_f1 = evaluate(val_loader)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_accs_original, val_accs_original = [], []
for epoch in range(1, 201):
    train(train_loader_original)
    train_acc = test(train_loader_original)
    val_acc = test(val_loader)
    train_accs_original.append(train_acc)
    val_accs_original.append(val_acc)

no_rs_acc, no_rs_rec, no_rs_pre, no_rs_f1 = evaluate(val_loader)

st.title("Heart Failure Prediction App")
st.write("Enter the patient's clinical features:")

anaemia = st.selectbox('Anaemia', [0, 1])
sex = st.selectbox('Sex (Woman: 0, Man: 1)', [0, 1])
age = st.number_input('Age (years)', min_value=0, max_value=120, step=1)
cpk = st.number_input('CPK (mcg/L)', min_value=0, step=1)
ejection_fraction = st.number_input('Ejection Fraction (%)', min_value=0, max_value=100, step=1)
serum_creatinine = st.number_input('Serum Creatinine (mg/dL)', min_value=0.0, step=0.1)
serum_sodium = st.number_input('Serum Sodium (mEq/L)', min_value=0, step=1)

input_data = pd.DataFrame([[anaemia, sex, age, cpk, ejection_fraction, serum_creatinine, serum_sodium]], 
                          columns=all_features)
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

edge_index = create_edges(input_data.values, k=5)
if edge_index.numel() == 0: 
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
input_tensor = torch.tensor(input_data.values, dtype=torch.float)
edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

data = Data(x=input_tensor, edge_index=edge_index_tensor)

if st.button('Predict'):
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1).item()
        if pred == 1:
            st.write("The model predicts that the patient is at risk of a death event.")
        else:
            st.write("The model predicts that the patient is not at risk of a death event.")