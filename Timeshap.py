import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy

from timeshap.wrappers import TorchModelWrapper
from timeshap.utils import calc_avg_event, calc_avg_sequence, get_avg_score_with_avg_event
from timeshap.explainer import local_report, global_report
from IPython.display import display, HTML

# Setup
np.random.seed(42)
warnings.filterwarnings('ignore')
display(HTML("<style>.container { width:100% !important; }</style>"))

# Load data
data_directories = next(os.walk("timeshap/notebooks/ARem/AReM"))[1]

all_csvs = []
for folder in data_directories:
    if folder in ['bending1', 'bending2']:
        continue
    folder_csvs = next(os.walk(f"timeshap/notebooks/ARem/AReM/{folder}"))[2]
    for data_csv in folder_csvs:
        if data_csv == 'dataset8.csv' and folder == 'sitting':
            continue
        loaded_data = pd.read_csv(f"timeshap/notebooks/ARem/AReM/{folder}/{data_csv}", skiprows=4)
        print(f"{folder}/{data_csv} ------ {loaded_data.shape}")

        csv_id = re.findall(r'\d+', data_csv)[0]
        loaded_data['id'] = csv_id
        loaded_data['all_id'] = f"{folder}_{csv_id}"
        loaded_data['activity'] = folder
        all_csvs.append(loaded_data)

all_data = pd.concat(all_csvs)

# Feature setup
raw_model_features = ['avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23']
all_data.columns = ['timestamp', 'avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23', 'id',
                    'all_id', 'activity']

# Train/test split
ids_for_test = np.random.choice(all_data['id'].unique(), size=4, replace=False)
d_train = all_data[~all_data['id'].isin(ids_for_test)]
d_test = all_data[all_data['id'].isin(ids_for_test)]

# Normalization
class NumericalNormalizer:
    def __init__(self, fields: list):
        self.metrics = {}
        self.fields = fields

    def fit(self, df: pd.DataFrame) -> list:
        means = df[self.fields].mean()
        std = df[self.fields].std()
        for field in self.fields:
            self.metrics[field] = {'mean': means[field], 'std': std[field]}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for field in self.fields:
            f_mean = self.metrics[field]['mean']
            f_stddev = self.metrics[field]['std']
            df[field] = df[field].apply(lambda x: f_mean - 3 * f_stddev if x < f_mean - 3 * f_stddev else x)
            df[field] = df[field].apply(lambda x: f_mean + 3 * f_stddev if x > f_mean + 3 * f_stddev else x)
            if f_stddev > 1e-5:
                df[f'p_{field}_normalized'] = df[field].apply(lambda x: ((x - f_mean)/f_stddev))
            else:
                df[f'p_{field}_normalized'] = df[field].apply(lambda x: x * 0)
        return df

normalizor = NumericalNormalizer(raw_model_features)
normalizor.fit(d_train)
d_train_normalized = normalizor.transform(d_train)
d_test_normalized = normalizor.transform(d_test)

model_features = [f"p_{x}_normalized" for x in raw_model_features]
time_feat = 'timestamp'
label_feat = 'activity'
sequence_id_feat = 'all_id'

plot_feats = {
    'p_avg_rss12_normalized': "Mean Chest <-> Right Ankle",
    'p_var_rss12_normalized': "STD Chest <-> Right Ankle",
    'p_avg_rss13_normalized': "Mean Chest <-> Left Ankle",
    'p_var_rss13_normalized': "STD Chest <-> Left Ankle",
    'p_avg_rss23_normalized': "Mean Right Ankle <-> Left Ankle",
    'p_var_rss23_normalized': "STD Right Ankle <-> Left Ankle",
}

# Label processing
chosen_activity = 'cycling'
d_train_normalized['label'] = d_train_normalized['activity'].apply(lambda x: int(x == chosen_activity))
d_test_normalized['label'] = d_test_normalized['activity'].apply(lambda x: int(x == chosen_activity))

# Convert to tensor
def df_to_Tensor(df, model_feats, label_feat, group_by_feat, timestamp_Feat):
    sequence_length = len(df[timestamp_Feat].unique())
    data_tensor = np.zeros((len(df[group_by_feat].unique()), sequence_length, len(model_feats)))
    labels_tensor = np.zeros((len(df[group_by_feat].unique()), 1))

    for i, name in enumerate(df[group_by_feat].unique()):
        name_data = df[df[group_by_feat] == name].sort_values(timestamp_Feat)
        data_tensor[i, :, :] = name_data[model_feats].values
        labels_tensor[i, :] = name_data[label_feat].values[0]

    return torch.FloatTensor(data_tensor), torch.FloatTensor(labels_tensor)

train_data, train_labels = df_to_Tensor(d_train_normalized, model_features, 'label', sequence_id_feat, time_feat)
test_data, test_labels = df_to_Tensor(d_test_normalized, model_features, 'label', sequence_id_feat, time_feat)

# Model
def build_model(input_size):
    class ExplainedRNN(nn.Module):
        def __init__(self, input_size: int, cfg: dict):
            super().__init__()
            self.hidden_dim = cfg.get('hidden_dim', 32)
            torch.manual_seed(cfg.get('random_seed', 42))
            self.recurrent_block = nn.GRU(input_size, self.hidden_dim, batch_first=True, num_layers=2)
            self.classifier_block = nn.Linear(self.hidden_dim, 1)
            self.output_activation_func = nn.Sigmoid()

        def forward(self, x, hidden_states=None):
            if hidden_states is not None:
                output, hidden = self.recurrent_block(x, hidden_states)
            else:
                output, hidden = self.recurrent_block(x)
            assert torch.equal(output[:, -1, :], hidden[-1, :, :])
            y = self.output_activation_func(self.classifier_block(hidden[-1, :, :]))
            return y, hidden

    return ExplainedRNN(input_size, {})

model = build_model(len(model_features))
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
EPOCHS = 5
for epoch in tqdm.trange(EPOCHS):
    y_pred, _ = model(copy.deepcopy(train_data))
    train_loss = loss_function(y_pred, copy.deepcopy(train_labels))

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    with torch.no_grad():
        test_preds, _ = model(copy.deepcopy(test_data))
        test_loss = loss_function(test_preds, copy.deepcopy(test_labels))
        print(f"Train loss: {train_loss.item()} --- Test loss {test_loss.item()} ")

# TimeSHAP setup
model_wrapped = TorchModelWrapper(model)
f_hs = lambda x, y=None: model_wrapped.predict_last_hs(x, y)

average_event = calc_avg_event(d_train_normalized, numerical_feats=model_features, categorical_feats=[])
average_sequence = calc_avg_sequence(d_train_normalized, numerical_feats=model_features, categorical_feats=[], model_features=model_features, entity_col=sequence_id_feat)
avg_score_over_len = get_avg_score_with_avg_event(f_hs, average_event, top=480)

# Local explanation
positive_sequence_id = f"cycling_{np.random.choice(ids_for_test)}"
pos_x_pd = d_test_normalized[d_test_normalized['all_id'] == positive_sequence_id]
pos_x_data = np.expand_dims(pos_x_pd[model_features].to_numpy().copy(), axis=0)

pruning_dict = {'tol': 0.025}
event_dict = {'rs': 42, 'nsamples': 32000}
feature_dict = {'rs': 42, 'nsamples': 32000, 'feature_names': model_features, 'plot_features': plot_feats}
cell_dict = {'rs': 42, 'nsamples': 32000, 'top_x_feats': 2, 'top_x_events': 2}

# local_report実行
report = local_report(
    f_hs, pos_x_data, pruning_dict, event_dict, feature_dict, cell_dict,
    average_event, entity_uuid=positive_sequence_id, entity_col='all_id'
)

report.save('local_report.html')  # Altairのsaveメソッド
