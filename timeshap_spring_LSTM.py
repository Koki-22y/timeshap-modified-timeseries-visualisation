import warnings
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy

from timeshap.src.timeshap.wrappers import TorchModelWrapper
from timeshap.src.timeshap.utils import calc_avg_event, calc_avg_sequence, get_avg_score_with_avg_event
from timeshap.src.timeshap.explainer import local_report
from IPython.display import display, HTML

# Setup
np.random.seed(42)
warnings.filterwarnings('ignore')
display(HTML("<style>.container { width:100% !important; }</style>"))

# データ読み込み
data = pd.read_csv('csv/combined_15m_2024_spring.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])  # タイムスタンプ列があると仮定
data['id'] = 1  # 単一シーケンスを仮定（必要に応じて分割）
data['all_id'] = 'combined_1'  # ユニークID

# Feature setup
raw_model_features = ['combined']
data = data[['timestamp', 'combined', 'id', 'all_id']]  # 必要な列のみ

# Train/test split
sequence_length = 24
total_length = len(data)
train_size = int(0.8 * (total_length - sequence_length))
d_train = data[:train_size]
d_test = data[train_size:]

# Normalization
class NumericalNormalizer:
    def __init__(self, fields: list):
        self.metrics = {}
        self.fields = fields
    def fit(self, df: pd.DataFrame):
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
            df[f'p_{field}_normalized'] = df[field].apply(lambda x: ((x - f_mean)/f_stddev) if f_stddev > 1e-5 else 0)
        return df

normalizor = NumericalNormalizer(raw_model_features)
normalizor.fit(d_train)
d_train_normalized = normalizor.transform(d_train)
d_test_normalized = normalizor.transform(d_test)

model_features = [f"p_{x}_normalized" for x in raw_model_features]
time_feat = 'timestamp'
sequence_id_feat = 'all_id'

plot_feats = {'p_combined_normalized': 'Combined Value'}

# Data preparation for next-step prediction
def df_to_Tensor(df, model_feats, group_by_feat, timestamp_Feat):
    data_tensor = []
    labels_tensor = []
    for name in df[group_by_feat].unique():
        name_data = df[df[group_by_feat] == name].sort_values(timestamp_Feat)[model_feats].values
        for i in range(len(name_data) - sequence_length):
            data_tensor.append(name_data[i:i+sequence_length])
            labels_tensor.append(name_data[i+sequence_length])
    return torch.FloatTensor(np.array(data_tensor)), torch.FloatTensor(np.array(labels_tensor))

train_data, train_labels = df_to_Tensor(d_train_normalized, model_features, sequence_id_feat, time_feat)
test_data, test_labels = df_to_Tensor(d_test_normalized, model_features, sequence_id_feat, time_feat)

# Model
def build_model(input_size):
    class ExplainedLSTM(nn.Module):
        def __init__(self, input_size: int, cfg: dict):
            super().__init__()
            self.hidden_dim = cfg.get('hidden_dim', 32)
            torch.manual_seed(cfg.get('random_seed', 42))
            self.lstm = nn.LSTM(input_size, self.hidden_dim, batch_first=True, num_layers=2)
            self.linear = nn.Linear(self.hidden_dim, input_size)
        def forward(self, x, hidden_states=None):
            if hidden_states is not None:
                output, (hn, cn) = self.lstm(x, hidden_states)
            else:
                output, (hn, cn) = self.lstm(x)
            y = self.linear(output[:, -1, :])
            return y, (hn, cn)
    return ExplainedLSTM(input_size, {})

model = build_model(len(model_features))
loss_function = nn.MSELoss()
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

# モデルの保存
torch.save(model.state_dict(), 'LSTM_model.pth')

# TimeSHAP setup
model_wrapped = TorchModelWrapper(model)
f_hs = lambda x, y=None: (np.mean(model_wrapped.predict_last_hs(x, y)[0], axis=-1, keepdims=True), model_wrapped.predict_last_hs(x, y)[1])

# 数値特徴量とentity_colを選択
d_train_numeric = d_train_normalized[model_features + [sequence_id_feat]]
average_event = calc_avg_event(d_train_numeric, numerical_feats=model_features, categorical_feats=[])
average_sequence = calc_avg_sequence(d_train_numeric, numerical_feats=model_features, categorical_feats=[], model_features=model_features, entity_col=sequence_id_feat)
avg_score_over_len = get_avg_score_with_avg_event(f_hs, average_event, top=672)

# Local explanation
positive_sequence_id = 'combined_1'
num_test = 50
pos_x_pd = d_test_normalized.sort_values(time_feat)[num_test:num_test + sequence_length]
pos_x_data = np.expand_dims(pos_x_pd[model_features].to_numpy().copy(), axis=0)

pruning_dict = {'tol': 0.025}
event_dict = {'rs': 42, 'nsamples': 32000}
feature_dict = {'rs': 42, 'nsamples': 32000, 'feature_names': model_features, 'plot_features': plot_feats}
cell_dict = {'rs': 42, 'nsamples': 32000, 'top_x_feats': 1, 'top_x_events': 2}

# Local report
report = local_report(
    f_hs, pos_x_data, pruning_dict, event_dict, feature_dict, cell_dict,
    average_event, entity_uuid=positive_sequence_id, entity_col='all_id',
)

report.save('local_report_spring_LSTM.html')

def plot_event_shap_dot(event_data, observed_series=None):
    import altair as alt
    import pandas as pd
    import numpy as np

    event_data = event_data.rename(columns={'Feature': 'event_name', 'Shapley Value': 'shap_value'})

    # 🔧 SHAP値の順番を過去→現在に揃える
    event_data = event_data.iloc[::-1].reset_index(drop=True)

    if 'time_step' not in event_data.columns:
        event_data['time_step'] = range(len(event_data))

    event_data['event_index'] = list(range(len(event_data)))  # 過去→現在

    if observed_series is not None:
        observed_series = np.ravel(observed_series)

        observed_df = pd.DataFrame({
            'value': observed_series,
            'event_index': list(range(len(observed_series)))
        })

        event_data['observed_value'] = observed_series

        line_chart = alt.Chart(observed_df).mark_line(
            color='black', strokeWidth=2
        ).encode(
            x=alt.X('event_index:O', title='Time Step (Past → Recent)', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('value:Q', title='Observed Value'),
            tooltip=['value']
        )

        dot_chart = alt.Chart(event_data).mark_circle(size=150, opacity=1.0).encode(
            x=alt.X('event_index:O'),
            y=alt.Y('observed_value:Q'),
            color=alt.Color('shap_value:Q', scale=alt.Scale(scheme='Plasma')),
            tooltip=['event_name', 'shap_value', 'observed_value']
        )

        chart = alt.layer(line_chart, dot_chart).resolve_scale(y='shared')

    else:
        raise ValueError("observed_series is required to align dots with line")

    return chart.properties(
        width=600,
        height=300,
        title='Observed Data with SHAP-colored Dots (Past → Recent)'
    ).interactive()


def custom_heatmap_report(
        f, data, event_dict, baseline=None, model_features=None,
        entity_col=None, entity_uuid=None, time_col=None, verbose=False
):
    from timeshap.src.timeshap.explainer import validate_local_input, local_event
    import numpy as np
    import pandas as pd

    validate_local_input(f, data, None, event_dict, {}, None, baseline, model_features, entity_col, time_col,
                         entity_uuid)

    if isinstance(data, pd.DataFrame):
        if time_col is not None:
            data[time_col] = data[[time_col]].apply(pd.to_numeric, errors='coerce')
            data = data.sort_values(time_col)
        if model_features is not None:
            data = data[model_features]
        data = np.expand_dims(data.to_numpy().copy(), axis=0).astype(float)

    event_data = local_event(f, data, event_dict, entity_uuid, entity_col, baseline, pruned_idx=0)

    # デバッグ用出力
    print(event_data.head())
    print(event_data.dtypes)

    # observed_series を渡すように修正
    observed_series = data[0][:len(event_data)]  # 必要なら変換

    dot_plot = plot_event_shap_dot(event_data, observed_series=observed_series)
    return dot_plot


# 呼び出し
heatmap = custom_heatmap_report(
    f_hs,
    pos_x_pd,
    event_dict,
    baseline=average_event,
    model_features=model_features,
    entity_col='all_id',
    entity_uuid=positive_sequence_id,
    time_col='timestamp',
    verbose=False
)

# 折線に使う観測データを準備（非正規化ではなく正規化された値）
observed_values = observed_series = pos_x_pd[model_features[0]].values.ravel()

# 修正済み plot 関数を再呼び出し（イベントデータを heatmap から取得するか、再取得）
from timeshap.src.timeshap.explainer import local_event
event_data = local_event(f_hs, np.expand_dims(pos_x_pd[model_features].values, axis=0), event_dict,
                         entity_uuid=positive_sequence_id, entity_col='all_id',
                         baseline=average_event, pruned_idx=0)

final_chart = plot_event_shap_dot(event_data, observed_series=observed_values)
final_chart.save('dotplot_with_observed_LSTM.html')