'''
this file is documentating some code used for handling the transformer model 
and this code is incompletely, only record some important code,
doing this for protecting the client datat privacy, meanwhile, keep the source code that
used in this case

author: Ying ZHANG, on 19Feb,2025
'''

# to reselect the features in a df
df = df[['feature1', 'feature2']]

# drop NA value base on one sub feature
df.dropna(subset=['feature_name']).reset_index(drop=True)

#  create a new feature by calculating how many minutes left before the disturbance happens
# find the label with 1, and store the time value at that moment,
# calculate the minutes within each event, by group the event number, total over 500 hundres unique value

# create the target value feature, calculate the minutes left within each event.
# 1. store the timestamp with the label value is 1
# 2. if until the end, there is no more disturbance, take the end_time of this event to calculate the minutes.
# 3. calculate the minutes left before the disturbance happens within each event

def calculate_minutes_left(group):
    # Sort the group by time
    group = group.sort_values('time')
    
    # Find all rows with target=1 and their times
    disturbance_times = group[group['target'] == 1]['time'].tolist()
    
    # Create a list to store minutes left
    minutes_left = []
    
    for idx, row in group.iterrows():
        if row['target'] == 1:
            # For disturbance rows, set minutes left to 0
            minutes_left.append(0)
        else:
            # Find next disturbance time in this event
            next_disturbance = next((t for t in disturbance_times if t > row['time']), None)
            
            if next_disturbance:
                # Calculate time until next disturbance
                delta = (next_disturbance - row['time']).total_seconds() / 60
            else:
                # Calculate time until event end
                delta = (row['Event_Project completion time'] - row['time']).total_seconds() / 60
                
            minutes_left.append(round(delta, 2))
    
    group['minutes_left'] = minutes_left
    return group

# Apply the function to each event group
preprocessing_df = preprocessing_df.groupby('event_number', group_keys=False).apply(calculate_minutes_left)

## this is how to create a new feature when you processing a time series dataset

def create_features(df, target_col='feature1'):
    """Add time-based and difference features"""
    # 1. Cyclical time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)  # Assuming 0-6
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # 2. Difference features
    df['diff_1'] = df[target_col].diff(1)
    df['diff_24'] = df[target_col].diff(24)  # Daily difference
    
    # 3. Rolling statistics (example)
    df['rolling_6h_mean'] = df[target_col].rolling(6).mean()
    
    # 4. Handle NaNs from differencing
    df = df.dropna().reset_index(drop=True)
    
    return df

# Apply to your dataframe
df = create_features(preprocessing_df1)


# the following part is about scaler and encoder the data
target_col = 'feeature1'
X = preprocessing_df_head.drop(columns=[target_col])
y = preprocessing_df_head[target_col]

# 2. Split data first to prevent leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 3. Define different preprocessing for numerical and categorical features
numerical_cols = ['feature1', 'feature2', 'feature2']
categorical_cols = ['feature4', 'feature5']

# 4. Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', ce.TargetEncoder(cols=categorical_cols), categorical_cols)
], remainder='passthrough')

# 5. Fit transformers ONLY on training data
X_train_preprocessed = preprocessor.fit_transform(X_train, y_train)
X_test_preprocessed = preprocessor.transform(X_test)

# 6. Scale target separately (if needed)
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

## due to the structure of transfomer model, this is meant to create sequence of data
def create_sequences(data, targets, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
        labels.append(targets[i+window_size])
    return torch.FloatTensor(sequences), torch.FloatTensor(labels)

window_size = 10
X_train_seq, y_train_seq = create_sequences(X_train_preprocessed, y_train_scaled, window_size)
X_test_seq, y_test_seq = create_sequences(X_test_preprocessed, y_test_scaled, window_size)


# after create the sequence, the following part is creatre the dataloader for training and testing
# 8. Create DataLoaders
batch_size = 240
train_dataset = torch.utils.data.TensorDataset(X_train_seq, y_train_seq)
test_dataset = torch.utils.data.TensorDataset(X_test_seq, y_test_seq)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# both train and test setted shuffle as False, cause I want the data trained according to
# the sequence since they are time series dataset

# import the libaries for the model, the torch is used for transfomer, and the hugface is also quite popular
# you may try

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import category_encoders as ce
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Enhanced Positional Encoding with Dropout
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 2. Improved Transformer Architecture
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, num_layers=4, nhead=8, 
                 dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        
        # Input processing
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer layers with causal masking
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Temporal attention pooling
        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False))
        
        # Non-linear prediction head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, input_size)
        # print(f"Input shape after permute: {x.shape}")
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Causal mask
        mask = torch.triu(torch.ones(x.size(0), x.size(0)), diagonal=1).bool().to(x.device)
        
        # Transformer processing
        output = self.transformer(x, mask=mask)  # (seq_len, batch_size, d_model)
        
        # Temporal attention pooling
        attn_weights = F.softmax(self.temporal_attention(output), dim=0)
        context = torch.sum(attn_weights * output, dim=0)  # (batch_size, d_model)
        
        # Final prediction
        return self.fc(context).squeeze(-1)  # (batch_size,)

# 3. Head Selection Function (fixed for d_model)
def find_valid_nhead(d_model, max_heads=8):
    for n in range(max_heads, 0, -1):
        if d_model % n == 0:
            return n
    return 1

# 4. Model Initialization
input_size = X_train_preprocessed.shape[1]
d_model = 64
nhead = find_valid_nhead(d_model)

model = TimeSeriesTransformer(
    input_size=input_size,
    d_model=d_model,
    nhead=nhead,
    num_layers=4,
    dropout=0.2
).to(device)

# 5. Enhanced Training Configuration
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
best_mae = float('inf')
early_stopping_counter = 0
patience = 15

# 6. Modified Training Loop with Huber Loss
for epoch in range(200):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        outputs = model(batch_X)
        loss = F.smooth_l1_loss(outputs, batch_y)  # Huber loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_preds = []
    val_trues = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X.to(device))
            val_preds.extend(y_scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)))
            val_trues.extend(y_scaler.inverse_transform(batch_y.numpy().reshape(-1, 1)))
    
    val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_trues)))
    mae = mean_absolute_error(val_trues, val_preds)
    r2 = r2_score(val_trues, val_preds)
    scheduler.step(val_mae)
    
    # Early stopping check
    if val_mae < best_mae:
        best_mae = val_mae
        torch.save(model.state_dict(), 'best_model.pth')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    print(f"Epoch {epoch+1:03d} | Train Loss: {total_loss/len(train_loader):.4f} | Val MAE: {val_mae:.4f}| R²: {r2:.4f}")
    
    if early_stopping_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# the best model is saved during training

# this model is for handling the non-linear regression

# and you can check the feature importance by use the following code
import matplotlib.pyplot as plt
import seaborn as sns

# After training
with torch.no_grad():
    projection_weights = model.input_proj[0].weight.cpu().numpy()
    feature_importance = np.abs(projection_weights).mean(axis=0)

plt.barh(X.columns, feature_importance)
plt.title("Input Feature Importance")

## this is how you store the prediction and actual value into a df
val_trues_np = np.array([item[0] for item in val_trues])
val_preds_np = np.array([item[0] for item in val_preds])

df = pd.DataFrame({'val_predicts':val_preds_np,
                    'val_trues':val_trues_np})