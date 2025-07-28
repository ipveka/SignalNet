# Time Series Forecasting Model Design

## Overview

This design implements proper time series forecasting practices to fix the critical prediction issues. The current model violates fundamental time series principles: no normalization, inappropriate architecture, and poor feature engineering. This redesign follows established neural network time series forecasting best practices.

## Architecture

### Time Series Forecasting Pipeline

```
Raw Time Series → Per-Series Normalization → Time Series Features → Causal Transformer → Autoregressive Prediction → Denormalization
```

### Key Design Principles

1. **Temporal Causality**: No future information leakage
2. **Per-Series Normalization**: Proper scaling for each time series
3. **Cyclical Time Features**: Sin/cos encoding for periodic patterns
4. **Autoregressive Generation**: Step-by-step prediction
5. **Temporal Data Splits**: No random shuffling

## Components and Interfaces

### 1. TimeSeriesTransformer (Redesigned)

```python
class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        model_dim: int = 64,  # Appropriate size for time series
        num_heads: int = 4,   # Fewer heads, more focused attention
        num_layers: int = 3,  # Moderate depth
        output_dim: int = 1,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        # Causal (masked) self-attention for time series
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(model_dim, max_seq_len)
        
        # Causal transformer layers
        self.transformer_layers = nn.ModuleList([
            CausalTransformerLayer(model_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, output_dim)
        
    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        x = self.embedding(x) + self.pos_encoding(x.size(1))
        
        for layer in self.transformer_layers:
            x = layer(x, mask)
            
        x = self.layer_norm(x)
        return self.output_proj(x)
```

### 2. TimeSeriesNormalizer (Per-Series)

```python
class TimeSeriesNormalizer:
    def __init__(self, method: str = 'zscore'):
        self.method = method
        self.series_stats = {}  # Store stats per series
        
    def fit(self, data: pd.DataFrame, series_col: str = 'series_id', value_col: str = 'value'):
        """Fit normalizer per series to avoid data leakage"""
        for series_id in data[series_col].unique():
            series_data = data[data[series_col] == series_id][value_col].values
            if self.method == 'zscore':
                mean = np.mean(series_data)
                std = np.std(series_data)
                self.series_stats[series_id] = {'mean': mean, 'std': std}
                
    def transform(self, values: np.ndarray, series_id: int) -> np.ndarray:
        """Transform values for specific series"""
        stats = self.series_stats[series_id]
        if self.method == 'zscore':
            return (values - stats['mean']) / (stats['std'] + 1e-8)
            
    def inverse_transform(self, values: np.ndarray, series_id: int) -> np.ndarray:
        """Inverse transform to original scale"""
        stats = self.series_stats[series_id]
        if self.method == 'zscore':
            return values * stats['std'] + stats['mean']
```

### 3. TimeSeriesFeatureExtractor

```python
class TimeSeriesFeatureExtractor:
    def __init__(self):
        pass
        
    def extract_features(self, timestamps: pd.Series, values: np.ndarray) -> np.ndarray:
        """Extract time series specific features"""
        features = []
        
        # Cyclical time features (sin/cos encoding)
        hour_sin = np.sin(2 * np.pi * timestamps.dt.hour / 24)
        hour_cos = np.cos(2 * np.pi * timestamps.dt.hour / 24)
        dow_sin = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
        dow_cos = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
        
        # Lag features (previous values)
        lag_1 = np.roll(values, 1)
        lag_24 = np.roll(values, 24)  # Daily lag for hourly data
        
        # Rolling statistics
        rolling_mean = pd.Series(values).rolling(window=24, min_periods=1).mean().values
        rolling_std = pd.Series(values).rolling(window=24, min_periods=1).std().fillna(0).values
        
        features = np.column_stack([
            hour_sin, hour_cos, dow_sin, dow_cos,
            lag_1, lag_24, rolling_mean, rolling_std
        ])
        
        return features
```

### 4. TimeSeriesTrainer

```python
class TimeSeriesTrainer:
    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        self.normalizer = TimeSeriesNormalizer()
        
    def temporal_train_val_test_split(self, data: pd.DataFrame, 
                                    train_ratio: float = 0.6,
                                    val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Temporal split - no random shuffling"""
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return (
            data.iloc[:train_end],
            data.iloc[train_end:val_end], 
            data.iloc[val_end:]
        )
        
    def train_autoregressive(self, model, train_loader, val_loader):
        """Train with autoregressive loss"""
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay)
        
        for epoch in range(self.config.epochs):
            model.train()
            for batch in train_loader:
                context, target = batch
                
                # Autoregressive training
                predictions = []
                current_context = context
                
                for step in range(target.size(1)):
                    pred = model(current_context)[:, -1:]  # Last prediction
                    predictions.append(pred)
                    
                    # Use ground truth for next step (teacher forcing with decay)
                    if np.random.random() < self.config.teacher_forcing_ratio:
                        next_input = target[:, step:step+1]
                    else:
                        next_input = pred
                        
                    current_context = torch.cat([current_context[:, 1:], next_input], dim=1)
                
                predictions = torch.cat(predictions, dim=1)
                loss = F.mse_loss(predictions, target)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()
```

## Data Models

### TimeSeriesConfig

```python
@dataclass
class TimeSeriesConfig:
    # Model architecture
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # Time series specific
    context_length: int = 24
    prediction_length: int = 6
    teacher_forcing_ratio: float = 0.5
    normalization_method: str = 'zscore'
```

## Critical Fixes for Current Issues

### 1. **Proper Normalization** (Most Critical)
```python
# Current Issue: Raw values (-1 to 2) → Model predicts ~0
# Fix: Per-series z-score normalization
normalizer = TimeSeriesNormalizer('zscore')
normalizer.fit(train_data)
normalized_values = normalizer.transform(values, series_id)
```

### 2. **Causal Architecture**
```python
# Current Issue: Encoder-decoder allows future information leakage
# Fix: Causal (masked) self-attention only
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

### 3. **Proper Time Features**
```python
# Current Issue: Linear normalization of cyclical features
# Fix: Sin/cos encoding for cyclical patterns
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
```

### 4. **Temporal Data Splitting**
```python
# Current Issue: Random train/test split causes data leakage
# Fix: Temporal split
train_data = data[:int(0.6 * len(data))]
val_data = data[int(0.6 * len(data)):int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]
```

### 5. **Autoregressive Prediction**
```python
# Current Issue: Teacher forcing during inference
# Fix: True autoregressive generation
def predict_autoregressive(self, context, steps):
    predictions = []
    current_context = context
    
    for _ in range(steps):
        pred = self.model(current_context)[:, -1:]
        predictions.append(pred)
        current_context = torch.cat([current_context[:, 1:], pred], dim=1)
    
    return torch.cat(predictions, dim=1)
```

## Expected Results After Implementation

1. **Predictions in correct range**: Values should be ~1-2, not ~0
2. **Reduced overfitting**: Test loss should be closer to train loss
3. **Meaningful temporal patterns**: Model should learn daily/weekly cycles
4. **Better generalization**: Performance on unseen future data should improve significantly