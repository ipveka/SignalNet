# Time Features and Feature Engineering

SignalNet automatically detects data frequency and extracts only relevant temporal features, ensuring optimal model performance by excluding meaningless features for different data granularities.

## 🕒 Frequency-Aware Feature Generation

SignalNet intelligently selects features based on detected data frequency:

```python
def _detect_frequency(self, timestamps: pd.Series) -> str:
    # Automatically detects: 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'
    time_diffs = timestamps.diff().dropna()
    most_common_diff = time_diffs.mode().iloc[0]
    total_seconds = most_common_diff.total_seconds()
    
    if total_seconds <= 60: return 'minute'
    elif total_seconds <= 3600: return 'hour'
    elif total_seconds <= 86400: return 'day'
    elif total_seconds <= 604800: return 'week'
    elif total_seconds <= 2592000: return 'month'
    elif total_seconds <= 7776000: return 'quarter'
    else: return 'year'
```

## 📊 Frequency-Specific Feature Selection

### **Always Included Features** (All Frequencies)
- **Day of Week** (Normalized: 0-1): Weekly patterns and cycles
- **Month** (Normalized: 0-1): Seasonal patterns and monthly trends  
- **Is Weekend** (Binary: 0/1): Weekend vs. weekday patterns

### **High-Frequency Data** (Minutes/Hours)
```python
# Minute frequency data
if frequency == 'minute':
    features = [
        'day_of_week (normalized)',
        'month (normalized)', 
        'is_weekend',
        'hour_of_day (normalized)',      # ✅ Relevant
        'minute_of_hour (normalized)',   # ✅ Relevant
        'is_business_hour'               # ✅ Relevant
    ]

# Hour frequency data  
if frequency == 'hour':
    features = [
        'day_of_week (normalized)',
        'month (normalized)',
        'is_weekend', 
        'hour_of_day (normalized)',      # ✅ Relevant
        'is_business_hour'               # ✅ Relevant
        # ❌ minute_of_hour excluded (always 0)
    ]
```

### **Medium-Frequency Data** (Days/Weeks)
```python
# Daily frequency data
if frequency == 'day':
    features = [
        'day_of_week (normalized)',
        'month (normalized)',
        'is_weekend',
        'day_of_month (normalized)',     # ✅ Relevant
        'is_business_hour',              # ✅ Relevant
        'is_month_end'                   # ✅ Relevant
        # ❌ hour_of_day excluded (always 0)
        # ❌ minute_of_hour excluded (always 0)
    ]

# Weekly frequency data
if frequency == 'week':
    features = [
        'day_of_week (normalized)',
        'month (normalized)', 
        'is_weekend',
        'day_of_month (normalized)',     # ✅ Relevant
        'is_month_end'                   # ✅ Relevant
        # ❌ hour_of_day excluded (meaningless for weekly data)
        # ❌ minute_of_hour excluded (meaningless for weekly data)
        # ❌ is_business_hour excluded (not relevant for weekly data)
    ]
```

### **Low-Frequency Data** (Months/Quarters/Years)
```python
# Monthly frequency data
if frequency == 'month':
    features = [
        'day_of_week (normalized)',
        'month (normalized)',
        'is_weekend',
        'day_of_month (normalized)',
        'quarter (normalized)',          # ✅ Relevant
        'day_of_year (normalized)',      # ✅ Relevant
        'is_month_end'                   # ✅ Relevant
        # ❌ All hourly/minute features excluded
    ]
```

## 🎯 Why This Matters

### **Before (Fixed Features)**
```python
# Weekly data with irrelevant features
weekly_data_features = [
    'day_of_week',      # ✅ Relevant
    'hour_of_day',      # ❌ Always 0 (meaningless)
    'minute_of_hour',   # ❌ Always 0 (meaningless) 
    'month',            # ✅ Relevant
    'day_of_month',     # ✅ Relevant
    'is_weekend'        # ✅ Relevant
]
# Result: 50% irrelevant features, wasted model capacity
```

### **After (Frequency-Aware)**
```python
# Weekly data with only relevant features
weekly_data_features = [
    'day_of_week',      # ✅ Relevant
    'month',            # ✅ Relevant
    'is_weekend',       # ✅ Relevant
    'day_of_month',     # ✅ Relevant
    'is_month_end'      # ✅ Relevant
]
# Result: 100% relevant features, optimal model performance
```

## 🔧 Feature Detection Examples

### **Example 1: Weekly Data**
```python
# Input: Weekly timestamps
timestamps = pd.date_range('2023-01-01', periods=52, freq='W')

# Detection output:
[INFO] Detected frequency: week
[INFO] Features used: ['day_of_week (normalized)', 'month (normalized)', 'is_weekend', 'day_of_month (normalized)', 'is_month_end']
[INFO] Number of features: 5
```

### **Example 2: Hourly Data**
```python
# Input: Hourly timestamps  
timestamps = pd.date_range('2023-01-01', periods=8760, freq='H')

# Detection output:
[INFO] Detected frequency: hour
[INFO] Features used: ['day_of_week (normalized)', 'month (normalized)', 'is_weekend', 'hour_of_day (normalized)', 'is_business_hour']
[INFO] Number of features: 5
```

### **Example 3: Minute Data**
```python
# Input: 5-minute timestamps
timestamps = pd.date_range('2023-01-01', periods=288, freq='5min')

# Detection output:
[INFO] Detected frequency: minute
[INFO] Features used: ['day_of_week (normalized)', 'month (normalized)', 'is_weekend', 'hour_of_day (normalized)', 'minute_of_hour (normalized)', 'is_business_hour']
[INFO] Number of features: 6
```

## 📈 Performance Benefits

### **Model Efficiency**
- **Reduced Feature Dimensionality**: Only relevant features included
- **Better Convergence**: No noise from irrelevant features
- **Improved Accuracy**: Model focuses on meaningful patterns
- **Faster Training**: Smaller feature matrices

### **Memory Optimization**
- **Lower Memory Usage**: Fewer features = smaller tensors
- **Efficient Batching**: Reduced memory per batch
- **Scalable Processing**: Better handling of large datasets

## 🔍 Feature Analysis and Validation

### **Inspecting Frequency Detection**
```python
from signalnet.data.loader import SignalDataLoader

# Load data and check frequency detection
loader = SignalDataLoader('input/weekly_data.csv', context_length=24, prediction_length=6)
dataset = loader.get_dataset()

# Check detected frequency and features
print(f"Detected frequency: {dataset._detect_frequency(dataset.timestamps)}")
print(f"Features used: {dataset.features_used}")
print(f"Feature matrix shape: {dataset.time_features.shape}")
```

### **Feature Relevance Analysis**
```python
# Analyze feature variance to confirm relevance
for i, feature_name in enumerate(dataset.features_used):
    feature_values = dataset.time_features[:, i]
    variance = np.var(feature_values)
    print(f"{feature_name}: variance = {variance:.6f}")
    
    # Low variance indicates constant/irrelevant feature
    if variance < 1e-6:
        print(f"  ⚠️  {feature_name} has very low variance - may be irrelevant")
```

## ⚡ Advanced Customization

### **Custom Frequency Detection**
```python
def custom_frequency_detection(timestamps):
    """Custom frequency detection for specific use cases"""
    time_diffs = timestamps.diff().dropna()
    
    # Custom thresholds for your domain
    if time_diffs.std() < pd.Timedelta(minutes=5):
        return 'high_frequency'
    elif time_diffs.std() < pd.Timedelta(hours=1):
        return 'medium_frequency'
    else:
        return 'low_frequency'
```

### **Domain-Specific Features**
```python
# Add domain-specific features based on frequency
if frequency == 'day' and domain == 'financial':
    # Add market-specific features
    is_market_open = ((timestamps.dt.hour >= 9) & 
                     (timestamps.dt.hour <= 16) & 
                     (timestamps.dt.dayofweek < 5)).values.reshape(-1, 1).astype(float)
    features.append(is_market_open)
    features_used.append('is_market_open')
```

## 🎯 Best Practices

### **Data Quality**
1. **Consistent Frequency**: Ensure regular time intervals
2. **Clean Timestamps**: Remove duplicates and gaps
3. **Proper Format**: Use standard datetime format

### **Feature Selection**
1. **Trust Auto-Detection**: Let SignalNet detect frequency automatically
2. **Validate Features**: Check that selected features make sense
3. **Monitor Performance**: Compare with manual feature selection

### **Model Configuration**
1. **Dynamic Dimensions**: Model automatically adapts to feature count
2. **No Manual Tuning**: No need to specify feature dimensions
3. **Consistent Interface**: Same API regardless of frequency

---

**SignalNet Features**: Intelligent frequency-aware feature engineering that automatically selects only relevant temporal features for optimal model performance. 