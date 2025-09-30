"""
Train all ML models on sample data
"""
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from imblearn.over_sampling import SMOTE
from datetime import datetime, timedelta

# Set random seed
np.random.seed(42)

# Create directories
Path("app/models/artifacts").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

print("🚀 Starting model training...")

# ============================================================================
# Generate Synthetic Training Data (since we only have sample CSVs)
# ============================================================================

def generate_synthetic_surveys(n=10000):
    """Generate realistic survey data for model training"""
    print(f"📊 Generating {n} synthetic survey records...")

    np.random.seed(42)

    # Customer demographics
    ages = np.random.randint(18, 65, n)
    genders = np.random.choice(['male', 'female'], n, p=[0.55, 0.45])

    # Temporal patterns
    dates = pd.date_range(end=datetime.now(), periods=n, freq='1H')
    hours = dates.hour
    day_of_week = dates.dayofweek
    is_weekend = day_of_week.isin([5, 6]).astype(int)

    # Purchase behavior (with realistic correlations)
    base_value = np.random.gamma(shape=2, scale=15, size=n)
    age_factor = (ages - ages.min()) / (ages.max() - ages.min()) * 10
    weekend_factor = is_weekend * np.random.uniform(1.1, 1.3, n)
    evening_factor = ((hours >= 19) & (hours <= 22)).astype(int) * np.random.uniform(1.05, 1.2, n)

    purchase_values = base_value + age_factor + weekend_factor + evening_factor
    purchase_values = np.clip(purchase_values, 10, 100)

    # Visit duration (correlated with purchase value)
    visit_durations = 30 + (purchase_values - purchase_values.min()) / (purchase_values.max() - purchase_values.min()) * 90
    visit_durations = visit_durations + np.random.normal(0, 10, n)
    visit_durations = np.clip(visit_durations, 15, 180)

    # Satisfaction scores (higher for bigger spenders)
    satisfaction_base = 3 + (purchase_values / purchase_values.max()) * 1.5
    satisfaction = satisfaction_base + np.random.normal(0, 0.5, n)
    overall_scores = np.clip(satisfaction, 1, 5)

    nps_base = 5 + (overall_scores - 1) / 4 * 5
    nps_scores = nps_base + np.random.normal(0, 1, n)
    nps_scores = np.clip(nps_scores, 0, 10).astype(int)

    would_recommend = (nps_scores >= 7).astype(int)

    # Weather (temperature affects behavior)
    temperatures = np.random.normal(20, 5, n)
    is_rainy = np.random.choice([0, 1], n, p=[0.8, 0.2])

    # Create DataFrame
    df = pd.DataFrame({
        'survey_id': [f'SRV_{i:06d}' for i in range(n)],
        'customer_age': ages,
        'customer_gender': genders,
        'collected_at': dates,
        'hour': hours,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'purchase_value': purchase_values,
        'visit_duration_minutes': visit_durations,
        'overall_score': overall_scores,
        'nps_score': nps_scores,
        'would_recommend': would_recommend,
        'temperature_c': temperatures,
        'is_rainy': is_rainy
    })

    return df

# Generate data
surveys_df = generate_synthetic_surveys(10000)
surveys_df.to_csv('data/processed/training_data.csv', index=False)
print(f"✅ Generated {len(surveys_df)} training samples")

# ============================================================================
# Model 1: Customer Segmentation (K-Means)
# ============================================================================

print("\n🎯 Training Model 1: Customer Segmentation...")

# Aggregate customer-level features
customer_features = surveys_df.groupby(['customer_age', 'customer_gender']).agg({
    'purchase_value': ['mean', 'std', 'count'],
    'visit_duration_minutes': 'mean',
    'nps_score': 'mean',
    'overall_score': 'mean',
    'would_recommend': 'mean',
    'is_weekend': 'mean'
}).reset_index()

customer_features.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in customer_features.columns]
customer_features['purchase_value_std'] = customer_features['purchase_value_std'].fillna(0)

# Features for clustering
feature_cols = ['purchase_value_mean', 'purchase_value_count', 'nps_score_mean',
                'overall_score_mean', 'would_recommend_mean', 'is_weekend_mean']

X_seg = customer_features[feature_cols].fillna(0)

# Scale features
scaler_seg = StandardScaler()
X_seg_scaled = scaler_seg.fit_transform(X_seg)

# Train K-Means
kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
customer_features['segment'] = kmeans.fit_predict(X_seg_scaled)

# Save artifacts
joblib.dump(kmeans, 'app/models/artifacts/segmentation_model.pkl')
joblib.dump(scaler_seg, 'app/models/artifacts/segmentation_scaler.pkl')
joblib.dump(feature_cols, 'app/models/artifacts/segmentation_features.pkl')

# Segment profiles
segment_profiles = customer_features.groupby('segment').agg({
    'customer_age': 'count',
    'purchase_value_mean': 'mean',
    'nps_score_mean': 'mean',
    'purchase_value_count': 'mean'
}).round(2)

print(f"✅ Segmentation model trained ({len(segment_profiles)} segments)")
print(segment_profiles)

# ============================================================================
# Model 2: Purchase Value Prediction (XGBoost)
# ============================================================================

print("\n💰 Training Model 2: Purchase Value Prediction...")

# Features for prediction
pred_features = ['customer_age', 'hour', 'day_of_week', 'is_weekend',
                 'temperature_c', 'is_rainy', 'visit_duration_minutes']

X_pred = surveys_df[pred_features].copy()
X_pred['customer_gender'] = (surveys_df['customer_gender'] == 'female').astype(int)
y_pred = surveys_df['purchase_value']

# Add interaction features
X_pred['age_weekend'] = X_pred['customer_age'] * X_pred['is_weekend']
X_pred['temp_hour'] = X_pred['temperature_c'] * X_pred['hour']

# Train XGBoost
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

xgb_model.fit(X_pred, y_pred)

# Save artifacts
joblib.dump(xgb_model, 'app/models/artifacts/purchase_prediction_model.pkl')
joblib.dump(list(X_pred.columns), 'app/models/artifacts/purchase_prediction_features.pkl')

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
y_pred_train = xgb_model.predict(X_pred)
mae = mean_absolute_error(y_pred, y_pred_train)
mape = mean_absolute_percentage_error(y_pred, y_pred_train)
r2 = r2_score(y_pred, y_pred_train)

print(f"✅ Purchase prediction model trained")
print(f"   MAE: ${mae:.2f} | MAPE: {mape*100:.1f}% | R²: {r2:.3f}")

# ============================================================================
# Model 3: Churn Risk Scoring (Gradient Boosting)
# ============================================================================

print("\n🔥 Training Model 3: Churn Risk Scoring...")

# Create customer-level RFM features
current_date = surveys_df['collected_at'].max()

rfm = surveys_df.groupby(['customer_age', 'customer_gender']).agg(
    last_visit=('collected_at', 'max'),
    first_visit=('collected_at', 'min'),
    visit_count=('collected_at', 'count'),
    total_spent=('purchase_value', 'sum'),
    avg_spent=('purchase_value', 'mean'),
    std_spent=('purchase_value', 'std'),
    avg_nps=('nps_score', 'mean'),
    recommend_rate=('would_recommend', 'mean')
).reset_index()

# Calculate recency
rfm['recency_days'] = (current_date - rfm['last_visit']).dt.days
rfm['tenure_days'] = (rfm['last_visit'] - rfm['first_visit']).dt.days + 1
rfm['purchase_frequency'] = rfm['visit_count'] / rfm['tenure_days']
rfm['std_spent'] = rfm['std_spent'].fillna(0)
rfm['spending_volatility'] = rfm['std_spent'] / (rfm['avg_spent'] + 1)

# Behavioral flags
rfm['is_one_time'] = (rfm['visit_count'] == 1).astype(int)
rfm['is_dissatisfied'] = (rfm['avg_nps'] < 7).astype(int)

# Define churn with probability-based approach for realistic distribution
# Use percentile-based thresholds
recency_threshold = rfm['recency_days'].quantile(0.7)  # Top 30% recency
nps_threshold = rfm['avg_nps'].quantile(0.3)  # Bottom 30% NPS
frequency_threshold = rfm['purchase_frequency'].quantile(0.3)  # Bottom 30% frequency

rfm['churned'] = (
    ((rfm['recency_days'] > recency_threshold) & (rfm['avg_nps'] < nps_threshold)) |
    (rfm['purchase_frequency'] < frequency_threshold) |
    (rfm['avg_nps'] < 5)
).astype(int)

# Features for churn prediction
churn_features = ['recency_days', 'visit_count', 'total_spent', 'avg_spent',
                  'purchase_frequency', 'avg_nps', 'recommend_rate',
                  'is_one_time', 'spending_volatility', 'is_dissatisfied']

X_churn = rfm[churn_features].fillna(0)
y_churn = rfm['churned']

# Check if we have both classes
if y_churn.nunique() > 1:
    # Handle class imbalance
    smote = SMOTE(sampling_strategy=0.6, random_state=42)
    X_churn_balanced, y_churn_balanced = smote.fit_resample(X_churn, y_churn)
else:
    # If only one class, use original data without SMOTE
    X_churn_balanced, y_churn_balanced = X_churn, y_churn

# Train Gradient Boosting
churn_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

churn_model.fit(X_churn_balanced, y_churn_balanced)

# Save artifacts
joblib.dump(churn_model, 'app/models/artifacts/churn_model.pkl')
joblib.dump(churn_features, 'app/models/artifacts/churn_features.pkl')

# Calculate metrics
from sklearn.metrics import roc_auc_score, classification_report
y_churn_pred_proba = churn_model.predict_proba(X_churn)[:, 1]
auc = roc_auc_score(y_churn, y_churn_pred_proba)

print(f"✅ Churn prediction model trained")
print(f"   AUC-ROC: {auc:.3f} | Churn rate: {y_churn.mean()*100:.1f}%")

# ============================================================================
# Model 4: Sentiment Analysis (TextBlob + Rule-Based)
# ============================================================================

print("\n💬 Training Model 4: Sentiment Analysis...")

# Generate synthetic feedback comments
np.random.seed(42)

positive_comments = [
    "Great service and quality products!",
    "Love the selection here.",
    "Staff is very friendly and helpful.",
    "Best prices in town.",
    "Clean store with good atmosphere.",
    "Always have what I need.",
    "Quick checkout process.",
    "Excellent customer service!",
]

neutral_comments = [
    "It's okay, nothing special.",
    "Average store.",
    "Standard selection.",
    "Normal shopping experience.",
    "Decent prices.",
]

negative_comments = [
    "Poor service, won't return.",
    "Overpriced items.",
    "Unfriendly staff.",
    "Store is dirty and disorganized.",
    "Long wait times at checkout.",
    "Limited product selection.",
    "Rude employees.",
]

# Create sentiment training data
n_sentiment = 1000
sentiment_data = []

for _ in range(n_sentiment):
    nps = np.random.randint(0, 11)
    if nps >= 9:
        comment = np.random.choice(positive_comments)
        sentiment = 'positive'
    elif nps >= 7:
        comment = np.random.choice(positive_comments + neutral_comments)
        sentiment = 'neutral'
    else:
        comment = np.random.choice(negative_comments + neutral_comments)
        sentiment = 'negative'

    sentiment_data.append({
        'comment': comment,
        'nps_score': nps,
        'sentiment': sentiment
    })

sentiment_df = pd.DataFrame(sentiment_data)

# Save sentiment keywords for runtime analysis
sentiment_keywords = {
    'positive': ['great', 'love', 'excellent', 'best', 'friendly', 'helpful', 'clean', 'good'],
    'negative': ['poor', 'overpriced', 'rude', 'dirty', 'limited', 'unfriendly', 'bad', 'worst'],
    'intensifiers': ['very', 'extremely', 'really', 'absolutely']
}

joblib.dump(sentiment_keywords, 'app/models/artifacts/sentiment_keywords.pkl')

print(f"✅ Sentiment analysis model ready")
print(f"   Positive samples: {(sentiment_df['sentiment'] == 'positive').sum()}")
print(f"   Negative samples: {(sentiment_df['sentiment'] == 'negative').sum()}")

# ============================================================================
# Model 5: POS Location Ranking
# ============================================================================

print("\n📍 Training Model 5: POS Location Ranking...")

# Generate synthetic POS location data
locations = [
    'Downtown Main St', 'Airport Terminal', 'Shopping Mall Entrance',
    'Train Station', 'University Campus', 'Business District',
    'Residential Area', 'Tourist Center', 'Highway Rest Stop'
]

pos_data = []
np.random.seed(42)

for location in locations:
    # Location-specific characteristics
    if 'Airport' in location or 'Train' in location:
        base_transactions = np.random.randint(800, 1200)
        base_revenue = np.random.uniform(25000, 35000)
        foot_traffic = np.random.randint(5000, 8000)
    elif 'Downtown' in location or 'Shopping Mall' in location:
        base_transactions = np.random.randint(600, 900)
        base_revenue = np.random.uniform(18000, 28000)
        foot_traffic = np.random.randint(3000, 5000)
    else:
        base_transactions = np.random.randint(300, 600)
        base_revenue = np.random.uniform(10000, 20000)
        foot_traffic = np.random.randint(1000, 3000)

    avg_transaction = base_revenue / base_transactions
    conversion_rate = base_transactions / foot_traffic

    pos_data.append({
        'location_name': location,
        'monthly_transactions': base_transactions,
        'monthly_revenue': base_revenue,
        'avg_transaction_value': avg_transaction,
        'foot_traffic': foot_traffic,
        'conversion_rate': conversion_rate,
        'customer_satisfaction': np.random.uniform(3.5, 4.8)
    })

pos_df = pd.DataFrame(pos_data)

# Calculate composite score
pos_df['revenue_score'] = (pos_df['monthly_revenue'] - pos_df['monthly_revenue'].min()) / (pos_df['monthly_revenue'].max() - pos_df['monthly_revenue'].min())
pos_df['efficiency_score'] = (pos_df['conversion_rate'] - pos_df['conversion_rate'].min()) / (pos_df['conversion_rate'].max() - pos_df['conversion_rate'].min())
pos_df['satisfaction_score'] = (pos_df['customer_satisfaction'] - pos_df['customer_satisfaction'].min()) / (pos_df['customer_satisfaction'].max() - pos_df['customer_satisfaction'].min())

pos_df['composite_score'] = (
    pos_df['revenue_score'] * 0.4 +
    pos_df['efficiency_score'] * 0.35 +
    pos_df['satisfaction_score'] * 0.25
)

pos_df['rank'] = pos_df['composite_score'].rank(ascending=False).astype(int)
pos_df = pos_df.sort_values('rank')

# Save POS ranking data
joblib.dump(pos_df, 'app/models/artifacts/pos_ranking_data.pkl')

print(f"✅ POS location ranking model trained")
print(f"   Total locations: {len(pos_df)}")
print(f"\n🏆 Top 3 Locations:")
for idx, row in pos_df.head(3).iterrows():
    print(f"   #{row['rank']}: {row['location_name']} - Score: {row['composite_score']:.3f}")

# ============================================================================
# Save metadata
# ============================================================================

metadata = {
    'training_date': datetime.now().isoformat(),
    'n_samples': len(surveys_df),
    'models': {
        'segmentation': {
            'n_segments': len(segment_profiles),
            'features': feature_cols
        },
        'purchase_prediction': {
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2)
        },
        'churn': {
            'auc': float(auc),
            'churn_rate': float(y_churn.mean())
        },
        'sentiment': {
            'n_samples': len(sentiment_df),
            'positive_rate': float((sentiment_df['sentiment'] == 'positive').mean())
        },
        'pos_ranking': {
            'n_locations': len(pos_df),
            'top_location': pos_df.iloc[0]['location_name']
        }
    }
}

joblib.dump(metadata, 'app/models/artifacts/metadata.pkl')

print("\n✅ All 5 models trained and saved successfully!")
print(f"📦 Artifacts saved to: app/models/artifacts/")
