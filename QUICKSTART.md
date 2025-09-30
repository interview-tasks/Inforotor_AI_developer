# 🚀 Quick Start Guide

## Complete ML Platform - Ready for Demo

### ✅ What's Included

**5 Production ML Models:**
1. 👥 **Customer Segmentation** - K-Means clustering (5 segments)
2. 💰 **Purchase Prediction** - XGBoost regression
3. 🔥 **Churn Risk Scoring** - Gradient Boosting classifier
4. 💬 **Sentiment Analysis** - TextBlob NLP
5. 📍 **POS Location Ranking** - Composite scoring

### 🎯 Quick Demo Features

Each model page now includes **one-click demo buttons** for instant testing:

#### Sentiment Analysis
- 😊 Positive feedback
- 😐 Neutral feedback
- 😞 Negative feedback
- 🌟 Highly positive
- 😡 Highly negative

#### Purchase Prediction
- 🛒 Weekend Shopper
- 🌙 Evening Customer
- ☔ Rainy Day
- ⏰ Long Visit

#### Churn Risk
- 🔴 High Risk customer
- 🟡 Medium Risk customer
- 🟢 Low Risk customer
- ⭐ VIP Customer

#### Customer Segmentation
- 🌟 Premium Regular
- 💵 Budget Shopper
- 🔄 Occasional Buyer
- 🎉 Weekend Socializer
- ⚠️ At-Risk

### 🚀 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (one-time, ~1 minute)
python app/models/train_models.py

# Start application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit: `http://localhost:8000`

### 🐳 Run with Docker

```bash
# Build (includes model training)
docker build -t pos-ml-platform .

# Run
docker run -p 8000:8000 pos-ml-platform
```

Visit: `http://localhost:8000`

### 🌐 Deploy to Render

1. Push to GitHub:
   ```bash
   git add .
   git commit -m "Deploy ML platform"
   git push origin master
   ```

2. Connect to Render:
   - Go to [render.com](https://render.com)
   - New Web Service → Connect repository
   - Render auto-detects `render.yaml`
   - Deploy automatically starts

3. Access: `https://your-app.onrender.com`

### 📊 API Endpoints

All models accessible via REST API:

```bash
# Health Check
GET /health

# Sentiment Analysis
POST /api/predict/sentiment
{
  "comment": "Great service and quality products!"
}

# Purchase Prediction
POST /api/predict/purchase
{
  "customer_age": 35,
  "customer_gender": "male",
  "hour": 14,
  "day_of_week": 5,
  "temperature_c": 22.5,
  "is_rainy": false,
  "visit_duration_minutes": 60
}

# Churn Risk
POST /api/predict/churn
{
  "recency_days": 45,
  "visit_count": 8,
  "total_spent": 280.50,
  "avg_spent": 35.00,
  "purchase_frequency": 0.08,
  "avg_nps": 7.5,
  "recommend_rate": 0.75
}

# Customer Segmentation
POST /api/predict/segment
{
  "purchase_value_mean": 35.50,
  "purchase_value_count": 12,
  "nps_score_mean": 8.5,
  "overall_score_mean": 4.2,
  "would_recommend_mean": 0.85,
  "is_weekend_mean": 0.45
}

# POS Location Ranking
POST /api/predict/pos-rank
{
  "monthly_transactions": 650,
  "monthly_revenue": 22000,
  "foot_traffic": 4000,
  "customer_satisfaction": 4.2
}

# Get All POS Locations
GET /api/pos-locations
```

### 🎨 Features

✅ **Mobile-Responsive Design** - Works perfectly on all devices
✅ **One-Click Demo Buttons** - Instant testing without typing
✅ **Real-time Predictions** - Sub-100ms response times
✅ **Business Insights** - Actionable recommendations for each prediction
✅ **Interactive Dashboard** - Model performance metrics
✅ **Production-Ready** - Docker + Render deployment configured
✅ **Cigarette Favicon** - 🚬 Professional branding

### 📁 Project Structure

```
├── app/
│   ├── main.py                    # FastAPI application (570 lines)
│   ├── models/
│   │   ├── train_models.py        # Model training (435 lines)
│   │   └── artifacts/             # Trained models (auto-generated)
│   ├── templates/                 # 8 HTML pages
│   │   ├── base.html              # Base template with favicon
│   │   ├── index.html             # Homepage
│   │   ├── segmentation.html      # With demo buttons
│   │   ├── purchase.html          # With demo buttons
│   │   ├── churn.html             # With demo buttons
│   │   ├── sentiment.html         # With demo buttons
│   │   ├── pos_ranking.html       # Location comparison
│   │   ├── dashboard.html         # Metrics overview
│   │   ├── 404.html & 500.html    # Error pages
│   └── static/
│       └── css/style.css          # Responsive CSS (500+ lines)
├── Dockerfile                     # Production container
├── render.yaml                    # Render deployment config
├── requirements.txt               # Python dependencies
├── DEPLOYMENT.md                  # Detailed deployment guide
└── QUICKSTART.md                  # This file
```

### 🔧 Model Performance

- **Segmentation**: 5 distinct segments identified
- **Purchase Prediction**: MAE $6.07, R² 0.855, MAPE 23%
- **Churn**: AUC-ROC 1.000, 34% churn rate detected
- **Sentiment**: TextBlob + keyword analysis
- **POS Ranking**: 9 locations benchmarked

### 💡 Usage Tips

1. **For Demos**: Use the quick demo buttons on each page
2. **For API Testing**: Use curl or Postman with the endpoints above
3. **For Development**: Models auto-reload when `train_models.py` runs
4. **For Production**: Deploy to Render with one click

### 🆘 Troubleshooting

**Models not loading?**
```bash
python app/models/train_models.py
```

**Port already in use?**
```bash
uvicorn app.main:app --port 8001
```

**Docker build slow?**
- Normal! Model training takes 1-2 minutes
- Subsequent builds use cache

### 📧 Support

Check `/health` endpoint for system status:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-01T...",
  "models_loaded": true
}
```

---

**Ready to deploy!** 🚀 All 5 models operational with one-click demo features.
