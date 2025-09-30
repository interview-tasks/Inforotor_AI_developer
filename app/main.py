"""
FastAPI Main Application
"""
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Initialize FastAPI
app = FastAPI(
    title="Cigarette POS ML Platform",
    description="Production ML models for retail optimization",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Load models
MODELS_DIR = Path("app/models/artifacts")

try:
    # Segmentation
    segmentation_model = joblib.load(MODELS_DIR / "segmentation_model.pkl")
    segmentation_scaler = joblib.load(MODELS_DIR / "segmentation_scaler.pkl")
    segmentation_features = joblib.load(MODELS_DIR / "segmentation_features.pkl")

    # Purchase prediction
    purchase_model = joblib.load(MODELS_DIR / "purchase_prediction_model.pkl")
    purchase_features = joblib.load(MODELS_DIR / "purchase_prediction_features.pkl")

    # Churn prediction
    churn_model = joblib.load(MODELS_DIR / "churn_model.pkl")
    churn_features = joblib.load(MODELS_DIR / "churn_features.pkl")

    # Sentiment analysis
    sentiment_keywords = joblib.load(MODELS_DIR / "sentiment_keywords.pkl")

    # POS ranking
    pos_ranking_data = joblib.load(MODELS_DIR / "pos_ranking_data.pkl")

    # Metadata
    metadata = joblib.load(MODELS_DIR / "metadata.pkl")

    models_loaded = True
except Exception as e:
    print(f"⚠️ Warning: Could not load models: {e}")
    models_loaded = False

# ============================================================================
# Pydantic Models
# ============================================================================

class PurchasePredictionRequest(BaseModel):
    customer_age: int
    customer_gender: str
    hour: int
    day_of_week: int
    temperature_c: float
    is_rainy: bool
    visit_duration_minutes: int = 60

class ChurnPredictionRequest(BaseModel):
    recency_days: int
    visit_count: int
    total_spent: float
    avg_spent: float
    purchase_frequency: float
    avg_nps: float
    recommend_rate: float

class SegmentationRequest(BaseModel):
    purchase_value_mean: float
    purchase_value_count: int
    nps_score_mean: float
    overall_score_mean: float
    would_recommend_mean: float
    is_weekend_mean: float

class SentimentRequest(BaseModel):
    comment: str
    nps_score: Optional[int] = None

class POSRankingRequest(BaseModel):
    monthly_transactions: int
    monthly_revenue: float
    foot_traffic: int
    customer_satisfaction: float

# ============================================================================
# Routes - Frontend
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Homepage"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models_loaded": models_loaded,
        "metadata": metadata if models_loaded else None
    })

@app.get("/segmentation", response_class=HTMLResponse)
async def segmentation_page(request: Request):
    """Customer Segmentation Page"""
    return templates.TemplateResponse("segmentation.html", {
        "request": request
    })

@app.get("/purchase", response_class=HTMLResponse)
async def purchase_page(request: Request):
    """Purchase Prediction Page"""
    return templates.TemplateResponse("purchase.html", {
        "request": request
    })

@app.get("/churn", response_class=HTMLResponse)
async def churn_page(request: Request):
    """Churn Prediction Page"""
    return templates.TemplateResponse("churn.html", {
        "request": request
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Dashboard Page"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "metadata": metadata if models_loaded else None
    })

@app.get("/sentiment", response_class=HTMLResponse)
async def sentiment_page(request: Request):
    """Sentiment Analysis Page"""
    return templates.TemplateResponse("sentiment.html", {
        "request": request
    })

@app.get("/pos-ranking", response_class=HTMLResponse)
async def pos_ranking_page(request: Request):
    """POS Location Ranking Page"""
    return templates.TemplateResponse("pos_ranking.html", {
        "request": request,
        "locations": pos_ranking_data.to_dict('records') if models_loaded else []
    })

# ============================================================================
# Routes - API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_loaded
    }

@app.post("/api/predict/segment")
async def predict_segment(data: SegmentationRequest):
    """Predict customer segment"""
    if not models_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Models not loaded"}
        )

    try:
        # Prepare features
        features = np.array([[
            data.purchase_value_mean,
            data.purchase_value_count,
            data.nps_score_mean,
            data.overall_score_mean,
            data.would_recommend_mean,
            data.is_weekend_mean
        ]])

        # Scale and predict
        features_scaled = segmentation_scaler.transform(features)
        segment = int(segmentation_model.predict(features_scaled)[0])

        # Segment mapping
        segment_names = {
            0: "Premium Regulars",
            1: "Budget Shoppers",
            2: "Occasional Buyers",
            3: "Weekend Socializers",
            4: "At-Risk Customers"
        }

        segment_descriptions = {
            0: "High-value customers with frequent purchases and excellent satisfaction scores.",
            1: "Price-sensitive customers looking for value deals and promotions.",
            2: "Infrequent visitors with moderate spending patterns.",
            3: "Social customers who prefer weekend visits and group activities.",
            4: "Customers at risk of churning - need immediate retention efforts."
        }

        segment_actions = {
            0: "VIP loyalty program, premium brand offers, exclusive lounge access",
            1: "Volume discounts: Buy 2 Get 15% off, budget brand promotions",
            2: "Re-engagement campaigns: 20% welcome back offer",
            3: "Group discounts, weekend special offers, social media marketing",
            4: "Win-back campaign: 30% discount + manager outreach call"
        }

        return {
            "segment_id": segment,
            "segment_name": segment_names.get(segment, "Unknown"),
            "description": segment_descriptions.get(segment, ""),
            "recommended_action": segment_actions.get(segment, ""),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/predict/purchase")
async def predict_purchase(data: PurchasePredictionRequest):
    """Predict purchase value"""
    if not models_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Models not loaded"}
        )

    try:
        # Prepare features
        is_weekend = 1 if data.day_of_week in [5, 6] else 0
        customer_gender = 1 if data.customer_gender.lower() == 'female' else 0

        features_dict = {
            'customer_age': data.customer_age,
            'hour': data.hour,
            'day_of_week': data.day_of_week,
            'is_weekend': is_weekend,
            'temperature_c': data.temperature_c,
            'is_rainy': int(data.is_rainy),
            'visit_duration_minutes': data.visit_duration_minutes,
            'customer_gender': customer_gender,
            'age_weekend': data.customer_age * is_weekend,
            'temp_hour': data.temperature_c * data.hour
        }

        # Ensure correct feature order
        features = np.array([[features_dict[f] for f in purchase_features]])

        # Predict
        prediction = float(purchase_model.predict(features)[0])

        # Confidence interval (±15%)
        ci_lower = prediction * 0.85
        ci_upper = prediction * 1.15

        # Upselling recommendation
        if prediction < 30:
            upsell = "Offer premium brand upgrade (+$15 potential)"
            tier = "Low Value"
        elif prediction < 50:
            upsell = "Suggest multi-pack bundle (+$20 potential)"
            tier = "Medium Value"
        else:
            upsell = "Customer is already high-value, focus on retention"
            tier = "High Value"

        return {
            "predicted_value": round(prediction, 2),
            "confidence_interval": {
                "lower": round(ci_lower, 2),
                "upper": round(ci_upper, 2)
            },
            "revenue_tier": tier,
            "upselling_recommendation": upsell,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/predict/churn")
async def predict_churn(data: ChurnPredictionRequest):
    """Predict churn risk"""
    if not models_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Models not loaded"}
        )

    try:
        # Calculate derived features
        is_one_time = 1 if data.visit_count == 1 else 0
        spending_volatility = 0 if data.avg_spent == 0 else abs(data.total_spent - data.avg_spent * data.visit_count) / (data.avg_spent + 1)
        is_dissatisfied = 1 if data.avg_nps < 7 else 0

        # Prepare features
        features_dict = {
            'recency_days': data.recency_days,
            'visit_count': data.visit_count,
            'total_spent': data.total_spent,
            'avg_spent': data.avg_spent,
            'purchase_frequency': data.purchase_frequency,
            'avg_nps': data.avg_nps,
            'recommend_rate': data.recommend_rate,
            'is_one_time': is_one_time,
            'spending_volatility': spending_volatility,
            'is_dissatisfied': is_dissatisfied
        }

        # Ensure correct feature order
        features = np.array([[features_dict[f] for f in churn_features]])

        # Predict
        churn_prob = float(churn_model.predict_proba(features)[0][1])

        # Risk tiering
        if churn_prob > 0.7:
            risk_tier = "HIGH"
            priority = 1
            action = "Immediate intervention required"
            offer = "30% discount + manager outreach call"
            recovery_rate = 0.35
        elif churn_prob > 0.4:
            risk_tier = "MEDIUM"
            priority = 2
            action = "Schedule retention campaign within 7 days"
            offer = "20% off next purchase via SMS"
            recovery_rate = 0.25
        else:
            risk_tier = "LOW"
            priority = 3
            action = "Add to standard loyalty program"
            offer = "10% off 10th purchase (loyalty card)"
            recovery_rate = 0.10

        # Calculate ROI
        intervention_cost = 15 if risk_tier == "HIGH" else (8 if risk_tier == "MEDIUM" else 3)
        expected_recovery = data.total_spent * recovery_rate
        expected_roi = expected_recovery / intervention_cost if intervention_cost > 0 else 0

        return {
            "churn_probability": round(churn_prob, 3),
            "risk_tier": risk_tier,
            "priority": priority,
            "recommended_action": action,
            "offer": offer,
            "metrics": {
                "customer_ltv": round(data.total_spent, 2),
                "intervention_cost": intervention_cost,
                "expected_recovery": round(expected_recovery, 2),
                "expected_roi": round(expected_roi, 2)
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/predict/sentiment")
async def predict_sentiment(data: SentimentRequest):
    """Analyze sentiment of customer feedback"""
    if not models_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Models not loaded"}
        )

    try:
        from textblob import TextBlob

        comment = data.comment.lower()

        # TextBlob sentiment analysis
        blob = TextBlob(data.comment)
        polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
        subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)

        # Keyword-based analysis
        positive_count = sum(1 for word in sentiment_keywords['positive'] if word in comment)
        negative_count = sum(1 for word in sentiment_keywords['negative'] if word in comment)
        intensifier_count = sum(1 for word in sentiment_keywords['intensifiers'] if word in comment)

        # Combined score
        keyword_score = (positive_count - negative_count) * (1 + intensifier_count * 0.2)
        combined_score = (polarity + keyword_score / 5) / 2  # Normalize

        # Classification
        if combined_score > 0.2:
            sentiment = "Positive"
            emoji = "😊"
            color = "success"
        elif combined_score < -0.2:
            sentiment = "Negative"
            emoji = "😞"
            color = "danger"
        else:
            sentiment = "Neutral"
            emoji = "😐"
            color = "warning"

        # Actionable insights
        if sentiment == "Negative":
            action = "Immediate follow-up required. Contact customer within 24 hours."
            priority = "HIGH"
        elif sentiment == "Neutral":
            action = "Monitor feedback trends. Consider satisfaction survey."
            priority = "MEDIUM"
        else:
            action = "Request review/testimonial. Offer referral discount."
            priority = "LOW"

        return {
            "sentiment": sentiment,
            "emoji": emoji,
            "confidence": abs(combined_score),
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "color": color,
            "priority": priority,
            "recommended_action": action,
            "keyword_analysis": {
                "positive_keywords": positive_count,
                "negative_keywords": negative_count,
                "intensifiers": intensifier_count
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/predict/pos-rank")
async def predict_pos_rank(data: POSRankingRequest):
    """Calculate POS location ranking score"""
    if not models_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Models not loaded"}
        )

    try:
        # Calculate metrics
        avg_transaction_value = data.monthly_revenue / data.monthly_transactions if data.monthly_transactions > 0 else 0
        conversion_rate = data.monthly_transactions / data.foot_traffic if data.foot_traffic > 0 else 0

        # Normalize against existing locations
        revenue_min = pos_ranking_data['monthly_revenue'].min()
        revenue_max = pos_ranking_data['monthly_revenue'].max()
        revenue_score = (data.monthly_revenue - revenue_min) / (revenue_max - revenue_min) if revenue_max > revenue_min else 0.5

        conv_min = pos_ranking_data['conversion_rate'].min()
        conv_max = pos_ranking_data['conversion_rate'].max()
        efficiency_score = (conversion_rate - conv_min) / (conv_max - conv_min) if conv_max > conv_min else 0.5

        sat_min = pos_ranking_data['customer_satisfaction'].min()
        sat_max = pos_ranking_data['customer_satisfaction'].max()
        satisfaction_score = (data.customer_satisfaction - sat_min) / (sat_max - sat_min) if sat_max > sat_min else 0.5

        # Composite score (40% revenue, 35% efficiency, 25% satisfaction)
        composite_score = (
            revenue_score * 0.4 +
            efficiency_score * 0.35 +
            satisfaction_score * 0.25
        )

        # Performance tier
        if composite_score >= 0.75:
            tier = "Excellent"
            tier_color = "success"
            recommendation = "Replicate success factors to other locations. Consider expansion."
        elif composite_score >= 0.5:
            tier = "Good"
            tier_color = "primary"
            recommendation = "Monitor performance. Optimize conversion through staff training."
        elif composite_score >= 0.3:
            tier = "Fair"
            tier_color = "warning"
            recommendation = "Improvement needed. Focus on customer experience and marketing."
        else:
            tier = "Poor"
            tier_color = "danger"
            recommendation = "Critical intervention required. Consider relocation or major restructuring."

        # Rank prediction
        better_locations = (pos_ranking_data['composite_score'] > composite_score).sum()
        predicted_rank = better_locations + 1

        return {
            "composite_score": round(composite_score, 3),
            "predicted_rank": int(predicted_rank),
            "total_locations": len(pos_ranking_data) + 1,
            "performance_tier": tier,
            "tier_color": tier_color,
            "recommended_action": recommendation,
            "metrics": {
                "avg_transaction_value": round(avg_transaction_value, 2),
                "conversion_rate": round(conversion_rate * 100, 2),
                "revenue_score": round(revenue_score, 3),
                "efficiency_score": round(efficiency_score, 3),
                "satisfaction_score": round(satisfaction_score, 3)
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/pos-locations")
async def get_pos_locations():
    """Get all POS location rankings"""
    if not models_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Models not loaded"}
        )

    try:
        locations_list = pos_ranking_data.to_dict('records')
        return {
            "locations": locations_list,
            "total": len(locations_list),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def internal_error(request: Request, exc):
    return templates.TemplateResponse("500.html", {"request": request}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
