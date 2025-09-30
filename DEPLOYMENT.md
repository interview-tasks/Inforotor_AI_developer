# Deployment Guide

## Quick Start - Local Development

### Prerequisites
- Python 3.11+
- Docker (optional)

### Option 1: Run with Python

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python app/models/train_models.py

# Run application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Application will be available at `http://localhost:8000`

### Option 2: Run with Docker

```bash
# Build Docker image
docker build -t pos-ml-platform .

# Run container
docker run -p 8000:8000 pos-ml-platform
```

Application will be available at `http://localhost:8000`

## Deployment to Render

### Prerequisites
- GitHub account
- Render account (free tier available)

### Steps

1. **Push Code to GitHub**
   ```bash
   git add .
   git commit -m "Deploy ML platform"
   git push origin master
   ```

2. **Connect to Render**
   - Go to [render.com](https://render.com)
   - Sign in with GitHub
   - Click "New +" → "Web Service"
   - Connect your repository

3. **Configure Service**
   - **Name**: `cigarette-pos-ml-platform`
   - **Environment**: `Docker`
   - **Region**: Choose closest to your users
   - **Branch**: `master`
   - **Plan**: Free (or paid for better performance)

4. **Deploy**
   - Render will automatically detect `render.yaml` configuration
   - Docker build process includes model training
   - First deployment takes 5-10 minutes
   - Subsequent deployments are faster

5. **Access Application**
   - Your app will be available at: `https://cigarette-pos-ml-platform.onrender.com`
   - Health check: `https://cigarette-pos-ml-platform.onrender.com/health`

## Environment Variables

No environment variables required for basic setup. Optional configurations:

- `PORT`: Application port (default: 8000)
- `PYTHONUNBUFFERED`: Set to 1 for real-time logs

## Health Check

The application includes a health check endpoint at `/health`:

```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00",
  "models_loaded": true
}
```

## Architecture

```
├── app/
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   ├── train_models.py     # Model training script
│   │   └── artifacts/          # Trained model files (generated)
│   ├── templates/              # Jinja2 HTML templates
│   └── static/                 # CSS and JS files
├── Dockerfile                  # Container configuration
├── render.yaml                 # Render deployment config
└── requirements.txt            # Python dependencies
```

## Model Training

Models are automatically trained during Docker build process. To retrain manually:

```bash
python app/models/train_models.py
```

This generates:
- **Segmentation**: K-Means clustering model
- **Purchase Prediction**: XGBoost regression model
- **Churn Prediction**: Gradient Boosting classifier
- **Sentiment Analysis**: TextBlob + keyword dictionaries
- **POS Ranking**: Composite scoring data

## Performance Optimization

### For Production:

1. **Upgrade Render Plan**: Free tier sleeps after inactivity
2. **Add Redis Caching**: Cache model predictions
3. **Enable CDN**: For static assets
4. **Add Monitoring**: Sentry or similar for error tracking

### Resource Requirements:

- **Memory**: 512MB minimum (1GB recommended)
- **CPU**: 0.5 vCPU minimum (1 vCPU recommended)
- **Storage**: 1GB for models and dependencies

## Troubleshooting

### Models Not Loading
- Check `app/models/artifacts/` directory exists
- Verify all `.pkl` files are present
- Check Docker build logs for training errors

### Slow Performance
- Upgrade Render plan (free tier has limited resources)
- Check if service is sleeping (free tier sleeps after 15min inactivity)
- Monitor response times in Render dashboard

### Build Failures
- Verify Python version (3.11)
- Check requirements.txt dependencies
- Review Docker build logs in Render dashboard

## Support

For issues or questions:
- Check application logs in Render dashboard
- Review error messages at `/health` endpoint
- Verify model artifacts are generated correctly
