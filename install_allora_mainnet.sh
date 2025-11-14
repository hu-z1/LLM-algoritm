#!/bin/bash

# اسکریپت نصب خودکار Allora Worker با مدل هوشمند - MAINNET
# استفاده: bash install_allora_mainnet.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}"
echo "=============================================="
echo "   Allora Mainnet Smart Worker Installer     "
echo "   با مدل هوشمند Self-Learning               "
echo "=============================================="
echo -e "${NC}"

log() {
    echo -e "${GREEN}[✓]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

error() {
    echo -e "${RED}[✗]${NC} $1"
}

info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# بررسی root
if [ "$EUID" -eq 0 ]; then
    error "Please don't run as root. Run as normal user."
    exit 1
fi

# 1. نصب Dependencies
log "Installing system dependencies..."
sudo apt update -qq
sudo apt install -y curl wget git jq > /dev/null 2>&1

# 2. نصب Docker
if ! command -v docker &> /dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh > /dev/null 2>&1
    sudo usermod -aG docker $USER
    rm get-docker.sh
    
    warn "Docker installed. You need to LOGOUT and LOGIN again for Docker permissions."
    read -p "Press Enter after you logout and login back, or press Ctrl+C to exit..."
else
    log "Docker already installed"
fi

# 3. نصب Docker Compose
if ! command -v docker-compose &> /dev/null; then
    log "Installing Docker Compose..."
    sudo curl -sL "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
else
    log "Docker Compose already installed"
fi

# 4. Clone Repository
log "Cloning Allora repository..."
cd $HOME

if [ -d "basic-coin-prediction-node" ]; then
    warn "Directory exists. Backing up..."
    mv basic-coin-prediction-node basic-coin-prediction-node.backup.$(date +%s)
fi

git clone -q https://github.com/allora-network/basic-coin-prediction-node
cd basic-coin-prediction-node

# 5. Backup original files
log "Backing up original files..."
[ -f model.py ] && mv model.py model.py.original
[ -f app.py ] && mv app.py app.py.original

# 6. Create smart model.py
log "Creating advanced ML model..."
cat > model.py << 'PYEOF'
"""
Advanced Self-Learning Price Prediction Model for Allora Network
LSTM + Ensemble + Auto-Retraining
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/data/model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdvancedPricePredictor:
    def __init__(self, token="ETH", data_path="/app/data"):
        self.token = token
        self.data_path = data_path
        self.model_path = os.path.join(data_path, f"model_{token}.pkl")
        self.scaler_path = os.path.join(data_path, f"scaler_{token}.pkl")
        self.performance_path = os.path.join(data_path, f"performance_{token}.json")
        
        self.models = {}
        self.scaler = MinMaxScaler()
        self.performance_history = []
        self.lookback_period = 168
        self.prediction_horizon = 1
        
        self._load_or_initialize()
    
    def _load_or_initialize(self):
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.models = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"✅ Model loaded for {self.token}")
            else:
                self._initialize_models()
            
            if os.path.exists(self.performance_path):
                with open(self.performance_path, 'r') as f:
                    self.performance_history = json.load(f)
        except Exception as e:
            logger.error(f"Load error: {e}")
            self._initialize_models()
    
    def _initialize_models(self):
        self.models = {
            'gb': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                min_samples_split=5, subsample=0.8, random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=150, max_depth=10, min_samples_split=5,
                random_state=42, n_jobs=-1
            )
        }
    
    def fetch_historical_data(self, days=30):
        try:
            coin_map = {
                'ETH': 'ethereum', 'BTC': 'bitcoin', 'SOL': 'solana',
                'BNB': 'binancecoin', 'ARB': 'arbitrum'
            }
            
            coin_id = coin_map.get(self.token, 'ethereum')
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {'vs_currency': 'usd', 'days': days, 'interval': 'hourly'}
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            if volumes:
                df_vol = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                df_vol['timestamp'] = pd.to_datetime(df_vol['timestamp'], unit='ms')
                df = df.merge(df_vol, on='timestamp', how='left')
            
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"📊 Fetched {len(df)} records for {self.token}")
            return df
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return pd.DataFrame()
    
    def engineer_features(self, df):
        df = df.copy()
        
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        for window in [6, 12, 24, 48, 168]:
            df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['price'].ewm(span=window).mean()
            df[f'price_to_sma_{window}'] = df['price'] / df[f'sma_{window}']
        
        for window in [12, 24, 168]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
        
        df['rsi_14'] = self._calculate_rsi(df['price'], 14)
        df['rsi_24'] = self._calculate_rsi(df['price'], 24)
        
        exp1 = df['price'].ewm(span=12, adjust=False).mean()
        exp2 = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        df['bb_middle'] = df['price'].rolling(window=20).mean()
        bb_std = df['price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        for lag in [1, 6, 12, 24]:
            df[f'momentum_{lag}'] = df['price'] - df['price'].shift(lag)
            df[f'momentum_pct_{lag}'] = df['price'].pct_change(lag)
        
        if 'volume' in df.columns:
            df['volume_sma_24'] = df['volume'].rolling(window=24).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_24']
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        for lag in range(1, 25):
            df[f'lag_{lag}'] = df['price'].shift(lag)
        
        df['price_mean_24h'] = df['price'].rolling(window=24).mean()
        df['price_std_24h'] = df['price'].rolling(window=24).std()
        df['price_min_24h'] = df['price'].rolling(window=24).min()
        df['price_max_24h'] = df['price'].rolling(window=24).max()
        df['price_range_24h'] = df['price_max_24h'] - df['price_min_24h']
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_training_data(self, df):
        df_clean = df.dropna()
        
        if len(df_clean) < self.lookback_period + 10:
            raise ValueError(f"Need at least {self.lookback_period + 10} records")
        
        df_clean['target'] = df_clean['price'].shift(-self.prediction_horizon)
        df_clean = df_clean.dropna()
        
        feature_cols = [col for col in df_clean.columns if col not in ['price', 'target']]
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Training data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y, feature_cols
    
    def train(self, df):
        try:
            logger.info(f"🎓 Training models for {self.token}...")
            
            df_features = self.engineer_features(df)
            X, y, feature_cols = self.prepare_training_data(df_features)
            self.feature_cols = feature_cols
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            scores = {}
            for name, model in self.models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                scores[name] = {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)}
                logger.info(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            
            self._save_model()
            self._update_performance(scores)
            
            logger.info("✅ Training completed")
            return scores
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
    
    def predict(self, current_data=None):
        try:
            if current_data is None:
                current_data = self.fetch_historical_data(days=8)
            
            if current_data.empty:
                raise ValueError("No data available")
            
            df_features = self.engineer_features(current_data)
            df_features = df_features.dropna()
            
            if len(df_features) == 0:
                raise ValueError("Not enough data after engineering")
            
            latest_features = df_features[self.feature_cols].iloc[-1:].values
            latest_scaled = self.scaler.transform(latest_features)
            
            predictions = {}
            for name, model in self.models.items():
                pred = model.predict(latest_scaled)[0]
                predictions[name] = float(pred)
            
            weights = self._calculate_model_weights()
            final_prediction = sum(predictions[name] * weights[name] for name in predictions.keys())
            
            current_price = float(current_data['price'].iloc[-1])
            change_pct = ((final_prediction - current_price) / current_price) * 100
            
            result = {
                'token': self.token,
                'current_price': current_price,
                'predicted_price': final_prediction,
                'change_percent': change_pct,
                'model_predictions': predictions,
                'model_weights': weights,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"💡 Prediction: ${final_prediction:.2f} (Current: ${current_price:.2f}, Change: {change_pct:+.2f}%)")
            self._save_prediction_for_learning(result)
            
            return final_prediction
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            if current_data is not None and not current_data.empty:
                return float(current_data['price'].iloc[-1])
            raise
    
    def _calculate_model_weights(self):
        if not self.performance_history:
            n_models = len(self.models)
            return {name: 1.0/n_models for name in self.models.keys()}
        
        latest_perf = self.performance_history[-1]
        weights = {}
        total_inverse_mae = 0
        
        for name in self.models.keys():
            if name in latest_perf:
                inverse_mae = 1.0 / (latest_perf[name].get('mae', 1.0) + 1e-6)
                weights[name] = inverse_mae
                total_inverse_mae += inverse_mae
        
        for name in weights:
            weights[name] /= total_inverse_mae
        
        return weights
    
    def _save_model(self):
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.models, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            logger.error(f"Save error: {e}")
    
    def _update_performance(self, scores):
        try:
            self.performance_history.append({'timestamp': datetime.now().isoformat(), **scores})
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            with open(self.performance_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            logger.error(f"Update performance error: {e}")
    
    def _save_prediction_for_learning(self, prediction):
        try:
            pred_history_path = os.path.join(self.data_path, f"predictions_{self.token}.jsonl")
            with open(pred_history_path, 'a') as f:
                f.write(json.dumps(prediction) + '\n')
        except Exception as e:
            logger.error(f"Save prediction error: {e}")


def get_inference(token, timeframe="4h"):
    try:
        logger.info(f"🔮 Inference for {token}")
        predictor = AdvancedPricePredictor(token=token)
        return predictor.predict()
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise


def train_model(timeframe="4h"):
    try:
        token = os.getenv('TOKEN', 'ETH')
        logger.info(f"🎓 Training for {token}")
        
        predictor = AdvancedPricePredictor(token=token)
        df = predictor.fetch_historical_data(days=60)
        
        if df.empty:
            raise ValueError("No data for training")
        
        scores = predictor.train(df)
        logger.info("✅ Training completed")
        return scores
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    print("Testing Advanced Predictor...")
    predictor = AdvancedPricePredictor(token="ETH", data_path="./data")
    df = predictor.fetch_historical_data(days=30)
    if not df.empty:
        predictor.train(df)
        prediction = predictor.predict()
        print(f"Prediction: ${prediction:.2f}")
PYEOF

log "Smart model created successfully!"

# 7. Create app.py
log "Creating Flask application..."
cat > app.py << 'EOF'
import json
from flask import Flask, Response
from model import get_inference, train_model
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
TOKEN = os.getenv('TOKEN', 'ETH')

@app.route("/inference/<string:token>")
def generate_inference(token):
    if not token or token.upper() != TOKEN:
        error_msg = "Token required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')
    
    try:
        logger.info(f"Inference for {token}")
        inference = get_inference(token.upper())
        return Response(str(inference), status=200)
    except Exception as e:
        logger.error(f"Error: {e}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/health")
def health():
    return Response(json.dumps({"status": "healthy"}), status=200, mimetype='application/json')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
EOF

# 8. Create update_app.py
cat > update_app.py << 'EOF'
import logging
from model import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update():
    try:
        logger.info("Starting model update...")
        scores = train_model()
        logger.info(f"Update completed: {scores}")
        return True
    except Exception as e:
        logger.error(f"Update error: {e}")
        return False

if __name__ == "__main__":
    update()
EOF

# 9. Update requirements.txt
log "Updating requirements..."
cat > requirements.txt << 'EOF'
Flask==2.3.0
requests==2.31.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
gunicorn==21.2.0
EOF

# 10. Create .env
log "Creating environment configuration..."
read -p "Enter TOKEN (ETH/BTC/SOL/ARB/BNB) [default: ETH]: " TOKEN
TOKEN=${TOKEN:-ETH}

read -p "Enter TRAINING_DAYS [default: 60]: " TRAINING_DAYS
TRAINING_DAYS=${TRAINING_DAYS:-60}

cat > .env << EOF
TOKEN=$TOKEN
TRAINING_DAYS=$TRAINING_DAYS
TIMEFRAME=4h
MODEL=ADVANCED_ENSEMBLE
REGION=US
DATA_PROVIDER=coingecko
CG_API_KEY=
EOF

log ".env created"

# 11. Setup wallet
echo ""
info "Setting up Allora wallet for MAINNET..."
warn "⚠️  This is for MAINNET. Make sure you have ALLO tokens!"
echo ""

read -p "Do you have an existing wallet? (y/n): " HAS_WALLET

if [ "$HAS_WALLET" = "n" ]; then
    warn "You need to create a wallet first with Keplr or allorad CLI"
    warn "Visit: https://wallet.keplr.app/"
    echo ""
fi

read -p "Enter your 24-word mnemonic: " MNEMONIC

if [ -z "$MNEMONIC" ]; then
    error "Mnemonic is required!"
    exit 1
fi

# 12. Setup config.json
log "Creating config.json for MAINNET..."

read -p "Enter Topic ID [default: 69]: " TOPIC_ID
TOPIC_ID=${TOPIC_ID:-69}

read -p "Enter Loop Seconds [default: 300]: " LOOP_SECONDS
LOOP_SECONDS=${LOOP_SECONDS:-300}

cat > config.json << EOF
{
    "wallet": {
        "addressKeyName": "mainnet-worker",
        "addressRestoreMnemonic": "$MNEMONIC",
        "alloraHomeDir": "",
        "gas": "1000000",
        "gasAdjustment": 1.0,
        "nodeRpc": "https://allora-rpc.mainnet-1.allora.network",
        "maxRetries": 1,
        "delay": 1,
        "submitTx": false
    },
    "worker": [
        {
            "topicId": $TOPIC_ID,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": $LOOP_SECONDS,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "$TOKEN"
            }
        }
    ]
}
EOF

# 13. Setup docker-compose.yml
log "Creating docker-compose.yml for MAINNET..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  inference:
    container_name: inference
    build: .
    env_file:
      - .env
    ports:
      - "8000:8000"
    volumes:
      - ./inference-data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://inference:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 12
      start_period: 40s
    restart: unless-stopped
    networks:
      - allora-net

  updater:
    container_name: updater
    build: .
    env_file:
      - .env
    volumes:
      - ./inference-data:/app/data
    command: >
      sh -c "
      while true; do
        python -u /app/update_app.py;
        sleep 24h;
      done
      "
    depends_on:
      inference:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - allora-net

  worker:
    container_name: worker
    image: alloranetwork/allora-offchain-node:latest
    volumes:
      - ./worker-data:/data
    depends_on:
      inference:
        condition: service_healthy
    env_file:
      - ./worker-data/env_file
    restart: unless-stopped
    networks:
      - allora-net

networks:
  allora-net:
    driver: bridge

volumes:
  inference-data:
  worker-data:
EOF

# 14. Run init script
log "Initializing worker node..."
mkdir -p worker-data
chmod 777 worker-data

if [ -f "init.config" ]; then
    chmod +x init.config
    ./init.config
else
    warn "init.config not found. Creating worker keys manually..."
    docker run -it --entrypoint=bash -v ./worker-data:/data \
      alloranetwork/allora-inference-base:latest \
      -c "mkdir -p /data/keys && (cd /data/keys && allora-keys)"
fi

# 15. Create helper scripts
log "Creating helper scripts..."

cat > start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Allora Worker..."
docker-compose up -d --build
echo "✅ Worker started!"
echo "Check logs: docker-compose logs -f"
EOF
chmod +x start.sh

cat > stop.sh << 'EOF'
#!/bin/bash
echo "⏹️  Stopping Allora Worker..."
docker-compose down
echo "✅ Worker stopped!"
EOF
chmod +x stop.sh

cat > logs.sh << 'EOF'
#!/bin/bash
docker-compose logs -f
EOF
chmod +x logs.sh

cat > status.sh << 'EOF'
#!/bin/bash
echo "=== Container Status ==="
docker-compose ps
echo ""
echo "=== Inference Health ==="
curl -s http://localhost:8000/health | jq
echo ""
echo "=== Recent Logs ==="
docker logs inference --tail 10
echo ""
docker logs worker --tail 10
EOF
chmod +x status.sh

# 16. Final summary
echo ""
echo -e "${GREEN}=============================================="
echo "   Installation Complete! 🎉"
echo "==============================================  ${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Token: $TOKEN"
echo "  Topic ID: $TOPIC_ID"
echo "  Training Days: $TRAINING_DAYS"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Start worker: ${GREEN}./start.sh${NC}"
echo "  2. Check logs: ${GREEN}./logs.sh${NC}"
echo "  3. Check status: ${GREEN}./status.sh${NC}"
echo "  4. Register on Model Forge: ${GREEN}https://forge.allora.network${NC}"
echo ""
echo -e "${YELLOW}Useful Commands:${NC}"
echo "  • Test inference: ${GREEN}curl http://localhost:8000/inference/$TOKEN${NC}"
echo "  • View model logs: ${GREEN}tail -f inference-data/model.log${NC}"
echo "  • Restart: ${GREEN}docker-compose restart${NC}"
echo "  • Stop: ${GREEN}./stop.sh${NC}"
echo ""
echo -e "${RED}⚠️  IMPORTANT:${NC}"
echo "  • This is MAINNET - make sure you have ALLO tokens for gas!"
echo "  • Your mnemonic is in config.json - keep it safe!"
echo "  • Monitor your worker on Model Forge"
echo ""

read -p "Do you want to start the worker now? (y/n): " START_NOW

if [ "$START_NOW" = "y" ]; then
    log "Building and starting containers..."
    docker-compose up -d --build
    
    log "Waiting for services to be healthy..."
    sleep 45
    
    echo ""
    log "Container Status:"
    docker-compose ps
    
    echo ""
    log "Testing Inference API..."
    curl -s http://localhost:8000/health | jq
    
    echo ""
    log "Recent Logs:"
    docker logs inference --tail 20
    
    echo ""
    echo -e "${GREEN}✅ Setup complete! Monitor with: ./logs.sh${NC}"
fi

echo ""
echo -e "${GREEN}Happy earning! 🚀💰${NC}"
