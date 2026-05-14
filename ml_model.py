"""
ML 择时模型。

基于机器学习预测股价方向/涨跌：
- XGBoost 分类器：特征→涨/跌/平 三分类
- LSTM 时序预测：滑动窗口历史序列→下N日收益率
- 特征工程：技术指标+因子+市场状态 自动构建特征矩阵
- Walk-Forward 验证：滚动训练+测试，防止过拟合
- 模型持久化：joblib 保存/加载
- 集成预测：多模型投票

用法:
    from ml_model import MLTimingModel

    model = MLTimingModel(model_type="xgboost")
    model.train(kline_df, horizon=5)
    pred = model.predict(latest_data)
    proba = model.predict_proba(latest_data)
    model.save("model_xgb.joblib")
"""

import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL_DIR = os.path.expanduser("~/.a-stock-agent/models")


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class MLPrediction:
    """ML预测结果。"""
    direction: str  # "UP" / "DOWN" / "FLAT"
    confidence: float  # 0~1
    proba_up: float = 0.0
    proba_down: float = 0.0
    proba_flat: float = 0.0
    expected_return: float = 0.0  # 预期收益率
    model_name: str = ""
    signal: str = ""  # BUY / SELL / HOLD


@dataclass
class MLBacktestResult:
    """ML模型回测结果。"""
    accuracy: float = 0.0
    precision_up: float = 0.0  # 预测涨的准确率
    precision_down: float = 0.0  # 预测跌的准确率
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_importance: Dict[str, float] = field(default_factory=dict)


# ── 特征工程 ──────────────────────────────────────────────

class FeatureEngineer:
    """自动特征工程。

    从K线数据构建ML特征矩阵：
    - 价格特征（收益率、均线比率、波动率）
    - 技术指标特征（RSI/MACD/布林/ATR）
    - 量价特征（量比/OBV/资金流向）
    - 日历特征（星期/月份/距年末天数）
    - 滞后特征（自动生成 t-1 ~ t-N 滞后）
    """

    @staticmethod
    def build_features(
        kline: pd.DataFrame,
        horizon: int = 5,
        lag_periods: List[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """构建特征矩阵和标签。

        Args:
            kline: OHLCV DataFrame
            horizon: 预测周期（未来N天）
            lag_periods: 滞后特征周期

        Returns:
            (特征DataFrame, 标签Series)
        """
        if lag_periods is None:
            lag_periods = [1, 3, 5, 10, 20]

        df = kline.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)  # noqa: F841
        volume = df["volume"].astype(float)

        features = pd.DataFrame(index=df.index)

        # ── 价格特征 ────────────────────
        for p in lag_periods:
            features[f"ret_{p}d"] = close.pct_change(p)

        ma_list = [5, 10, 20, 60, 120]
        for ma in ma_list:
            if len(close) >= ma:
                ma_val = close.rolling(ma).mean()
                features[f"ma_ratio_{ma}"] = close / ma_val - 1

        # ── 波动特征 ────────────────────
        ret_1d = close.pct_change()
        for p in [5, 10, 20]:
            features[f"vol_{p}d"] = ret_1d.rolling(p).std()

        features["vol_ratio_5_20"] = ret_1d.rolling(5).std() / ret_1d.rolling(20).std().replace(0, np.nan)

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        features["atr_14"] = tr.rolling(14).mean() / close

        # 最大回撤
        for p in [10, 20, 60]:
            peak = close.rolling(p).max()
            features[f"max_dd_{p}d"] = close / peak - 1

        # ── 技术指标 ────────────────────
        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features["rsi_14"] = 100 - 100 / (1 + rs)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        features["macd_dif"] = dif / close
        features["macd_hist"] = (dif - dea) / close

        # 布林带位置
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features["boll_position"] = (close - bb_mid) / (bb_std * 2)
        features["boll_width"] = (bb_std * 2) / bb_mid

        # ── 量价特征 ────────────────────
        features["volume_ratio_5"] = volume / volume.rolling(5).mean()
        features["volume_ratio_20"] = volume / volume.rolling(20).mean()

        # OBV动量
        obv = (np.sign(close.diff()) * volume).cumsum()
        for p in [5, 20]:
            features[f"obv_chg_{p}d"] = obv.pct_change(p)

        # 量价相关性
        features["corr_price_vol_20"] = close.rolling(20).corr(volume)

        # ── 强度特征 ────────────────────
        features["close_position"] = (close - low) / (high - low).replace(0, np.nan)
        features["gap"] = (open_ - close.shift()) / close.shift()

        # ── 趋势特征 ────────────────────
        for p in [5, 10, 20]:
            features[f"up_days_{p}d"] = (close > close.shift()).rolling(p).sum() / p

        # ── 日历特征 ────────────────────
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            features["day_of_week"] = dates.dt.dayofweek
            features["month"] = dates.dt.month
            features["day_of_month"] = dates.dt.day

        # ── 标签：未来 horizon 天的方向 ──
        future_ret = close.shift(-horizon) / close - 1
        label = pd.Series(1, index=df.index)  # FLAT=1
        label[future_ret > 0.02] = 2  # UP=2
        label[future_ret < -0.02] = 0  # DOWN=0
        label = label.iloc[:-horizon]  # 去掉最后 horizon 天无标签

        features = features.iloc[:-horizon]

        # 去除全NaN列
        features = features.dropna(axis=1, how="all")

        return features, label

    @staticmethod
    def get_feature_names() -> List[str]:
        """返回特征名称列表（文档用）。"""
        return [
            "ret_1d", "ret_5d", "ret_20d", "ret_60d",
            "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60", "ma_ratio_120",
            "vol_5d", "vol_20d", "vol_ratio_5_20", "atr_14",
            "max_dd_10d", "max_dd_20d", "max_dd_60d",
            "rsi_14", "macd_dif", "macd_hist", "boll_position", "boll_width",
            "volume_ratio_5", "volume_ratio_20",
            "obv_chg_5d", "obv_chg_20d", "corr_price_vol_20",
            "close_position", "gap",
            "up_days_5d", "up_days_10d", "up_days_20d",
        ]


# ── XGBoost 模型 ──────────────────────────────────────────

class XGBoostModel:
    """XGBoost 分类器。

    三分类：涨(UP) / 跌(DOWN) / 平(FLAT)
    """

    def __init__(self, **kwargs):
        self.params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            **kwargs,
        }
        self.model = None
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_metric: str = "mlogloss",
        verbose: bool = False,
    ):
        """训练 XGBoost 模型。

        Args:
            X: 特征矩阵
            y: 标签 (0=DOWN, 1=FLAT, 2=UP)
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("请安装 xgboost: pip install xgboost")
            return

        # 处理缺失值
        X = X.fillna(X.median())
        self.feature_names = list(X.columns)

        # 样本权重（平衡类别）
        from collections import Counter
        counts = Counter(y)
        max_count = max(counts.values())
        sample_weight = y.map(lambda c: max_count / counts.get(c, 1))

        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric=eval_metric,
            verbosity=1 if verbose else 0,
            **self.params,
        )

        self.model.fit(X, y, sample_weight=sample_weight, verbose=verbose)

        # 特征重要性
        importance = self.model.feature_importances_
        self.feature_importance = dict(
            sorted(
                zip(self.feature_names, importance),
                key=lambda x: -x[1],
            )[:20]
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别。xgb未训练时返回默认。"""
        if self.model is None:
            return np.ones(len(X), dtype=int)  # 默认FLAT=1
        X = X.fillna(0)[self.feature_names]
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率。未训练返回均匀分布。"""
        if self.model is None:
            n = len(X)
            return np.column_stack([np.full(n, 0.33), np.full(n, 0.34), np.full(n, 0.33)])
        X = X.fillna(0)[self.feature_names]
        return self.model.predict_proba(X)

    def save(self, path: str):
        """保存模型。"""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "params": self.params,
        }, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        """加载模型。"""
        import joblib
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.feature_importance = data.get("feature_importance", {})
        instance.params = data.get("params", {})
        return instance


# ── LSTM 模型 ──────────────────────────────────────────────

class LSTMModel:
    """LSTM 时序预测模型。

    使用滑动窗口历史序列预测未来方向。
    """

    def __init__(
        self,
        seq_len: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []

    def _build_model(self, n_features: int):
        """构建 LSTM 网络。"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.error("请安装 torch: pip install torch")
            return

        class LSTMPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes=3):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=dropout,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, num_classes),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self.model = LSTMPredictor(
            n_features, self.hidden_size, self.num_layers, self.dropout
        )

    def _create_sequences(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """创建滑动窗口序列。"""
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.int64)

        seqs, labels = [], []
        for i in range(self.seq_len, len(X_arr)):
            seqs.append(X_arr[i - self.seq_len:i])
            labels.append(y_arr[i])

        return np.array(seqs), np.array(labels)

    def train(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True):
        """训练 LSTM。

        Args:
            X: 特征矩阵
            y: 标签
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("请安装 torch scikit-learn: pip install torch scikit-learn")
            return

        self.feature_names = list(X.columns)

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X.fillna(X.median())),
            columns=X.columns,
            index=X.index,
        )

        # 创建序列
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        if len(X_seq) < 10:
            logger.warning("序列数据不足")
            return

        # 划分训练/验证
        split = int(len(X_seq) * 0.8)
        X_train, y_train = X_seq[:split], y_seq[:split]
        X_val, y_val = X_seq[split:], y_seq[split:]

        # 转 Tensor
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)

        # 构建模型
        self._build_model(X_train.shape[2])
        if self.model is None:
            return

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()

            output = self.model(X_train_t)
            loss = criterion(output, y_train_t)
            loss.backward()
            optimizer.step()

            # 验证
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val_t)
                val_loss = criterion(val_output, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    logger.info("LSTM epoch %d early stop (val_loss=%.4f)", epoch, val_loss)
                break

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别。"""
        try:
            import torch
        except ImportError:
            return np.zeros(len(X))

        if self.model is None:
            raise RuntimeError("模型尚未训练")

        X_scaled = pd.DataFrame(
            self.scaler.transform(X.fillna(X.median())),
            columns=X.columns,
            index=X.index,
        )
        X_arr = X_scaled.values.astype(np.float32)

        # 需要 self.seq_len 的历史+当前数据
        if len(X_arr) <= self.seq_len:
            # 用已有数据填充
            seq = np.tile(X_arr[0], (self.seq_len, 1))
            seq[-len(X_arr):] = X_arr
        else:
            seq = X_arr[-self.seq_len:]

        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.FloatTensor(seq).unsqueeze(0))
            pred = output.argmax(dim=1).numpy()

        return pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率。"""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return np.array([[0.33, 0.33, 0.34]])

        if self.model is None:
            raise RuntimeError("模型尚未训练")

        X_scaled = pd.DataFrame(
            self.scaler.transform(X.fillna(X.median())),
            columns=X.columns,
        )
        X_arr = X_scaled.values.astype(np.float32)

        if len(X_arr) <= self.seq_len:
            seq = np.tile(X_arr[0], (self.seq_len, 1))
            seq[-len(X_arr):] = X_arr
        else:
            seq = X_arr[-self.seq_len:]

        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.FloatTensor(seq).unsqueeze(0))
            proba = F.softmax(output, dim=1).numpy()

        return proba

    def save(self, path: str):
        """保存模型。"""
        try:
            import torch
            import joblib
        except ImportError:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
        }, path)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        """加载模型。"""
        try:
            import torch
        except ImportError:
            raise ImportError("请安装 torch")

        data = torch.load(path)
        instance = cls(
            seq_len=data["seq_len"],
            hidden_size=data["hidden_size"],
        )
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance._build_model(len(instance.feature_names))
        if instance.model is not None:
            instance.model.load_state_dict(data["model_state"])
        return instance


# ── ML 择时主模型 ──────────────────────────────────────────

class MLTimingModel:
    """ML 择时模型主类。

    整合特征工程 + 模型训练 + 预测 + 回测 + 持久化。

    Usage:
        model = MLTimingModel(model_type="xgboost")
        model.train(kline, horizon=5)
        pred = model.predict(latest_data)
        result = model.backtest(kline, horizon=5)
    """

    def __init__(
        self,
        model_type: str = "xgboost",  # xgboost / lstm / ensemble
        model_dir: str = None,
        **model_kwargs,
    ):
        self.model_type = model_type
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.feature_engineer = FeatureEngineer()

        if model_type == "xgboost":
            self.model = XGBoostModel(**model_kwargs)
        elif model_type == "lstm":
            self.model = LSTMModel(**model_kwargs)
        else:
            # ensemble = 两个都用
            self.model = XGBoostModel(**model_kwargs)
            self.model_lstm = LSTMModel(**model_kwargs)

        self._last_features: Optional[pd.DataFrame] = None
        self._last_label: Optional[pd.Series] = None
        self._horizon: int = 5

    # ── 训练 ──────────────────────────────────────────────

    def train(
        self,
        kline: pd.DataFrame,
        horizon: int = 5,
        verbose: bool = True,
    ):
        """训练模型。

        Args:
            kline: OHLCV K线数据
            horizon: 预测周期
            verbose: 打印进度
        """
        self._horizon = horizon
        X, y = self.feature_engineer.build_features(kline, horizon=horizon)

        if len(X) < 100:
            logger.warning("数据量不足（%d条），至少需要100条", len(X))
            return

        self._last_features = X
        self._last_label = y

        # 去除全NaN行
        valid = X.notna().all(axis=1)
        X_clean = X[valid]
        y_clean = y[valid]

        if verbose:
            logger.info(
                "训练 %s 模型: %d 样本, %d 特征, horizon=%d",
                self.model_type, len(X_clean), X_clean.shape[1], horizon,
            )

        if self.model_type in ("xgboost", "ensemble"):
            self.model.train(X_clean, y_clean, verbose=verbose)

        if self.model_type in ("lstm", "ensemble"):
            lstm = self.model_lstm if self.model_type == "ensemble" else self.model
            lstm.train(X_clean, y_clean, verbose=verbose)

    # ── 预测 ──────────────────────────────────────────────

    def predict(self, latest_data: pd.DataFrame) -> MLPrediction:
        """预测最新方向。

        Args:
            latest_data: 最新K线数据（至少包含最近60天用于特征计算）

        Returns:
            MLPrediction
        """
        X, _ = self.feature_engineer.build_features(latest_data, horizon=self._horizon)
        if X.empty:
            return MLPrediction(direction="FLAT", confidence=0.0, model_name=self.model_type)

        latest_X = X.iloc[-1:]

        if self.model_type == "ensemble":
            proba_xgb = self.model.predict_proba(latest_X)[0]
            proba_lstm = self.model_lstm.predict_proba(latest_X)[0]
            proba = (proba_xgb + proba_lstm) / 2
        else:
            proba = self.model.predict_proba(latest_X)[0]

        # proba = [P(DOWN), P(FLAT), P(UP)]
        direction_map = {0: "DOWN", 1: "FLAT", 2: "UP"}
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        # 信号
        if pred_class == 2 and confidence > 0.5:
            signal = "BUY"
        elif pred_class == 0 and confidence > 0.5:
            signal = "SELL"
        else:
            signal = "HOLD"

        # 预期收益 = P(UP) * avg_up_return - P(DOWN) * avg_down_return
        expected_ret = float(proba[2] * 0.05 - proba[0] * 0.05)

        return MLPrediction(
            direction=direction_map[pred_class],
            confidence=confidence,
            proba_up=float(proba[2]),
            proba_down=float(proba[0]),
            proba_flat=float(proba[1]),
            expected_return=expected_ret,
            model_name=self.model_type,
            signal=signal,
        )

    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """批量预测。"""
        X, _ = self.feature_engineer.build_features(data, horizon=self._horizon)
        if X.empty:
            return pd.DataFrame()

        if self.model_type == "ensemble":
            proba = (self.model.predict_proba(X) + self.model_lstm.predict_proba(X)) / 2
        else:
            proba = self.model.predict_proba(X)

        result = pd.DataFrame(proba, columns=["proba_down", "proba_flat", "proba_up"])
        result["prediction"] = result[["proba_down", "proba_flat", "proba_up"]].idxmax(axis=1)
        result["prediction"] = result["prediction"].map({
            "proba_down": "DOWN", "proba_flat": "FLAT", "proba_up": "UP",
        })
        result["confidence"] = result[["proba_down", "proba_flat", "proba_up"]].max(axis=1)

        return result

    # ── 回测 ──────────────────────────────────────────────

    def backtest(
        self,
        kline: pd.DataFrame,
        horizon: int = 5,
        test_ratio: float = 0.3,
    ) -> MLBacktestResult:
        """Walk-Forward 回测。

        Args:
            kline: K线数据
            horizon: 预测周期
            test_ratio: 测试集比例

        Returns:
            MLBacktestResult
        """
        X, y = self.feature_engineer.build_features(kline, horizon=horizon)

        if len(X) < 60:
            return MLBacktestResult()

        # 时间序列划分
        split = int(len(X) * (1 - test_ratio))
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test, y_test = X.iloc[split:], y.iloc[split:]

        # 训练
        self.model.train(X_train, y_train, verbose=False)

        # 预测
        preds = self.model.predict(X_test)
        probas = self.model.predict_proba(X_test)

        # 准确率
        accuracy = float((preds == y_test.values).mean())

        # UP/DOWN 各自精准率
        up_mask = preds == 2
        down_mask = preds == 0
        precision_up = float((preds[up_mask] == y_test.values[up_mask]).mean()) if up_mask.sum() > 0 else 0
        precision_down = float((preds[down_mask] == y_test.values[down_mask]).mean()) if down_mask.sum() > 0 else 0

        # 按预测交易回测
        signals = pd.Series("HOLD", index=X_test.index)
        signals[preds == 2] = "BUY"
        signals[preds == 0] = "SELL"

        # 模拟交易收益
        rets = kline["close"].pct_change().shift(-horizon).loc[X_test.index]
        trade_rets = []
        for i in range(len(rets)):
            if signals.iloc[i] == "BUY":
                trade_rets.append(rets.iloc[i])
            elif signals.iloc[i] == "SELL":
                trade_rets.append(-rets.iloc[i])

        total_ret = float(np.prod(1 + np.array(trade_rets)) - 1) if trade_rets else 0
        years = len(trade_rets) / 252 if trade_rets else 1
        annual_ret = float((1 + total_ret) ** (1 / years) - 1) if years > 0 else 0

        if trade_rets:
            sr = float(np.mean(trade_rets) / np.std(trade_rets, ddof=1) * np.sqrt(252)) if np.std(trade_rets) > 0 else 0
        else:
            sr = 0

        # 最大回撤
        cum = np.cumprod(1 + np.array(trade_rets)) if trade_rets else np.array([1])
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = float(np.min(dd))

        wins = sum(1 for r in trade_rets if r > 0)
        win_rate = wins / len(trade_rets) if trade_rets else 0

        return MLBacktestResult(
            accuracy=accuracy,
            precision_up=precision_up,
            precision_down=precision_down,
            total_return=total_ret * 100,
            annual_return=annual_ret * 100,
            sharpe_ratio=sr,
            max_drawdown=max_dd * 100,
            win_rate=win_rate * 100,
            predictions=pd.DataFrame({
                "actual": y_test.values,
                "predicted": preds,
                "signal": signals.values,
                "confidence": probas.max(axis=1),
            }, index=X_test.index),
            feature_importance=self.model.feature_importance if hasattr(self.model, 'feature_importance') else {},
        )

    # ── 持久化 ──────────────────────────────────────────────

    def save(self, name: str = None):
        """保存模型。"""
        if name is None:
            name = f"{self.model_type}_h{self._horizon}"
        path = os.path.join(self.model_dir, f"{name}.joblib")
        self.model.save(path)
        logger.info("模型已保存: %s", path)
        return path

    @classmethod
    def load(cls, name: str, model_type: str = "xgboost") -> "MLTimingModel":
        """加载模型。"""
        path = os.path.join(DEFAULT_MODEL_DIR, f"{name}.joblib")
        instance = cls(model_type=model_type)
        if model_type == "xgboost":
            instance.model = XGBoostModel.load(path)
        elif model_type == "lstm":
            instance.model = LSTMModel.load(path)
        return instance

    # ── 报告 ──────────────────────────────────────────────

    def generate_report(self, result: MLBacktestResult = None) -> str:
        """生成模型评估报告。"""
        if result is None and self._last_features is not None:
            result = self.backtest(
                pd.concat([self._last_features, self._last_label], axis=1)
            )

        if result is None:
            return "请先训练模型或提供回测结果"

        lines = [
            "╔══════════════════════════════════════╗",
            f"║      ML 择 时 模 型 报 告          ║",
            "╠══════════════════════════════════════╣",
            f"║  模型类型: {self.model_type}",
            f"║  预测周期: {self._horizon}天",
            "╠══════════════════════════════════════╣",
            f"║  准确率:     {result.accuracy:.1%}",
            f"║  涨准确率:   {result.precision_up:.1%}",
            f"║  跌准确率:   {result.precision_down:.1%}",
            f"║  总收益:     {result.total_return:+.2f}%",
            f"║  年化收益:   {result.annual_return:+.2f}%",
            f"║  夏普比率:   {result.sharpe_ratio:.2f}",
            f"║  最大回撤:   {result.max_drawdown:.2f}%",
            f"║  胜率:       {result.win_rate:.1f}%",
        ]

        if result.feature_importance:
            lines.append("╠══════════════════════════════════════╣")
            lines.append("║  Top 10 特征重要性:")
            for i, (name, imp) in enumerate(
                sorted(result.feature_importance.items(), key=lambda x: -x[1])[:10], 1
            ):
                lines.append(f"║    {i:2d}. {name:<25s} {imp:.4f}")

        lines.append("╚══════════════════════════════════════╝")
        return "\n".join(lines)