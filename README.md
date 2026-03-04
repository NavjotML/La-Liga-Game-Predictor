# La Liga Game Predictor

**🔗 [Live Demo](https://la-liga-game-predictor.onrender.com/)**

Advanced football team rating system for La Liga matches (2012–present) achieving **71.3% ROC-AUC** and **83.6% accuracy** on high-confidence predictions.

## Rating Algorithms
- **Elo** – Dynamic K-factor with home advantage weighting
- **TrueSkill** – Bayesian rating system with uncertainty modeling
- **Glicko-2** – Rating deviation and volatility tracking

## Features
- Historical rating evolution across 13+ seasons (980+ matches)
- Predictive features: `elo_diff`, `ts_conservative_diff`, `glicko_diff`
- Gradient Boosting classifier with probability calibration
- Interactive Streamlit dashboard with temporal visualizations
- Confidence-stratified predictions (≥70% threshold: 83.6% accuracy)

## Tech Stack
- **Core:** Python, pandas, numpy, matplotlib
- **ML:** scikit-learn (Gradient Boosting, Calibrated Classifiers)
- **Ratings:** trueskill, glicko2
- **Deployment:** Streamlit, Docker, Render

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

Access at `http://localhost:8501`

## Performance Highlights
- **Overall Accuracy:** 65.8%
- **ROC-AUC:** 0.7133
- **Dataset:** 2012–2025 (980+ matches)
- **Best Season:** 77.8% (2025)

