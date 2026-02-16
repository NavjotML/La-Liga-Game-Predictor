import json
from datetime import datetime
from collections import defaultdict

class ModelHealthTracker:
    def __init__(self):
        self.predictions=[]
        self.errors=[]

    def log_prediction(self,home_team,away_team,prediction,probabilities,latency,features):
        """Log each prediction"""
        log_entry={
            "timestamp":datetime.now().isoformat(),
            "home_team":home_team,
            "away_team":away_team,
            "prediction":int(prediction),
            "prob_home_win":float(probabilities[1]),
            "prob_not_home_win":float(probabilities[0]),
            "confidence":float(max(probabilities)),
            "latency_ms":float(latency*1000),
            "elo_diff":features["elo_diff"],
            "form_diff":features['form']


        }
        self.predictions.append(log_entry)

        #save
        with open('model_predictions.json','a') as f:
            f.write(json.dumps(log_entry)+ '\n')

    def get_health_metrics(self):
        """calculate health metrics"""
        if not self.predictions:
            return None
        
        latencies=[p['latency_ms'] for p in self.predictions]
        confidences=[p['confidence'] for p in self.predictions]
        predictions_dist=[p['prediction']for p in self.predictions]

        return {
            "total_predictions":len(self.predictions),
            "avg_latency_ms":sum(latencies) / len(latencies),
            "max_latency_ms":max(latencies),
            "min_latency_ms":min(latencies),
            "avg_confidence":sum(confidences)/len(confidences),
            "home_win_predictions":sum(predictions_dist),
            "not_home_win_predictions":len(predictions_dist)-sum(predictions_dist),
            "error_count":len(self.errors),
            "last_updated":datetime.now().isoformat()

        }
    
tracker=ModelHealthTracker()