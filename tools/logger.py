"""
Logging Utilities Module
"""
import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def setup_logger(name="cat_recognition", log_dir="logs", level=logging.INFO):
    """Setup logger"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    log_file = log_dir / f'{name}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Log file: {log_file}")
    
    return logger


class TrainingResultSaver:
    """Training result saver"""
    
    def __init__(self, output_dir="outputs"):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(output_dir) / f"run_{self.timestamp}"
        
        self.results_dir = self.run_dir / "results"
        self.viz_dir = self.run_dir / "visualizations"
        
        for d in [self.results_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results, name, logger=None):
        """Save training results"""
        path = self.results_dir / f"{name}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._make_serializable(results), f, ensure_ascii=False, indent=2)
        if logger:
            logger.info(f"Results saved: {path}")
    
    def save_summary(self, summary, logger=None):
        """Save training summary"""
        path = self.run_dir / "summary.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._make_serializable(summary), f, ensure_ascii=False, indent=2)
        if logger:
            logger.info(f"Summary saved: {path}")
    
    def _make_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
    
    def get_viz_dir(self):
        """Get visualization directory"""
        return str(self.viz_dir)
    
    def get_run_dir(self):
        """Get run directory"""
        return str(self.run_dir)
