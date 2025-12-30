"""
Data Loader Module - Load cat meow audio dataset
"""
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Category mapping
CATEGORIES = {
    'Brushing': 0,
    'UnfamiliarSurroundings': 1,
    'WaitForFood': 2
}

CATEGORY_NAMES = {v: k for k, v in CATEGORIES.items()}


class CatMeowsDataset:
    """Cat meows dataset class"""
    
    def __init__(self, dataset_path, sr=22050):
        self.dataset_path = Path(dataset_path)
        self.sr = sr
        self.audio_files = []
        self.labels = []
        self.category_counts = {}
        
        self._scan_dataset()
    
    def _scan_dataset(self):
        """Scan dataset directory and get all audio files"""
        for category, label in CATEGORIES.items():
            category_path = self.dataset_path / category
            if not category_path.exists():
                logger.warning(f"Category directory not found: {category_path}")
                continue
            
            wav_files = list(category_path.glob('*.wav'))
            self.category_counts[category] = len(wav_files)
            
            for wav_file in wav_files:
                self.audio_files.append(wav_file)
                self.labels.append(label)
        
        logger.info("Dataset loaded:")
        for category, count in self.category_counts.items():
            logger.info(f"  - {category}: {count} samples")
        logger.info(f"  Total: {len(self.audio_files)} samples")
    
    def load_audio(self, file_path):
        """Load a single audio file"""
        return librosa.load(file_path, sr=self.sr)
    
    def load_all_audio(self, show_progress=True):
        """Load all audio data"""
        iterator = tqdm(self.audio_files, desc="Loading audio") if show_progress else self.audio_files
        audio_data = [self.load_audio(f)[0] for f in iterator]
        return audio_data, np.array(self.labels)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        y, _ = self.load_audio(self.audio_files[idx])
        return y, self.labels[idx]


def load_dataset(dataset_path="dataset", sr=22050):
    """Convenience function: load dataset"""
    return CatMeowsDataset(dataset_path, sr)
