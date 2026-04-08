import logging
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton cache to avoid loading the same models multiple times
    across different analytics modules (like yolov8m-pose.pt).
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._cache = {}
        return cls._instance

    def get_model(self, model_path: str, device: str = None):
        """
        Get or load a YOLO model from the cache.
        """
        if YOLO is None:
            raise ImportError("ultralytics is required to construct YOLO models")

        if model_path in self._cache:
            logger.debug(f"Returning cached model: {model_path}")
            return self._cache[model_path]
            
        logger.info(f"Loading model into cache: {model_path} on {device}")
        try:
            model = YOLO(model_path)
            if device:
                model.to(device)
            self._cache[model_path] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise

    def clear(self):
        """Clear cache and free GPU memory."""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global accessor
model_manager = ModelManager()
