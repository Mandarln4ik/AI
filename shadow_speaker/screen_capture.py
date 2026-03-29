"""
Модуль захвата экрана для получения визуального контекста
"""
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import mss
    import mss.tools
    from PIL import Image
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    logging.warning("mss not installed, screen capture disabled")

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Захват экрана для анализа контекста"""
    
    def __init__(self, monitor_index: int = 1):
        self.monitor_index = monitor_index
        self.last_capture_time = 0
        self.capture_interval = 10.0  # Захват каждые 10 секунд
        self.last_screenshot: Optional[Image.Image] = None
        
        if not MSS_AVAILABLE:
            logger.error("mss library not available, screen capture disabled")
        
        logger.info(f"ScreenCapture initialized (monitor {monitor_index})")
    
    def capture(self) -> Optional[Image.Image]:
        """Сделать скриншот"""
        if not MSS_AVAILABLE:
            return None
        
        try:
            with mss.mss() as sct:
                # Получаем список мониторов
                monitors = sct.monitors
                
                if self.monitor_index >= len(monitors):
                    logger.warning(f"Monitor {self.monitor_index} not found, using primary")
                    self.monitor_index = 0
                
                monitor = monitors[self.monitor_index]
                
                # Делаем скриншот
                screenshot = sct.grab(monitor)
                
                # Конвертируем в PIL Image
                img = Image.frombytes(
                    "RGB",
                    screenshot.size,
                    screenshot.bgra,
                    "raw",
                    "BGRX"
                )
                
                self.last_screenshot = img
                self.last_capture_time = time.time()
                
                logger.debug(f"Screen captured: {img.size}")
                return img
                
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    def should_capture(self) -> bool:
        """Проверить нужно ли делать новый захват"""
        return (time.time() - self.last_capture_time) >= self.capture_interval
    
    def get_context_description(self) -> str:
        """Получить описание текущего скриншота (для отправки в LLM)"""
        if not self.last_screenshot:
            return ""
        
        # Базовая информация об изображении
        width, height = self.last_screenshot.size
        timestamp = time.strftime("%H:%M:%S", time.localtime(self.last_capture_time))
        
        description = f"[Экран: {width}x{height}, время: {timestamp}]"
        
        # В будущем можно добавить:
        # - OCR для чтения текста с экрана
        # - Анализ цветов/объектов
        # - Интеграция с мультимодальной LLM
        
        return description
    
    def save_screenshot(self, path: Optional[str] = None) -> Optional[str]:
        """Сохранить последний скриншот"""
        if not self.last_screenshot:
            return None
        
        if path is None:
            path = f"screenshot_{int(time.time())}.png"
        
        try:
            self.last_screenshot.save(path)
            logger.info(f"Screenshot saved: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            return None
    
    def set_capture_interval(self, seconds: float):
        """Установить интервал захвата"""
        self.capture_interval = max(1.0, seconds)
        logger.info(f"Capture interval set to {seconds}s")
