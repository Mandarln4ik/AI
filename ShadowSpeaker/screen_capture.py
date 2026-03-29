import mss
import mss.tools
from PIL import Image
import io
import base64
import threading
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from config import Config


class ScreenCapture:
    """Захват экрана для визуального контекста"""
    
    def __init__(self, config: Config):
        self.config = config
        self.enabled = config.screen.enabled
        self.capture_interval = config.screen.capture_interval
        self.monitor_index = config.screen.monitor_index
        self.resize_dims = (config.screen.resize_width, config.screen.resize_height)
        
        self.last_screenshot: Optional[Image.Image] = None
        self.last_screenshot_time: Optional[datetime] = None
        self.screenshot_history: List[Dict[str, Any]] = []  # История скриншотов
        self.max_history = 5  # Хранить последние 5 скриншотов
        
        self.is_running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Callback функции для новых скриншотов
        self.callbacks = []
    
    def register_callback(self, callback) -> None:
        """Регистрация callback функции для новых скриншотов"""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, screenshot: Image.Image, timestamp: datetime) -> None:
        """Уведомление callback функций"""
        for callback in self.callbacks:
            try:
                callback(screenshot, timestamp)
            except Exception as e:
                print(f"Ошибка в callback скриншота: {e}")
    
    def start_capture(self) -> None:
        """Запуск захвата экрана"""
        if not self.enabled:
            print("Захват экрана отключен в конфигурации")
            return
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print(f"Запуск захвата экрана (монитор {self.monitor_index}, интервал {self.capture_interval}с)")
    
    def stop_capture(self) -> None:
        """Остановка захвата экрана"""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        print("Захват экрана остановлен")
    
    def _capture_loop(self) -> None:
        """Основной цикл захвата экрана"""
        while self.is_running:
            try:
                self._take_screenshot()
                time.sleep(self.capture_interval)
            except Exception as e:
                print(f"Ошибка захвата экрана: {e}")
                time.sleep(1.0)
    
    def _take_screenshot(self) -> Optional[Image.Image]:
        """Сделать скриншот"""
        try:
            with mss.mss() as sct:
                # Получение информации о мониторе
                monitors = sct.monitors
                if self.monitor_index >= len(monitors):
                    print(f"Монитор с индексом {self.monitor_index} не найден, используем первый")
                    monitor = monitors[1] if len(monitors) > 1 else monitors[0]
                else:
                    monitor = monitors[self.monitor_index] if self.monitor_index > 0 else monitors[1]
                
                # Захват области монитора
                screenshot = sct.grab(monitor)
                
                # Конвертация в PIL Image
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                
                # Изменение размера для оптимизации
                img_resized = img.resize(self.resize_dims, Image.Resampling.LANCZOS)
                
                timestamp = datetime.now()
                
                with self.lock:
                    self.last_screenshot = img_resized
                    self.last_screenshot_time = timestamp
                    
                    # Добавление в историю
                    self.screenshot_history.append({
                        "image": img_resized,
                        "timestamp": timestamp
                    })
                    
                    # Ограничение истории
                    if len(self.screenshot_history) > self.max_history:
                        self.screenshot_history.pop(0)
                
                # Уведомление callback функций
                self._notify_callbacks(img_resized, timestamp)
                
                return img_resized
                
        except Exception as e:
            print(f"Ошибка при создании скриншота: {e}")
            return None
    
    def get_latest_screenshot(self) -> Optional[Image.Image]:
        """Получение последнего скриншота"""
        with self.lock:
            return self.last_screenshot
    
    def get_latest_screenshot_base64(self) -> Optional[str]:
        """Получение последнего скриншота в формате base64"""
        with self.lock:
            if self.last_screenshot is None:
                return None
            
            buffered = io.BytesIO()
            self.last_screenshot.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def get_screenshot_context(self) -> str:
        """Получение текстового описания контекста экрана (для LLM)"""
        with self.lock:
            if self.last_screenshot is None:
                return "Визуальный контекст недоступен."
            
            time_str = self.last_screenshot_time.strftime("%H:%M:%S") if self.last_screenshot_time else "неизвестно"
            return f"[Скриншот сделан в {time_str}, разрешение {self.resize_dims[0]}x{self.resize_dims[1]}]"
    
    def get_recent_screenshots(self, count: int = 3) -> List[Dict[str, Any]]:
        """Получение последних N скриншотов"""
        with self.lock:
            return self.screenshot_history[-count:] if self.screenshot_history else []
