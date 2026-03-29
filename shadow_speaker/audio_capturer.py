"""
Захват аудио из конкретных приложений на Windows
Использует pycaw и pywin32 для захвата звука из выбранных процессов
"""
import numpy as np
import sounddevice as sd
from loguru import logger
from typing import Optional, Callable, Generator
import psutil
import threading
import queue
import time

try:
    from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    import ctypes
    from ctypes import cast, POINTER
    from comtypes import IUnknown
except ImportError as e:
    logger.warning(f"Windows-specific audio modules not available: {e}")


class AudioCapturer:
    """
    Захватывает аудио из конкретного приложения или системного выхода
    Для работы с конкретными приложениями требуется Virtual Audio Cable
    """
    
    def __init__(self, config):
        self.config = config
        self.audio_queue = queue.Queue()
        self.is_capturing = False
        self.stream = None
        self.target_process_name = None
        self._co_initialized = False
        
    def find_application_process(self, app_name: str) -> Optional[psutil.Process]:
        """Найти процесс по имени приложения"""
        for proc in psutil.process_iter(['name', 'pid']):
            try:
                if app_name.lower() in proc.info['name'].lower():
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def list_available_devices(self) -> list:
        """Список доступных аудио устройств"""
        devices = sd.query_devices()
        available = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0 or dev['max_output_channels'] > 0:
                available.append({
                    'index': i,
                    'name': dev['name'],
                    'input_channels': dev['max_input_channels'],
                    'output_channels': dev['max_output_channels'],
                    'sample_rate': dev['default_samplerate']
                })
        return available
    
    def select_virtual_cable_device(self) -> Optional[int]:
        """
        Найти устройство Virtual Audio Cable
        Пользователь должен настроить маршрутизацию звука из Discord/Teamspeak в VAC
        """
        devices = self.list_available_devices()
        
        # Ищем устройства с "CABLE" или "Virtual" в названии
        for dev in devices:
            name_lower = dev['name'].lower()
            if 'cable' in name_lower or 'virtual' in name_lower:
                logger.info(f"Found virtual cable device: {dev['name']}")
                return dev['index']
        
        # Если не найдено, используем устройство по умолчанию
        default_device = sd.default.device[0]
        if default_device is not None:
            logger.warning("Virtual Audio Cable not found, using default input device")
            logger.warning("Please install Virtual Audio Cable and route application audio to it")
            return default_device
        
        return None
    
    def start_capture(self, callback: Callable[[np.ndarray], None]) -> bool:
        """
        Начать захват аудио
        
        Args:
            callback: Функция вызываемая с аудио данными (numpy array)
        
        Returns:
            bool: Успешность запуска
        """
        if self.is_capturing:
            logger.warning("Already capturing audio")
            return False
        
        device_index = self.select_virtual_cable_device()
        if device_index is None:
            logger.error("No suitable audio device found")
            return False
        
        self.callback = callback
        self.is_capturing = True
        
        try:
            # Инициализируем COM для работы с Windows Audio
            if psutil.os == "windows":
                try:
                    CoInitialize()
                    self._co_initialized = True
                except Exception as e:
                    logger.debug(f"COM initialization: {e}")
            
            # Создаем поток захвата
            self.stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=self.config.audio_sample_rate,
                blocksize=int(self.config.audio_sample_rate * self.config.audio_chunk_duration),
                callback=self._audio_callback,
                dtype=np.float32
            )
            self.stream.start()
            logger.info(f"Started audio capture on device {device_index}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self.is_capturing = False
            return False
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback для обработки аудио данных"""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        if self.is_capturing and hasattr(self, 'callback'):
            # Копируем данные и передаем в callback
            audio_data = indata.copy().flatten()
            self.callback(audio_data)
    
    def stop_capture(self):
        """Остановить захват аудио"""
        self.is_capturing = False
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
            finally:
                self.stream = None
        
        # Освобождаем COM
        if self._co_initialized:
            try:
                CoUninitialize()
                self._co_initialized = False
            except Exception as e:
                logger.debug(f"COM uninitialization: {e}")
        
        logger.info("Audio capture stopped")
    
    def get_audio_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Генератор для получения аудио чанков
        Используется для потоковой обработки
        """
        audio_queue = queue.Queue()
        
        def callback(audio_data):
            audio_queue.put(audio_data)
        
        self.start_capture(callback)
        
        try:
            while self.is_capturing:
                try:
                    data = audio_queue.get(timeout=1.0)
                    yield data
                except queue.Empty:
                    continue
        finally:
            self.stop_capture()
    
    def detect_speech_activity(self, audio_data: np.ndarray) -> bool:
        """
        Детектирование наличия речи в аудио данных
        Простая проверка по уровню громкости
        """
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > self.config.silence_threshold
    
    def set_target_application(self, app_name: str):
        """Установить целевое приложение для захвата"""
        process = self.find_application_process(app_name)
        if process:
            self.target_process_name = app_name
            logger.info(f"Target application set to: {app_name} (PID: {process.pid})")
        else:
            logger.warning(f"Application '{app_name}' not found")
            self.target_process_name = None
