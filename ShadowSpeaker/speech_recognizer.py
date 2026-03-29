import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from faster_whisper import WhisperModel
from datetime import datetime, timedelta
import threading
import queue
import sounddevice as sd
from config import Config, load_config, MODELS_DIR
import os


class DialogueMessage:
    """Сообщение диалога с метаданными"""
    def __init__(self, speaker_id: str, text: str, timestamp: datetime, confidence: float = 1.0):
        self.speaker_id = speaker_id
        self.text = text
        self.timestamp = timestamp
        self.confidence = confidence
    
    def to_dict(self) -> Dict:
        return {
            "speaker_id": self.speaker_id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence
        }


class DialogueMemory:
    """Память диалога со скользящим окном"""
    def __init__(self, max_duration_seconds: int = 300, max_messages: int = 50):
        self.messages: List[DialogueMessage] = []
        self.max_duration = timedelta(seconds=max_duration_seconds)
        self.max_messages = max_messages
        self.lock = threading.Lock()
    
    def add_message(self, message: DialogueMessage) -> None:
        """Добавление сообщения в память"""
        with self.lock:
            self.messages.append(message)
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Очистка старых сообщений"""
        now = datetime.now()
        cutoff_time = now - self.max_duration
        
        # Удаляем сообщения старше cutoff_time
        self.messages = [msg for msg in self.messages if msg.timestamp > cutoff_time]
        
        # Ограничиваем количество сообщений
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> str:
        """Получение контекста диалога для LLM"""
        with self.lock:
            if not self.messages:
                return ""
            
            context_lines = []
            for msg in self.messages:
                speaker_label = f"Спикер {msg.speaker_id}"
                time_str = msg.timestamp.strftime("%H:%M:%S")
                context_lines.append(f"[{time_str}] {speaker_label}: {msg.text}")
            
            return "\n".join(context_lines)
    
    def get_recent_messages(self, count: int = 10) -> List[DialogueMessage]:
        """Получение последних N сообщений"""
        with self.lock:
            return self.messages[-count:] if self.messages else []
    
    def clear(self) -> None:
        """Очистка всей памяти"""
        with self.lock:
            self.messages.clear()


class SpeechRecognizer:
    """Распознавание речи с поддержкой пользовательских моделей"""
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.recognition_thread = None
        self.memory = DialogueMemory(
            max_duration_seconds=config.memory.max_duration_seconds,
            max_messages=config.memory.max_messages
        )
        self.callbacks = []  # Callback функции для новых транскрипций
        
        # Инициализация модели
        self._load_model()
    
    def _load_model(self) -> None:
        """Загрузка модели Whisper"""
        try:
            model_path = self.config.whisper.model_path
            
            # Проверка на пользовательскую .pt модель
            if model_path and model_path.endswith('.pt'):
                if os.path.exists(model_path):
                    print(f"Загрузка пользовательской модели: {model_path}")
                    # Для кастомных моделей используем путь напрямую
                    self.model = WhisperModel(model_path, device=self.config.whisper.device, 
                                            compute_type=self.config.whisper.compute_type)
                else:
                    print(f"Модель не найдена: {model_path}, пробуем загрузить по имени")
                    model_name = self.config.whisper.model_name
                    self.model = WhisperModel(model_name, device=self.config.whisper.device,
                                            compute_type=self.config.whisper.compute_type)
            else:
                # Загрузка стандартной модели
                model_name = self.config.whisper.model_name or "large-v3-turbo"
                print(f"Загрузка модели Whisper: {model_name}")
                self.model = WhisperModel(model_name, device=self.config.whisper.device,
                                        compute_type=self.config.whisper.compute_type)
            
            print("Модель Whisper успешно загружена")
        except Exception as e:
            print(f"Ошибка загрузки модели Whisper: {e}")
            print("Попробуйте установить модель вручную или проверить путь к файлу")
            raise
    
    def register_callback(self, callback) -> None:
        """Регистрация callback функции для новых транскрипций"""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, messages: List[DialogueMessage]) -> None:
        """Уведомление всех зарегистрированных callback функций"""
        for callback in self.callbacks:
            try:
                callback(messages)
            except Exception as e:
                print(f"Ошибка в callback функции: {e}")
    
    def start_listening(self, device_name: Optional[str] = None) -> None:
        """Запуск прослушивания аудио"""
        self.is_running = True
        
        # Настройка устройства ввода
        device_info = None
        if device_name:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if device_name.lower() in dev['name'].lower():
                    device_info = i
                    break
        
        if device_info is None and device_name:
            print(f"Устройство '{device_name}' не найдено, используем устройство по умолчанию")
        
        # Запуск потока записи аудио
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Статус аудио: {status}")
            if self.is_running:
                self.audio_queue.put(indata.copy())
        
        sample_rate = self.config.audio.sample_rate
        channels = self.config.audio.channels
        
        self.stream = sd.InputStream(
            device=device_info,
            samplerate=sample_rate,
            channels=channels,
            callback=audio_callback,
            blocksize=int(sample_rate * self.config.audio.chunk_duration)
        )
        self.stream.start()
        
        # Запуск потока распознавания
        self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self.recognition_thread.start()
        
        print("Начато прослушивание аудио...")
    
    def stop_listening(self) -> None:
        """Остановка прослушивания"""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2.0)
        print("Прослушивание остановлено")
    
    def _recognition_loop(self) -> None:
        """Основной цикл распознавания речи"""
        while self.is_running:
            try:
                # Получение аудио чанка
                audio_data = self.audio_queue.get(timeout=1.0)
                
                # Преобразование в нужный формат
                audio_float = audio_data.flatten().astype(np.float32)
                audio_normalized = audio_float / np.max(np.abs(audio_float))
                
                # Распознавание
                segments, info = self.model.transcribe(
                    audio_normalized,
                    language=self.config.whisper.language,
                    vad_filter=True,
                    word_timestamps=False
                )
                
                messages = []
                for segment in segments:
                    if segment.text.strip():
                        # Простая эвристика для определения спикера
                        # В реальной реализации здесь будет pyannote для diarization
                        speaker_id = "1"  # Заглушка, будет заменено на реальную диаризацию
                        
                        message = DialogueMessage(
                            speaker_id=speaker_id,
                            text=segment.text.strip(),
                            timestamp=datetime.now(),
                            confidence=segment.avg_logprob
                        )
                        messages.append(message)
                        self.memory.add_message(message)
                
                if messages:
                    self._notify_callbacks(messages)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка распознавания: {e}")
    
    def get_memory(self) -> DialogueMemory:
        """Получение объекта памяти диалога"""
        return self.memory
