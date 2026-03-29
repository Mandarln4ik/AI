import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Базовая директория для хранения данных
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
CONFIG_FILE = DATA_DIR / "config.json"

# Создаем директории если не существуют
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class WhisperConfig(BaseModel):
    """Конфигурация модели распознавания речи"""
    model_path: Optional[str] = None  # Путь к .pt файлу или название модели
    model_name: str = "large-v3-turbo"  # По умолчанию
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "ru"  # Язык по умолчанию
    use_speaker_diarization: bool = True


class LLMProviderConfig(BaseModel):
    """Конфигурация провайдера LLM"""
    provider: str = "ollama"  # ollama, lmstudio, llama_cpp
    model_name: str = "phi3"  # Модель по умолчанию
    base_url: str = "http://localhost:11434"  # Для Ollama
    lmstudio_port: int = 1234  # Для LM Studio
    context_length: int = 4096
    temperature: float = 0.7
    max_tokens: int = 512


class ScreenCaptureConfig(BaseModel):
    """Конфигурация захвата экрана"""
    enabled: bool = True
    capture_interval: float = 10.0  # Секунды между захватами
    monitor_index: int = 0  # Индекс монитора
    resize_width: int = 800  # Изменение размера для оптимизации
    resize_height: int = 600


class MemoryConfig(BaseModel):
    """Конфигурация памяти диалога"""
    enabled: bool = True
    max_duration_seconds: int = 300  # 5 минут
    max_messages: int = 50  # Максимальное количество сообщений в памяти


class AudioConfig(BaseModel):
    """Конфигурация аудио"""
    input_device: Optional[str] = None  # Название устройства ввода (Virtual Cable)
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 5.0  # Длительность чанка для обработки


class OverlayConfig(BaseModel):
    """Конфигурация оверлея"""
    position_x: int = 100
    position_y: int = 100
    width: int = 400
    height: int = 300
    opacity: float = 0.8
    font_size: int = 14
    hotkey_toggle: str = "Ctrl+Shift+S"
    hotkey_accept_1: str = "Ctrl+1"
    hotkey_accept_2: str = "Ctrl+2"
    hotkey_accept_3: str = "Ctrl+3"


class Config(BaseModel):
    """Основная конфигурация приложения"""
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    screen: ScreenCaptureConfig = Field(default_factory=ScreenCaptureConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    
    # Список доступных моделей (заполняется автоматически)
    available_whisper_models: List[str] = []
    available_llm_models: List[str] = []

    class Config:
        arbitrary_types_allowed = True


def load_config() -> Config:
    """Загрузка конфигурации из файла"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Config(**data)
        except Exception as e:
            print(f"Ошибка загрузки конфига: {e}, используем значения по умолчанию")
    return Config()


def save_config(config: Config) -> None:
    """Сохранение конфигурации в файл"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Ошибка сохранения конфига: {e}")


def get_available_whisper_models() -> List[str]:
    """Получение списка доступных моделей Whisper"""
    models = ["large-v3-turbo", "medium", "small", "base", "tiny"]
    
    # Добавляем пользовательские .pt файлы
    if MODELS_DIR.exists():
        for file in MODELS_DIR.glob("*.pt"):
            models.append(file.name)
    
    return models


def get_available_llm_models(provider: str, base_url: str = "") -> List[str]:
    """Получение списка доступных LLM моделей от провайдера"""
    # Это будет заполняться динамически через API
    return []
