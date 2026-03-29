"""
Конфигурация ShadowSpeaker
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class Config(BaseModel):
    # Пути
    base_dir: Path = Path(__file__).parent
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "models")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "logs")
    
    # Аудио настройки
    audio_sample_rate: int = 16000
    audio_chunk_duration: float = 5.0  # секунд
    silence_threshold: float = 0.01  # порог тишины
    min_speech_duration: float = 0.5  # минимальная длительность речи
    
    # Настройки STT (Faster-Whisper)
    whisper_model: str = "medium"  # tiny, base, small, medium, large-v2
    whisper_device: str = "cuda"  # cuda, cpu
    whisper_compute_type: str = "float16"  # float16, int8, int8_float16
    
    # Диаризация спикеров
    use_diarization: bool = True
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    max_speakers: int = 5  # максимальное количество спикеров
    
    # LLM настройки
    llm_provider: str = "ollama"  # ollama, lmstudio, llama_cpp
    llm_model: str = "llama3.2"  # llama3.2, phi3, mistral или имя модели в LM Studio
    llm_host: str = "http://localhost:11434"  # Ollama: 11434, LM Studio: 1234
    llm_context_length: int = 4096
    llm_temperature: float = 0.7
    llm_max_tokens: int = 150
    
    # Whisper модель - путь к файлу или название
    whisper_model_path: Optional[str] = None  # Путь к .pt файлу (например large-v3-turbo.pt)
    whisper_model_name: str = "medium"  # tiny, base, small, medium, large-v2, large-v3-turbo
    
    # Захват экрана для визуального контекста
    enable_screen_capture: bool = True
    screen_capture_interval: float = 10.0  # Захватывать экран каждые N секунд
    screen_monitor_index: int = 1  # Индекс монитора (0 - основной)
    
    # Количество вариантов ответов
    response_variants_count: int = 3
    
    # Оверлей настройки
    overlay_position: str = "bottom_right"  # top_left, top_right, bottom_left, bottom_right
    overlay_opacity: float = 0.9
    overlay_font_size: int = 14
    overlay_max_width: int = 600
    overlay_margin: int = 20
    
    # Горячие клавиши (Windows virtual key codes)
    hotkey_select_1: str = "Ctrl+1"
    hotkey_select_2: str = "Ctrl+2"
    hotkey_select_3: str = "Ctrl+3"
    hotkey_refresh: str = "Ctrl+R"
    hotkey_toggle: str = "Ctrl+H"
    hotkey_settings: str = "Ctrl+S"  # Горячая клавиша для настроек
    hotkey_quit: str = "Ctrl+Q"
    
    # Профиль пользователя для идентификации
    user_voice_profile: Optional[str] = None  # имя/метка пользователя
    user_speaker_id: Optional[int] = None  # ID спикера если известен
    
    # Приложения для захвата звука
    target_applications: List[str] = Field(default_factory=lambda: [
        "Discord",
        "TeamSpeak",
        "Skype",
        "Zoom"
    ])
    
    # Логирование
    log_level: str = "INFO"
    log_file: str = "shadow_speaker.log"
    
    class Config:
        arbitrary_types_allowed = True


# Глобальный экземпляр конфигурации
config = Config()

# Создаем директории если не существуют
config.models_dir.mkdir(exist_ok=True)
config.logs_dir.mkdir(exist_ok=True)
