"""
Распознавание речи с диаризацией спикеров
Использует Faster-Whisper для STT и pyannote для разделения спикеров
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger
from dataclasses import dataclass
import threading
import queue

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed")

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("pyannote.audio not installed")


@dataclass
class SpeechSegment:
    """Сегмент речи с информацией о спикере"""
    text: str
    speaker: str  # ID спикера (SPEAKER_0, SPEAKER_1, etc.)
    start_time: float
    end_time: float
    confidence: float = 1.0
    
    def __post_init__(self):
        """Установить временную метку если не указана"""
        import time
        if not hasattr(self, 'timestamp') or self.end_time == 0:
            self.end_time = time.time()
        if self.start_time == 0:
            self.start_time = self.end_time - 1.0


class SpeechRecognizer:
    """
    Распознавание речи с автоматическим определением спикеров
    """
    
    def __init__(self, config):
        self.config = config
        self.whisper_model = None
        self.diarization_pipeline = None
        self.is_initialized = False
        self._init_lock = threading.Lock()
        
        # Буфер для накопления аудио перед обработкой
        self.audio_buffer = []
        self.buffer_duration = 0.0
        
        # История распознанных сегментов
        self.recognized_segments: List[SpeechSegment] = []
        
        # Кэш спикеров для идентификации пользователя
        self.speaker_profiles: Dict[str, np.ndarray] = {}
        
    def initialize(self) -> bool:
        """Инициализация моделей STT и диаризации"""
        with self._init_lock:
            if self.is_initialized:
                return True
            
            try:
                # Инициализация Whisper
                if WHISPER_AVAILABLE:
                    # Проверяем есть ли путь к пользовательской модели
                    if self.config.whisper_model_path:
                        import os
                        if os.path.exists(self.config.whisper_model_path):
                            logger.info(f"Loading custom Whisper model from: {self.config.whisper_model_path}")
                            model_path = self.config.whisper_model_path
                        else:
                            logger.warning(f"Custom model path not found: {self.config.whisper_model_path}")
                            logger.warning(f"Falling back to built-in model: {self.config.whisper_model}")
                            model_path = self.config.whisper_model
                    else:
                        model_path = self.config.whisper_model
                    
                    logger.info(f"Loading Whisper model: {model_path}")
                    self.whisper_model = WhisperModel(
                        model_path,
                        device=self.config.whisper_device,
                        compute_type=self.config.whisper_compute_type
                    )
                    logger.info("Whisper model loaded successfully")
                else:
                    logger.error("faster-whisper is not available")
                    return False
                
                # Инициализация диаризации (опционально)
                if self.config.use_diarization and PYANNOTE_AVAILABLE:
                    logger.info("Loading diarization pipeline...")
                    logger.warning("Note: pyannote requires HuggingFace token for speaker diarization")
                    logger.warning("Set HF_TOKEN environment variable or use without diarization")
                    # Диаризация будет загружена при первом использовании если есть токен
                else:
                    logger.info("Diarization disabled or unavailable")
                
                self.is_initialized = True
                logger.info("SpeechRecognizer initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize SpeechRecognizer: {e}")
                return False
    
    def _ensure_initialized(self):
        """Проверка инициализации"""
        if not self.is_initialized:
            self.initialize()
    
    def transcribe_audio(self, audio_data: np.ndarray) -> List[SpeechSegment]:
        """
        Транскрибация аудио с диаризацией
        
        Args:
            audio_data: Аудио данные (numpy array, sample_rate=16000)
        
        Returns:
            Список сегментов речи с текстом и спикерами
        """
        self._ensure_initialized()
        
        if not self.whisper_model:
            logger.error("Whisper model not available")
            return []
        
        segments_list = []
        
        try:
            # Транскрибация через Whisper
            segments, info = self.whisper_model.transcribe(
                audio_data,
                beam_size=5,
                language="ru" if self._detect_language(audio_data) == "ru" else "en"
            )
            
            # Если диаризация включена и доступна
            if self.config.use_diarization and self._diarization_available():
                segments_list = self._transcribe_with_diarization(audio_data, segments)
            else:
                # Без диаризации - все речь от одного спикера
                for segment in segments:
                    speech_segment = SpeechSegment(
                        text=segment.text.strip(),
                        speaker="SPEAKER_0",
                        start_time=segment.start,
                        end_time=segment.end,
                        confidence=segment.avg_logprob
                    )
                    segments_list.append(speech_segment)
            
            # Сохраняем в историю
            self.recognized_segments.extend(segments_list)
            
            return segments_list
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return []
    
    def _transcribe_with_diarization(self, audio_data: np.ndarray, whisper_segments) -> List[SpeechSegment]:
        """Транскрибация с разделением спикеров"""
        try:
            # Загружаем pipeline если еще не загружен
            if not self.diarization_pipeline:
                import os
                hf_token = os.getenv("HF_TOKEN")
                if hf_token:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        self.config.diarization_model,
                        use_auth_token=hf_token
                    )
                else:
                    logger.warning("HF_TOKEN not set, skipping diarization")
                    return list(self._segments_to_speech_segments(whisper_segments))
            
            # Запускаем диаризацию
            diarization = self.diarization_pipeline(
                audio_data,
                min_speakers=1,
                max_speakers=self.config.max_speakers
            )
            
            # Сопоставляем сегменты Whisper со спикерами
            result_segments = []
            for segment in whisper_segments:
                # Определяем спикера для этого сегмента
                speaker = self._find_speaker_for_segment(diarization, segment.start, segment.end)
                
                speech_segment = SpeechSegment(
                    text=segment.text.strip(),
                    speaker=speaker,
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=segment.avg_logprob
                )
                result_segments.append(speech_segment)
            
            return result_segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}, falling back to no diarization")
            return list(self._segments_to_speech_segments(whisper_segments))
    
    def _find_speaker_for_segment(self, diarization, start: float, end: float) -> str:
        """Найти спикера для временного сегмента"""
        best_speaker = "SPEAKER_0"
        best_overlap = 0.0
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Вычисляем перекрытие
            overlap_start = max(start, turn.start)
            overlap_end = min(end, turn.end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        
        return best_speaker
    
    def _segments_to_speech_segments(self, whisper_segments):
        """Конвертация сегментов Whisper в SpeechSegment без диаризации"""
        for segment in whisper_segments:
            yield SpeechSegment(
                text=segment.text.strip(),
                speaker="SPEAKER_0",
                start_time=segment.start,
                end_time=segment.end,
                confidence=segment.avg_logprob
            )
    
    def _detect_language(self, audio_data: np.ndarray) -> str:
        """Простая эвристика для определения языка (можно улучшить)"""
        # В реальной реализации можно использовать langdetect или анализ частот
        # Пока возвращаем русский как основной для проекта
        return "ru"
    
    def _diarization_available(self) -> bool:
        """Проверка доступности диаризации"""
        if not PYANNOTE_AVAILABLE:
            return False
        
        import os
        return os.getenv("HF_TOKEN") is not None
    
    def add_audio_chunk(self, audio_data: np.ndarray) -> Optional[List[SpeechSegment]]:
        """
        Добавить аудио чанк в буфер и обработать если набралось достаточно
        """
        self.audio_buffer.append(audio_data)
        chunk_duration = len(audio_data) / self.config.audio_sample_rate
        self.buffer_duration += chunk_duration
        
        # Если набрали достаточно аудио для обработки
        if self.buffer_duration >= self.config.audio_chunk_duration:
            # Конкатенируем буфер
            full_audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            self.buffer_duration = 0.0
            
            # Обрабатываем
            return self.transcribe_audio(full_audio)
        
        return None
    
    def clear_buffer(self):
        """Очистить аудио буфер"""
        self.audio_buffer = []
        self.buffer_duration = 0.0
    
    def get_conversation_context(self, last_n_segments: int = 10, max_age_seconds: float = 300.0) -> str:
        """
        Получить контекст последних реплик для LLM
        
        Args:
            last_n_segments: Максимальное количество сегментов
            max_age_seconds: Максимальный возраст сегментов в секундах (по умолчанию 5 минут)
        
        Returns:
            Строка контекста диалога
        """
        if not self.recognized_segments:
            return ""
        
        import time
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        # Фильтруем по времени и берем последние N сегментов
        recent = [
            seg for seg in self.recognized_segments 
            if seg.end_time >= cutoff_time
        ][-last_n_segments:]
        
        if not recent:
            return "Диалог только начался."
        
        context_lines = []
        for seg in recent:
            context_lines.append(f"{seg.speaker}: {seg.text}")
        
        return "\n".join(context_lines)
    
    def identify_user_speaker(self, user_profile_name: str) -> Optional[str]:
        """
        Попытаться определить какой спикер является пользователем
        На основе профиля или статистики участия в диалоге
        """
        if not self.recognized_segments:
            return None
        
        # Простая эвристика: пользователь чаще всего говорит вторым после вопроса
        # Или можно использовать голосовой отпечаток если есть профиль
        
        speaker_counts = {}
        for seg in self.recognized_segments:
            speaker_counts[seg.speaker] = speaker_counts.get(seg.speaker, 0) + 1
        
        # Возвращаем наиболее активного спикера как кандидата
        if speaker_counts:
            most_active = max(speaker_counts.items(), key=lambda x: x[1])[0]
            logger.info(f"Identified user as likely speaker: {most_active}")
            return most_active
        
        return None
