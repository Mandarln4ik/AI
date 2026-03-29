"""
Менеджер спикеров - отслеживает участников диалога и идентифицирует пользователя
"""
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import json


@dataclass
class SpeakerInfo:
    """Информация о спикере"""
    speaker_id: str
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    utterance_count: int = 0
    total_speech_duration: float = 0.0
    is_user: bool = False
    voice_profile: Optional[dict] = None  # можно добавить голосовой отпечаток


class SpeakerManager:
    """
    Управляет информацией об участниках диалога
    """
    
    def __init__(self, config):
        self.config = config
        self.speakers: Dict[str, SpeakerInfo] = {}
        self.user_speaker_id: Optional[str] = None
        
        # История активностей для анализа паттернов
        self.activity_history: List[Dict] = []
        
        # Максимум истории
        self.max_history = 100
    
    def register_speaker(self, speaker_id: str) -> SpeakerInfo:
        """Зарегистрировать нового спикера или получить существующего"""
        if speaker_id not in self.speakers:
            self.speakers[speaker_id] = SpeakerInfo(speaker_id=speaker_id)
            logger.info(f"New speaker registered: {speaker_id}")
        
        # Обновляем время последнего появления
        self.speakers[speaker_id].last_seen = datetime.now()
        
        return self.speakers[speaker_id]
    
    def record_utterance(self, speaker_id: str, text: str, duration: float):
        """Записать реплику спикера"""
        speaker = self.register_speaker(speaker_id)
        speaker.utterance_count += 1
        speaker.total_speech_duration += duration
        
        # Добавляем в историю
        self.activity_history.append({
            "speaker_id": speaker_id,
            "text": text,
            "duration": duration,
            "timestamp": datetime.now()
        })
        
        # Ограничиваем историю
        if len(self.activity_history) > self.max_history:
            self.activity_history = self.activity_history[-self.max_history:]
        
        # Пытаемся определить пользователя если еще не определен
        if not self.user_speaker_id and len(self.speakers) >= 2:
            self._try_identify_user()
    
    def _try_identify_user(self):
        """Попытка автоматически определить какой спикер является пользователем"""
        if self.user_speaker_id:
            return
        
        # Эвристики для определения пользователя:
        # 1. Пользователь обычно говорит после вопросов других
        # 2. Пользователь может быть наиболее активным
        # 3. Если задан профиль пользователя, используем его
        
        if self.config.user_voice_profile:
            # Если есть профиль, пытаемся сопоставить
            logger.info("User profile specified, attempting identification")
            # Здесь можно добавить логику сопоставления с профилем
        
        # Простая эвристика: второй по активности спикер (первый часто собеседник)
        sorted_speakers = sorted(
            self.speakers.items(),
            key=lambda x: x[1].utterance_count,
            reverse=True
        )
        
        if len(sorted_speakers) >= 2:
            # Предполагаем что пользователь - один из активных спикеров
            candidate = sorted_speakers[1][0]  # второй по активности
            self.set_user_speaker(candidate)
            logger.info(f"Auto-identified user as speaker: {candidate}")
    
    def set_user_speaker(self, speaker_id: str):
        """Явно указать какой спикер является пользователем"""
        if speaker_id in self.speakers:
            self.user_speaker_id = speaker_id
            self.speakers[speaker_id].is_user = True
            logger.info(f"User speaker set to: {speaker_id}")
        else:
            logger.warning(f"Speaker {speaker_id} not found")
    
    def get_active_speakers(self, time_window_seconds: int = 60) -> List[str]:
        """Получить список активных спикеров за последнее время"""
        now = datetime.now()
        active = []
        
        for speaker_id, info in self.speakers.items():
            if (now - info.last_seen).total_seconds() < time_window_seconds:
                active.append(speaker_id)
        
        return active
    
    def get_speaker_count(self) -> int:
        """Получить количество уникальных спикеров"""
        return len(self.speakers)
    
    def get_conversation_summary(self) -> dict:
        """Получить сводку о текущем диалоге"""
        return {
            "total_speakers": len(self.speakers),
            "user_speaker": self.user_speaker_id,
            "speakers": [
                {
                    "id": sp.speaker_id,
                    "is_user": sp.is_user,
                    "utterances": sp.utterance_count,
                    "duration_sec": round(sp.total_speech_duration, 2)
                }
                for sp in self.speakers.values()
            ],
            "recent_activity": len([
                a for a in self.activity_history
                if (datetime.now() - a["timestamp"]).total_seconds() < 60
            ])
        }
    
    def reset(self):
        """Сбросить все данные о спикерах"""
        self.speakers.clear()
        self.user_speaker_id = None
        self.activity_history.clear()
        logger.info("Speaker manager reset")
    
    def export_state(self) -> str:
        """Экспортировать состояние в JSON"""
        state = {
            "user_speaker_id": self.user_speaker_id,
            "speakers": {
                sid: {
                    "first_seen": info.first_seen.isoformat(),
                    "last_seen": info.last_seen.isoformat(),
                    "utterance_count": info.utterance_count,
                    "total_speech_duration": info.total_speech_duration,
                    "is_user": info.is_user
                }
                for sid, info in self.speakers.items()
            }
        }
        return json.dumps(state, indent=2)
    
    def import_state(self, json_str: str):
        """Импортировать состояние из JSON"""
        try:
            state = json.loads(json_str)
            self.user_speaker_id = state.get("user_speaker_id")
            
            for sid, data in state.get("speakers", {}).items():
                info = SpeakerInfo(
                    speaker_id=sid,
                    first_seen=datetime.fromisoformat(data["first_seen"]),
                    last_seen=datetime.fromisoformat(data["last_seen"]),
                    utterance_count=data["utterance_count"],
                    total_speech_duration=data["total_speech_duration"],
                    is_user=data.get("is_user", False)
                )
                self.speakers[sid] = info
            
            logger.info("Speaker state imported successfully")
        except Exception as e:
            logger.error(f"Failed to import speaker state: {e}")
