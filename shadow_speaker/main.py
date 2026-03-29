"""
ShadowSpeaker - Главный файл приложения
Локальный ИИ-ассистент для помощи в диалогах
"""
import sys
import time
import threading
from loguru import logger
from typing import Optional

from config import config
from audio_capturer import AudioCapturer
from speech_recognizer import SpeechRecognizer, SpeechSegment
from llm_engine import LLMEngine
from speaker_manager import SpeakerManager
from overlay_ui import OverlayManager


class ShadowSpeaker:
    """
    Основное приложение ShadowSpeaker
    """
    
    def __init__(self, config):
        self.config = config
        
        # Настраиваем логирование
        self._setup_logging()
        
        # Компоненты
        self.audio_capturer: Optional[AudioCapturer] = None
        self.speech_recognizer: Optional[SpeechRecognizer] = None
        self.llm_engine: Optional[LLMEngine] = None
        self.speaker_manager: Optional[SpeakerManager] = None
        self.overlay_manager: Optional[OverlayManager] = None
        self.screen_capture = None  # Захват экрана для визуального контекста
        
        # Состояние
        self.is_running = False
        self.is_processing = False
        self.current_variants = []
        
        logger.info("=" * 50)
        logger.info("ShadowSpeaker initialized")
        logger.info(f"Config: whisper={config.whisper_model_name}, llm={config.llm_model}")
        logger.info(f"Screen capture: {config.enable_screen_capture}")
        logger.info("=" * 50)
    
    def _setup_logging(self):
        """Настройка логирования"""
        logger.remove()  # Убираем стандартный handler
        
        # Консоль
        logger.add(
            sys.stderr,
            level=self.config.log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        )
        
        # Файл
        logger.add(
            self.config.logs_dir / self.config.log_file,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days"
        )
    
    def initialize(self) -> bool:
        """Инициализация всех компонентов"""
        try:
            logger.info("Initializing components...")
            
            # Audio Capturer
            self.audio_capturer = AudioCapturer(self.config)
            devices = self.audio_capturer.list_available_devices()
            logger.info(f"Available audio devices: {len(devices)}")
            
            # Speech Recognizer
            self.speech_recognizer = SpeechRecognizer(self.config)
            if not self.speech_recognizer.initialize():
                logger.error("Failed to initialize speech recognizer")
                return False
            
            # LLM Engine
            self.llm_engine = LLMEngine(self.config)
            if not self.llm_engine.initialize():
                logger.warning("LLM engine not available, will use fallback responses")
            
            # Speaker Manager
            self.speaker_manager = SpeakerManager(self.config)
            
            # Screen Capture (для визуального контекста)
            if self.config.enable_screen_capture:
                try:
                    from screen_capture import ScreenCapture
                    self.screen_capture = ScreenCapture(
                        monitor_index=self.config.screen_monitor_index
                    )
                    self.screen_capture.set_capture_interval(
                        self.config.screen_capture_interval
                    )
                    logger.info("Screen capture enabled")
                except Exception as e:
                    logger.warning(f"Screen capture initialization failed: {e}")
                    self.screen_capture = None
            
            # Overlay Manager
            self.overlay_manager = OverlayManager(self.config)
            self.overlay_manager.start(
                on_variant_selected=self._on_variant_selected,
                on_refresh=self._on_refresh_requested,
                on_toggle=self._on_toggle_overlay,
                on_quit=self._on_quit
            )
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def start(self):
        """Запуск приложения"""
        if not self.initialize():
            logger.error("Failed to initialize, exiting")
            return
        
        self.is_running = True
        
        # Запускаем захват аудио в отдельном потоке
        audio_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
        audio_thread.start()
        
        logger.info("ShadowSpeaker started - listening for conversations")
        logger.info("Press Ctrl+Q or close the window to exit")
        
        # Запускаем Qt event loop (блокирующий)
        self.overlay_manager.run()
    
    def _audio_processing_loop(self):
        """Основной цикл обработки аудио"""
        def on_audio_chunk(audio_data):
            if self.is_processing:
                return
            
            # Проверяем наличие речи
            if not self.audio_capturer.detect_speech_activity(audio_data):
                return
            
            # Обрабатываем аудио
            self.is_processing = True
            try:
                segments = self.speech_recognizer.transcribe_audio(audio_data)
                
                for segment in segments:
                    self._process_speech_segment(segment)
                    
            finally:
                self.is_processing = False
        
        # Запускаем захват
        if not self.audio_capturer.start_capture(on_audio_chunk):
            logger.error("Failed to start audio capture")
            return
        
        # Держим поток активным
        while self.is_running:
            time.sleep(0.1)
        
        self.audio_capturer.stop_capture()
    
    def _process_speech_segment(self, segment: SpeechSegment):
        """Обработка распознанного сегмента речи"""
        logger.info(f"Speech detected: [{segment.speaker}] {segment.text}")
        
        # Регистрируем спикера
        self.speaker_manager.record_utterance(
            segment.speaker,
            segment.text,
            segment.end_time - segment.start_time
        )
        
        # Добавляем в историю LLM с временной меткой
        self.llm_engine.add_to_history(segment.speaker, segment.text)
        
        # Если это не пользователь и есть контекст - генерируем ответ
        user_speaker = self.speaker_manager.user_speaker_id
        
        if user_speaker and segment.speaker != user_speaker:
            # Получаем контекст диалога за последние 5 минут
            context = self.speech_recognizer.get_conversation_context(last_n_segments=20)
            
            # Получаем визуальный контекст если включено
            screen_context = ""
            if hasattr(self, 'screen_capture') and self.screen_capture:
                screen_context = self.screen_capture.get_context_description()
            
            # Генерируем варианты ответов
            variants = self.llm_engine.generate_response_variants(
                context, 
                user_speaker,
                screen_context=screen_context
            )
            
            if variants:
                self.current_variants = variants
                self.overlay_manager.update_variants(variants)
                logger.info(f"Generated {len(variants)} response variants")
    
    def _on_variant_selected(self, index: int):
        """Обработчик выбора варианта ответа"""
        if 0 <= index < len(self.current_variants):
            selected_text = self.current_variants[index]["text"]
            logger.info(f"User selected variant {index + 1}: {selected_text}")
            
            # Копируем в буфер обмена (можно добавить автоввод)
            try:
                import pyperclip
                pyperclip.copy(selected_text)
                logger.info("Response copied to clipboard")
            except ImportError:
                logger.debug("pyperclip not installed, cannot copy to clipboard")
            
            # Добавляем выбранный ответ в историю
            if self.speaker_manager.user_speaker_id:
                self.llm_engine.add_to_history(
                    self.speaker_manager.user_speaker_id,
                    selected_text
                )
            
            # Очищаем варианты после выбора
            self.overlay_manager.clear_variants()
            self.current_variants = []
    
    def _on_refresh_requested(self):
        """Запрос на обновление вариантов"""
        logger.info("Refresh requested")
        
        if self.speaker_manager.user_speaker_id:
            context = self.speech_recognizer.get_conversation_context()
            variants = self.llm_engine.generate_response_variants(
                context,
                self.speaker_manager.user_speaker_id
            )
            
            if variants:
                self.current_variants = variants
                self.overlay_manager.update_variants(variants)
    
    def _on_toggle_overlay(self):
        """Переключение видимости оверлея"""
        logger.info("Toggle overlay requested")
        if self.overlay_manager.window:
            self.overlay_manager.window.toggle_visibility()
    
    def _on_quit(self):
        """Выход из приложения"""
        logger.info("Quit requested")
        self.stop()
    
    def stop(self):
        """Остановка приложения"""
        logger.info("Stopping ShadowSpeaker...")
        self.is_running = False
        
        if self.audio_capturer:
            self.audio_capturer.stop_capture()
        
        if self.overlay_manager:
            self.overlay_manager.stop()
        
        logger.info("ShadowSpeaker stopped")


def main():
    """Точка входа"""
    app = ShadowSpeaker(config)
    
    try:
        app.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        app.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        app.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
