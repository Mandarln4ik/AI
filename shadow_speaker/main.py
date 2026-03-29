import sys
import asyncio
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from config import load_config, save_config, Config
from speech_recognizer import SpeechRecognizer
from screen_capture import ScreenCapture
from llm_engine import LLMEngine
from overlay_ui import OverlayWindow
from settings_gui import SettingsWindow


class WorkerThread(QThread):
    """Рабочий поток для обработки задач ИИ"""
    response_ready = pyqtSignal(list)  # Сигнал готовности вариантов ответа
    
    def __init__(self, llm_engine: LLMEngine, speech_recognizer: SpeechRecognizer, screen_capture: ScreenCapture):
        super().__init__()
        self.llm_engine = llm_engine
        self.speech_recognizer = speech_recognizer
        self.screen_capture = screen_capture
        self.is_running = False
    
    def run(self):
        """Основной цикл обработки"""
        self.is_running = True
        
        while self.is_running:
            try:
                # Получение контекста диалога
                dialogue_context = self.speech_recognizer.get_memory().get_context()
                
                if not dialogue_context:
                    # Если нет диалога, ждем
                    self.msleep(1000)
                    continue
                
                # Получение визуального контекста
                screen_context = ""
                screenshot_base64 = None
                
                if self.screen_capture.enabled:
                    screen_context = self.screen_capture.get_screenshot_context()
                    screenshot_base64 = self.screen_capture.get_latest_screenshot_base64()
                
                # Генерация ответов
                options = self.llm_engine.generate_response(
                    dialogue_context=dialogue_context,
                    screen_context=screen_context,
                    screenshot_base64=screenshot_base64,
                    num_options=3
                )
                
                # Отправка результатов в главный поток
                self.response_ready.emit(options)
                
                # Пауза перед следующей генерацией (чтобы не спамить)
                self.msleep(5000)
                
            except Exception as e:
                print(f"Ошибка в рабочем потоке: {e}")
                self.msleep(2000)
    
    def stop(self):
        """Остановка потока"""
        self.is_running = False
        self.wait(2000)


class ShadowSpeakerApp:
    """Основное приложение ShadowSpeaker"""
    
    def __init__(self):
        # Загрузка конфигурации
        self.config = load_config()
        
        # Инициализация Qt приложения
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("ShadowSpeaker")
        
        # Компоненты
        self.speech_recognizer = None
        self.screen_capture = None
        self.llm_engine = None
        self.overlay = None
        self.settings_window = None
        self.worker_thread = None
        
        # Флаг активности
        self.is_active = False
    
    def initialize_components(self):
        """Инициализация всех компонентов"""
        print("Инициализация компонентов...")
        
        # Распознавание речи
        try:
            self.speech_recognizer = SpeechRecognizer(self.config)
            self.speech_recognizer.register_callback(self.on_speech_detected)
            print("✓ Распознавание речи инициализировано")
        except Exception as e:
            print(f"✗ Ошибка инициализации распознавания речи: {e}")
            self.speech_recognizer = None
        
        # Захват экрана
        try:
            self.screen_capture = ScreenCapture(self.config)
            print("✓ Захват экрана инициализирован")
        except Exception as e:
            print(f"✗ Ошибка инициализации захвата экрана: {e}")
            self.screen_capture = None
        
        # LLM движок
        try:
            self.llm_engine = LLMEngine(self.config)
            print(f"✓ LLM движок инициализирован ({self.config.llm.provider})")
        except Exception as e:
            print(f"✗ Ошибка инициализации LLM движка: {e}")
            self.llm_engine = None
        
        # Оверлей
        try:
            self.overlay = OverlayWindow(self.config)
            self.overlay.option_selected.connect(self.on_option_selected)
            self.overlay.settings_requested.connect(self.show_settings)
            print("✓ Оверлей инициализирован")
        except Exception as e:
            print(f"✗ Ошибка инициализации оверлея: {e}")
            self.overlay = None
        
        # Рабочий поток
        if self.llm_engine and self.speech_recognizer and self.screen_capture:
            self.worker_thread = WorkerThread(
                self.llm_engine,
                self.speech_recognizer,
                self.screen_capture
            )
            self.worker_thread.response_ready.connect(self.on_response_ready)
            print("✓ Рабочий поток инициализирован")
    
    def start(self):
        """Запуск приложения"""
        print("\n🚀 Запуск ShadowSpeaker...")
        
        self.initialize_components()
        
        # Запуск компонентов
        if self.speech_recognizer:
            self.speech_recognizer.start_listening(self.config.audio.input_device)
        
        if self.screen_capture:
            self.screen_capture.start_capture()
        
        if self.worker_thread:
            self.worker_thread.start()
        
        self.is_active = True
        
        # Показ оверлея (скрытого по умолчанию)
        if self.overlay:
            self.overlay.hide()
        
        print("\n✅ ShadowSpeaker запущен!")
        print("Горячие клавиши:")
        print("  Ctrl+S - Настройки")
        print("  Ctrl+1/2/3 - Выбор варианта ответа")
        print("\nНажмите Ctrl+C для выхода\n")
        
        # Запуск Qt цикла
        sys.exit(self.app.exec())
    
    def stop(self):
        """Остановка приложения"""
        print("\n🛑 Остановка ShadowSpeaker...")
        
        self.is_active = False
        
        # Остановка компонентов
        if self.speech_recognizer:
            self.speech_recognizer.stop_listening()
        
        if self.screen_capture:
            self.screen_capture.stop_capture()
        
        if self.worker_thread:
            self.worker_thread.stop()
        
        print("✅ ShadowSpeaker остановлен")
    
    def on_speech_detected(self, messages):
        """Обработчик обнаружения речи"""
        for msg in messages:
            print(f"[{msg.timestamp.strftime('%H:%M:%S')}] Спикер {msg.speaker_id}: {msg.text}")
    
    def on_response_ready(self, options):
        """Обработчик готовности вариантов ответа"""
        if self.overlay and options:
            # Обновление оверлея в главном потоке
            QTimer.singleShot(0, lambda: self.overlay.update_options(options))
    
    def on_option_selected(self, index):
        """Обработчик выбора варианта ответа"""
        if self.overlay and index < len(self.overlay.current_options):
            selected_text = self.overlay.current_options[index]
            print(f"\n💬 Выбран вариант {index + 1}: {selected_text}")
            # Здесь можно добавить копирование в буфер обмена или автоматический ввод
    
    def show_settings(self):
        """Показ окна настроек"""
        if self.settings_window is None:
            self.settings_window = SettingsWindow(self.config)
            self.settings_window.settings_saved.connect(self.on_settings_saved)
        
        self.settings_window.show()
        self.settings_window.activateWindow()
    
    def on_settings_saved(self, new_config: Config):
        """Обработчик сохранения настроек"""
        print("Настройки сохранены, применяем изменения...")
        self.config = new_config
        
        # Перезапуск компонентов с новыми настройками
        # (в полной версии нужно корректно перезапустить компоненты)


def main():
    """Точка входа"""
    try:
        app = ShadowSpeakerApp()
        app.start()
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
