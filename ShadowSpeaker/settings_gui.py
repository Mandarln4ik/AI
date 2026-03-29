import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, QTabWidget,
                             QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QFileDialog,
                             QGroupBox, QFormLayout, QCheckBox, QMessageBox, QListWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from config import Config, load_config, save_config, get_available_whisper_models, MODELS_DIR
from llm_engine import LLMEngine


class SettingsWindow(QMainWindow):
    """Окно настроек приложения"""
    
    settings_saved = pyqtSignal(Config)  # Сигнал сохранения настроек
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.setWindowTitle("Настройки ShadowSpeaker")
        self.setMinimumSize(700, 500)
        
        self._setup_ui()
        self._load_current_settings()
    
    def _setup_ui(self):
        """Настройка интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Заголовок
        title_label = QLabel("⚙ Настройки ShadowSpeaker")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Вкладки
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Вкладка Whisper STT
        whisper_tab = self._create_whisper_tab()
        self.tabs.addTab(whisper_tab, "🎤 Whisper STT")
        
        # Вкладка LLM Провайдер
        llm_tab = self._create_llm_tab()
        self.tabs.addTab(llm_tab, "🧠 LLM Провайдер")
        
        # Вкладка Аудио
        audio_tab = self._create_audio_tab()
        self.tabs.addTab(audio_tab, "🔊 Аудио")
        
        # Вкладка Экран
        screen_tab = self._create_screen_tab()
        self.tabs.addTab(screen_tab, "🖥 Экран")
        
        # Вкладка Оверлей
        overlay_tab = self._create_overlay_tab()
        self.tabs.addTab(overlay_tab, "📟 Оверлей")
        
        # Кнопки
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Отмена")
        cancel_btn.setFixedSize(120, 40)
        cancel_btn.clicked.connect(self.close)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("💾 Сохранить")
        save_btn.setFixedSize(120, 40)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        save_btn.clicked.connect(self._save_settings)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
    
    def _create_whisper_tab(self) -> QWidget:
        """Создание вкладки Whisper"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Выбор модели
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.setEditable(True)
        layout.addRow("Модель Whisper:", self.whisper_model_combo)
        
        # Кнопка импорта .pt файла
        import_btn = QPushButton("📁 Импорт .pt модели")
        import_btn.clicked.connect(self._import_whisper_model)
        layout.addRow("", import_btn)
        
        # Путь к модели
        self.whisper_path_edit = QLineEdit()
        self.whisper_path_edit.setPlaceholderText("Путь к файлу модели (.pt)")
        layout.addRow("Путь к модели:", self.whisper_path_edit)
        
        # Устройство
        self.whisper_device_combo = QComboBox()
        self.whisper_device_combo.addItems(["cuda", "cpu"])
        layout.addRow("Устройство:", self.whisper_device_combo)
        
        # Тип вычислений
        self.compute_type_combo = QComboBox()
        self.compute_type_combo.addItems(["float16", "float32", "int8"])
        layout.addRow("Тип вычислений:", self.compute_type_combo)
        
        # Язык
        self.language_edit = QLineEdit()
        self.language_edit.setText("ru")
        layout.addRow("Язык по умолчанию:", self.language_edit)
        
        # Диаризация спикеров
        self.diarization_check = QCheckBox("Использовать диаризацию спикеров")
        layout.addRow("", self.diarization_check)
        
        return widget
    
    def _create_llm_tab(self) -> QWidget:
        """Создание вкладки LLM"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Выбор провайдера
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["ollama", "lmstudio", "llama_cpp"])
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        layout.addRow("Провайдер LLM:", self.provider_combo)
        
        # Модель
        self.llm_model_combo = QComboBox()
        self.llm_model_combo.setEditable(True)
        layout.addRow("Модель LLM:", self.llm_model_combo)
        
        # Кнопка обновления списка моделей
        refresh_btn = QPushButton("🔄 Обновить список моделей")
        refresh_btn.clicked.connect(self._refresh_llm_models)
        layout.addRow("", refresh_btn)
        
        # Base URL (для Ollama)
        self.base_url_edit = QLineEdit()
        self.base_url_edit.setText("http://localhost:11434")
        layout.addRow("Base URL:", self.base_url_edit)
        
        # Порт LM Studio
        self.lmstudio_port_spin = QSpinBox()
        self.lmstudio_port_spin.setRange(1000, 65535)
        self.lmstudio_port_spin.setValue(1234)
        layout.addRow("Порт LM Studio:", self.lmstudio_port_spin)
        
        # Температура
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(0.7)
        layout.addRow("Температура:", self.temperature_spin)
        
        # Макс токенов
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 8192)
        self.max_tokens_spin.setSingleStep(64)
        self.max_tokens_spin.setValue(512)
        layout.addRow("Макс токенов:", self.max_tokens_spin)
        
        # Тест подключения
        test_btn = QPushButton("🔍 Тест подключения")
        test_btn.clicked.connect(self._test_llm_connection)
        layout.addRow("", test_btn)
        
        return widget
    
    def _create_audio_tab(self) -> QWidget:
        """Создание вкладки Аудио"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Устройство ввода
        self.audio_device_combo = QComboBox()
        self.audio_device_combo.setEditable(True)
        layout.addRow("Устройство ввода:", self.audio_device_combo)
        
        # Частота дискретизации
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 48000)
        self.sample_rate_spin.setSingleStep(8000)
        self.sample_rate_spin.setValue(16000)
        layout.addRow("Частота дискретизации:", self.sample_rate_spin)
        
        # Каналы
        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(1, 2)
        self.channels_spin.setValue(1)
        layout.addRow("Каналы:", self.channels_spin)
        
        # Длительность чанка
        self.chunk_duration_spin = QDoubleSpinBox()
        self.chunk_duration_spin.setRange(1.0, 30.0)
        self.chunk_duration_spin.setSingleStep(1.0)
        self.chunk_duration_spin.setValue(5.0)
        layout.addRow("Длительность чанка (сек):", self.chunk_duration_spin)
        
        return widget
    
    def _create_screen_tab(self) -> QWidget:
        """Создание вкладки Экран"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Включение захвата
        self.screen_enabled_check = QCheckBox("Включить захват экрана")
        layout.addRow("", self.screen_enabled_check)
        
        # Индекс монитора
        self.monitor_index_spin = QSpinBox()
        self.monitor_index_spin.setRange(0, 10)
        self.monitor_index_spin.setValue(0)
        layout.addRow("Индекс монитора:", self.monitor_index_spin)
        
        # Интервал захвата
        self.capture_interval_spin = QDoubleSpinBox()
        self.capture_interval_spin.setRange(1.0, 60.0)
        self.capture_interval_spin.setSingleStep(1.0)
        self.capture_interval_spin.setValue(10.0)
        layout.addRow("Интервал захвата (сек):", self.capture_interval_spin)
        
        # Ширина
        self.resize_width_spin = QSpinBox()
        self.resize_width_spin.setRange(320, 1920)
        self.resize_width_spin.setSingleStep(80)
        self.resize_width_spin.setValue(800)
        layout.addRow("Ширина скриншота:", self.resize_width_spin)
        
        # Высота
        self.resize_height_spin = QSpinBox()
        self.resize_height_spin.setRange(240, 1080)
        self.resize_height_spin.setSingleStep(60)
        self.resize_height_spin.setValue(600)
        layout.addRow("Высота скриншота:", self.resize_height_spin)
        
        return widget
    
    def _create_overlay_tab(self) -> QWidget:
        """Создание вкладки Оверлей"""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Позиция X
        self.overlay_x_spin = QSpinBox()
        self.overlay_x_spin.setRange(0, 3840)
        self.overlay_x_spin.setValue(100)
        layout.addRow("Позиция X:", self.overlay_x_spin)
        
        # Позиция Y
        self.overlay_y_spin = QSpinBox()
        self.overlay_y_spin.setRange(0, 2160)
        self.overlay_y_spin.setValue(100)
        layout.addRow("Позиция Y:", self.overlay_y_spin)
        
        # Ширина
        self.overlay_width_spin = QSpinBox()
        self.overlay_width_spin.setRange(200, 800)
        self.overlay_width_spin.setValue(400)
        layout.addRow("Ширина окна:", self.overlay_width_spin)
        
        # Высота
        self.overlay_height_spin = QSpinBox()
        self.overlay_height_spin.setRange(150, 600)
        self.overlay_height_spin.setValue(300)
        layout.addRow("Высота окна:", self.overlay_height_spin)
        
        # Прозрачность
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.1, 1.0)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.setValue(0.8)
        layout.addRow("Прозрачность:", self.opacity_spin)
        
        # Размер шрифта
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 24)
        self.font_size_spin.setValue(14)
        layout.addRow("Размер шрифта:", self.font_size_spin)
        
        return widget
    
    def _load_current_settings(self):
        """Загрузка текущих настроек в интерфейс"""
        # Whisper
        available_models = get_available_whisper_models()
        self.whisper_model_combo.clear()
        self.whisper_model_combo.addItems(available_models)
        if self.config.whisper.model_name in available_models:
            self.whisper_model_combo.setCurrentText(self.config.whisper.model_name)
        
        if self.config.whisper.model_path:
            self.whisper_path_edit.setText(self.config.whisper.model_path)
        
        self.whisper_device_combo.setCurrentText(self.config.whisper.device)
        self.compute_type_combo.setCurrentText(self.config.whisper.compute_type)
        self.language_edit.setText(self.config.whisper.language)
        self.diarization_check.setChecked(self.config.whisper.use_speaker_diarization)
        
        # LLM
        self.provider_combo.setCurrentText(self.config.llm.provider)
        self.llm_model_combo.setCurrentText(self.config.llm.model_name)
        self.base_url_edit.setText(self.config.llm.base_url)
        self.lmstudio_port_spin.setValue(self.config.llm.lmstudio_port)
        self.temperature_spin.setValue(self.config.llm.temperature)
        self.max_tokens_spin.setValue(self.config.llm.max_tokens)
        
        # Audio
        if self.config.audio.input_device:
            self.audio_device_combo.setCurrentText(self.config.audio.input_device)
        self.sample_rate_spin.setValue(self.config.audio.sample_rate)
        self.channels_spin.setValue(self.config.audio.channels)
        self.chunk_duration_spin.setValue(self.config.audio.chunk_duration)
        
        # Screen
        self.screen_enabled_check.setChecked(self.config.screen.enabled)
        self.monitor_index_spin.setValue(self.config.screen.monitor_index)
        self.capture_interval_spin.setValue(self.config.screen.capture_interval)
        self.resize_width_spin.setValue(self.config.screen.resize_width)
        self.resize_height_spin.setValue(self.config.screen.resize_height)
        
        # Overlay
        self.overlay_x_spin.setValue(self.config.overlay.position_x)
        self.overlay_y_spin.setValue(self.config.overlay.position_y)
        self.overlay_width_spin.setValue(self.config.overlay.width)
        self.overlay_height_spin.setValue(self.config.overlay.height)
        self.opacity_spin.setValue(self.config.overlay.opacity)
        self.font_size_spin.setValue(self.config.overlay.font_size)
        
        # Заполнение аудио устройств
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            self.audio_device_combo.clear()
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    self.audio_device_combo.addItem(f"{dev['name']} ({i})")
        except Exception as e:
            print(f"Ошибка получения списка аудио устройств: {e}")
    
    def _import_whisper_model(self):
        """Импорт .pt модели Whisper"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите модель Whisper (.pt)",
            "",
            "PyTorch Models (*.pt);;All Files (*)"
        )
        
        if file_path:
            # Копирование файла в директорию моделей
            import shutil
            filename = os.path.basename(file_path)
            dest_path = MODELS_DIR / filename
            
            try:
                shutil.copy2(file_path, dest_path)
                self.whisper_path_edit.setText(str(dest_path))
                
                # Обновление списка моделей
                available_models = get_available_whisper_models()
                self.whisper_model_combo.clear()
                self.whisper_model_combo.addItems(available_models)
                self.whisper_model_combo.setCurrentText(filename)
                
                QMessageBox.information(self, "Успех", f"Модель {filename} успешно импортирована!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось импортировать модель: {e}")
    
    def _on_provider_changed(self, provider: str):
        """Обработчик изменения провайдера"""
        # Можно добавить логику для автоматического обновления URL и портов
    
    def _refresh_llm_models(self):
        """Обновление списка LLM моделей"""
        provider = self.provider_combo.currentText()
        
        try:
            temp_config = Config()
            temp_config.llm.provider = provider
            temp_config.llm.base_url = self.base_url_edit.text()
            temp_config.llm.lmstudio_port = self.lmstudio_port_spin.value()
            
            engine = LLMEngine(temp_config)
            models = engine.get_available_models()
            
            self.llm_model_combo.clear()
            self.llm_model_combo.addItems(models)
            
            if models:
                QMessageBox.information(self, "Успех", f"Найдено моделей: {len(models)}")
            else:
                QMessageBox.warning(self, "Предупреждение", "Модели не найдены. Проверьте подключение.")
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось получить список моделей: {e}")
    
    def _test_llm_connection(self):
        """Тестирование подключения к LLM"""
        provider = self.provider_combo.currentText()
        
        try:
            temp_config = Config()
            temp_config.llm.provider = provider
            temp_config.llm.base_url = self.base_url_edit.text()
            temp_config.llm.lmstudio_port = self.lmstudio_port_spin.value()
            temp_config.llm.model_name = self.llm_model_combo.currentText()
            
            engine = LLMEngine(temp_config)
            result = engine.test_connection()
            
            if result["success"]:
                QMessageBox.information(self, "Успех", result["message"])
            else:
                QMessageBox.warning(self, "Ошибка подключения", result["message"])
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Тест не удался: {e}")
    
    def _save_settings(self):
        """Сохранение настроек"""
        # Whisper
        self.config.whisper.model_name = self.whisper_model_combo.currentText()
        self.config.whisper.model_path = self.whisper_path_edit.text()
        self.config.whisper.device = self.whisper_device_combo.currentText()
        self.config.whisper.compute_type = self.compute_type_combo.currentText()
        self.config.whisper.language = self.language_edit.text()
        self.config.whisper.use_speaker_diarization = self.diarization_check.isChecked()
        
        # LLM
        self.config.llm.provider = self.provider_combo.currentText()
        self.config.llm.model_name = self.llm_model_combo.currentText()
        self.config.llm.base_url = self.base_url_edit.text()
        self.config.llm.lmstudio_port = self.lmstudio_port_spin.value()
        self.config.llm.temperature = self.temperature_spin.value()
        self.config.llm.max_tokens = self.max_tokens_spin.value()
        
        # Audio
        device_text = self.audio_device_combo.currentText()
        if device_text:
            # Извлечение имени устройства из строки "name (index)"
            if " (" in device_text:
                self.config.audio.input_device = device_text.split(" (")[0]
            else:
                self.config.audio.input_device = device_text
        
        self.config.audio.sample_rate = self.sample_rate_spin.value()
        self.config.audio.channels = self.channels_spin.value()
        self.config.audio.chunk_duration = self.chunk_duration_spin.value()
        
        # Screen
        self.config.screen.enabled = self.screen_enabled_check.isChecked()
        self.config.screen.monitor_index = self.monitor_index_spin.value()
        self.config.screen.capture_interval = self.capture_interval_spin.value()
        self.config.screen.resize_width = self.resize_width_spin.value()
        self.config.screen.resize_height = self.resize_height_spin.value()
        
        # Overlay
        self.config.overlay.position_x = self.overlay_x_spin.value()
        self.config.overlay.position_y = self.overlay_y_spin.value()
        self.config.overlay.width = self.overlay_width_spin.value()
        self.config.overlay.height = self.overlay_height_spin.value()
        self.config.overlay.opacity = self.opacity_spin.value()
        self.config.overlay.font_size = self.font_size_spin.value()
        
        # Сохранение в файл
        save_config(self.config)
        
        # Отправка сигнала
        self.settings_saved.emit(self.config)
        
        QMessageBox.information(self, "Успех", "Настройки успешно сохранены!")
        self.close()
