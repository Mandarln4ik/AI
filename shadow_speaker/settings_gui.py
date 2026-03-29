"""
GUI для настройки ShadowSpeaker
Управление провайдерами LLM, выбор моделей, импорт Whisper моделей
"""
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger

try:
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
        QLabel, QPushButton, QComboBox, QLineEdit, 
        QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
        QTabWidget, QDialog, QDialogButtonBox, QFileDialog,
        QMessageBox, QListWidget, QListWidgetItem, QCheckBox,
        QProgressBar, QTextEdit
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logger.warning("PyQt6 not available, settings GUI disabled")

import requests
import json


class ModelDownloader(QThread):
    """Поток для загрузки моделей"""
    progress = pyqtSignal(int, str)  # percent, message
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, provider: str, host: str, model_name: str):
        super().__init__()
        self.provider = provider
        self.host = host
        self.model_name = model_name
    
    def run(self):
        try:
            if self.provider == "ollama":
                # Ollama pull через API
                self.progress.emit(0, f"Starting download of {self.model_name}...")
                
                response = requests.post(
                    f"{self.host}/api/pull",
                    json={"name": self.model_name},
                    stream=True,
                    timeout=600
                )
                
                if response.status_code != 200:
                    self.finished.emit(False, f"Failed to start download: {response.status_code}")
                    return
                
                total_size = 0
                downloaded = 0
                
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "total" in data:
                            total_size = data["total"]
                        if "completed" in data:
                            downloaded = data["completed"]
                        
                        if total_size > 0:
                            percent = int((downloaded / total_size) * 100)
                            status = data.get("status", "downloading")
                            self.progress.emit(percent, f"{status}: {percent}%")
                        
                        if data.get("status") == "success":
                            self.progress.emit(100, "Download complete!")
                            self.finished.emit(True, f"Model {self.model_name} downloaded successfully")
                            return
                
                self.finished.emit(True, "Download completed")
                
            elif self.provider == "lmstudio":
                # LM Studio не поддерживает загрузку через API, только уведомление
                self.progress.emit(50, "LM Studio requires manual model download")
                self.progress.emit(100, "Open LM Studio UI to download models")
                self.finished.emit(
                    False, 
                    "LM Studio: Please download models through LM Studio interface. "
                    "The local server only serves already loaded models."
                )
            
        except Exception as e:
            self.finished.emit(False, f"Download error: {str(e)}")


class SettingsDialog(QDialog):
    """Диалог настроек приложения"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("ShadowSpeaker - Настройки")
        self.setMinimumSize(700, 600)
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Создаем вкладки
        tabs = QTabWidget()
        
        # Вкладка LLM
        llm_tab = self.create_llm_tab()
        tabs.addTab(llm_tab, "LLM Провайдер")
        
        # Вкладка Whisper
        whisper_tab = self.create_whisper_tab()
        tabs.addTab(whisper_tab, "Whisper STT")
        
        # Вкладка Аудио
        audio_tab = self.create_audio_tab()
        tabs.addTab(audio_tab, "Аудио")
        
        # Вкладка Оверлей
        overlay_tab = self.create_overlay_tab()
        tabs.addTab(overlay_tab, "Оверлей")
        
        layout.addWidget(tabs)
        
        # Кнопки
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save_settings)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def create_llm_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Выбор провайдера
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["ollama", "lmstudio", "llama_cpp"])
        self.provider_combo.setCurrentText(self.config.llm_provider)
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        layout.addRow("Провайдер:", self.provider_combo)
        
        # Хост
        self.host_edit = QLineEdit(self.config.llm_host)
        layout.addRow("Хост API:", self.host_edit)
        
        # Модель
        self.model_edit = QLineEdit(self.config.llm_model)
        layout.addRow("Модель:", self.model_edit)
        
        # Кнопка проверки соединения
        self.check_conn_btn = QPushButton("Проверить соединение")
        self.check_conn_btn.clicked.connect(self.check_connection)
        layout.addRow("", self.check_conn_btn)
        
        # Кнопка загрузки модели
        self.download_model_btn = QPushButton("Загрузить модель")
        self.download_model_btn.clicked.connect(self.download_model)
        layout.addRow("", self.download_model_btn)
        
        # Статус загрузки
        self.download_status = QLabel("")
        self.download_progress = QProgressBar()
        self.download_progress.setVisible(False)
        layout.addRow("", self.download_status)
        layout.addRow("", self.download_progress)
        
        # Температура
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(self.config.llm_temperature)
        layout.addRow("Температура:", self.temp_spin)
        
        # Макс токенов
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(50, 4096)
        self.max_tokens_spin.setValue(self.config.llm_max_tokens)
        layout.addRow("Макс токенов:", self.max_tokens_spin)
        
        # Список доступных моделей
        models_group = QGroupBox("Доступные модели")
        models_layout = QVBoxLayout()
        self.models_list = QListWidget()
        self.refresh_models_btn = QPushButton("Обновить список")
        self.refresh_models_btn.clicked.connect(self.refresh_models_list)
        models_layout.addWidget(self.models_list)
        models_layout.addWidget(self.refresh_models_btn)
        models_group.setLayout(models_layout)
        layout.addRow("", models_group)
        
        return tab
    
    def create_whisper_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Путь к модели Whisper
        whisper_layout = QHBoxLayout()
        self.whisper_path_edit = QLineEdit(self.config.whisper_model_path or "")
        whisper_layout.addWidget(self.whisper_path_edit)
        browse_btn = QPushButton("Обзор...")
        browse_btn.clicked.connect(self.browse_whisper_model)
        whisper_layout.addWidget(browse_btn)
        layout.addRow("Путь к .pt файлу:", whisper_layout)
        
        # Built-in модель
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems([
            "tiny", "base", "small", "medium", "large-v2", "large-v3-turbo"
        ])
        if self.config.whisper_model in [item.text() for item in range(self.whisper_model_combo.count())]:
            self.whisper_model_combo.setCurrentText(self.config.whisper_model)
        layout.addRow("Built-in модель:", self.whisper_model_combo)
        
        # Устройство
        self.whisper_device_combo = QComboBox()
        self.whisper_device_combo.addItems(["cuda", "cpu"])
        self.whisper_device_combo.setCurrentText(self.config.whisper_device)
        layout.addRow("Устройство:", self.whisper_device_combo)
        
        # Тип вычислений
        self.compute_type_combo = QComboBox()
        self.compute_type_combo.addItems(["float16", "int8", "int8_float16"])
        self.compute_type_combo.setCurrentText(self.config.whisper_compute_type)
        layout.addRow("Тип вычислений:", self.compute_type_combo)
        
        # Информация
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(100)
        info_text.setText(
            "Для импорта вашей модели large-v3-turbo.pt:\n"
            "1. Нажмите 'Обзор...' и выберите файл .pt\n"
            "2. Убедитесь что файл совместим с faster-whisper\n"
            "3. Перезапустите приложение для применения настроек"
        )
        layout.addRow("", info_text)
        
        return tab
    
    def create_audio_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Sample rate
        self.sr_spin = QSpinBox()
        self.sr_spin.setRange(8000, 48000)
        self.sr_spin.setSingleStep(8000)
        self.sr_spin.setValue(self.config.audio_sample_rate)
        layout.addRow("Sample Rate:", self.sr_spin)
        
        # Длительность чанка
        self.chunk_spin = QDoubleSpinBox()
        self.chunk_spin.setRange(1.0, 30.0)
        self.chunk_spin.setSingleStep(1.0)
        self.chunk_spin.setValue(self.config.audio_chunk_duration)
        layout.addRow("Длительность чанка (сек):", self.chunk_spin)
        
        # Порог тишины
        self.silence_spin = QDoubleSpinBox()
        self.silence_spin.setRange(0.001, 1.0)
        self.silence_spin.setSingleStep(0.01)
        self.silence_spin.setValue(self.config.silence_threshold)
        layout.addRow("Порог тишины:", self.silence_spin)
        
        # Диаризация
        self.diarization_check = QCheckBox("Включить диаризацию спикеров")
        self.diarization_check.setChecked(self.config.use_diarization)
        layout.addRow("", self.diarization_check)
        
        # Макс спикеров
        self.max_speakers_spin = QSpinBox()
        self.max_speakers_spin.setRange(1, 10)
        self.max_speakers_spin.setValue(self.config.max_speakers)
        layout.addRow("Макс спикеров:", self.max_speakers_spin)
        
        return tab
    
    def create_overlay_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Позиция
        self.position_combo = QComboBox()
        self.position_combo.addItems([
            "top_left", "top_right", "bottom_left", "bottom_right"
        ])
        self.position_combo.setCurrentText(self.config.overlay_position)
        layout.addRow("Позиция:", self.position_combo)
        
        # Прозрачность
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.1, 1.0)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.setValue(self.config.overlay_opacity)
        layout.addRow("Прозрачность:", self.opacity_spin)
        
        # Размер шрифта
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 72)
        self.font_size_spin.setValue(self.config.overlay_font_size)
        layout.addRow("Размер шрифта:", self.font_size_spin)
        
        # Ширина
        self.max_width_spin = QSpinBox()
        self.max_width_spin.setRange(200, 2000)
        self.max_width_spin.setValue(self.config.overlay_max_width)
        layout.addRow("Макс ширина:", self.max_width_spin)
        
        return tab
    
    def on_provider_changed(self, provider: str):
        """Изменение провайдера"""
        if provider == "lmstudio":
            self.host_edit.setText("http://localhost:1234")
        elif provider == "ollama":
            self.host_edit.setText("http://localhost:11434")
    
    def check_connection(self):
        """Проверка соединения с провайдером"""
        provider = self.provider_combo.currentText()
        host = self.host_edit.text()
        
        try:
            if provider == "ollama":
                response = requests.get(f"{host}/api/tags", timeout=5)
                if response.status_code == 200:
                    QMessageBox.information(self, "Успех", "Ollama подключен!")
                    self.refresh_models_list()
                else:
                    QMessageBox.warning(self, "Ошибка", f"Ollama ответил: {response.status_code}")
            
            elif provider == "lmstudio":
                response = requests.get(f"{host}/v1/models", timeout=5)
                if response.status_code == 200:
                    QMessageBox.information(self, "Успех", "LM Studio подключен!")
                    self.refresh_models_list()
                else:
                    QMessageBox.warning(self, "Ошибка", f"LM Studio ответил: {response.status_code}")
            
            elif provider == "llama_cpp":
                QMessageBox.information(
                    self, "Инфо", 
                    "llama.cpp работает локально без сервера. "
                    "Укажите путь к .gguf файлу в поле 'Модель'."
                )
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось подключиться: {str(e)}")
    
    def refresh_models_list(self):
        """Обновление списка доступных моделей"""
        provider = self.provider_combo.currentText()
        host = self.host_edit.text()
        
        self.models_list.clear()
        
        try:
            if provider == "ollama":
                response = requests.get(f"{host}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    for model in models:
                        name = model.get("name", "unknown")
                        size = model.get("size", 0)
                        size_gb = size / (1024**3) if size > 0 else 0
                        self.models_list.addItem(f"{name} ({size_gb:.1f} GB)")
            
            elif provider == "lmstudio":
                response = requests.get(f"{host}/v1/models", timeout=5)
                if response.status_code == 200:
                    models_data = response.json().get("data", [])
                    for model in models_data:
                        model_id = model.get("id", "unknown")
                        self.models_list.addItem(model_id)
            
            if self.models_list.count() == 0:
                self.models_list.addItem("Нет доступных моделей")
        
        except Exception as e:
            self.models_list.addItem(f"Ошибка: {str(e)}")
    
    def download_model(self):
        """Загрузка модели"""
        provider = self.provider_combo.currentText()
        host = self.host_edit.text()
        model_name = self.model_edit.text()
        
        if not model_name:
            QMessageBox.warning(self, "Ошибка", "Введите имя модели")
            return
        
        self.downloader = ModelDownloader(provider, host, model_name)
        self.downloader.progress.connect(self.on_download_progress)
        self.downloader.finished.connect(self.on_download_finished)
        
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)
        self.download_status.setText("Starting download...")
        self.download_model_btn.setEnabled(False)
        
        self.downloader.start()
    
    def on_download_progress(self, percent: int, message: str):
        """Обновление прогресса загрузки"""
        self.download_progress.setValue(percent)
        self.download_status.setText(message)
    
    def on_download_finished(self, success: bool, message: str):
        """Завершение загрузки"""
        self.download_model_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Успех", message)
            self.refresh_models_list()
        else:
            QMessageBox.warning(self, "Инфо", message)
    
    def browse_whisper_model(self):
        """Выбор файла Whisper модели"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите Whisper модель",
            "",
            "PyTorch Models (*.pt);;All Files (*)"
        )
        
        if file_path:
            self.whisper_path_edit.setText(file_path)
    
    def save_settings(self):
        """Сохранение настроек"""
        # LLM
        self.config.llm_provider = self.provider_combo.currentText()
        self.config.llm_host = self.host_edit.text()
        self.config.llm_model = self.model_edit.text()
        self.config.llm_temperature = self.temp_spin.value()
        self.config.llm_max_tokens = self.max_tokens_spin.value()
        
        # Whisper
        self.config.whisper_model_path = self.whisper_path_edit.text() or None
        self.config.whisper_model = self.whisper_model_combo.currentText()
        self.config.whisper_device = self.whisper_device_combo.currentText()
        self.config.whisper_compute_type = self.compute_type_combo.currentText()
        
        # Audio
        self.config.audio_sample_rate = self.sr_spin.value()
        self.config.audio_chunk_duration = self.chunk_spin.value()
        self.config.silence_threshold = self.silence_spin.value()
        self.config.use_diarization = self.diarization_check.isChecked()
        self.config.max_speakers = self.max_speakers_spin.value()
        
        # Overlay
        self.config.overlay_position = self.position_combo.currentText()
        self.config.overlay_opacity = self.opacity_spin.value()
        self.config.overlay_font_size = self.font_size_spin.value()
        self.config.overlay_max_width = self.max_width_spin.value()
        
        logger.info("Settings saved")
        self.accept()


def show_settings_dialog(config):
    """Показать диалог настроек"""
    if not PYQT_AVAILABLE:
        logger.error("PyQt6 is required for settings GUI")
        print("Error: PyQt6 is required for settings GUI")
        print("Install with: pip install PyQt6")
        return False
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    dialog = SettingsDialog(config)
    result = dialog.exec()
    
    return result == QDialog.DialogCode.Accepted


if __name__ == "__main__":
    # Тестовый запуск
    from config import config
    
    app = QApplication(sys.argv)
    dialog = SettingsDialog(config)
    dialog.show()
    sys.exit(app.exec())
