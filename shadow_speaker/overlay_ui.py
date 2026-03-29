"""
Оверлей интерфейс для отображения вариантов ответов
Использует PyQt6 для создания прозрачного окна поверх всех приложений
"""
import sys
from typing import List, Dict, Optional, Callable
from loguru import logger

try:
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
        QFrame, QGraphicsOpacityEffect
    )
    from PyQt6.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal
    from PyQt6.QtGui import QFont, QColor, QPalette, QKeySequence, QShortcut
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logger.warning("PyQt6 not installed")


class OverlayWindow(QWidget):
    """
    Прозрачное оверлей окно с вариантами ответов
    """
    
    # Сигналы для выбора вариантов
    variant_selected = pyqtSignal(int)  # индекс выбранного варианта
    refresh_requested = pyqtSignal()
    toggle_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.variants = []
        self.is_visible = True
        
        if not PYQT_AVAILABLE:
            logger.error("PyQt6 is required for overlay")
            return
        
        self._setup_ui()
        self._setup_hotkeys()
        self._position_window()
        
    def _setup_ui(self):
        """Настройка интерфейса"""
        # Базовые настройки окна
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        # Основной контейнер
        main_frame = QFrame(self)
        main_frame.setObjectName("overlayFrame")
        main_frame.setStyleSheet("""
            #overlayFrame {
                background-color: rgba(30, 30, 30, 230);
                border-radius: 10px;
                border: 1px solid rgba(100, 100, 100, 150);
            }
        """)
        
        # Вертикальный layout
        layout = QVBoxLayout(main_frame)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Заголовок
        title_label = QLabel("💡 Варианты ответов")
        title_label.setStyleSheet("""
            color: #4FC3F7;
            font-size: 16px;
            font-weight: bold;
            padding-bottom: 5px;
        """)
        layout.addWidget(title_label)
        
        # Контейнер для вариантов
        self.variants_container = QWidget()
        self.variants_layout = QVBoxLayout(self.variants_container)
        self.variants_layout.setSpacing(8)
        layout.addWidget(self.variants_container)
        
        # Кнопка настроек
        settings_btn = QPushButton("⚙ Настройки")
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(50, 50, 50, 200);
                color: #4FC3F7;
                border: 1px solid rgba(100, 100, 100, 100);
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(70, 70, 70, 230);
                border: 1px solid #4FC3F7;
            }
        """)
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.clicked.connect(self._on_settings_clicked)
        layout.addWidget(settings_btn)
        
        # Подсказка по горячим клавишам
        hint_label = QLabel("Ctrl+1/2/3 - выбрать | Ctrl+R - обновить | Ctrl+H - скрыть | Ctrl+S - настройки")
        hint_label.setStyleSheet("""
            color: #90A4AE;
            font-size: 11px;
            padding-top: 8px;
        """)
        layout.addWidget(hint_label)
        
        # Устанавливаем главный виджет
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(main_frame)
        
        # Ограничиваем размер
        self.setMaximumWidth(self.config.overlay_max_width)
        
    def _setup_hotkeys(self):
        """Настройка горячих клавиш"""
        if not PYQT_AVAILABLE:
            return
        
        # Ctrl+1, Ctrl+2, Ctrl+3 - выбор вариантов
        for i in range(3):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{i+1}"), self)
            shortcut.activated.connect(lambda idx=i: self._on_variant_hotkey(idx))
        
        # Ctrl+R - обновить
        shortcut_refresh = QShortcut(QKeySequence("Ctrl+R"), self)
        shortcut_refresh.activated.connect(self.refresh_requested.emit)
        
        # Ctrl+H - показать/скрыть
        shortcut_toggle = QShortcut(QKeySequence("Ctrl+H"), self)
        shortcut_toggle.activated.connect(self.toggle_requested.emit)
        
        # Ctrl+S - настройки
        shortcut_settings = QShortcut(QKeySequence("Ctrl+S"), self)
        shortcut_settings.activated.connect(self._on_settings_clicked)
        
        # Ctrl+Q - выход
        shortcut_quit = QShortcut(QKeySequence("Ctrl+Q"), self)
        shortcut_quit.activated.connect(self.quit_requested.emit)
    
    def _on_settings_clicked(self):
        """Открыть диалог настроек"""
        try:
            from settings_gui import show_settings_dialog
            logger.info("Opening settings dialog")
            show_settings_dialog(self.config)
        except Exception as e:
            logger.error(f"Failed to open settings: {e}")
    
    def _position_window(self):
        """Позиционирование окна согласно настройкам"""
        if not PYQT_AVAILABLE:
            return
        
        screen = QApplication.primaryScreen().geometry()
        margin = self.config.overlay_margin
        
        positions = {
            "top_left": (screen.left() + margin, screen.top() + margin),
            "top_right": (screen.right() - self.width() - margin, screen.top() + margin),
            "bottom_left": (screen.left() + margin, screen.bottom() - self.height() - margin),
            "bottom_right": (screen.right() - self.width() - margin, screen.bottom() - self.height() - margin)
        }
        
        pos = positions.get(self.config.overlay_position, positions["bottom_right"])
        self.move(pos[0], pos[1])
    
    def update_variants(self, variants: List[Dict[str, str]]):
        """Обновить отображаемые варианты ответов"""
        self.variants = variants
        
        # Очищаем текущие виджеты
        while self.variants_layout.count():
            item = self.variants_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Добавляем новые варианты
        styles = {
            "neutral": ("🔹", "#B0BEC5"),
            "friendly": ("😊", "#81C784"),
            "concise": ("⚡", "#FFB74D"),
            "variant_1": ("1️⃣", "#64B5F6"),
            "variant_2": ("2️⃣", "#4DB6AC"),
            "variant_3": ("3️⃣", "#BA68C8")
        }
        
        for i, variant in enumerate(variants[:3]):
            style_key = variant.get("style", f"variant_{i+1}")
            icon, color = styles.get(style_key, ("•", "#FFFFFF"))
            
            text = variant.get("text", "")
            
            btn = QPushButton(f"{icon} [{i+1}] {text}")
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(50, 50, 50, 200);
                    color: {color};
                    border: 1px solid rgba(100, 100, 100, 100);
                    border-radius: 5px;
                    padding: 8px 12px;
                    text-align: left;
                    font-size: {self.config.overlay_font_size}px;
                }}
                QPushButton:hover {{
                    background-color: rgba(70, 70, 70, 230);
                    border: 1px solid {color};
                }}
                QPushButton:pressed {{
                    background-color: rgba(90, 90, 90, 255);
                }}
            """)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, idx=i: self._on_variant_clicked(idx))
            
            self.variants_layout.addWidget(btn)
        
        # Показываем окно если есть варианты
        if variants and not self.isVisible():
            self.show()
    
    def _on_variant_clicked(self, index: int):
        """Обработчик клика по варианту"""
        if 0 <= index < len(self.variants):
            logger.info(f"Variant {index+1} selected: {self.variants[index].get('text', '')}")
            self.variant_selected.emit(index)
    
    def _on_variant_hotkey(self, index: int):
        """Обработчик горячей клавиши выбора"""
        if self.isVisible() and 0 <= index < len(self.variants):
            self._on_variant_clicked(index)
    
    def toggle_visibility(self):
        """Переключить видимость"""
        self.is_visible = not self.is_visible
        if self.is_visible:
            self.show()
            self.activateWindow()
        else:
            self.hide()
    
    def clear_variants(self):
        """Очистить варианты"""
        self.variants = []
        while self.variants_layout.count():
            item = self.variants_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class OverlayManager:
    """
    Менеджер оверлея - управляет созданием и жизненным циклом
    """
    
    def __init__(self, config):
        self.config = config
        self.app = None
        self.window = None
        self.is_running = False
        
    def start(self, 
              on_variant_selected: Optional[Callable[[int], None]] = None,
              on_refresh: Optional[Callable[[], None]] = None,
              on_toggle: Optional[Callable[[], None]] = None,
              on_quit: Optional[Callable[[], None]] = None):
        """
        Запустить оверлей
        
        Args:
            on_variant_selected: Callback при выборе варианта
            on_refresh: Callback при запросе обновления
            on_toggle: Callback при переключении видимости
            on_quit: Callback при выходе
        """
        if not PYQT_AVAILABLE:
            logger.error("Cannot start overlay: PyQt6 not available")
            return False
        
        try:
            # Создаем QApplication если еще нет
            if not QApplication.instance():
                self.app = QApplication(sys.argv)
            else:
                self.app = QApplication.instance()
            
            # Создаем окно
            self.window = OverlayWindow(self.config)
            
            # Подключаем сигналы
            if on_variant_selected:
                self.window.variant_selected.connect(on_variant_selected)
            if on_refresh:
                self.window.refresh_requested.connect(on_refresh)
            if on_toggle:
                self.window.toggle_requested.connect(on_toggle)
            if on_quit:
                self.window.quit_requested.connect(on_quit)
            
            # Показываем окно (изначально скрыто пока нет вариантов)
            self.window.hide()
            
            self.is_running = True
            logger.info("Overlay manager started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start overlay: {e}")
            return False
    
    def update_variants(self, variants: List[Dict[str, str]]):
        """Обновить варианты в оверлее"""
        if self.window:
            self.window.update_variants(variants)
    
    def clear_variants(self):
        """Очистить варианты"""
        if self.window:
            self.window.clear_variants()
    
    def run(self):
        """Запустить цикл событий Qt"""
        if self.app and self.is_running:
            logger.info("Starting Qt event loop")
            self.app.exec()
    
    def stop(self):
        """Остановить оверлей"""
        self.is_running = False
        if self.window:
            self.window.close()
        if self.app:
            self.app.quit()
        logger.info("Overlay manager stopped")
