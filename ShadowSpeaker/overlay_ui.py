import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QColor, QFont, QPalette, QKeySequence, QShortcut


class OverlayWindow(QMainWindow):
    """Прозрачное оверлейное окно с вариантами ответов"""
    
    option_selected = pyqtSignal(int)  # Сигнал выбора варианта (0, 1, 2)
    settings_requested = pyqtSignal()  # Сигнал запроса настроек
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_options = ["", "", ""]
        
        self._setup_ui()
        self._setup_hotkeys()
        
        # Автоскрытие через 30 секунд без активности
        self.hide_timer = QTimer()
        self.hide_timer.timeout.connect(self.hide)
        self.hide_timer.setInterval(30000)  # 30 секунд
    
    def _setup_ui(self):
        """Настройка интерфейса"""
        # Убираем рамки и делаем окно прозрачным
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Установка позиции и размера
        self.setGeometry(
            self.config.overlay.position_x,
            self.config.overlay.position_y,
            self.config.overlay.width,
            self.config.overlay.height
        )
        
        # Главный виджет с полупрозрачным фоном
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Стилизация
        palette = central_widget.palette()
        bg_color = QColor(30, 30, 30, int(255 * self.config.overlay.opacity))
        palette.setColor(QPalette.ColorRole.Window, bg_color)
        central_widget.setAutoFillBackground(True)
        central_widget.setPalette(palette)
        
        # Основной layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Заголовок
        header_layout = QHBoxLayout()
        
        title_label = QLabel("💡 Варианты ответов")
        title_label.setStyleSheet("""
            color: #4CAF50;
            font-size: 16px;
            font-weight: bold;
        """)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Кнопка настроек
        settings_btn = QPushButton("⚙")
        settings_btn.setFixedSize(30, 30)
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: white;
                border-radius: 15px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        settings_btn.clicked.connect(lambda: self.settings_requested.emit())
        header_layout.addWidget(settings_btn)
        
        layout.addLayout(header_layout)
        
        # Разделитель
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #666;")
        line.setFixedHeight(1)
        layout.addWidget(line)
        
        # Варианты ответов
        self.option_labels = []
        for i in range(3):
            option_frame = QFrame()
            option_frame.setStyleSheet("""
                QFrame {
                    background-color: rgba(60, 60, 60, 0.8);
                    border-radius: 8px;
                    padding: 5px;
                }
                QFrame:hover {
                    background-color: rgba(80, 80, 80, 0.9);
                }
            """)
            
            option_layout = QHBoxLayout(option_frame)
            option_layout.setContentsMargins(10, 8, 10, 8)
            
            # Номер варианта
            num_label = QLabel(f"{i+1}.")
            num_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
            option_layout.addWidget(num_label)
            
            # Текст варианта
            text_label = QLabel("Ожидание ответа...")
            text_label.setWordWrap(True)
            text_label.setStyleSheet(f"""
                color: white;
                font-size: {self.config.overlay.font_size}px;
            """)
            text_label.setMinimumHeight(40)
            option_layout.addWidget(text_label)
            
            self.option_labels.append(text_label)
            layout.addWidget(option_frame)
        
        # Подсказка по горячим клавишам
        hint_label = QLabel("Ctrl+1/2/3 - выбрать вариант | Ctrl+S - настройки")
        hint_label.setStyleSheet("color: #888; font-size: 11px;")
        hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint_label)
    
    def _setup_hotkeys(self):
        """Настройка горячих клавиш"""
        # Выбор вариантов
        QShortcut(QKeySequence("Ctrl+1"), self).activated.connect(lambda: self._select_option(0))
        QShortcut(QKeySequence("Ctrl+2"), self).activated.connect(lambda: self._select_option(1))
        QShortcut(QKeySequence("Ctrl+3"), self).activated.connect(lambda: self._select_option(2))
        
        # Настройки
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(lambda: self.settings_requested.emit())
    
    def _select_option(self, index: int):
        """Выбор варианта ответа"""
        if 0 <= index < len(self.current_options):
            self.option_selected.emit(index)
            self.hide_timer.stop()
            self.hide()
    
    def update_options(self, options: list):
        """Обновление вариантов ответов"""
        self.current_options = options
        
        for i, label in enumerate(self.option_labels):
            if i < len(options):
                label.setText(options[i])
            else:
                label.setText("Нет варианта")
        
        # Показываем окно и сбрасываем таймер
        self.show()
        self.hide_timer.start()
    
    def mousePressEvent(self, event):
        """Перетаскивание окна"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Перемещение окна при перетаскивании"""
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, 'drag_position'):
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
