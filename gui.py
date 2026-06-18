import argparse
import base64
import json
import os
import shutil
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None

try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    from sign_detect.sign_detect import (  # type: ignore
        Detection,
        clue_type_for_detection,
        detect_signs,
        model_ready as yolo_model_ready,
        model_status as yolo_model_status,
        warmup_model as yolo_warmup_model,
    )

    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False
    Detection = None  # type: ignore

    def yolo_model_ready() -> bool:  # type: ignore
        return False

    def yolo_model_status() -> str:  # type: ignore
        return "YOLO: модуль sign_detect недоступен"

    def detect_signs(*_a, **_k):  # type: ignore
        raise RuntimeError("sign_detect недоступен")

    def yolo_warmup_model() -> None:  # type: ignore
        return

    def clue_type_for_detection(_d):  # type: ignore
        return "auto"

try:
    from ocr_pipeline import (  # type: ignore
        _score_text,
        easyocr_available,
        easyocr_status_message,
        last_ocr_engine,
        run_ocr_on_pil,
    )
except Exception:
    run_ocr_on_pil = None  # type: ignore

    def _score_text(t: str) -> int:  # type: ignore
        return len((t or "").strip())

    def last_ocr_engine() -> str:  # type: ignore
        return "unknown"

    def easyocr_available() -> bool:  # type: ignore
        return False

    def easyocr_status_message() -> str:  # type: ignore
        return ""


@dataclass
class Fragment:
    created_at_ms: int
    rect: QtCore.QRect
    pixmap: QtGui.QPixmap
    note: str = ""
    ocr_text: str = ""
    clue_type: str = "auto"
    detected_kind: str = ""
    detected_label: str = ""
    id: str = field(default_factory=lambda: f"frag_{int(time.time() * 1000)}")


class ScreenGrabOverlay(QtWidgets.QWidget):
    grabbed = QtCore.Signal(QtCore.QRect, QtGui.QPixmap)
    cancelled = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)

        self._origin: Optional[QtCore.QPoint] = None
        self._current: Optional[QtCore.QPoint] = None
        self._dragging = False
        self._screen: Optional[QtGui.QScreen] = None

    def start(self) -> None:
        self._screen = QtGui.QGuiApplication.primaryScreen()
        if self._screen is None:
            QtWidgets.QMessageBox.critical(None, "Ошибка", "Не удалось получить primaryScreen().")
            self.cancelled.emit()
            self.hide()
            return

        geom = self._screen.geometry()
        self.setGeometry(geom)
        self._origin = None
        self._current = None
        self._dragging = False
        self.show()
        self.raise_()
        self.activateWindow()

    def cancel(self) -> None:
        self._origin = None
        self._current = None
        self._dragging = False
        self.hide()
        self.cancelled.emit()

    def _selection_rect(self) -> Optional[QtCore.QRect]:
        if self._origin is None or self._current is None:
            return None
        rect = QtCore.QRect(self._origin, self._current).normalized()
        if rect.width() < 3 or rect.height() < 3:
            return None
        return rect

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self._dragging = True
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._dragging:
            return
        self._current = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return

        self._dragging = False
        rect = self._selection_rect()
        if rect is None:
            self.cancel()
            return

        screen = self._screen or QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            self.cancel()
            return

        # Important: grabWindow(0, x, y, w, h) uses global screen coordinates.
        pm = screen.grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height())
        self.hide()
        self.grabbed.emit(rect, pm)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.cancel()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        _ = event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        # Dim entire screen.
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 90))

        sel = self._selection_rect()
        if sel is None:
            painter.end()
            return

        # Clear selected area a bit.
        painter.fillRect(sel, QtGui.QColor(0, 0, 0, 20))

        pen = QtGui.QPen(QtGui.QColor(0, 170, 255, 220), 2)
        painter.setPen(pen)
        painter.drawRect(sel)

        label = f"{sel.width()}×{sel.height()}"
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 160)))

        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(label)
        text_h = fm.height()
        pad = 6
        box = QtCore.QRect(sel.x(), max(0, sel.y() - (text_h + pad * 2 + 4)), text_w + pad * 2, text_h + pad * 2)
        painter.drawRoundedRect(box, 6, 6)
        painter.drawText(
            box.adjusted(pad, pad + fm.ascent() - fm.height(), -pad, -pad),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            label,
        )
        painter.end()


try:
    from fragment_clues import CLUE_TYPE_LABELS, clue_type_label, effective_clue_type, guess_clue_type_from_text
except Exception:
    CLUE_TYPE_LABELS = [
        ("auto", "Авто"),
        ("plate", "Номер"),
        ("heraldry", "Герб"),
        ("flag", "Флаг"),
        ("highway", "Трасса"),
        ("street", "Улица / адрес"),
        ("text", "Текст"),
        ("ignore", "Не анализировать"),
    ]

    def clue_type_label(code: str) -> str:
        for c, label in CLUE_TYPE_LABELS:
            if c == code:
                return label
        return code or "Авто"

    def guess_clue_type_from_text(text: str) -> str:
        return "auto"

    def effective_clue_type(
        clue_type: str,
        ocr_text: str,
        note: str,
        width: int = 0,
        height: int = 0,
    ) -> str:
        return clue_type or "auto"


_CLUE_ITEM_BG = {
    "plate": QtGui.QColor(214, 232, 255),
    "heraldry": QtGui.QColor(255, 244, 210),
    "flag": QtGui.QColor(255, 244, 210),
    "highway": QtGui.QColor(214, 255, 228),
    "street": QtGui.QColor(255, 232, 214),
    "text": QtGui.QColor(240, 240, 240),
    "auto": QtGui.QColor(245, 245, 245),
    "ignore": QtGui.QColor(230, 230, 230),
}

_KIND_LABEL_RU = {
    "city": "Город",
    "house": "Дом",
    "intersection": "Пересечение",
    "intersection_approx": "Пересечение",
    "street_centroid": "Улица",
    "plate_region": "Номер региона",
    "highway": "Трасса",
    "phone_code": "Тел. код",
    "heraldry": "Герб",
    "heraldry_region": "Герб региона",
}


def _pixmap_to_temp_png(pm: QtGui.QPixmap) -> str:
    """PNG во временный файл — тот же путь, что у sign_detect.py --image."""
    import tempfile

    if pm.isNull():
        raise ValueError("Пустое изображение")
    fd, path = tempfile.mkstemp(suffix=".png", prefix="diplom_yolo_")
    os.close(fd)
    if not pm.save(path, "PNG"):
        try:
            os.unlink(path)
        except OSError:
            pass
        raise RuntimeError("Не удалось сохранить кадр во временный PNG")
    return path


class _YoloWorker(QtCore.QThread):
    finished_ok = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, image_path: str) -> None:
        super().__init__()
        self._image_path = image_path

    def run(self) -> None:
        try:
            dets = detect_signs(self._image_path)
            self.finished_ok.emit(dets)
        except Exception as e:
            self.failed.emit(repr(e))
        finally:
            try:
                os.unlink(self._image_path)
            except OSError:
                pass


def _pixmap_to_b64_jpeg(pm: QtGui.QPixmap, max_side: int = 384, quality: int = 82) -> str:
    if pm.isNull():
        return ""
    img = pm.toImage()
    w, h = img.width(), img.height()
    if w <= 0 or h <= 0:
        return ""
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.scaled(
            int(w * scale),
            int(h * scale),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    if not img.save(buf, "JPEG", quality=quality):
        return ""
    return base64.b64encode(bytes(buf.data())).decode("ascii")


def _city_and_kind_from_topk_name(name: str, kind: str) -> tuple[str, str]:
    """Короткая подпись для кнопки: город и тип результата."""
    n = (name or "").strip()
    city = n
    if ":" in n:
        city = n.split(":", 1)[0].strip()
    elif "," in n:
        city = n.split(",", 1)[0].strip()
    kind_ru = _KIND_LABEL_RU.get(kind or "", kind or "")
    return city, kind_ru


def _format_topk_score(raw: Any) -> str:
    """
    Подпись к score из API: для значений 0…1 — как «уверенность» в процентах,
    иначе произвольная шкала (например, ранжирование городов > 1).
    """
    try:
        s = float(raw)
    except (TypeError, ValueError):
        return ""
    if 0.0 <= s <= 1.0:
        pct = s * 100.0
        if pct >= 10.0:
            return f"{pct:.0f}%"
        return f"{pct:.1f}%"
    return f"{s:.2f}"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ui_mode: Literal["full", "user"] = "full") -> None:
        super().__init__()
        self._ui_mode = ui_mode
        self.setWindowTitle("DIPLOM Overlay MVP" if ui_mode == "full" else "DIPLOM — упрощённый режим")
        self.resize(980, 640)

        self._overlay = ScreenGrabOverlay()
        self._overlay.grabbed.connect(self._on_grabbed)
        self._overlay.cancelled.connect(self._on_overlay_cancelled)

        self._fragments: list[Fragment] = []
        self._last_best: Optional[dict] = None
        self._last_topk: List[Dict[str, Any]] = []
        self._yolo_busy = False

        self._build_ui()
        self._bind_shortcuts()
        self._show_ocr_startup_hint()
        if _HAS_YOLO and yolo_model_ready():
            QtCore.QTimer.singleShot(300, self._warmup_yolo)

    def _round_icon_button(
        self,
        icon: QtGui.QIcon,
        tooltip: str,
        *,
        diameter: int = 40,
    ) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setIcon(icon)
        b.setIconSize(QtCore.QSize(diameter - 14, diameter - 14))
        b.setFixedSize(diameter, diameter)
        b.setToolTip(tooltip)
        b.setStyleSheet(
            f"""
            QToolButton {{
                border-radius: {diameter // 2}px;
                border: 1px solid #9aa0a6;
                background: #f3f4f6;
            }}
            QToolButton:hover {{ background: #e5e7eb; }}
            QToolButton:pressed {{ background: #d1d5db; }}
            """
        )
        return b

    def _build_ui(self) -> None:
        if self._ui_mode == "user":
            self._build_ui_user()
            return

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)

        main = QtWidgets.QHBoxLayout(root)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(12)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)
        main.addLayout(left, 3)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)
        main.addLayout(right, 2)

        # Controls
        controls = QtWidgets.QGroupBox("Захват")
        controls_l = QtWidgets.QVBoxLayout(controls)

        self.btn_capture = QtWidgets.QPushButton("Выделить фрагмент (Ctrl+Shift+S)")
        self.btn_capture.clicked.connect(self.start_capture)
        controls_l.addWidget(self.btn_capture)

        self.btn_clear = QtWidgets.QPushButton("Очистить всё")
        self.btn_clear.clicked.connect(self.clear_all)
        controls_l.addWidget(self.btn_clear)

        hint = QtWidgets.QLabel("В режиме выделения, Esc чтобы отменить.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        controls_l.addWidget(hint)

        left.addWidget(controls)

        # Manual note
        manual = QtWidgets.QGroupBox("Ручная подсказка")
        manual_l = QtWidgets.QVBoxLayout(manual)
        self.edit_manual = QtWidgets.QPlainTextEdit()
        self.edit_manual.setPlaceholderText("Например: «Новосибирск, Академгородок»")
        self.edit_manual.setTabChangesFocus(True)
        manual_l.addWidget(self.edit_manual)
        left.addWidget(manual, 1)

        # Export / send
        actions = QtWidgets.QGroupBox("Запрос")
        actions_l = QtWidgets.QGridLayout(actions)
        actions_l.setHorizontalSpacing(8)
        actions_l.setVerticalSpacing(8)

        self.btn_export = QtWidgets.QPushButton("Экспорт JSON")
        self.btn_export.clicked.connect(self.export_json)
        actions_l.addWidget(self.btn_export, 0, 0)

        self.btn_copy = QtWidgets.QPushButton("Копировать JSON")
        self.btn_copy.clicked.connect(self.copy_json)
        actions_l.addWidget(self.btn_copy, 0, 1)

        self.edit_endpoint = QtWidgets.QLineEdit()
        self.edit_endpoint.setPlaceholderText("Endpoint (например http://127.0.0.1:8000/locate)")
        self.edit_endpoint.setText("http://127.0.0.1:8000/locate")
        actions_l.addWidget(self.edit_endpoint, 1, 0, 1, 2)

        self.btn_send = QtWidgets.QPushButton("Отправить")
        self.btn_send.clicked.connect(self.send_payload)
        actions_l.addWidget(self.btn_send, 2, 0)

        self.btn_dry_run = QtWidgets.QPushButton("Проверить (без сети)")
        self.btn_dry_run.clicked.connect(self.dry_run_payload)
        actions_l.addWidget(self.btn_dry_run, 2, 1)

        self.btn_map = QtWidgets.QPushButton("Открыть на карте")
        self.btn_map.clicked.connect(self.open_map)
        self.btn_map.setEnabled(False)
        actions_l.addWidget(self.btn_map, 3, 0, 1, 2)

        left.addWidget(actions)

        # Output
        out = QtWidgets.QGroupBox("Лог / вывод")
        out_l = QtWidgets.QVBoxLayout(out)
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        out_l.addWidget(self.out)
        left.addWidget(out, 1)

        # Fragments list
        fragments_box = QtWidgets.QGroupBox("Фрагменты")
        fragments_l = QtWidgets.QVBoxLayout(fragments_box)

        self.list = QtWidgets.QListWidget()
        self.list.setIconSize(QtCore.QSize(220, 140))
        self.list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list.setMovement(QtWidgets.QListView.Movement.Static)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list.currentRowChanged.connect(self._on_select_fragment)
        fragments_l.addWidget(self.list, 1)

        buttons_row = QtWidgets.QHBoxLayout()
        self.btn_delete = QtWidgets.QPushButton("Удалить выбранный")
        self.btn_delete.clicked.connect(self.delete_selected)
        buttons_row.addWidget(self.btn_delete)

        self.btn_edit_note = QtWidgets.QPushButton("Заметка к фрагменту…")
        self.btn_edit_note.clicked.connect(self.edit_fragment_note)
        buttons_row.addWidget(self.btn_edit_note)

        self.btn_ocr = QtWidgets.QPushButton("OCR выбранного")
        self.btn_ocr.clicked.connect(self.ocr_selected)
        buttons_row.addWidget(self.btn_ocr)

        self.btn_detect_signs = QtWidgets.QPushButton("Найти знаки (YOLO)")
        self.btn_detect_signs.setToolTip(
            "На выбранном кадре панорамы найти щиты и таблички, нарезать на фрагменты + OCR"
        )
        self.btn_detect_signs.clicked.connect(self.detect_signs_selected)
        buttons_row.addWidget(self.btn_detect_signs)

        fragments_l.addLayout(buttons_row)
        right.addWidget(fragments_box, 3)

        frag_details = QtWidgets.QGroupBox("Детали выбранного")
        frag_details_l = QtWidgets.QFormLayout(frag_details)
        self.lbl_rect = QtWidgets.QLabel("—")
        self.lbl_created = QtWidgets.QLabel("—")
        self.txt_ocr = QtWidgets.QPlainTextEdit()
        self.txt_ocr.setPlaceholderText("OCR-текст (можно редактировать вручную)")
        self.txt_ocr.textChanged.connect(self._on_ocr_edited)
        self.combo_clue_type = QtWidgets.QComboBox()
        for code, label in CLUE_TYPE_LABELS:
            self.combo_clue_type.addItem(label, code)
        self.combo_clue_type.currentIndexChanged.connect(self._on_clue_type_changed)
        frag_details_l.addRow("Rect:", self.lbl_rect)
        frag_details_l.addRow("Создан:", self.lbl_created)
        frag_details_l.addRow("Тип следа:", self.combo_clue_type)
        frag_details_l.addRow("OCR:", self.txt_ocr)
        right.addWidget(frag_details, 2)

        self.lbl_send_status = QtWidgets.QLabel("")
        self.lbl_send_status.setStyleSheet("color: #666; font-size: 11px;")
        left.addWidget(self.lbl_send_status)

    def _build_ui_user(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        style = self.style()
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)

        self.btn_capture = self._round_icon_button(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DesktopIcon),
            "Выделить фрагмент (Ctrl+Shift+S)",
        )
        self.btn_capture.clicked.connect(self.start_capture)
        top.addWidget(self.btn_capture)

        self.btn_clear = self._round_icon_button(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon),
            "Очистить всё",
        )
        self.btn_clear.clicked.connect(self.clear_all)
        top.addWidget(self.btn_clear)

        top.addStretch(1)

        self.edit_endpoint = QtWidgets.QLineEdit()
        self.edit_endpoint.setPlaceholderText("http://127.0.0.1:8000/locate")
        self.edit_endpoint.setText("http://127.0.0.1:8000/locate")
        top.addWidget(self.edit_endpoint, 1)

        self.btn_send = self._round_icon_button(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowForward),
            "Отправить на сервер",
        )
        self.btn_send.clicked.connect(self.send_payload)
        top.addWidget(self.btn_send)

        self.lbl_send_status = QtWidgets.QLabel("")
        self.lbl_send_status.setStyleSheet("color: #666; font-size: 11px;")
        self.lbl_send_status.setMinimumWidth(180)
        top.addWidget(self.lbl_send_status, 1)
        outer.addLayout(top)

        res_label = QtWidgets.QLabel("Варианты по убыванию оценки (нажатие — карта)")
        res_label.setStyleSheet("font-weight: 600;")
        outer.addWidget(res_label)

        self._results_host = QtWidgets.QWidget()
        self.results_layout = QtWidgets.QVBoxLayout(self._results_host)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(6)
        self.results_layout.addStretch(1)

        res_scroll = QtWidgets.QScrollArea()
        res_scroll.setWidgetResizable(True)
        res_scroll.setWidget(self._results_host)
        res_scroll.setMinimumHeight(100)
        res_scroll.setMaximumHeight(220)
        outer.addWidget(res_scroll)

        manual = QtWidgets.QGroupBox("Текст (подсказка и распознанное)")
        manual_l = QtWidgets.QVBoxLayout(manual)
        self.edit_manual = QtWidgets.QPlainTextEdit()
        self.edit_manual.setPlaceholderText(
            "Текст можно ввести вручную"
            "Или 2–3 фрагмента (17 / улица / название). OCR только подсказка."
        )
        self.edit_manual.setTabChangesFocus(True)
        manual_l.addWidget(self.edit_manual)
        outer.addWidget(manual, 1)

        split = QtWidgets.QHBoxLayout()
        split.setSpacing(10)

        fragments_box = QtWidgets.QGroupBox("Фрагменты экрана")
        fragments_l = QtWidgets.QVBoxLayout(fragments_box)
        self.list = QtWidgets.QListWidget()
        self.list.setIconSize(QtCore.QSize(200, 120))
        self.list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list.setMovement(QtWidgets.QListView.Movement.Static)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list.currentRowChanged.connect(self._on_select_fragment)
        fragments_l.addWidget(self.list, 1)

        frag_tb = QtWidgets.QHBoxLayout()
        self.btn_delete = self._round_icon_button(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogCancelButton),
            "Удалить выбранный",
            diameter=36,
        )
        self.btn_delete.clicked.connect(self.delete_selected)
        frag_tb.addWidget(self.btn_delete)

        self.btn_edit_note = self._round_icon_button(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView),
            "Заметка к фрагменту",
            diameter=36,
        )
        self.btn_edit_note.clicked.connect(self.edit_fragment_note)
        frag_tb.addWidget(self.btn_edit_note)

        self.btn_ocr = self._round_icon_button(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogInfoView),
            "OCR выбранного фрагмента",
            diameter=36,
        )
        self.btn_ocr.clicked.connect(self.ocr_selected)
        frag_tb.addWidget(self.btn_ocr)

        self.btn_detect_signs = self._round_icon_button(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_CommandLink),
            "Найти знаки на кадре (YOLO - фрагменты + OCR)",
            diameter=36,
        )
        self.btn_detect_signs.clicked.connect(self.detect_signs_selected)
        frag_tb.addStretch(1)
        frag_tb.addWidget(self.btn_detect_signs)
        fragments_l.addLayout(frag_tb)
        split.addWidget(fragments_box, 3)

        ocr_box = QtWidgets.QGroupBox("Текст выбранного фрагмента")
        ocr_l = QtWidgets.QVBoxLayout(ocr_box)
        clue_row = QtWidgets.QHBoxLayout()
        clue_row.addWidget(QtWidgets.QLabel("Тип следа:"))
        self.combo_clue_type = QtWidgets.QComboBox()
        for code, label in CLUE_TYPE_LABELS:
            self.combo_clue_type.addItem(label, code)
        self.combo_clue_type.currentIndexChanged.connect(self._on_clue_type_changed)
        clue_row.addWidget(self.combo_clue_type, 1)
        ocr_l.addLayout(clue_row)
        self.txt_ocr = QtWidgets.QPlainTextEdit()
        self.txt_ocr.setPlaceholderText("OCR (можно править вручную)")
        self.txt_ocr.textChanged.connect(self._on_ocr_edited)
        ocr_l.addWidget(self.txt_ocr)
        split.addWidget(ocr_box, 2)

        outer.addLayout(split, 2)

        hint = QtWidgets.QLabel("Захват: Esc — отмена. Окно скрывается на время выделения.")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        self.lbl_rect = None
        self.lbl_created = None
        self.out = None
        self.btn_export = None
        self.btn_copy = None
        self.btn_dry_run = None
        self.btn_map = None

    def _bind_shortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+S"), self, activated=self.start_capture)
        QtGui.QShortcut(QtGui.QKeySequence("Delete"), self, activated=self.delete_selected)

    def log(self, msg: str) -> None:
        if self._ui_mode == "full" and self.out is not None:
            self.out.appendPlainText(msg)

    def _on_overlay_cancelled(self) -> None:
        if self._ui_mode == "user":
            self.show()
            self.raise_()
            self.activateWindow()

    def start_capture(self) -> None:
        if self._ui_mode == "user":
            self.hide()
            QtCore.QTimer.singleShot(120, self._overlay.start)
        else:
            self.log("Режим выделения (Esc — отмена).")
            self._overlay.start()

    def _on_grabbed(self, rect: QtCore.QRect, pixmap: QtGui.QPixmap) -> None:
        if self._ui_mode == "user":
            self.show()
            self.raise_()
            self.activateWindow()
        frag = Fragment(
            created_at_ms=int(time.time() * 1000),
            rect=rect,
            pixmap=pixmap,
        )
        self._fragments.append(frag)
        self._add_fragment_item(frag)
        self.log(f"Добавлен фрагмент: {rect.x()},{rect.y()} {rect.width()}×{rect.height()}")

    def _fragment_list_title(self, frag: Fragment, index: int) -> str:
        type_lbl = clue_type_label(frag.clue_type)
        base = f"{index}. {frag.rect.width()}×{frag.rect.height()} · {type_lbl}"
        if frag.detected_label:
            kind_lbl = _KIND_LABEL_RU.get(frag.detected_kind, frag.detected_kind)
            if kind_lbl:
                return f"{base} → {frag.detected_label} [{kind_lbl}]"
            return f"{base} → {frag.detected_label}"
        return base

    def _refresh_fragment_item(self, index: int) -> None:
        if index < 0 or index >= len(self._fragments) or index >= self.list.count():
            return
        frag = self._fragments[index]
        item = self.list.item(index)
        item.setText(self._fragment_list_title(frag, index + 1))
        bg = _CLUE_ITEM_BG.get(frag.clue_type) or _CLUE_ITEM_BG.get("auto")
        if frag.detected_kind == "plate_region":
            bg = _CLUE_ITEM_BG.get("plate", bg)
        elif frag.detected_kind in ("heraldry", "heraldry_region"):
            bg = _CLUE_ITEM_BG.get("heraldry", bg)
        if bg is not None:
            item.setBackground(bg)

    def _add_fragment_item(self, frag: Fragment) -> None:
        item = QtWidgets.QListWidgetItem()
        idx = len(self._fragments)
        item.setText(self._fragment_list_title(frag, idx))

        thumb = frag.pixmap
        if thumb.width() > 500:
            thumb = thumb.scaledToWidth(500, QtCore.Qt.TransformationMode.SmoothTransformation)
        icon = QtGui.QIcon(thumb)
        item.setIcon(icon)

        item.setData(QtCore.Qt.ItemDataRole.UserRole, frag.id)
        bg = _CLUE_ITEM_BG.get(frag.clue_type, _CLUE_ITEM_BG["auto"])
        item.setBackground(bg)
        self.list.addItem(item)
        self.list.setCurrentItem(item)

    def _selected_index(self) -> int:
        return self.list.currentRow()

    def _selected_fragment(self) -> Optional[Fragment]:
        idx = self._selected_index()
        if idx < 0 or idx >= len(self._fragments):
            return None
        return self._fragments[idx]

    def _on_select_fragment(self, row: int) -> None:
        if row < 0 or row >= len(self._fragments):
            if self.lbl_rect is not None:
                self.lbl_rect.setText("—")
            if self.lbl_created is not None:
                self.lbl_created.setText("—")
            self.txt_ocr.blockSignals(True)
            self.txt_ocr.setPlainText("")
            self.txt_ocr.blockSignals(False)
            return

        frag = self._fragments[row]
        if self.lbl_rect is not None:
            self.lbl_rect.setText(f"{frag.rect.x()},{frag.rect.y()} {frag.rect.width()}×{frag.rect.height()}")
        if self.lbl_created is not None:
            self.lbl_created.setText(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(frag.created_at_ms / 1000)))
        self.txt_ocr.blockSignals(True)
        self.txt_ocr.setPlainText(frag.ocr_text)
        self.txt_ocr.blockSignals(False)
        if hasattr(self, "combo_clue_type"):
            self.combo_clue_type.blockSignals(True)
            ci = self.combo_clue_type.findData(frag.clue_type)
            self.combo_clue_type.setCurrentIndex(ci if ci >= 0 else 0)
            self.combo_clue_type.blockSignals(False)

    def _on_ocr_edited(self) -> None:
        frag = self._selected_fragment()
        if frag is None:
            return
        frag.ocr_text = self.txt_ocr.toPlainText()
        if frag.clue_type == "auto":
            guessed = guess_clue_type_from_text(frag.ocr_text)
            if guessed != "auto":
                frag.clue_type = guessed
                if hasattr(self, "combo_clue_type"):
                    self.combo_clue_type.blockSignals(True)
                    ci = self.combo_clue_type.findData(guessed)
                    if ci >= 0:
                        self.combo_clue_type.setCurrentIndex(ci)
                    self.combo_clue_type.blockSignals(False)
        row = self._selected_index()
        if row >= 0:
            self._refresh_fragment_item(row)

    def _on_clue_type_changed(self, _index: int) -> None:
        frag = self._selected_fragment()
        if frag is None or not hasattr(self, "combo_clue_type"):
            return
        code = self.combo_clue_type.currentData()
        if not code:
            return
        frag.clue_type = str(code)
        frag.detected_kind = ""
        frag.detected_label = ""
        row = self._selected_index()
        if row >= 0:
            self._refresh_fragment_item(row)

    def _warmup_yolo(self) -> None:
        if not _HAS_YOLO or not yolo_model_ready():
            return
        try:
            self._set_send_status("Загрузка YOLO…", ok=None)
            QtWidgets.QApplication.processEvents()
            yolo_warmup_model()
        except Exception as e:
            if self._ui_mode == "full":
                self.log(f"YOLO warmup: {e!r}")
        finally:
            self._show_ocr_startup_hint()

    def _show_ocr_startup_hint(self) -> None:
        parts: List[str] = []
        if _HAS_YOLO and yolo_model_ready():
            parts.append("YOLO готов")
        elif _HAS_YOLO:
            parts.append("YOLO: нет весов")
        msg = easyocr_status_message()
        if easyocr_available() and (not msg or msg == "EasyOCR готов."):
            parts.append("EasyOCR готов")
        elif run_ocr_on_pil is not None:
            parts.append("OCR: Tesseract")
        if parts:
            self._set_send_status(" · ".join(parts), ok=yolo_model_ready() if _HAS_YOLO else None)
        if msg and msg != "EasyOCR готов." and self._ui_mode == "full":
            self.log(msg)
        if _HAS_YOLO and not yolo_model_ready() and self._ui_mode == "full":
            self.log(yolo_model_status())

    def _payload_has_text_clues(self, payload: Dict[str, Any]) -> bool:
        manual = (payload.get("manual_text") or "").strip()
        if manual:
            return True
        for frag in payload.get("fragments") or []:
            if not isinstance(frag, dict):
                continue
            for key in ("ocr_text", "note", "text"):
                if (frag.get(key) or "").strip():
                    return True
        return False

    def _set_send_status(self, text: str, ok: Optional[bool] = None) -> None:
        if not hasattr(self, "lbl_send_status"):
            return
        self.lbl_send_status.setText(text)
        if ok is True:
            self.lbl_send_status.setStyleSheet("color: #1a7f37; font-size: 11px; font-weight: 600;")
        elif ok is False:
            self.lbl_send_status.setStyleSheet("color: #b42318; font-size: 11px; font-weight: 600;")
        else:
            self.lbl_send_status.setStyleSheet("color: #666; font-size: 11px;")

    def _apply_fragment_analysis(self, data: Dict[str, Any]) -> None:
        analysis = data.get("fragment_analysis")
        if not isinstance(analysis, list):
            ev = data.get("evidence")
            if isinstance(ev, dict):
                analysis = ev.get("fragment_analysis")
        if not isinstance(analysis, list):
            return
        by_id = {f.id: f for f in self._fragments}
        for row in analysis:
            if not isinstance(row, dict):
                continue
            fid = str(row.get("id") or "")
            frag = by_id.get(fid)
            if frag is None:
                continue
            frag.detected_kind = str(row.get("detected_kind") or "")
            frag.detected_label = str(row.get("detected_label") or "")
            eff = str(row.get("clue_type") or "")
            if eff and frag.clue_type == "auto":
                frag.clue_type = eff
        for i in range(len(self._fragments)):
            self._refresh_fragment_item(i)
        row = self._selected_index()
        if row >= 0:
            self._on_select_fragment(row)

    def edit_fragment_note(self) -> None:
        frag = self._selected_fragment()
        if frag is None:
            return

        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self,
            "Заметка к фрагменту",
            "Например: «вывеска магазина», «номер трассы», «флаг региона»",
            frag.note,
        )
        if not ok:
            return
        frag.note = text.strip()
        self.log("Заметка обновлена.")

    def _clear_result_buttons(self) -> None:
        if self._ui_mode != "user" or not hasattr(self, "results_layout"):
            return
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.results_layout.addStretch(1)

    def _populate_result_buttons(self, data: Dict[str, Any]) -> None:
        self._clear_result_buttons()
        topk = data.get("topk") if isinstance(data.get("topk"), list) else []
        self._last_topk = [x for x in topk if isinstance(x, dict)]
        self._last_best = data.get("best") if isinstance(data.get("best"), dict) else None

        if not self._last_topk and self._last_best:
            self._last_topk = [self._last_best]

        if not self._last_topk:
            lbl = QtWidgets.QLabel("Нет вариантов (пустой topk / нет координат).")
            lbl.setStyleSheet("color: #666; padding: 6px;")
            lbl.setWordWrap(True)
            self.results_layout.insertWidget(self.results_layout.count() - 1, lbl)
            return

        for rank, item in enumerate(self._last_topk, start=1):
            name = str(item.get("name") or "")
            kind = str(item.get("kind") or "")
            city, kind_ru = _city_and_kind_from_topk_name(name, kind)
            score_s = _format_topk_score(item.get("score"))
            if score_s:
                label = f"{rank}. {city}. [{kind_ru}]  ·  {score_s}"
            else:
                label = f"{rank}. {city}. [{kind_ru}]"
            btn = QtWidgets.QPushButton(label)
            raw_score = item.get("score")
            tip_lines = [name, f"score (API): {raw_score!r}"]
            if len(self._last_topk) > 1:
                tip_lines.append("Порядок: от более к менее подходящему.")
            btn.setToolTip("\n".join(tip_lines))
            btn.setStyleSheet(
                "QPushButton { text-align: left; padding: 8px 10px; }"
            )
            try:
                lat = float(item.get("lat"))
                lon = float(item.get("lon"))
            except (TypeError, ValueError):
                btn.setEnabled(False)
                btn.setToolTip((name or "") + "\n(нет координат)")
            else:
                btn.clicked.connect(
                    lambda _checked=False, la=lat, lo=lon: self._open_map_coords(la, lo)
                )
            self.results_layout.insertWidget(self.results_layout.count() - 1, btn)

    def _open_map_coords(self, lat: float, lon: float) -> None:
        url = f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=17/{lat:.6f}/{lon:.6f}"
        webbrowser.open(url)

    def delete_selected(self) -> None:
        idx = self._selected_index()
        if idx < 0 or idx >= len(self._fragments):
            return
        frag = self._fragments.pop(idx)
        self.list.takeItem(idx)
        self.log(f"Удалён фрагмент {frag.id}.")
        if self._fragments:
            self.list.setCurrentRow(min(idx, len(self._fragments) - 1))

    def clear_all(self) -> None:
        self._fragments.clear()
        self.list.clear()
        self.log("Все фрагменты удалены.")
        if self._ui_mode == "user":
            self._last_best = None
            self._last_topk = []
            self._clear_result_buttons()

    def _combined_text_for_server(self) -> str:
        """Ручной текст + OCR всех фрагментов (основной вход для геокодера)."""
        parts: List[str] = []
        manual = self.edit_manual.toPlainText().strip()
        if manual:
            parts.append(manual)
        for frag in self._fragments:
            t = (frag.ocr_text or "").strip()
            if t and t not in parts:
                parts.append(t)
        return "\n".join(parts)

    def _fragment_image_b64(self, frag: Fragment) -> str:
        """Картинка на сервер — только для гербов; иначе лишний ResNet и таймаут."""
        ct = (frag.clue_type or "auto").strip().lower()
        if ct in ("street", "highway", "text", "plate", "ignore"):
            return ""
        if ct in ("heraldry", "flag"):
            return _pixmap_to_b64_jpeg(frag.pixmap)
        if ct == "auto":
            for other in self._fragments:
                if other is frag:
                    continue
                oct_ = (other.clue_type or "auto").strip().lower()
                if oct_ in ("street", "highway", "text") and (other.ocr_text or "").strip():
                    return ""
        return _pixmap_to_b64_jpeg(frag.pixmap)

    def build_payload(self) -> dict:
        fragments_payload = []
        for frag in self._fragments:
            fragments_payload.append(
                {
                    "id": frag.id,
                    "created_at_ms": frag.created_at_ms,
                    "rect": {"x": frag.rect.x(), "y": frag.rect.y(), "w": frag.rect.width(), "h": frag.rect.height()},
                    "note": frag.note,
                    "ocr_text": frag.ocr_text,
                    "image_b64": self._fragment_image_b64(frag),
                    "clue_type": frag.clue_type or "auto",
                }
            )

        payload = {
            "schema": "diplom.overlay.v1",
            "created_at_ms": int(time.time() * 1000),
            "manual_text": self._combined_text_for_server(),
            "mode": "geoguess",
            "fragments": fragments_payload,
        }
        return payload

    def payload_json(self) -> str:
        return json.dumps(self.build_payload(), ensure_ascii=False, indent=2)

    def export_json(self) -> None:
        if self._ui_mode == "user":
            return
        data = self.payload_json()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить JSON", "overlay_request.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
        self.log(f"JSON сохранён: {path}")

    def copy_json(self) -> None:
        if self._ui_mode == "user":
            return
        data = self.payload_json()
        QtGui.QGuiApplication.clipboard().setText(data)
        self.log("JSON скопирован в буфер обмена.")

    def dry_run_payload(self) -> None:
        if self._ui_mode == "user":
            return
        payload = self.build_payload()
        fragments = payload.get("fragments", [])
        manual = payload.get("manual_text", "")
        ocr_chars = sum(len(f.get("ocr_text", "") or "") for f in fragments)
        self.log(
            f"Проверка: fragments={len(fragments)}, manual_len={len(manual)}, ocr_total_len={ocr_chars}. "
            f"Схема: {payload.get('schema')}"
        )

    def _prepare_fragments_for_send(self) -> None:
        for i, frag in enumerate(self._fragments):
            if frag.clue_type == "auto":
                eff = effective_clue_type(
                    "auto",
                    frag.ocr_text,
                    frag.note,
                    frag.rect.width(),
                    frag.rect.height(),
                )
                if eff != "auto":
                    frag.clue_type = eff
            if not (frag.ocr_text or "").strip():
                ct = (frag.clue_type or "auto").strip().lower()
                # Не гонять EasyOCR по большой панораме «Авто» при отправке — только типизированные следы.
                if ct in ("street", "text", "highway", "plate", "heraldry", "flag"):
                    self._ocr_fragment(frag, show_errors=False)
            if frag.clue_type == "auto":
                blob = "\n".join(p for p in (frag.note, frag.ocr_text) if p)
                guessed = guess_clue_type_from_text(blob)
                if guessed != "auto":
                    frag.clue_type = guessed
            self._refresh_fragment_item(i)

    def send_payload(self) -> None:
        if requests is None:
            QtWidgets.QMessageBox.warning(
                self,
                "requests не установлен",
                "Чтобы отправлять на сервер, установить requests:\n\npython -m pip install requests",
            )
            return

        url = self.edit_endpoint.text().strip()
        if not url:
            QtWidgets.QMessageBox.warning(self, "Endpoint пустой", "Указать URL эндпоинта, например http://127.0.0.1:8000/locate")
            return

        self.btn_send.setEnabled(False)
        self._set_send_status("OCR фрагментов…")
        QtWidgets.QApplication.processEvents()
        self._prepare_fragments_for_send()
        payload = self.build_payload()
        if not self._payload_has_text_clues(payload):
            self._set_send_status("Нет текста для поиска — ввести вручную", ok=False)
            self.btn_send.setEnabled(True)
            return

        self._set_send_status("Отправка на сервер…")
        QtWidgets.QApplication.processEvents()
        self.log(f"POST {url} …")
        try:
            r = requests.post(url, json=payload, timeout=120)
            self.log(f"Ответ: HTTP {r.status_code}")
            ct = (r.headers.get("content-type") or "").lower()
            if "application/json" in ct:
                data = r.json()
                if isinstance(data, dict):
                    self._last_best = data.get("best") if isinstance(data.get("best"), dict) else None
                    self._apply_fragment_analysis(data)
                    n = len(data.get("topk") or [])
                    if r.status_code == 200:
                        if n:
                            self._set_send_status(f"Ответ получен · {n} вариант(ов)", ok=True)
                        else:
                            hint = ""
                            if not self._payload_has_text_clues(payload):
                                hint = " — добавить текст вручную"
                            self._set_send_status(f"Ответ получен · нет вариантов{hint}", ok=True)
                    else:
                        self._set_send_status(f"HTTP {r.status_code}", ok=False)
                    if self._ui_mode == "full" and self.btn_map is not None:
                        self.btn_map.setEnabled(
                            bool(
                                self._last_best
                                and self._last_best.get("lat") is not None
                                and self._last_best.get("lon") is not None
                            )
                        )
                    if self._ui_mode == "user":
                        self._populate_result_buttons(data)
                    else:
                        self.log(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    if self._ui_mode != "user":
                        self.log(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                self.log((r.text or "").strip()[:2000])
                self._set_send_status("Сервер вернул не JSON", ok=False)
                if self._ui_mode == "user":
                    self._clear_result_buttons()
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Ответ сервера",
                        "Сервер вернул не JSON. Проверить URL и что запущен uvicorn.",
                    )
        except Exception as e:
            self.log(f"Ошибка отправки: {e!r}")
            self._set_send_status(f"Ошибка: {e!r}", ok=False)
            self._last_best = None
            self._last_topk = []
            if self._ui_mode == "full" and self.btn_map is not None:
                self.btn_map.setEnabled(False)
            if self._ui_mode == "user":
                self._clear_result_buttons()
                QtWidgets.QMessageBox.critical(self, "Ошибка сети", f"{e!r}")
        finally:
            self.btn_send.setEnabled(True)

    def open_map(self) -> None:
        if self._ui_mode == "user":
            best = self._last_best or {}
            try:
                lat = float(best.get("lat"))
                lon = float(best.get("lon"))
            except Exception:
                QtWidgets.QMessageBox.information(
                    self,
                    "Нет координат",
                    "Сначала отправить запрос — выбрать вариант на карте или дождаться ответа.",
                )
                return
            self._open_map_coords(lat, lon)
            return

        best = self._last_best or {}
        try:
            lat = float(best.get("lat"))
            lon = float(best.get("lon"))
        except Exception:
            QtWidgets.QMessageBox.information(self, "Нет координат", "Сначала получить ответ сервера с координатами (best.lat/best.lon).")
            return

        url = f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=17/{lat:.6f}/{lon:.6f}"
        webbrowser.open(url)

    def _ocr_fragment(self, frag: Fragment, show_errors: bool = True) -> bool:
        if pytesseract is None:
            if show_errors:
                QtWidgets.QMessageBox.warning(
                    self,
                    "OCR недоступен",
                    "Не найден модуль pytesseract.\n\n"
                    "Поставить:\n  python -m pip install pytesseract pillow\n\n"
                    "И установить сам Tesseract OCR (Windows).",
                )
            return False

        tcmd = os.environ.get("TESSERACT_CMD", "").strip()
        if not tcmd:
            tcmd = shutil.which("tesseract") or ""
        if not tcmd:
            candidates = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            ]
            for c in candidates:
                if c and os.path.isfile(c):
                    tcmd = c
                    break
        if tcmd:
            pytesseract.pytesseract.tesseract_cmd = tcmd

        if run_ocr_on_pil is None:
            if show_errors:
                QtWidgets.QMessageBox.warning(
                    self,
                    "ocr_pipeline",
                    "Не найден модуль ocr_pipeline.py рядом с gui.py.\n"
                    "Скопировать файл из репозитория или обновить проект.",
                )
            return False

        try:
            ba = QtCore.QByteArray()
            buf = QtCore.QBuffer(ba)
            buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
            ok = frag.pixmap.save(buf, "PNG")
            buf.close()
            if not ok:
                raise RuntimeError("Не удалось сериализовать фрагмент в PNG.")

            import io
            import PIL.Image  # type: ignore

            pil = PIL.Image.open(io.BytesIO(bytes(ba))).convert("RGB")
            if (frag.clue_type or "auto") == "auto":
                eff = effective_clue_type(
                    "auto",
                    frag.ocr_text,
                    frag.note,
                    frag.rect.width(),
                    frag.rect.height(),
                )
                if eff != "auto":
                    frag.clue_type = eff
            ct = (frag.clue_type or "auto").strip().lower()
            if ct == "plate":
                ocr_lang = (os.environ.get("DIPLOM_OCR_LANG", "rus+eng") or "rus+eng").strip()
                hint = "plate"
            elif ct in ("street", "text", "highway"):
                ocr_lang = (os.environ.get("DIPLOM_OCR_STREET_LANG", "rus") or "rus").strip()
                hint = "highway" if ct == "highway" else "street"
            else:
                ocr_lang = (os.environ.get("DIPLOM_OCR_LANG", "rus+eng") or "rus+eng").strip()
                hint = None
            work_pil = pil
            if ct in ("street", "text", "auto"):
                try:
                    from ocr_pipeline import _crop_sign_panel  # type: ignore

                    work_pil = _crop_sign_panel(pil)
                except Exception:
                    work_pil = pil

            text = run_ocr_on_pil(work_pil, lang=ocr_lang, hint=hint)
            text = (text or "").strip()

            # Слабый результат (mS, «и») — повтор через street pipeline.
            if hint != "plate" and _score_text(text) < 25:
                street_lang = (os.environ.get("DIPLOM_OCR_STREET_LANG", "rus") or "rus").strip()
                retry = run_ocr_on_pil(work_pil, lang=street_lang, hint="street")
                retry = (retry or "").strip()
                if _score_text(retry) > _score_text(text):
                    text = retry
                    if ct == "auto":
                        frag.clue_type = "street"

            if ct in ("street", "text") and not text and show_errors:
                extra = ""
                if not easyocr_available():
                    extra = (
                        "\n\nДля лучшего OCR на табличках:\n"
                        "  python download_easyocr_models.py\n"
                        "(один раз в терминале, затем перезапуск GUI)"
                    )
                QtWidgets.QMessageBox.information(
                    self,
                    "OCR не уверен",
                    "Не удалось надёжно распознать табличку.\n\n"
                    "Tesseract плохо работает на тенях и фоне (кирпич).\n"
                    + extra,
                )
        except Exception as e:
            if show_errors:
                QtWidgets.QMessageBox.critical(self, "OCR ошибка", f"Не удалось распознать текст:\n\n{e!r}")
            return False

        frag.ocr_text = text
        if frag.clue_type == "auto":
            guessed = guess_clue_type_from_text(text)
            if guessed != "auto":
                frag.clue_type = guessed
        row = self._selected_index()
        if row >= 0 and self._fragments[row] is frag:
            self.txt_ocr.blockSignals(True)
            self.txt_ocr.setPlainText(text)
            self.txt_ocr.blockSignals(False)
            if hasattr(self, "combo_clue_type"):
                self.combo_clue_type.blockSignals(True)
                ci = self.combo_clue_type.findData(frag.clue_type)
                if ci >= 0:
                    self.combo_clue_type.setCurrentIndex(ci)
                self.combo_clue_type.blockSignals(False)
        for i, f in enumerate(self._fragments):
            if f is frag:
                self._refresh_fragment_item(i)
                break
        engine = last_ocr_engine()
        self.log(f"OCR [{engine}]: {len(text)} символов.")
        return True

    def ocr_selected(self) -> None:
        frag = self._selected_fragment()
        if frag is None:
            return
        self._ocr_fragment(frag, show_errors=True)

    def _set_detect_signs_busy(self, busy: bool) -> None:
        if hasattr(self, "btn_detect_signs"):
            self.btn_detect_signs.setEnabled(not busy)

    def _fragment_from_detection(self, parent: Fragment, det: Any) -> Optional[Fragment]:
        pw, ph = parent.pixmap.width(), parent.pixmap.height()
        if pw < 8 or ph < 8:
            return None
        x1 = max(0, min(int(det.x1), pw - 1))
        y1 = max(0, min(int(det.y1), ph - 1))
        x2 = max(x1 + 1, min(int(det.x2), pw))
        y2 = max(y1 + 1, min(int(det.y2), ph))
        w, h = x2 - x1, y2 - y1
        if w < 12 or h < 12:
            return None
        crop = parent.pixmap.copy(x1, y1, w, h)
        if crop.isNull():
            return None
        clue = clue_type_for_detection(det)
        return Fragment(
            created_at_ms=int(time.time() * 1000),
            rect=QtCore.QRect(parent.rect.x() + x1, parent.rect.y() + y1, w, h),
            pixmap=crop,
            clue_type=clue,
            note=f"YOLO {det.class_name} {det.confidence:.0%}",
        )

    def detect_signs_selected(self) -> None:
        if not _HAS_YOLO:
            QtWidgets.QMessageBox.warning(self, "YOLO", "Модуль sign_detect не найден.")
            return
        if not yolo_model_ready():
            QtWidgets.QMessageBox.warning(
                self,
                "YOLO",
                yolo_model_status() + "\n\nОбучить: python sign_detect\\train_yolo.py",
            )
            return
        parent = self._selected_fragment()
        if parent is None:
            QtWidgets.QMessageBox.information(
                self,
                "Найти знаки",
                "Сначала выделить крупный кадр панорамы и выберите его в списке фрагментов.",
            )
            return
        if self._yolo_busy:
            return
        try:
            tmp_path = _pixmap_to_temp_png(parent.pixmap)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "YOLO", f"Не удалось прочитать кадр:\n{e!r}")
            return

        self._yolo_busy = True
        self._set_detect_signs_busy(True)
        self._set_send_status("YOLO: поиск знаков…", ok=None)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

        worker = _YoloWorker(tmp_path)
        worker.setParent(self)

        def _end_yolo() -> None:
            self._yolo_busy = False
            self._set_detect_signs_busy(False)
            QtWidgets.QApplication.restoreOverrideCursor()

        def _done(dets: object) -> None:
            try:
                items = list(dets) if dets else []
                added = 0
                for det in items:
                    child = self._fragment_from_detection(parent, det)
                    if child is None:
                        continue
                    self._fragments.append(child)
                    self._add_fragment_item(child)
                    added += 1
                if added:
                    self._set_send_status(
                        f"YOLO: {added} фрагмент(ов) — OCR при «Отправить»",
                        ok=True,
                    )
                    self.log(f"YOLO: {added} фрагмент(ов) из кадра {parent.rect.width()}×{parent.rect.height()}.")
                else:
                    self._set_send_status("YOLO: знаки не найдены", ok=None)
                    pw, ph = parent.pixmap.width(), parent.pixmap.height()
                    hint = (
                        "На выбранном кадре знаки не найдены.\n\n"
                        "Выделить **крупный** кусок панорамы (не только щит)\n"
                        "Или вырезать щит вручную - OCR / Отправить\n"
                        f"Размер кадра: {pw}×{ph}"
                    )
                    QtWidgets.QMessageBox.information(self, "Найти знаки", hint)
            finally:
                _end_yolo()

        def _fail(err: str) -> None:
            try:
                self._set_send_status("YOLO: ошибка", ok=False)
                QtWidgets.QMessageBox.critical(self, "YOLO", err)
            finally:
                _end_yolo()

        worker.finished_ok.connect(_done)
        worker.failed.connect(_fail)
        worker.start()


def main() -> int:
    # GUI никогда не качает модели EasyOCR (зависает UI + SSL на Windows).
    os.environ.setdefault("DIPLOM_OCR_ALLOW_DOWNLOAD", "0")

    parser = argparse.ArgumentParser(description="DIPLOM: захват экрана + геокодинг")
    parser.add_argument(
        "--user",
        action="store_true",
        help="Упрощённый интерфейс (иконки, варианты кнопками, без лога)",
    )
    args, qt_argv = parser.parse_known_args()
    # Передать Qt только неизвестные флаги (например -platform on Windows).
    app = QtWidgets.QApplication([sys.argv[0], *qt_argv])
    app.setApplicationName("DIPLOM Overlay MVP")

    w = MainWindow(ui_mode="user" if args.user else "full")
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

