from __future__ import annotations

import logging
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from tkinter import *
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from database.db import Database
from database.repository import AttendanceRepository
from models.entities import Student
from services.camera_service import CameraService
from services.face_engine import FaceEngine
from services.recognition_service import RecognitionService
from services.training_service import TrainingService
from utils.config import (
    CALIBRATION_PATH,
    CAPTURE_MIN_FACE_HEIGHT,
    CAPTURE_MIN_FACE_WIDTH,
    CAPTURE_MIN_SHARPNESS,
    MIN_SAMPLES_PER_STUDENT,
    STABILITY_FRAMES,
)
from utils.csv_export import export_attendance_csv

logger = logging.getLogger(__name__)


class FaceAttendanceUI:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1240x760")
        self.root.minsize(1080, 680)
        self.root.configure(bg="#f5f7fb")

        self.db = Database()
        self.db.initialize()
        self.repository = AttendanceRepository(self.db)
        self.face_engine = FaceEngine()
        self.training_service = TrainingService(self.repository)
        self.recognition_service = RecognitionService(self.repository, self.face_engine)
        self.camera_service = CameraService(camera_index=0)

        self.active_page = StringVar(value="dashboard")
        self._camera_job: str | None = None
        self._camera_photo = None
        self._transition_job: str | None = None
        self._nav_pulse_job: str | None = None
        self._ambient_job: str | None = None
        self._clock_glow_job: str | None = None
        self._counter_jobs: list[str] = []
        self._pulse_tick = 0
        self._ambient_tick = 0
        self._active_nav_palette = ["#2f7de1", "#3487ef", "#3f93ff", "#3487ef"]
        self._frame_counter = 0
        self._cached_faces: list[tuple[int, int, int, int]] = []
        self._cached_predictions: list[tuple[str | None, float, float]] = []
        self._detect_stride = 2
        self._camera_mode = "idle"

        self.setup_styles()
        self.build_layout()
        self.start_ui_animations()
        self.show_page("dashboard")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("App.TFrame", background="#eef3fb")
        style.configure("Sidebar.TFrame", background="#16213a")
        style.configure("Content.TFrame", background="#eef3fb")
        style.configure("Card.TFrame", background="white")
        style.configure("CardTitle.TLabel", font=("Segoe UI Semibold", 11), background="white", foreground="#1e273a")
        style.configure("CardValue.TLabel", font=("Segoe UI Semibold", 20), background="white", foreground="#103362")
        style.configure("Heading.TLabel", font=("Segoe UI Semibold", 20), background="#eef3fb", foreground="#10264a")
        style.configure("Sub.TLabel", font=("Segoe UI", 10), background="#eef3fb", foreground="#53617c")
        style.configure("GlowSub.TLabel", font=("Segoe UI", 10), background="#eef3fb", foreground="#2d4f86")
        style.configure("White.TLabel", font=("Segoe UI", 10), background="#16213a", foreground="#e4ecfa")
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10))
        style.configure("PulseSub.TLabel", font=("Segoe UI Semibold", 10), background="#eef3fb", foreground="#2f7de1")

    def build_layout(self) -> None:
        self.main_frame = ttk.Frame(self.root, style="App.TFrame")
        self.main_frame.pack(fill=BOTH, expand=True)

        self.sidebar = ttk.Frame(self.main_frame, style="Sidebar.TFrame", width=240)
        self.sidebar.pack(side=LEFT, fill=Y)
        self.sidebar.pack_propagate(False)

        self.content = ttk.Frame(self.main_frame, style="Content.TFrame")
        self.content.pack(side=LEFT, fill=BOTH, expand=True)
        self.content.columnconfigure(0, weight=1)

        self.build_sidebar()
        self.build_content_area()

    def build_sidebar(self) -> None:
        Label(
            self.sidebar,
            text="FACE ATTENDANCE",
            font=("Segoe UI Semibold", 15),
            bg="#16213a",
            fg="white",
            padx=14,
            pady=18,
            anchor="w",
            wraplength=205,
        ).pack(fill=X)

        Label(
            self.sidebar,
            text="AI Production Edition",
            font=("Segoe UI", 9),
            bg="#16213a",
            fg="#a8b3c7",
            padx=14,
            pady=2,
            anchor="w",
            wraplength=205,
        ).pack(fill=X)

        self.nav_buttons: dict[str, Button] = {}
        nav_items = [
            ("dashboard", "Dashboard"),
            ("students", "Student Registry"),
            ("capture", "Face Capture"),
            ("live_attendance", "Live Attendance"),
            ("diagnostics", "Diagnostics"),
            ("attendance", "Attendance Log"),
        ]

        nav_wrap = Frame(self.sidebar, bg="#16213a")
        nav_wrap.pack(fill=X, pady=22)

        for key, text in nav_items:
            btn = Button(
                nav_wrap,
                text=text,
                font=("Segoe UI Semibold", 10),
                bg="#1f3157",
                fg="#eef3ff",
                activebackground="#3a4f80",
                activeforeground="white",
                relief=FLAT,
                bd=0,
                padx=12,
                pady=10,
                anchor="w",
                command=lambda k=key: self.show_page(k),
            )
            btn.pack(fill=X, padx=14, pady=5)
            self.nav_buttons[key] = btn

        footer_box = Frame(self.sidebar, bg="#16213a")
        footer_box.pack(side=BOTTOM, fill=X, padx=14, pady=14)
        ttk.Label(footer_box, text="MVC | SQLite | OpenCV", style="White.TLabel").pack(anchor="w")
        ttk.Label(footer_box, text="Realtime Recognition", style="White.TLabel").pack(anchor="w")

    def build_content_area(self) -> None:
        top = ttk.Frame(self.content, style="Content.TFrame")
        top.pack(fill=X, padx=26, pady=(22, 8))

        self.page_title = ttk.Label(top, text="Dashboard", style="Heading.TLabel")
        self.page_title.pack(side=LEFT)

        self.clock_label = ttk.Label(top, style="Sub.TLabel")
        self.clock_label.pack(side=RIGHT, padx=(0, 20))
        self.update_clock()

        self.ambient_canvas = Canvas(self.content, height=56, bg="#eef3fb", highlightthickness=0)
        self.ambient_canvas.pack(fill=X, padx=26, pady=(0, 0))

        self.transition_canvas = Canvas(self.content, height=5, bg="#eef3fb", highlightthickness=0)
        self.transition_canvas.pack(fill=X, padx=26, pady=(0, 6))

        self.body = ttk.Frame(self.content, style="Content.TFrame")
        self.body.pack(fill=BOTH, expand=True, padx=26, pady=(8, 20))

        self.pages = {
            "dashboard": self.create_dashboard_page,
            "students": self.create_students_page,
            "capture": self.create_capture_page,
            "live_attendance": self.create_live_attendance_page,
            "diagnostics": self.create_diagnostics_page,
            "training": self.create_training_page,
            "attendance": self.create_attendance_page,
        }

    def update_clock(self) -> None:
        self.clock_label.config(text=datetime.now().strftime("%d %b %Y | %I:%M:%S %p"))
        self.root.after(1000, self.update_clock)

    def clear_body(self) -> None:
        self.stop_camera_preview()
        self._camera_mode = "idle"
        self._cached_faces = []
        self._cached_predictions = []
        for child in self.body.winfo_children():
            child.destroy()

    def show_page(self, page_key: str) -> None:
        self.active_page.set(page_key)
        self.highlight_active_button()
        self.clear_body()

        page_titles = {
            "dashboard": "Dashboard",
            "students": "Student Registry",
            "capture": "Face Capture Samples",
            "live_attendance": "Live Attendance Recognition",
            "diagnostics": "Backend Diagnostics",
            "training": "Automatic Model Management",
            "attendance": "Attendance Log",
        }
        self.page_title.config(text=page_titles.get(page_key, "Dashboard"))
        self.animate_page_transition()
        self.pages[page_key]()

    def highlight_active_button(self) -> None:
        for key, button in self.nav_buttons.items():
            button.config(bg="#2f7de1" if key == self.active_page.get() else "#1f3157")

    def start_ui_animations(self) -> None:
        self.animate_active_nav_pulse()
        self.animate_ambient_glow()
        self.animate_clock_glow()

    def animate_active_nav_pulse(self) -> None:
        active = self.active_page.get()
        for key, button in self.nav_buttons.items():
            if key == active:
                palette_index = self._pulse_tick % len(self._active_nav_palette)
                button.config(bg=self._active_nav_palette[palette_index])
            else:
                button.config(bg="#1f3157")

        self._pulse_tick += 1
        self._nav_pulse_job = self.root.after(240, self.animate_active_nav_pulse)

    def animate_page_transition(self) -> None:
        if not hasattr(self, "transition_canvas"):
            return

        if self._transition_job:
            self.root.after_cancel(self._transition_job)
            self._transition_job = None

        canvas = self.transition_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        width = canvas.winfo_width()
        if width <= 1:
            width = max(self.content.winfo_width() - 52, 200)

        bar_width = max(width // 5, 160)

        def step(position: int) -> None:
            canvas.delete("all")
            canvas.create_rectangle(0, 0, width, 5, fill="#d8e5fb", outline="")
            canvas.create_rectangle(position, 0, position + bar_width, 5, fill="#2f7de1", outline="")
            if position < width:
                self._transition_job = self.root.after(16, lambda: step(position + 28))

        step(-bar_width)

    def animate_ambient_glow(self) -> None:
        if not hasattr(self, "ambient_canvas"):
            return

        canvas = self.ambient_canvas
        canvas.delete("all")
        width = canvas.winfo_width()
        if width <= 1:
            width = max(self.content.winfo_width() - 52, 300)

        canvas.create_rectangle(0, 0, width, 56, fill="#eef3fb", outline="")

        for idx, color in enumerate(["#d9e9ff", "#c7dcff", "#b7d2ff"]):
            x_center = (self._ambient_tick * (1.6 + idx * 0.25) + idx * 320) % (width + 260) - 130
            y_center = 20 + (idx * 9)
            radius = 110 - idx * 16
            canvas.create_oval(
                x_center - radius,
                y_center - radius,
                x_center + radius,
                y_center + radius,
                fill=color,
                outline="",
            )

        for idx, color in enumerate(["#eaf2ff", "#ddeafe"]):
            x_center = width - (((self._ambient_tick * (1.2 + idx * 0.2)) + idx * 260) % (width + 220)) + 80
            y_center = 36 + (idx * 8)
            radius = 84 - idx * 12
            canvas.create_oval(
                x_center - radius,
                y_center - radius,
                x_center + radius,
                y_center + radius,
                fill=color,
                outline="",
            )

        self._ambient_tick += 1
        self._ambient_job = self.root.after(72, self.animate_ambient_glow)

    def animate_clock_glow(self) -> None:
        wave = (math.sin(self._pulse_tick / 2.8) + 1) / 2
        style_name = "GlowSub.TLabel" if wave > 0.5 else "Sub.TLabel"
        self.clock_label.configure(style=style_name)
        self._clock_glow_job = self.root.after(280, self.animate_clock_glow)

    def create_card(self, parent, title: str, value: str, subtext: str):
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        value_label = ttk.Label(card, text=value, style="CardValue.TLabel")
        value_label.pack(anchor="w", pady=(6, 2))
        ttk.Label(card, text=subtext, style="Sub.TLabel").pack(anchor="w")

        self.animate_counter_label(value_label, value)
        return card

    def animate_counter_label(self, label: ttk.Label, raw_value: str) -> None:
        match = re.match(r"^(\d+)(.*)$", raw_value.strip())
        if not match:
            return

        target = int(match.group(1))
        suffix = match.group(2)
        steps = min(max(target, 10), 35)

        def tick(index: int = 0) -> None:
            value_now = int((target * index) / steps)
            label.config(text=f"{value_now}{suffix}")
            if index < steps:
                job = self.root.after(24, lambda: tick(index + 1))
                self._counter_jobs.append(job)

        tick()

    def create_dashboard_page(self) -> None:
        students = self.repository.list_students()
        attendance = self.repository.get_attendance(date_iso=datetime.now().date().isoformat())

        stats = ttk.Frame(self.body, style="Content.TFrame")
        stats.pack(fill=X)

        cards = [
            ("Total Students", str(len(students)), "Registered in database"),
            ("Stored Embeddings", str(len(self.repository.get_all_embeddings())), "Face vector samples"),
            ("Today Attendance", str(len(attendance)), "Detected as present"),
            ("Stability Rule", f"{STABILITY_FRAMES} frames", "Duplicate-safe marking"),
        ]

        for i, (title, value, subtext) in enumerate(cards):
            card = self.create_card(stats, title, value, subtext)
            card.grid(row=0, column=i, padx=(0, 14) if i < 3 else 0, sticky="nsew")
            stats.columnconfigure(i, weight=1)

        lower = ttk.Frame(self.body, style="Content.TFrame")
        lower.pack(fill=BOTH, expand=True, pady=(18, 0))
        lower.columnconfigure(0, weight=2)
        lower.columnconfigure(1, weight=1)

        log_card = ttk.Frame(lower, style="Card.TFrame", padding=16)
        log_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))

        ttk.Label(log_card, text="Recent Attendance", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 10))

        table_wrap = ttk.Frame(log_card, style="Card.TFrame")
        table_wrap.pack(fill=BOTH, expand=True)

        tree = ttk.Treeview(table_wrap, columns=("id", "name", "time", "status", "confidence"), show="headings", height=10)
        for col, text, width in [
            ("id", "Student ID", 120),
            ("name", "Name", 180),
            ("time", "Time", 180),
            ("status", "Status", 100),
            ("confidence", "Confidence", 100),
        ]:
            tree.heading(col, text=text)
            tree.column(col, width=width, anchor="center")

        scroll_y = ttk.Scrollbar(table_wrap, orient="vertical", command=tree.yview)
        scroll_x = ttk.Scrollbar(table_wrap, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        table_wrap.columnconfigure(0, weight=1)
        table_wrap.rowconfigure(0, weight=1)

        for row in self.repository.get_attendance()[:30]:
            tree.insert(
                "",
                "end",
                values=(
                    row["student_id"],
                    row["full_name"],
                    row["timestamp"],
                    row["status"],
                    f"{row['confidence']:.1f}%",
                ),
            )

        tips = ttk.Frame(lower, style="Card.TFrame", padding=16)
        tips.grid(row=0, column=1, sticky="nsew")
        ttk.Label(tips, text="Daily Workflow", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 10))

        workflow_points = [
            "1) Register student in Student Registry",
            "2) Open Face Capture and save 5-10 samples",
            "3) Open Live Attendance to auto-mark presence",
            "4) Review/export records in Attendance Log",
        ]
        for item in workflow_points:
            ttk.Label(tips, text=f"• {item}", style="Sub.TLabel", wraplength=300, justify=LEFT).pack(anchor="w", pady=3)

        ttk.Label(tips, text="", style="Sub.TLabel").pack(anchor="w", pady=2)
        ttk.Label(tips, text="System Status", style="CardTitle.TLabel").pack(anchor="w", pady=(6, 8))

        status_points = [
            f"Students: {len(students)}",
            f"Embeddings: {len(self.repository.get_all_embeddings())}",
            f"Matcher ready: {'Yes' if self.recognition_service.known_encodings.size > 0 else 'No'}",
            f"Embedding backend: {self.face_engine.embedding_mode}",
            "Realtime duplicate protection enabled",
        ]
        for item in status_points:
            ttk.Label(tips, text=f"• {item}", style="Sub.TLabel", wraplength=300, justify=LEFT).pack(anchor="w", pady=3)

    def create_students_page(self) -> None:
        wrapper = ttk.Frame(self.body, style="Card.TFrame", padding=18)
        wrapper.pack(fill=BOTH, expand=True)

        ttk.Label(wrapper, text="Add / Update Student", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", columnspan=2, pady=(0, 12))

        self.student_fields: dict[str, ttk.Entry] = {}
        fields = ["student_id", "full_name", "department", "semester", "email", "phone"]
        labels = ["Student ID", "Full Name", "Department", "Semester", "Email", "Phone"]

        for i, (field_key, label_text) in enumerate(zip(fields, labels)):
            ttk.Label(wrapper, text=label_text, style="Sub.TLabel").grid(row=i + 1, column=0, sticky="w", padx=(0, 12), pady=6)
            entry = ttk.Entry(wrapper, width=42)
            entry.grid(row=i + 1, column=1, sticky="w", pady=6)
            self.student_fields[field_key] = entry

        btn_row = Frame(wrapper, bg="white")
        btn_row.grid(row=len(fields) + 1, column=1, sticky="w", pady=(14, 8))

        Button(
            btn_row,
            text="Save Student",
            font=("Segoe UI", 9, "bold"),
            bg="#2f7de1",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=14,
            pady=7,
            command=self.save_student,
        ).pack(side=LEFT, padx=(0, 10))

        Button(
            btn_row,
            text="Clear",
            font=("Segoe UI", 9, "bold"),
            bg="#9099ab",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=14,
            pady=7,
            command=lambda: self.show_page("students"),
        ).pack(side=LEFT)

        Button(
            btn_row,
            text="Delete Selected",
            font=("Segoe UI", 9, "bold"),
            bg="#c64545",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=14,
            pady=7,
            command=self.delete_selected_student,
        ).pack(side=LEFT, padx=(10, 0))

        self.students_table = ttk.Treeview(wrapper, columns=("id", "name", "dept", "sem", "email", "phone"), show="headings", height=10)
        self.students_table.grid(row=len(fields) + 2, column=0, columnspan=2, sticky="nsew", pady=(14, 0))
        wrapper.rowconfigure(len(fields) + 2, weight=1)

        for col, text, width in [
            ("id", "Student ID", 110),
            ("name", "Name", 180),
            ("dept", "Department", 110),
            ("sem", "Semester", 100),
            ("email", "Email", 200),
            ("phone", "Phone", 120),
        ]:
            self.students_table.heading(col, text=text)
            self.students_table.column(col, width=width, anchor="center")

        self.refresh_students_table()

    def refresh_students_table(self) -> None:
        if not hasattr(self, "students_table"):
            return
        for item in self.students_table.get_children():
            self.students_table.delete(item)
        for student in self.repository.list_students():
            self.students_table.insert(
                "",
                "end",
                values=(
                    student["student_id"],
                    student["full_name"],
                    student["department"],
                    student["semester"],
                    student["email"],
                    student["phone"],
                ),
            )

    def save_student(self) -> None:
        payload = {k: v.get().strip() for k, v in self.student_fields.items()}

        if not payload["student_id"] or not payload["full_name"]:
            messagebox.showerror("Validation", "Student ID and Full Name are required")
            return

        student = Student(
            student_id=payload["student_id"],
            full_name=payload["full_name"],
            department=payload["department"] or "N/A",
            semester=payload["semester"] or "N/A",
            email=payload["email"],
            phone=payload["phone"],
        )
        self.repository.upsert_student(student)
        self.refresh_students_table()
        messagebox.showinfo("Student Registry", "Student saved successfully")

    def delete_selected_student(self) -> None:
        if not hasattr(self, "students_table"):
            return

        selected = self.students_table.selection()
        if not selected:
            messagebox.showwarning("Student Registry", "Select a student row to delete")
            return

        values = self.students_table.item(selected[0], "values")
        if not values:
            messagebox.showwarning("Student Registry", "Could not read selected student")
            return

        student_id = str(values[0])
        full_name = str(values[1]) if len(values) > 1 else student_id

        confirmed = messagebox.askyesno(
            "Confirm Delete",
            (
                f"Delete student '{full_name}' ({student_id})?\n\n"
                "This will permanently remove:\n"
                "- Student profile\n"
                "- Face embeddings\n"
                "- Attendance logs"
            ),
        )
        if not confirmed:
            return

        deleted, embeddings_deleted, attendance_deleted = self.repository.delete_student_data(student_id)
        if not deleted:
            messagebox.showwarning("Student Registry", "Student not found or already deleted")
            return

        self.refresh_students_table()
        self.refresh_capture_student_ids()
        self.recognition_service.refresh_model()
        messagebox.showinfo(
            "Student Registry",
            (
                f"Deleted {full_name} ({student_id})\n"
                f"Embeddings removed: {embeddings_deleted}\n"
                f"Attendance logs removed: {attendance_deleted}"
            ),
        )

    def create_capture_page(self) -> None:
        wrapper = ttk.Frame(self.body, style="Card.TFrame", padding=18)
        wrapper.pack(fill=BOTH, expand=True)

        ttk.Label(wrapper, text="Face Capture & Recognition", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(wrapper, text="Capture face samples only. Attendance marking is on the Live Attendance page.", style="Sub.TLabel").pack(anchor="w", pady=(4, 12))

        controls = Frame(wrapper, bg="white")
        controls.pack(fill=X, pady=(0, 10))

        guide = ttk.Label(
            wrapper,
            text="Tip: Select one student and capture 5-10 clear samples from slightly different angles.",
            style="Sub.TLabel",
            wraplength=980,
        )
        guide.pack(anchor="w", pady=(0, 10))

        ttk.Label(controls, text="Student ID for enrollment", style="Sub.TLabel").pack(side=LEFT, padx=(0, 8))
        ids = [s["student_id"] for s in self.repository.list_students()]
        self.capture_student_id = ttk.Combobox(controls, values=ids, width=22, state="readonly")
        if ids:
            self.capture_student_id.set(ids[0])
        self.capture_student_id.bind("<<ComboboxSelected>>", self.on_capture_student_changed)
        self.capture_student_id.pack(side=LEFT, padx=(0, 14))

        Button(
            controls,
            text="Refresh IDs",
            font=("Segoe UI", 9, "bold"),
            bg="#9099ab",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=self.refresh_capture_student_ids,
        ).pack(side=LEFT, padx=(0, 8))

        Button(
            controls,
            text="Start Camera",
            font=("Segoe UI", 9, "bold"),
            bg="#2f7de1",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=lambda: self.start_camera_preview(mode="capture"),
        ).pack(side=LEFT, padx=(0, 8))

        Button(
            controls,
            text="Capture Sample",
            font=("Segoe UI", 9, "bold"),
            bg="#2f7de1",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=self.capture_embedding_sample,
        ).pack(side=LEFT, padx=(0, 8))

        Button(
            controls,
            text="Stop Camera",
            font=("Segoe UI", 9, "bold"),
            bg="#9099ab",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=self.stop_camera_preview,
        ).pack(side=LEFT)

        Button(
            controls,
            text="Delete Selected Sample",
            font=("Segoe UI", 9, "bold"),
            bg="#c64545",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=self.delete_selected_sample,
        ).pack(side=LEFT, padx=(10, 8))

        Button(
            controls,
            text="Delete All Samples",
            font=("Segoe UI", 9, "bold"),
            bg="#8a2f2f",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=self.delete_all_samples_for_selected_student,
        ).pack(side=LEFT)

        self.camera_label = Label(wrapper, text="Camera not running", bg="#d9deea", fg="#4a5470", font=("Segoe UI", 12, "bold"))
        self.camera_label.pack(fill=BOTH, expand=True, pady=(10, 10))

        self.recognition_status = ttk.Label(wrapper, text="Status: Waiting", style="Sub.TLabel", wraplength=980)
        self.recognition_status.pack(anchor="w", pady=(8, 2))

        self.confidence_bar = ttk.Progressbar(wrapper, orient="horizontal", mode="determinate", length=500)
        self.confidence_bar.pack(anchor="w", pady=(2, 2))

        self.confidence_label = ttk.Label(wrapper, text="Confidence: 0%", style="Sub.TLabel")
        self.confidence_label.pack(anchor="w")

        self.stability_label = ttk.Label(wrapper, text="Detected Faces: 0", style="Sub.TLabel")
        self.stability_label.pack(anchor="w", pady=(2, 0))

        samples_card = ttk.Frame(wrapper, style="Card.TFrame")
        samples_card.pack(fill=BOTH, expand=True, pady=(10, 0))

        ttk.Label(samples_card, text="Saved Samples (Selected Student)", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))

        self.samples_table = ttk.Treeview(samples_card, columns=("sample_id", "student_id", "created_at"), show="headings", height=8)
        self.samples_table.pack(fill=BOTH, expand=True)
        for col, text, width in [
            ("sample_id", "Sample ID", 110),
            ("student_id", "Student ID", 160),
            ("created_at", "Captured At", 260),
        ]:
            self.samples_table.heading(col, text=text)
            self.samples_table.column(col, width=width, anchor="center")

        self.refresh_samples_table()

    def create_live_attendance_page(self) -> None:
        wrapper = ttk.Frame(self.body, style="Card.TFrame", padding=18)
        wrapper.pack(fill=BOTH, expand=True)

        ttk.Label(wrapper, text="Live Attendance Recognition", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            wrapper,
            text="Realtime face recognition to mark attendance. One face maps to one best match or Unknown.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(4, 12))

        controls = Frame(wrapper, bg="white")
        controls.pack(fill=X, pady=(0, 10))

        Button(
            controls,
            text="Start Camera",
            font=("Segoe UI", 9, "bold"),
            bg="#2f7de1",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=lambda: self.start_camera_preview(mode="attendance"),
        ).pack(side=LEFT, padx=(0, 8))

        Button(
            controls,
            text="Stop Camera",
            font=("Segoe UI", 9, "bold"),
            bg="#9099ab",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=self.stop_camera_preview,
        ).pack(side=LEFT)

        self.camera_label = Label(wrapper, text="Camera not running", bg="#d9deea", fg="#4a5470", font=("Segoe UI", 12, "bold"))
        self.camera_label.pack(fill=BOTH, expand=True, pady=(10, 10))

        self.recognition_status = ttk.Label(wrapper, text="Status: Waiting", style="Sub.TLabel", wraplength=980)
        self.recognition_status.pack(anchor="w", pady=(8, 2))

        self.confidence_bar = ttk.Progressbar(wrapper, orient="horizontal", mode="determinate", length=500)
        self.confidence_bar.pack(anchor="w", pady=(2, 2))

        self.confidence_label = ttk.Label(wrapper, text="Confidence: 0%", style="Sub.TLabel")
        self.confidence_label.pack(anchor="w")

        self.stability_label = ttk.Label(wrapper, text="Detected Faces: 0", style="Sub.TLabel")
        self.stability_label.pack(anchor="w", pady=(2, 0))

        if self.face_engine.embedding_mode != "face_recognition":
            if self.face_engine.embedding_mode == "orb_legacy":
                self.recognition_status.config(
                    text="Status: Running in legacy compatibility mode. Use Diagnostics to finalize 128-d migration."
                )
            else:
                self.recognition_status.config(
                    text=(
                        f"Status: Running with {self.face_engine.embedding_mode.upper()} backend. "
                        "Install face_recognition + dlib for best accuracy."
                    )
                )
        elif self.recognition_service.known_encodings.size == 0:
            self.recognition_status.config(
                text="Status: No compatible face_recognition samples found. Capture fresh samples in Face Capture."
            )

    def create_diagnostics_page(self) -> None:
        wrapper = ttk.Frame(self.body, style="Card.TFrame", padding=18)
        wrapper.pack(fill=BOTH, expand=True)

        ttk.Label(wrapper, text="Recognition Diagnostics", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            wrapper,
            text="Checks active backend, package availability, and sample readiness for accurate recognition.",
            style="Sub.TLabel",
            wraplength=980,
        ).pack(anchor="w", pady=(4, 12))

        diag = self.face_engine.get_backend_diagnostics()
        students = self.repository.list_students()
        embedding_rows = self.repository.get_all_embeddings()

        tools_row = Frame(wrapper, bg="white")
        tools_row.pack(fill=X, pady=(0, 10))
        Button(
            tools_row,
            text="Auto Calibrate Thresholds",
            font=("Segoe UI", 9, "bold"),
            bg="#2f7de1",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=self.run_auto_calibration,
        ).pack(side=LEFT)

        ttk.Label(tools_row, text="Student", style="Sub.TLabel").pack(side=LEFT, padx=(12, 6))
        self.personalize_student = ttk.Combobox(
            tools_row,
            values=["ALL"] + [s["student_id"] for s in students],
            width=16,
            state="readonly",
        )
        self.personalize_student.set("ALL")
        self.personalize_student.pack(side=LEFT)

        Button(
            tools_row,
            text="Personalize Calibration",
            font=("Segoe UI", 9, "bold"),
            bg="#2f7de1",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=self.run_personalized_calibration,
        ).pack(side=LEFT, padx=(8, 0))

        Button(
            tools_row,
            text="Reset Calibration",
            font=("Segoe UI", 9, "bold"),
            bg="#9099ab",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=self.run_reset_calibration,
        ).pack(side=LEFT, padx=(8, 0))

        Button(
            tools_row,
            text="Finalize 128-d Migration",
            font=("Segoe UI", 9, "bold"),
            bg="#1f8f4d",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=self.run_finalize_migration,
        ).pack(side=LEFT, padx=(8, 0))

        per_student_counts: dict[str, int] = {}
        for student_id, _ in embedding_rows:
            per_student_counts[student_id] = per_student_counts.get(student_id, 0) + 1

        stats_frame = ttk.Frame(wrapper, style="Card.TFrame")
        stats_frame.pack(fill=X, pady=(0, 10))

        lines = [
            f"Active backend: {diag.get('active_backend', 'unknown')}",
            f"deepface: {diag.get('deepface_status', 'unknown')}",
            f"face_recognition: {diag.get('face_recognition_status', 'unknown')}",
            f"Students registered: {len(students)}",
            f"Total samples: {len(embedding_rows)}",
            f"Backend-compatible samples: {self.recognition_service.known_encodings.shape[0]}",
            f"Calibration saved: {'Yes' if Path(CALIBRATION_PATH).exists() else 'No'}",
            f"Current distance threshold: {self.recognition_service.face_distance_threshold:.3f}",
            f"Current cosine threshold (deepface): {self.recognition_service.deepface_cosine_threshold:.3f}",
            f"Current cosine threshold (orb): {self.recognition_service.orb_cosine_threshold:.3f}",
            f"Current margin: {self.recognition_service.cosine_margin:.3f}",
            f"Personalized profiles: {len(self.recognition_service.per_student_distance_thresholds) + len(self.recognition_service.per_student_cosine_thresholds)}",
            diag.get("notes", ""),
        ]

        dim_breakdown: dict[int, int] = {}
        for _, emb in embedding_rows:
            dim = int(len(emb))
            dim_breakdown[dim] = dim_breakdown.get(dim, 0) + 1
        if dim_breakdown:
            parts = [f"{dim}d={count}" for dim, count in sorted(dim_breakdown.items())]
            lines.append(f"Dimension breakdown: {', '.join(parts)}")
        for line in lines:
            if line:
                ttk.Label(stats_frame, text=f"• {line}", style="Sub.TLabel", wraplength=1000, justify=LEFT).pack(anchor="w", pady=2)

        sample_card = ttk.Frame(wrapper, style="Card.TFrame")
        sample_card.pack(fill=BOTH, expand=True, pady=(4, 10))
        ttk.Label(sample_card, text="Sample Coverage by Student", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))

        sample_tree = ttk.Treeview(sample_card, columns=("student_id", "sample_count", "ready"), show="headings", height=8)
        sample_tree.pack(fill=BOTH, expand=True)
        for col, text, width in [
            ("student_id", "Student ID", 180),
            ("sample_count", "Samples", 120),
            ("ready", "Ready", 120),
        ]:
            sample_tree.heading(col, text=text)
            sample_tree.column(col, width=width, anchor="center")

        for student in students:
            student_id = student["student_id"]
            count = per_student_counts.get(student_id, 0)
            ready = "Yes" if count >= MIN_SAMPLES_PER_STUDENT else "No"
            sample_tree.insert("", "end", values=(student_id, count, ready))

        guide_card = ttk.Frame(wrapper, style="Card.TFrame")
        guide_card.pack(fill=X, pady=(4, 0))
        ttk.Label(guide_card, text="Recommended Setup (Windows)", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))

        guide_lines = [
            "1) Use Python 3.10 or 3.11 for best face_recognition compatibility.",
            "2) In your virtual environment, run: pip install cmake dlib face_recognition",
            "3) Restart the app and check this page until face_recognition is available.",
            f"4) Capture at least {MIN_SAMPLES_PER_STUDENT}-10 clean samples per student.",
            "5) Use Live Attendance page for marking after sample collection.",
        ]
        for line in guide_lines:
            ttk.Label(guide_card, text=f"• {line}", style="Sub.TLabel", wraplength=1000, justify=LEFT).pack(anchor="w", pady=2)

    def run_auto_calibration(self) -> None:
        result = self.recognition_service.calibrate_thresholds()
        if not result.get("updated"):
            messagebox.showwarning("Calibration", result.get("reason", "Calibration failed"))
            return

        mode = result.get("mode", self.face_engine.embedding_mode)
        if mode == "face_recognition":
            msg = (
                "Calibration complete\n"
                f"Mode: {mode}\n"
                f"Distance threshold: {result.get('face_distance_threshold')}\n"
                f"Distance margin: {result.get('face_distance_margin')}\n"
                f"Students: {result.get('students')} | Samples: {result.get('samples')}"
            )
        else:
            msg = (
                "Calibration complete\n"
                f"Mode: {mode}\n"
                f"Cosine threshold: {result.get('cosine_threshold')}\n"
                f"Cosine margin: {result.get('cosine_margin')}\n"
                f"Students: {result.get('students')} | Samples: {result.get('samples')}"
            )

        messagebox.showinfo("Calibration", msg)
        self.show_page("diagnostics")

    def run_reset_calibration(self) -> None:
        confirmed = messagebox.askyesno(
            "Reset Calibration",
            "Reset all calibration thresholds to default values?",
        )
        if not confirmed:
            return

        result = self.recognition_service.reset_calibration()
        msg = (
            "Calibration reset to defaults\n"
            f"Distance threshold: {result.get('face_distance_threshold')}\n"
            f"Distance margin: {result.get('face_distance_margin')}\n"
            f"Deepface cosine threshold: {result.get('deepface_cosine_threshold')}\n"
            f"ORB cosine threshold: {result.get('orb_cosine_threshold')}\n"
            f"Cosine margin: {result.get('cosine_margin')}\n"
            f"Min mark confidence: {result.get('min_mark_confidence')}"
        )
        messagebox.showinfo("Calibration", msg)
        self.show_page("diagnostics")

    def run_finalize_migration(self) -> None:
        records = self.repository.list_embedding_records()
        if not records:
            messagebox.showwarning("Migration", "No samples found")
            return

        if not self.face_engine.is_face_recognition_available():
            messagebox.showwarning("Migration", "face_recognition is not available in this environment")
            return

        per_student_128: dict[str, int] = {}
        delete_ids: list[int] = []

        for row in records:
            sid = row["student_id"]
            dim = len(row["embedding"])
            if dim == 128:
                per_student_128[sid] = per_student_128.get(sid, 0) + 1
            else:
                delete_ids.append(int(row["id"]))

        students = [s["student_id"] for s in self.repository.list_students()]
        weak_students = [sid for sid in students if per_student_128.get(sid, 0) < MIN_SAMPLES_PER_STUDENT]
        if weak_students:
            msg = (
                "Cannot finalize yet. These students need more 128-d samples:\n"
                + "\n".join([f"- {sid}: {per_student_128.get(sid, 0)}" for sid in weak_students])
            )
            messagebox.showwarning("Migration", msg)
            return

        if not delete_ids:
            messagebox.showinfo("Migration", "No legacy samples to remove. Migration already finalized.")
            return

        confirmed = messagebox.askyesno(
            "Finalize Migration",
            f"Delete {len(delete_ids)} legacy samples and keep only 128-d face_recognition samples?",
        )
        if not confirmed:
            return

        deleted = self.repository.delete_embeddings_by_ids(delete_ids)
        self.face_engine.embedding_mode = "face_recognition"
        self.recognition_service.refresh_model()
        self.recognition_service.calibrate_thresholds()
        self.recognition_service.calibrate_personalized_thresholds(None)
        messagebox.showinfo("Migration", f"Migration finalized. Removed {deleted} legacy samples.")
        self.show_page("diagnostics")

    def run_personalized_calibration(self) -> None:
        selected = "ALL"
        if hasattr(self, "personalize_student"):
            selected = self.personalize_student.get().strip() or "ALL"

        target_student = None if selected == "ALL" else selected
        result = self.recognition_service.calibrate_personalized_thresholds(target_student)
        if not result.get("updated"):
            messagebox.showwarning("Personalized Calibration", result.get("reason", "Personalization failed"))
            return

        msg = (
            "Personalized calibration complete\n"
            f"Mode: {result.get('mode')}\n"
            f"Target: {result.get('target_student')}\n"
            f"Students updated: {result.get('students_updated')}"
        )
        messagebox.showinfo("Personalized Calibration", msg)
        self.show_page("diagnostics")

    def start_camera_preview(self, mode: str = "attendance") -> None:
        try:
            self._frame_counter = 0
            self._camera_mode = mode
            self._cached_faces = []
            self._cached_predictions = []

            # Always refresh known embeddings so recognition does not stay stale/empty.
            self.recognition_service.refresh_model()

            self.camera_service.start()
            self.schedule_camera_update()
            if mode == "capture":
                self.recognition_status.config(text="Status: Camera running (sample capture mode)")
            else:
                known_count = len(set(self.recognition_service.known_student_ids))
                self.recognition_status.config(
                    text=(
                        f"Status: Camera running (attendance mode) | Known students: {known_count} "
                        f"(recommended >= {MIN_SAMPLES_PER_STUDENT} samples each)"
                    )
                )
        except Exception as exc:
            logger.exception("Failed to start camera")
            messagebox.showerror("Camera", f"Failed to start camera: {exc}")

    def stop_camera_preview(self) -> None:
        if self._camera_job:
            self.root.after_cancel(self._camera_job)
            self._camera_job = None
        self.camera_service.stop()

    def schedule_camera_update(self) -> None:
        self._camera_job = self.root.after(33, self.update_camera_frame)

    def update_camera_frame(self) -> None:
        frame = self.camera_service.get_frame()
        if frame is None:
            self.recognition_status.config(text="Status: Waiting for camera frame")
            self.schedule_camera_update()
            return

        display_frame = frame.copy()
        self._frame_counter += 1
        should_detect = (self._frame_counter % self._detect_stride) == 0
        faces: list[tuple[int, int, int, int]] = []
        if should_detect:
            faces = self.face_engine.detect_faces(display_frame)
            self._cached_faces = faces
        elif self._cached_faces:
            faces = self._cached_faces

        max_confidence = 0.0
        recognized_names: list[str] = []
        marked_names: list[str] = []

        if faces:
            if self._camera_mode == "capture":
                for x, y, w, h in faces:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (47, 125, 225), 2)
                self.stability_label.config(text=f"Detected Faces: {len(faces)}")
                self.recognition_status.config(text="Status: Face detected. Press Capture Sample to save.")
            else:
                if should_detect:
                    predictions: list[tuple[str | None, float, float]] = []
                    for face in faces:
                        embedding = self.face_engine.extract_embedding(frame, face)
                        if embedding is None:
                            predictions.append((None, 1.0, 0.0))
                            continue
                        predictions.append(self.recognition_service.predict_identity(embedding))
                    self._cached_predictions = predictions

                for idx, face in enumerate(faces):
                    x, y, w, h = face
                    recognized_id, distance, confidence = (
                        self._cached_predictions[idx]
                        if idx < len(self._cached_predictions)
                        else (None, 1.0, 0.0)
                    )

                    max_confidence = max(max_confidence, confidence)
                    if recognized_id:
                        name = self.repository.get_student_name(recognized_id) or recognized_id
                        recognized_names.append(name)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 170, 70), 2)
                        cv2.putText(
                            display_frame,
                            f"{name} ({confidence:.1f}%)",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 170, 70),
                            2,
                        )

                        if should_detect:
                            marked, _ = self.recognition_service.mark_if_eligible(recognized_id, confidence)
                            if marked:
                                marked_names.append(name)
                    else:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 90, 220), 2)
                        cv2.putText(
                            display_frame,
                            f"Unknown (score={distance:.3f})",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 90, 220),
                            2,
                        )

                unique_recognized = sorted(set(recognized_names))
                unique_marked = sorted(set(marked_names))
                if should_detect:
                    if unique_marked:
                        self.recognition_status.config(text=f"Status: Attendance marked for {', '.join(unique_marked)}")
                    elif unique_recognized:
                        self.recognition_status.config(text=f"Status: Recognized {', '.join(unique_recognized)}")
                    else:
                        self.recognition_status.config(text="Status: Unknown or low-confidence face detected")

                self.stability_label.config(text=f"Detected Faces: {len(faces)}")
        else:
            self._cached_faces = []
            self._cached_predictions = []
            self.stability_label.config(text="Detected Faces: 0")
            self.recognition_status.config(text="Status: No face detected")
            max_confidence = 0.0

        self.confidence_bar["value"] = max_confidence
        self.confidence_label.config(text=f"Confidence: {max_confidence:.1f}%")

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = image.resize((1000, 520))
        photo = ImageTk.PhotoImage(image)
        self._camera_photo = photo
        self.camera_label.configure(image=photo, text="")

        self.schedule_camera_update()

    def capture_embedding_sample(self) -> None:
        student_id = self.capture_student_id.get().strip() if hasattr(self, "capture_student_id") else ""
        if not student_id:
            messagebox.showerror("Capture", "Select a student ID before capturing")
            return

        frame = self.camera_service.get_frame()
        if frame is None:
            messagebox.showerror("Capture", "Camera frame not available")
            return

        faces = self.face_engine.detect_faces(frame)
        if not faces:
            messagebox.showwarning("Capture", "No face detected")
            return

        if len(faces) > 1:
            messagebox.showwarning("Capture", "Multiple faces detected. Keep only one face in frame.")
            return

        # Capture the most prominent face to reduce noisy samples.
        largest_face = max(faces, key=lambda b: b[2] * b[3])
        _, _, width, height = largest_face
        if width < CAPTURE_MIN_FACE_WIDTH or height < CAPTURE_MIN_FACE_HEIGHT:
            messagebox.showwarning("Capture", "Move closer to camera for a larger face sample")
            return

        x, y, w, h = largest_face
        crop = frame[max(y, 0): y + h, max(x, 0): x + w]
        if crop.size == 0:
            messagebox.showwarning("Capture", "Invalid face crop, try again")
            return

        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
        if sharpness < CAPTURE_MIN_SHARPNESS:
            messagebox.showwarning("Capture", "Image is blurry. Hold still and improve lighting.")
            return

        generated = 0
        augmented_crops = self.face_engine.generate_augmented_crops(crop)
        capture_mode = "face_recognition" if self.face_engine.is_face_recognition_available() else self.face_engine.embedding_mode
        for sample_crop in augmented_crops:
            embedding = self.face_engine.extract_embedding_from_crop_with_mode(sample_crop, capture_mode)
            if embedding is None:
                continue
            self.repository.add_embedding(student_id, embedding.tolist())
            generated += 1

        if generated == 0:
            messagebox.showwarning("Capture", "Could not extract embedding")
            return

        self.recognition_service.refresh_model()
        self.recognition_service.calibrate_thresholds()
        self.recognition_service.calibrate_personalized_thresholds(student_id)
        ready, message, trained_now = self.training_service.ensure_model_ready(force_retrain=False)

        if trained_now:
            messagebox.showinfo("Capture", f"Saved {generated} augmented samples ({capture_mode})\n{message}")
        elif ready:
            messagebox.showinfo("Capture", f"Saved {generated} augmented samples ({capture_mode})\nModel is ready for attendance")
        else:
            messagebox.showinfo("Capture", f"Saved {generated} augmented samples ({capture_mode})\n{message}")

        self.refresh_capture_student_ids()
        self.refresh_samples_table()

    def on_capture_student_changed(self, _event=None) -> None:
        self.refresh_samples_table()

    def refresh_samples_table(self) -> None:
        if not hasattr(self, "samples_table"):
            return

        for item in self.samples_table.get_children():
            self.samples_table.delete(item)

        student_id = self.capture_student_id.get().strip() if hasattr(self, "capture_student_id") else ""
        rows = self.repository.list_embedding_samples(student_id=student_id or None)
        for row in rows:
            self.samples_table.insert(
                "",
                "end",
                values=(row["id"], row["student_id"], row["created_at"]),
            )

    def delete_selected_sample(self) -> None:
        if not hasattr(self, "samples_table"):
            return

        selection = self.samples_table.selection()
        if not selection:
            messagebox.showwarning("Capture", "Select a sample row to delete")
            return

        values = self.samples_table.item(selection[0], "values")
        if not values:
            messagebox.showwarning("Capture", "Could not read selected sample")
            return

        sample_id = int(values[0])
        student_id = str(values[1])

        confirmed = messagebox.askyesno(
            "Confirm Delete",
            f"Delete sample #{sample_id} for student {student_id}?",
        )
        if not confirmed:
            return

        deleted = self.repository.delete_embedding_sample(sample_id)
        if not deleted:
            messagebox.showwarning("Capture", "Sample not found or already deleted")
            return

        self.recognition_service.refresh_model()
        self.refresh_samples_table()
        messagebox.showinfo("Capture", f"Deleted sample #{sample_id}")

    def delete_all_samples_for_selected_student(self) -> None:
        student_id = self.capture_student_id.get().strip() if hasattr(self, "capture_student_id") else ""
        if not student_id:
            messagebox.showwarning("Capture", "Select a student ID first")
            return

        confirmed = messagebox.askyesno(
            "Confirm Delete",
            f"Delete ALL saved samples for student {student_id}?",
        )
        if not confirmed:
            return

        deleted_count = self.repository.delete_embeddings_for_student(student_id)
        self.recognition_service.refresh_model()
        self.refresh_samples_table()
        messagebox.showinfo("Capture", f"Deleted {deleted_count} sample(s) for {student_id}")

    def refresh_capture_student_ids(self) -> None:
        if not hasattr(self, "capture_student_id"):
            return
        ids = [s["student_id"] for s in self.repository.list_students()]
        selected = self.capture_student_id.get().strip()
        self.capture_student_id["values"] = ids
        if selected in ids:
            self.capture_student_id.set(selected)
        elif ids:
            self.capture_student_id.set(ids[0])
        else:
            self.capture_student_id.set("")
        self.refresh_samples_table()

    def create_training_page(self) -> None:
        wrapper = ttk.Frame(self.body, style="Card.TFrame", padding=18)
        wrapper.pack(fill=BOTH, expand=True)

        ttk.Label(wrapper, text="Automatic Model Management", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(wrapper, text="Training runs automatically when you capture new face samples.", style="Sub.TLabel").pack(anchor="w", pady=(4, 12))

        controls = Frame(wrapper, bg="white")
        controls.pack(fill=X)

        Button(
            controls,
            text="Rebuild Model Now",
            font=("Segoe UI", 9, "bold"),
            bg="#2f7de1",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=14,
            pady=7,
            command=self.run_training,
        ).pack(side=LEFT)

        self.training_logs = Text(wrapper, height=16, bg="#f3f5fa", fg="#2c3448", relief=FLAT, padx=10, pady=8)
        self.training_logs.pack(fill=BOTH, expand=True, pady=(12, 0))
        self.training_logs.insert(END, "[INFO] Ready for training\n")
        self.training_logs.config(state=DISABLED)

    def append_training_log(self, text: str) -> None:
        self.training_logs.config(state=NORMAL)
        self.training_logs.insert(END, text + "\n")
        self.training_logs.see(END)
        self.training_logs.config(state=DISABLED)

    def run_training(self) -> None:
        self.append_training_log("[INFO] Rebuilding model from latest embeddings")
        ok, message, trained_now = self.training_service.ensure_model_ready(force_retrain=True)
        if ok:
            self.recognition_service.refresh_model()
            status = "[SUCCESS]" if trained_now else "[INFO]"
            self.append_training_log(f"{status} {message}")
            messagebox.showinfo("Training", message)
        else:
            self.append_training_log(f"[ERROR] {message}")
            messagebox.showerror("Training", message)

    def create_attendance_page(self) -> None:
        wrapper = ttk.Frame(self.body, style="Card.TFrame", padding=18)
        wrapper.pack(fill=BOTH, expand=True)

        controls = Frame(wrapper, bg="white")
        controls.pack(fill=X, pady=(0, 12))

        ttk.Label(
            wrapper,
            text="Tip: Keep the face centered for stable recognition before marking attendance.",
            style="Sub.TLabel",
            wraplength=980,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(controls, text="Class Date (YYYY-MM-DD)", style="Sub.TLabel").pack(side=LEFT, padx=(0, 8))
        self.filter_date = ttk.Entry(controls, width=16)
        self.filter_date.insert(0, datetime.now().date().isoformat())
        self.filter_date.pack(side=LEFT, padx=(0, 14))

        ttk.Label(controls, text="Semester", style="Sub.TLabel").pack(side=LEFT, padx=(0, 8))
        self.filter_semester = ttk.Entry(controls, width=10)
        self.filter_semester.pack(side=LEFT)

        Button(
            controls,
            text="Load",
            font=("Segoe UI", 9, "bold"),
            bg="#2f7de1",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=12,
            pady=7,
            command=self.load_attendance_table,
        ).pack(side=LEFT, padx=(10, 0))

        Button(
            controls,
            text="Export CSV",
            font=("Segoe UI", 9, "bold"),
            bg="#2f7de1",
            fg="white",
            relief=FLAT,
            bd=0,
            padx=14,
            pady=7,
            command=self.export_attendance,
        ).pack(side=RIGHT)

        self.attendance_tree = ttk.Treeview(wrapper, columns=("id", "name", "dept", "time", "remark", "confidence"), show="headings")
        self.attendance_tree.pack(fill=BOTH, expand=True)
        for col, label, width in [
            ("id", "Student ID", 120),
            ("name", "Name", 220),
            ("dept", "Department", 130),
            ("time", "Timestamp", 180),
            ("remark", "Status", 120),
            ("confidence", "Confidence", 100),
        ]:
            self.attendance_tree.heading(col, text=label)
            self.attendance_tree.column(col, width=width, anchor="center")

        self.attendance_summary = ttk.Label(wrapper, text="Records: 0", style="Sub.TLabel", wraplength=980, justify=LEFT)
        self.attendance_summary.pack(anchor="w", pady=(8, 0))

        self.load_attendance_table()

    def load_attendance_table(self) -> None:
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)

        date_val = self.filter_date.get().strip() if hasattr(self, "filter_date") else ""
        sem_val_raw = self.filter_semester.get().strip() if hasattr(self, "filter_semester") else ""
        sem_val = self._normalize_semester_filter(sem_val_raw)
        rows = self.repository.get_attendance(date_iso=date_val or None, section=sem_val or None)

        self._last_attendance_rows = rows

        for row in rows:
            self.attendance_tree.insert(
                "",
                "end",
                values=(
                    row["student_id"],
                    row["full_name"],
                    row["department"],
                    row["timestamp"],
                    row["status"],
                    f"{row['confidence']:.1f}%",
                ),
            )

        if hasattr(self, "attendance_summary"):
            summary = f"Records: {len(rows)} | Last refresh: {datetime.now().strftime('%I:%M:%S %p')}"

            # Helpful hint: date filter can hide semester records from previous days.
            if not rows and date_val and sem_val:
                semester_all_dates = self.repository.get_attendance(date_iso=None, section=sem_val)
                if semester_all_dates:
                    summary += f" | {len(semester_all_dates)} record(s) for semester {sem_val} on other date(s)"

            self.attendance_summary.config(text=summary)

    @staticmethod
    def _normalize_semester_filter(value: str) -> str:
        """Accept flexible semester input such as '6', 'Semester 6', or '6 semester'."""
        if not value:
            return ""

        lowered = value.lower().strip()
        lowered = lowered.replace("semester", "").replace("sem", "").strip()
        digit_match = re.search(r"\d+", lowered)
        if digit_match:
            return digit_match.group(0)
        return lowered

    def export_attendance(self) -> None:
        rows = getattr(self, "_last_attendance_rows", [])
        if not rows:
            messagebox.showwarning("Export", "No attendance rows to export")
            return

        default_name = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = filedialog.asksaveasfilename(
            title="Save attendance CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV Files", "*.csv")],
        )
        if not file_path:
            return

        exported = export_attendance_csv(rows, file_path)
        messagebox.showinfo("Export", f"CSV exported to:\n{exported}")

    def on_close(self) -> None:
        try:
            if self._transition_job:
                self.root.after_cancel(self._transition_job)
            if self._nav_pulse_job:
                self.root.after_cancel(self._nav_pulse_job)
            if self._ambient_job:
                self.root.after_cancel(self._ambient_job)
            if self._clock_glow_job:
                self.root.after_cancel(self._clock_glow_job)
            for job in self._counter_jobs:
                try:
                    self.root.after_cancel(job)
                except Exception:
                    pass
            self.stop_camera_preview()
        finally:
            self.root.destroy()


def run_app() -> None:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            # Mark process DPI-aware to avoid Windows bitmap scaling blur.
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            logger.debug("Could not enable DPI awareness", exc_info=True)

    root = Tk()
    FaceAttendanceUI(root)
    root.mainloop()
