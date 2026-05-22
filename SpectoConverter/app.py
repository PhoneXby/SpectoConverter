from __future__ import annotations

import io
import threading
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageTk

from spectro_core import (
    SpectroConfig,
    prepare_image,
    image_to_audio,
    load_image_from_path,
    render_spectrogram_preview,
    save_wav,
    text_to_image,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SpectroFlagApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CTF Spectrogram Flag Embedder")
        self.geometry("1100x720")
        self.minsize(960, 640)

        self._source_image: Image.Image | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._spectro_photo: ImageTk.PhotoImage | None = None
        self._busy = False
        self._action_buttons: list[ctk.CTkButton] = []

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        self._build_loading_overlay(main)

        ctk.CTkLabel(
            sidebar,
            text="CTF Spectrogram",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(padx=16, pady=(20, 4), anchor="w")
        ctk.CTkLabel(
            sidebar,
            text="Embed the flag in the\naudio file's spectrogram.",
            font=ctk.CTkFont(size=12),
            text_color="gray70",
        ).pack(padx=16, pady=(0, 16), anchor="w")

        self.mode_var = ctk.StringVar(value="text")
        ctk.CTkLabel(sidebar, text="Source").pack(padx=16, anchor="w")
        mode_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        mode_frame.pack(padx=16, pady=4, fill="x")
        ctk.CTkRadioButton(
            mode_frame,
            text="Text (flag)",
            variable=self.mode_var,
            value="text",
            command=self._on_mode_change,
        ).pack(side="left", padx=(0, 12))
        ctk.CTkRadioButton(
            mode_frame,
            text="Image",
            variable=self.mode_var,
            value="image",
            command=self._on_mode_change,
        ).pack(side="left")

        self.text_box = ctk.CTkTextbox(sidebar, height=100)
        self.text_box.pack(padx=16, pady=8, fill="x")
        self.text_box.insert("1.0", "FLAG{example_spectrogram_flag}")

        self.btn_load_img = ctk.CTkButton(
            sidebar,
            text="Select image…",
            command=self._pick_image,
            state="disabled",
        )
        self.btn_load_img.pack(padx=16, pady=4, fill="x")
        self._action_buttons.append(self.btn_load_img)

        ctk.CTkLabel(sidebar, text="Size & duration").pack(padx=16, pady=(12, 0), anchor="w")
        self._add_slider(sidebar, "Width (px)", "width", 400, 1400, 900)
        self._add_slider(sidebar, "Height (px)", "height", 200, 800, 450)
        self._add_slider(sidebar, "Font size", "font_size", 16, 120, 48)
        self._add_slider(sidebar, "Duration (s)", "duration", 4, 30, 12, is_float=True)
        self._add_slider(sidebar, "Min frequency (Hz)", "freq_min", 200, 2000, 400)
        self._add_slider(sidebar, "Max frequency (Hz)", "freq_max", 3000, 12000, 9000)

        self.invert_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            sidebar,
            text="Invert colors (white background)",
            variable=self.invert_var,
        ).pack(padx=16, pady=8, anchor="w")

        ctk.CTkLabel(sidebar, text="Export").pack(padx=16, pady=(8, 0), anchor="w")

        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.pack(padx=16, pady=8, fill="x")

        self.btn_export_wav = ctk.CTkButton(
            btn_frame,
            text="Generate and save WAV",
            command=self._export_wav,
            fg_color="#1d3557",
            hover_color="#457b9d",
            height=36,
        )
        self.btn_export_wav.pack(fill="x", pady=4)
        self._action_buttons.append(self.btn_export_wav)

        self.btn_export_png = ctk.CTkButton(
            btn_frame,
            text="Save source PNG",
            command=self._export_png,
            fg_color="#3d405b",
            hover_color="#5c5d7a",
        )
        self.btn_export_png.pack(fill="x", pady=4)
        self._action_buttons.append(self.btn_export_png)

        ctk.CTkLabel(sidebar, text="Preview").pack(padx=16, pady=(8, 0), anchor="w")
        preview_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        preview_frame.pack(padx=16, pady=8, fill="x")

        self.btn_preview = ctk.CTkButton(
            preview_frame,
            text="Preview spectrogram",
            command=self._preview,
            fg_color="#2d6a4f",
            hover_color="#40916c",
        )
        self.btn_preview.pack(fill="x", pady=4)
        self._action_buttons.append(self.btn_preview)

        ctk.CTkLabel(
            sidebar,
            text="Solve with: Audacity → Spectrogram\nor Sonic Visualiser",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
        ).pack(side="bottom", padx=16, pady=16, anchor="w")

        tabs = ctk.CTkTabview(main)
        tabs.grid(row=0, column=0, sticky="nsew")
        tabs.add("Source image")
        tabs.add("Spectrogram preview")

        self.lbl_source = ctk.CTkLabel(tabs.tab("Source image"), text="")
        self.lbl_source.pack(expand=True, fill="both", padx=8, pady=8)

        self.lbl_spectro = ctk.CTkLabel(tabs.tab("Spectrogram preview"), text="")
        self.lbl_spectro.pack(expand=True, fill="both", padx=8, pady=8)

        hint = (
            "Tip: Shorter duration + larger font = clearer spectrogram. "
            "Keep the frequency range around 400–9000 Hz."
        )
        ctk.CTkLabel(main, text=hint, text_color="gray65", wraplength=700).grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )

    def _build_loading_overlay(self, parent: ctk.CTkFrame) -> None:
        self._overlay = ctk.CTkFrame(
            parent,
            fg_color=("gray85", "gray12"),
            corner_radius=12,
            border_width=1,
            border_color=("gray70", "gray30"),
        )
        self._overlay.grid(row=0, column=0, sticky="nsew")
        self._overlay.grid_remove()
        self._overlay.grid_columnconfigure(0, weight=1)
        self._overlay.grid_rowconfigure(0, weight=1)

        box = ctk.CTkFrame(self._overlay, fg_color="transparent")
        box.place(relx=0.5, rely=0.5, anchor="center")

        self._loading_title = ctk.CTkLabel(
            box,
            text="Processing…",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self._loading_title.pack(pady=(0, 12))

        self._progress = ctk.CTkProgressBar(box, width=320, height=14)
        self._progress.pack(pady=(0, 8))
        self._progress.set(0)

        self._loading_status = ctk.CTkLabel(
            box,
            text="Preparing…",
            text_color="gray70",
        )
        self._loading_status.pack()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            for btn in self._action_buttons:
                btn.configure(state="disabled")
            self.text_box.configure(state="disabled")
            self._overlay.grid()
            self._progress.set(0)
        else:
            self._overlay.grid_remove()
            for btn in self._action_buttons:
                btn.configure(state="normal")
            self._on_mode_change()

    def _update_progress(self, value: float, message: str) -> None:
        self._progress.set(max(0.0, min(1.0, value)))
        if message:
            self._loading_status.configure(text=message)

    def _run_async(
        self,
        worker: Callable[[Callable[[float, str], None]], object],
        on_success: Callable[[object], None],
        *,
        title: str = "Processing…",
    ) -> None:
        if self._busy:
            return

        self._loading_title.configure(text=title)
        self._set_busy(True)

        def progress(value: float, message: str) -> None:
            self.after(0, lambda: self._update_progress(value, message))

        def run() -> None:
            try:
                result = worker(progress)
                self.after(0, lambda: self._on_worker_done(result, None, on_success))
            except Exception as exc:
                self.after(0, lambda: self._on_worker_done(None, exc, on_success))

        threading.Thread(target=run, daemon=True).start()

    def _on_worker_done(
        self,
        result: object | None,
        error: Exception | None,
        on_success: Callable[[object], None],
    ) -> None:
        self._set_busy(False)
        if error is not None:
            messagebox.showerror("Error", str(error))
            return
        if result is not None:
            on_success(result)

    def _add_slider(
        self,
        parent: ctk.CTkFrame,
        label: str,
        attr: str,
        low: int,
        high: int,
        default: int,
        is_float: bool = False,
    ) -> None:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(padx=16, pady=2, fill="x")
        lbl = ctk.CTkLabel(frame, text=label, width=140, anchor="w")
        lbl.pack(side="left")
        val_lbl = ctk.CTkLabel(frame, text=str(default), width=50)
        val_lbl.pack(side="right")

        def on_change(v: float) -> None:
            if is_float:
                setattr(self, f"_{attr}", round(v, 1))
                val_lbl.configure(text=f"{v:.1f}")
            else:
                setattr(self, f"_{attr}", int(v))
                val_lbl.configure(text=str(int(v)))

        slider = ctk.CTkSlider(
            parent,
            from_=low,
            to=high,
            number_of_steps=(high - low) if not is_float else (high - low) * 2,
            command=on_change,
        )
        slider.set(default)
        slider.pack(padx=16, fill="x")
        on_change(default)
        setattr(self, f"_slider_{attr}", slider)

    def _on_mode_change(self) -> None:
        if self._busy:
            return
        is_image = self.mode_var.get() == "image"
        state_text = "disabled" if is_image else "normal"
        state_img = "normal" if is_image else "disabled"
        self.text_box.configure(state=state_text)
        self.btn_load_img.configure(state=state_img)

    def _get_config(self) -> SpectroConfig:
        return SpectroConfig(
            width=getattr(self, "_width", 900),
            height=getattr(self, "_height", 450),
            font_size=getattr(self, "_font_size", 48),
            duration_sec=getattr(self, "_duration", 12.0),
            freq_min=getattr(self, "_freq_min", 400),
            freq_max=getattr(self, "_freq_max", 9000),
            invert=self.invert_var.get(),
        )

    def _build_source_image(self) -> Image.Image | None:
        cfg = self._get_config()
        if self.mode_var.get() == "text":
            text = self.text_box.get("1.0", "end").strip()
            if not text:
                return None
            return text_to_image(text, cfg)
        if self._source_image is None:
            return None
        return prepare_image(self._source_image, cfg)

    def _worker_generate_audio(
        self, progress: Callable[[float, str], None]
    ) -> tuple[Image.Image, object, int]:
        progress(0.02, "Preparing image…")
        src = self._build_source_image()
        if src is None:
            raise ValueError("Could not build source. Enter text or select an image.")
        cfg = self._get_config()
        audio, sr = image_to_audio(src, cfg, on_progress=progress)
        return src, audio, sr

    def _worker_preview(
        self, progress: Callable[[float, str], None]
    ) -> tuple[Image.Image, bytes]:
        src, audio, sr = self._worker_generate_audio(progress)

        progress(0.96, "Rendering spectrogram…")
        cfg = self._get_config()
        png_bytes = render_spectrogram_preview(audio, sr, n_fft=cfg.n_fft)
        progress(1.0, "Done")
        return src, png_bytes

    def _pick_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            cfg = self._get_config()
            self._source_image = load_image_from_path(path, cfg)
            messagebox.showinfo("OK", f"Image loaded:\n{Path(path).name}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _show_image_on_label(
        self, img: Image.Image, label: ctk.CTkLabel, attr: str, max_size: tuple[int, int]
    ) -> None:
        copy = img.copy()
        copy.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(copy)
        label.configure(image=photo, text="")
        setattr(self, attr, photo)

    def _preview(self) -> None:
        def on_success(result: object) -> None:
            src, png_bytes = result
            self._show_image_on_label(src, self.lbl_source, "_preview_photo", (900, 380))
            spec_img = Image.open(io.BytesIO(png_bytes))
            self._show_image_on_label(
                spec_img, self.lbl_spectro, "_spectro_photo", (900, 380)
            )

        self._run_async(
            self._worker_preview,
            on_success,
            title="Previewing spectrogram",
        )

    def _export_wav(self) -> None:
        def on_success(result: object) -> None:
            _, audio, sr = result
            path = filedialog.asksaveasfilename(
                title="Save WAV",
                defaultextension=".wav",
                filetypes=[("WAV audio", "*.wav")],
                initialfile="challenge.wav",
            )
            if not path:
                return
            try:
                save_wav(audio, sr, path)
                messagebox.showinfo("Saved", f"Audio file saved:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        self._run_async(
            self._worker_generate_audio,
            on_success,
            title="Generating WAV",
        )

    def _export_png(self) -> None:
        src = self._build_source_image()
        if src is None:
            messagebox.showwarning(
                "Warning", "Could not build source. Enter text or select an image."
            )
            return
        path = filedialog.asksaveasfilename(
            title="Save source PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile="source_flag.png",
        )
        if not path:
            return
        try:
            src.save(path)
            messagebox.showinfo("Saved", f"Source image saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main() -> None:
    app = SpectroFlagApp()
    app.mainloop()


if __name__ == "__main__":
    main()
