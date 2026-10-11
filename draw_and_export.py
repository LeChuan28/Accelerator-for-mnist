"""
draw_and_export.py — Vẽ chữ số bằng chuột, xuất ra file .mif để nạp vào
board DE2 qua Quartus In-System Memory Content Editor (dùng lại cáp
USB-Blaster sẵn có, KHÔNG cần cáp UART/RS-232).

Cài thư viện cần thiết trước khi chạy:
    pip install pillow numpy

Cách dùng:
    1. Chạy: python draw_and_export.py
    2. Vẽ chữ số bằng chuột trái trên canvas (nền đen, nét trắng)
    3. Bấm "Xuất file .mif" -> lưu file draw_image.mif
    4. Mở Quartus -> Tools -> In-System Memory Content Editor
       -> Import Data from File -> chọn draw_image.mif
       -> chuột phải vào bảng dữ liệu -> Write Data to In-System Memory
    5. Trên board: gạt SW[4]=1 (chế độ vẽ tay) -> bấm KEY[1] để chạy
       (chế độ này không tự động trigger như UART, cần bấm nút thủ công)
"""

import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import numpy as np

# Tương thích Pillow cũ (<10, dùng Image.LANCZOS) và mới (>=10, dùng
# Image.Resampling.LANCZOS) — tránh lỗi AttributeError khi resize.
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS

CANVAS_SIZE = 280       # canvas hiển thị 280x280 (10x phóng to so với 28x28 thật)
BRUSH_RADIUS = 10
IMAGE_SIZE = 784         # 28 x 28

class DrawApp:
    def __init__(self, root):
        self.root = root
        root.title("Vẽ chữ số -> xuất .mif nạp qua In-System Memory Content Editor")

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.pack()

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_xy = None
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "last_xy", None))

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="Xóa", command=self.clear).pack(side="left", expand=True, fill="x")
        tk.Button(btn_frame, text="Xem trước 28x28", command=self.preview).pack(side="left", expand=True, fill="x")
        tk.Button(btn_frame, text="Xuất file .mif", command=self.export_mif).pack(side="left", expand=True, fill="x")

        self.status = tk.Label(root, text="Vẽ 1 chữ số rồi bấm 'Xuất file .mif'", wraplength=400, justify="left")
        self.status.pack(fill="x")

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_xy is not None:
            self.canvas.create_line(self.last_xy[0], self.last_xy[1], x, y,
                                     fill="white", width=BRUSH_RADIUS*1.5,
                                     capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_xy, (x, y)], fill=255, width=BRUSH_RADIUS*2)
        else:
            r = BRUSH_RADIUS
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
            self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
        self.last_xy = (x, y)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.status.config(text="Đã xóa canvas.")

    def get_28x28_array(self):
        small = self.image.resize((28, 28), RESAMPLE_LANCZOS)
        arr = np.array(small, dtype=np.uint8)                          # 0-255
        arr7 = (arr.astype(np.uint16) * 127 // 255).astype(np.uint8)   # nén về 0-127
        return arr7

    def preview(self):
        try:
            arr7 = self.get_28x28_array()
            if arr7.max() == 0:
                messagebox.showwarning("Canvas trống", "Bạn chưa vẽ gì cả — hãy vẽ 1 chữ số trước.")
                return
            top = tk.Toplevel(self.root)
            top.title("Xem trước 28x28 (đã nén về thang 0-127)")
            pv_canvas = tk.Canvas(top, width=280, height=280, bg="black")
            pv_canvas.pack()
            big = Image.fromarray((arr7.astype(np.uint16) * 255 // 127).astype(np.uint8)) \
                       .resize((280, 280), Image.NEAREST)
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(big)
            pv_canvas.image = photo   # giữ tham chiếu tránh bị garbage-collect
            pv_canvas.create_image(0, 0, anchor="nw", image=photo)
            self.status.config(text=f"Preview OK — giá trị pixel cao nhất: {arr7.max()}")
        except Exception:
            err = traceback.format_exc()
            self.status.config(text="LỖI khi preview:\n" + err)
            messagebox.showerror("Lỗi preview", err)

    def export_mif(self):
        try:
            arr7 = self.get_28x28_array()

            if arr7.max() == 0:
                messagebox.showwarning("Canvas trống", "Bạn chưa vẽ gì cả — hãy vẽ 1 chữ số trước khi xuất.")
                return

            flat = arr7.flatten()   # 784 giá trị, row-major, khớp img_idx trong RTL
            assert len(flat) == IMAGE_SIZE, f"Số pixel không đúng: {len(flat)}"

            path = filedialog.asksaveasfilename(
                defaultextension=".mif",
                initialfile="draw_image.mif",
                filetypes=[("Memory Initialization File", "*.mif"), ("All files", "*.*")]
            )
            if not path:
                self.status.config(text="Đã hủy — không có file nào được lưu.")
                return

            lines = []
            lines.append("-- Anh ve tay, xuat boi draw_and_export.py\n")
            lines.append("WIDTH=8;\n")
            lines.append(f"DEPTH={IMAGE_SIZE};\n\n")
            lines.append("ADDRESS_RADIX=UNS;\n")
            lines.append("DATA_RADIX=HEX;\n\n")
            lines.append("CONTENT BEGIN\n")
            for addr, val in enumerate(flat):
                lines.append(f"\t{addr} : {int(val):02X};\n")
            lines.append("END;\n")

            with open(path, "w", encoding="ascii") as f:
                f.writelines(lines)

            # Kiểm tra lại ngay sau khi ghi — đảm bảo file không rỗng
            import os
            size = os.path.getsize(path)
            if size == 0:
                raise RuntimeError("File ghi ra có kích thước 0 byte — ghi thất bại!")

            self.status.config(
                text=f"Đã lưu {path} ({size} byte, {IMAGE_SIZE} pixel, "
                     f"giá trị cao nhất={arr7.max()}). Mở In-System Memory Content Editor để nạp."
            )

        except Exception:
            err = traceback.format_exc()
            self.status.config(text="LỖI khi xuất file:\n" + err)
            messagebox.showerror("Lỗi xuất file .mif", err)


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()