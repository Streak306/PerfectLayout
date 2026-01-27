"""Tkinter GUI for PerfectLayout."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import List

from .optimizer import ItemSpec, LayoutInput, optimize_layout


DEFAULT_ITEMS = """Sofa,2.2,0.9,1
Mesa,1.2,0.8,1
Rack,1.6,0.5,1
Tapete,1.8,1.2,1
Poltrona,0.8,0.8,2
"""

COLORS = [
    "#ffa69e",
    "#9bf6ff",
    "#bdb2ff",
    "#ffd6a5",
    "#caffbf",
    "#fdffb6",
    "#a0c4ff",
]


class PerfectLayoutApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PerfectLayout AI")
        self.geometry("980x640")
        self.configure(bg="#0f172a")
        self._build_ui()

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg="#0f172a")
        header.pack(fill="x", padx=24, pady=(24, 12))

        title = tk.Label(
            header,
            text="PerfectLayout AI",
            font=("Segoe UI", 20, "bold"),
            fg="#f8fafc",
            bg="#0f172a",
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            header,
            text="Inteligência avançada para encontrar o melhor layout possível.",
            font=("Segoe UI", 11),
            fg="#cbd5f5",
            bg="#0f172a",
        )
        subtitle.pack(anchor="w")

        content = tk.Frame(self, bg="#0f172a")
        content.pack(fill="both", expand=True, padx=24, pady=12)

        controls = tk.Frame(content, bg="#111827", padx=18, pady=18)
        controls.pack(side="left", fill="y")

        canvas_frame = tk.Frame(content, bg="#0f172a")
        canvas_frame.pack(side="right", fill="both", expand=True, padx=(18, 0))

        self.canvas = tk.Canvas(canvas_frame, bg="#0b1220", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        tk.Label(
            controls,
            text="Dimensões do ambiente (m)",
            font=("Segoe UI", 11, "bold"),
            fg="#e2e8f0",
            bg="#111827",
        ).pack(anchor="w")

        self.room_width_var = tk.StringVar(value="6")
        self.room_height_var = tk.StringVar(value="4")

        self._build_entry(controls, "Largura", self.room_width_var)
        self._build_entry(controls, "Comprimento", self.room_height_var)

        tk.Label(
            controls,
            text="Itens (nome, largura, comprimento, quantidade)",
            font=("Segoe UI", 11, "bold"),
            fg="#e2e8f0",
            bg="#111827",
        ).pack(anchor="w", pady=(18, 6))

        self.items_text = tk.Text(
            controls,
            width=30,
            height=12,
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
        )
        self.items_text.pack()
        self.items_text.insert("1.0", DEFAULT_ITEMS)

        tk.Label(
            controls,
            text="Dica: use metragem real para obter precisão máxima.",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#111827",
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=(8, 12))

        self.status_label = tk.Label(
            controls,
            text="Pronto para gerar o layout ideal.",
            font=("Segoe UI", 10),
            fg="#a5b4fc",
            bg="#111827",
            wraplength=260,
            justify="left",
        )
        self.status_label.pack(anchor="w", pady=(4, 12))

        action_button = tk.Button(
            controls,
            text="Gerar layout inteligente",
            font=("Segoe UI", 11, "bold"),
            bg="#6366f1",
            fg="#f8fafc",
            relief="flat",
            padx=12,
            pady=8,
            command=self._run_optimizer,
        )
        action_button.pack(fill="x", pady=(6, 6))

        self._draw_placeholder()

    def _build_entry(self, parent: tk.Frame, label: str, variable: tk.StringVar) -> None:
        frame = tk.Frame(parent, bg="#111827")
        frame.pack(anchor="w", fill="x", pady=4)
        tk.Label(
            frame,
            text=label,
            font=("Segoe UI", 10),
            fg="#cbd5f5",
            bg="#111827",
            width=12,
            anchor="w",
        ).pack(side="left")
        entry = tk.Entry(
            frame,
            textvariable=variable,
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#f8fafc",
            relief="flat",
            width=8,
        )
        entry.pack(side="left", padx=6)

    def _draw_placeholder(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            20,
            20,
            anchor="nw",
            fill="#94a3b8",
            font=("Segoe UI", 12),
            text="Visualização do layout aparecerá aqui.",
        )

    def _run_optimizer(self) -> None:
        try:
            room_width = float(self.room_width_var.get())
            room_height = float(self.room_height_var.get())
        except ValueError:
            messagebox.showerror("Erro", "Informe valores numéricos para o ambiente.")
            return

        try:
            items = self._parse_items()
        except ValueError as exc:
            messagebox.showerror("Erro", str(exc))
            return

        layout_input = LayoutInput(room_width=room_width, room_height=room_height, items=items)
        self.status_label.configure(text="Calculando o layout mais inteligente...")
        self.update_idletasks()

        result = optimize_layout(layout_input)
        self.status_label.configure(
            text=(
                "Layout otimizado com inteligência. "
                f"Pontuação: {result.score:.1f} | Iterações: {result.iterations}"
            )
        )
        self._draw_layout(layout_input, result.placements)

    def _parse_items(self) -> List[ItemSpec]:
        raw = self.items_text.get("1.0", "end").strip()
        if not raw:
            raise ValueError("Adicione ao menos um item.")

        items: List[ItemSpec] = []
        color_index = 0
        for line in raw.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) not in {3, 4}:
                raise ValueError(
                    "Formato inválido. Use: nome, largura, comprimento, quantidade (opcional)."
                )
            name = parts[0]
            width = float(parts[1])
            height = float(parts[2])
            count = int(parts[3]) if len(parts) == 4 else 1
            if width <= 0 or height <= 0 or count <= 0:
                raise ValueError("Largura, comprimento e quantidade devem ser positivos.")
            for index in range(count):
                item_name = f"{name} {index + 1}" if count > 1 else name
                items.append(
                    ItemSpec(
                        name=item_name,
                        width=width,
                        height=height,
                        color=COLORS[color_index % len(COLORS)],
                    )
                )
                color_index += 1
        return items

    def _draw_layout(self, layout: LayoutInput, placements) -> None:
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width() or 600
        canvas_height = self.canvas.winfo_height() or 400

        padding = 24
        scale_x = (canvas_width - padding * 2) / layout.room_width
        scale_y = (canvas_height - padding * 2) / layout.room_height
        scale = min(scale_x, scale_y)

        room_w = layout.room_width * scale
        room_h = layout.room_height * scale
        origin_x = (canvas_width - room_w) / 2
        origin_y = (canvas_height - room_h) / 2

        self.canvas.create_rectangle(
            origin_x,
            origin_y,
            origin_x + room_w,
            origin_y + room_h,
            outline="#94a3b8",
            width=2,
        )

        for item, placement in zip(layout.items, placements):
            width = item.height if placement.rotated else item.width
            height = item.width if placement.rotated else item.height
            x1 = origin_x + placement.x * scale
            y1 = origin_y + placement.y * scale
            x2 = x1 + width * scale
            y2 = y1 + height * scale
            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=item.color,
                outline="#0f172a",
                width=2,
            )
            self.canvas.create_text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                text=item.name,
                fill="#0f172a",
                font=("Segoe UI", 9, "bold"),
            )


if __name__ == "__main__":
    app = PerfectLayoutApp()
    app.mainloop()
