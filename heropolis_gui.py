# -*- coding: utf-8 -*-
"""
Legends of Heropolis DX - Visual Layout Planner (Tkinter)

Features:
- Tile grid (square tiles)
- 4x4 chunk visualization + rule: any footprint >1x1 must fit in a single 4x4 chunk
- Manual placement + drag-move (snapped to tiles)
- Block tiles (red) by toggling Block Mode and clicking tiles
- Auto optimizer that maximizes total % adjacency bonuses ("Spots")
"""
from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from core import (
    Placement,
    load_spots,
    build_pair_bonus,
    score_layout,
    is_valid_position,
    optimize_layout,
)

APP_TITLE = "Legends of Heropolis DX - Layout Planner"

DEFAULT_TILE_PX = 28

def _resource_path(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, *parts)

def load_inventory(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        inv = json.load(f)
    # normalize
    inv.setdefault("map", {"w": 12, "h": 12})
    inv.setdefault("chunk_size", 4)
    inv.setdefault("items", {})
    for name, info in inv["items"].items():
        info.setdefault("category", "Buildings")
        info.setdefault("count", 0)
        info.setdefault("w", None)
        info.setdefault("h", None)
    return inv


def load_items_catalog(path: str) -> dict:
    """Loads the shipped item catalog (name -> defaults)."""
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        catalog = {}
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict) and row.get("name"):
                    catalog[row["name"]] = row
        elif isinstance(data, dict):
            # allow {name: {..}}
            for k, v in data.items():
                if isinstance(v, dict):
                    catalog[k] = {"name": k, **v}
        return catalog
    except Exception:
        return {}


def merge_inventory(inv: dict, catalog: dict) -> bool:
    """Ensures inv['items'] contains every catalog item, without overwriting user values."""
    changed = False
    items = inv.setdefault("items", {})
    for name, defaults in (catalog or {}).items():
        if name not in items:
            items[name] = {
                "category": defaults.get("category", "Buildings"),
                "count": 0,
                "w": defaults.get("w", None),
                "h": defaults.get("h", None),
            }
            changed = True
        else:
            info = items[name]
            if "category" not in info and defaults.get("category"):
                info["category"] = defaults["category"]
                changed = True
            if "count" not in info:
                info["count"] = 0
                changed = True
            if "w" not in info:
                info["w"] = defaults.get("w", None)
                changed = True
            if "h" not in info:
                info["h"] = defaults.get("h", None)
                changed = True
    return changed

def save_inventory(path: str, inv: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(inv, f, ensure_ascii=False, indent=2)

def load_layout(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_layout(path: str, layout: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)

class ScrollFrame(ttk.Frame):
    """A small scrollable frame helper (Canvas + inner frame)."""
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.inner = ttk.Frame(self.canvas)

        self.inner_id = self.canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # mousewheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_inner_configure(self, _evt):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, evt):
        self.canvas.itemconfig(self.inner_id, width=evt.width)

    def _on_mousewheel(self, evt):
        # Windows: delta is 120 steps
        self.canvas.yview_scroll(int(-1*(evt.delta/120)), "units")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)

        self.inventory_path = _resource_path("inventory.json")
        self.spots_path = _resource_path("data", "spots_data.json")
        self.catalog_path = _resource_path("data", "items_catalog.json")

        self.inv = load_inventory(self.inventory_path)
        catalog = load_items_catalog(self.catalog_path)
        if merge_inventory(self.inv, catalog):
            save_inventory(self.inventory_path, self.inv)
        self.spots = load_spots(self.spots_path)
        self.pair_bonus = build_pair_bonus(self.spots)

        self.map_w = int(self.inv["map"]["w"])
        self.map_h = int(self.inv["map"]["h"])
        self.chunk_size = int(self.inv.get("chunk_size", 4))

        self.tile_px = DEFAULT_TILE_PX

        self.blocked = set()      # set[(x,y)]
        self.placements = []      # list[Placement]

        self.selected_item_name = None
        self.block_mode = tk.BooleanVar(value=False)

        # optimizer settings
        self.opt_quality = tk.StringVar(value="Normal")
        self.opt_random = tk.BooleanVar(value=False)
        self.opt_seed = tk.StringVar(value="0")

        # drag state
        self.drag_idx = None
        self.drag_start = None
        self.drag_preview = None

        self._build_ui()
        self._redraw()
        self._update_score()

    # ---------------- UI ----------------

    def _build_ui(self):
        self.geometry("1180x720")

        top = ttk.Frame(self)
        top.pack(fill="both", expand=True)

        left = ttk.Frame(top)
        left.pack(side="left", fill="y", padx=8, pady=8)

        right = ttk.Frame(top)
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        # ---- Left: controls ----
        hdr = ttk.Label(left, text="Inventário", font=("Segoe UI", 12, "bold"))
        hdr.pack(anchor="w", pady=(0,6))

        size_row = ttk.Frame(left)
        size_row.pack(fill="x", pady=4)
        ttk.Label(size_row, text="Mapa (W x H):").pack(side="left")
        self.map_w_var = tk.StringVar(value=str(self.map_w))
        self.map_h_var = tk.StringVar(value=str(self.map_h))
        w_ent = ttk.Entry(size_row, width=5, textvariable=self.map_w_var)
        h_ent = ttk.Entry(size_row, width=5, textvariable=self.map_h_var)
        w_ent.pack(side="left", padx=(6,2))
        ttk.Label(size_row, text="x").pack(side="left")
        h_ent.pack(side="left", padx=(2,6))
        ttk.Button(size_row, text="Aplicar", command=self.on_apply_map_size).pack(side="left")

        mode_row = ttk.Frame(left)
        mode_row.pack(fill="x", pady=(10,4))
        ttk.Checkbutton(mode_row, text="Block Mode (clicar tiles vermelhos)", variable=self.block_mode, command=self._redraw).pack(side="left")

        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", pady=6)
        ttk.Button(btn_row, text="Limpar mapa", command=self.on_clear_map).pack(side="left")
        ttk.Button(btn_row, text="Remover bloqueios", command=self.on_clear_blocks).pack(side="left", padx=6)

        opt_row = ttk.Frame(left)
        opt_row.pack(fill="x", pady=(10,4))
        ttk.Button(opt_row, text="Otimizar (meta)", command=self.on_optimize).pack(side="left")
        ttk.Button(opt_row, text="Melhorar (+10 tentativas)", command=self.on_improve).pack(side="left", padx=(8,0))
        self.progress = ttk.Progressbar(opt_row, length=140, mode="determinate")
        self.progress.pack(side="left", padx=8)

        opt2_row = ttk.Frame(left)
        opt2_row.pack(fill="x", pady=(0,6))

        # Reserved free space (deixa espaço pra minas/fazendas/etc.)
        self.reserve_chunks_var = tk.StringVar(value="0")
        self.reserve_2x2_var = tk.StringVar(value="0")
        self.reserve_1x1_var = tk.StringVar(value="0")

        reserve_row = ttk.LabelFrame(left, text="Reservar espaço livre")
        reserve_row.pack(fill="x", padx=6, pady=(0,6))

        ttk.Label(reserve_row, text="Chunks 4x4 vazios:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(reserve_row, width=6, textvariable=self.reserve_chunks_var).grid(row=0, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(reserve_row, text="Espaços 2x2 livres:").grid(row=0, column=2, sticky="w", padx=10, pady=2)
        ttk.Entry(reserve_row, width=6, textvariable=self.reserve_2x2_var).grid(row=0, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(reserve_row, text="Tiles 1x1 livres:").grid(row=0, column=4, sticky="w", padx=10, pady=2)
        ttk.Entry(reserve_row, width=6, textvariable=self.reserve_1x1_var).grid(row=0, column=5, sticky="w", padx=4, pady=2)
        ttk.Label(opt2_row, text="Qualidade:").pack(side="left")
        self.quality_cb = ttk.Combobox(opt2_row, textvariable=self.opt_quality, values=["Rápido", "Normal", "Insano"], width=8, state="readonly")
        self.quality_cb.pack(side="left", padx=(4,12))
        ttk.Checkbutton(opt2_row, text="Aleatório", variable=self.opt_random).pack(side="left")
        ttk.Label(opt2_row, text="Seed:").pack(side="left", padx=(12,0))
        ttk.Entry(opt2_row, textvariable=self.opt_seed, width=8).pack(side="left", padx=(4,0))
        ttk.Label(opt2_row, text="(seed vazio = 0 / random)").pack(side="left", padx=(8,0))
        self.score_var = tk.StringVar(value="Total buffs: 0%")
        ttk.Label(left, textvariable=self.score_var, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4,10))

        io_row = ttk.Frame(left)
        io_row.pack(fill="x", pady=(6,2))
        ttk.Button(io_row, text="Salvar inventário", command=self.on_save_inventory).pack(side="left")
        ttk.Button(io_row, text="Carregar inventário", command=self.on_load_inventory).pack(side="left", padx=6)

        io2_row = ttk.Frame(left)
        io2_row.pack(fill="x", pady=(2,10))
        ttk.Button(io2_row, text="Salvar layout", command=self.on_save_layout).pack(side="left")
        ttk.Button(io2_row, text="Carregar layout", command=self.on_load_layout).pack(side="left", padx=6)

        # Notebook for categories
        nb = ttk.Notebook(left)
        nb.pack(fill="both", expand=True)

        self.tab_frames = {}
        for cat in ["Environment", "Buildings", "Facilities"]:
            tab = ttk.Frame(nb)
            nb.add(tab, text=cat)
            sf = ScrollFrame(tab)
            sf.pack(fill="both", expand=True)
            self.tab_frames[cat] = sf

        self.item_widgets = {}  # name -> dict of vars/widgets
        self._populate_inventory_tabs()

        # ---- Right: canvas map ----
        self.canvas = tk.Canvas(right, bg="white", highlightthickness=1, highlightbackground="#aaa")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)  # quick block toggle

        hint = ttk.Label(
            right,
            text="Dicas:  • Clique para colocar  • Arraste para mover  • Block Mode ou botão direito para tile vermelho  • Objetos >1x1 não podem atravessar chunk 4x4",
            foreground="#444"
        )
        hint.pack(anchor="w", pady=(6,0))

    def _populate_inventory_tabs(self):
        # clear existing
        for cat, sf in self.tab_frames.items():
            for child in sf.inner.winfo_children():
                child.destroy()

        # group items by category
        cats = {"Environment": [], "Buildings": [], "Facilities": []}
        for name, info in self.inv["items"].items():
            cats.setdefault(info.get("category", "Buildings"), []).append(name)

        for cat, names in cats.items():
            names.sort(key=lambda s: s.lower())
            sf = self.tab_frames.get(cat)
            if not sf:
                continue
            for row_i, name in enumerate(names):
                info = self.inv["items"][name]
                row = ttk.Frame(sf.inner)
                row.grid(row=row_i, column=0, sticky="ew", padx=4, pady=2)
                row.columnconfigure(0, weight=1)

                # Name
                lbl = ttk.Label(row, text=name)
                lbl.grid(row=0, column=0, sticky="w")

                # Count
                cnt_var = tk.StringVar(value=str(info.get("count", 0)))
                cnt_ent = ttk.Entry(row, width=4, textvariable=cnt_var)
                cnt_ent.grid(row=0, column=1, padx=4)

                # Footprint
                fp_var = tk.StringVar(value=self._fp_to_text(info.get("w"), info.get("h")))
                fp_combo = ttk.Combobox(row, width=7, textvariable=fp_var, state="readonly",
                                        values=["unset", "1x1", "2x2", "4x4"])
                fp_combo.grid(row=0, column=2, padx=4)

                # Select
                sel_btn = ttk.Button(row, text="Selecionar", command=lambda n=name: self.select_item(n))
                sel_btn.grid(row=0, column=3, padx=4)

                # bind changes
                cnt_ent.bind("<FocusOut>", lambda _e, n=name: self._sync_item_from_widgets(n))
                cnt_ent.bind("<Return>", lambda _e, n=name: self._sync_item_from_widgets(n))
                fp_combo.bind("<<ComboboxSelected>>", lambda _e, n=name: self._sync_item_from_widgets(n))

                self.item_widgets[name] = {"cnt_var": cnt_var, "fp_var": fp_var}

    def _fp_to_text(self, w, h) -> str:
        if w is None or h is None:
            return "unset"
        return f"{int(w)}x{int(h)}"

    def _text_to_fp(self, s: str):
        s = (s or "").strip().lower()
        if s == "unset" or s == "":
            return None, None
        try:
            a, b = s.split("x")
            return int(a), int(b)
        except Exception:
            return None, None

    # -------------- State sync --------------

    def _sync_item_from_widgets(self, name: str):
        wdg = self.item_widgets.get(name)
        if not wdg:
            return
        info = self.inv["items"][name]

        # count
        try:
            cnt = int(wdg["cnt_var"].get().strip() or "0")
            if cnt < 0: cnt = 0
        except Exception:
            cnt = 0
        info["count"] = cnt

        # footprint
        w, h = self._text_to_fp(wdg["fp_var"].get())
        info["w"] = w
        info["h"] = h

        # if selected item became unset footprint, keep selection but warn via status
        self._update_score()

    def select_item(self, name: str):
        self._sync_item_from_widgets(name)
        self.selected_item_name = name
        self._redraw()

    # -------------- Map operations --------------

    def on_apply_map_size(self):
        try:
            w = int(self.map_w_var.get().strip())
            h = int(self.map_h_var.get().strip())
        except Exception:
            messagebox.showerror("Erro", "Tamanho do mapa inválido.")
            return
        if w <= 0 or h <= 0 or w > 80 or h > 80:
            messagebox.showerror("Erro", "Use um tamanho entre 1 e 80.")
            return

        # When resizing, drop anything out-of-bounds.
        self.map_w, self.map_h = w, h
        self.inv["map"]["w"] = w
        self.inv["map"]["h"] = h
        self.blocked = {(x,y) for (x,y) in self.blocked if x < w and y < h}
        self.placements = [p for p in self.placements if p.x + p.w <= w and p.y + p.h <= h]
        self._redraw()
        self._update_score()

    def on_clear_map(self):
        if messagebox.askyesno("Confirmar", "Remover todas as construções do mapa?"):
            self.placements = []
            self._redraw()
            self._update_score()

    def on_clear_blocks(self):
        self.blocked = set()
        self._redraw()

    # -------------- Save/Load --------------

    def on_save_inventory(self):
        # sync all widgets first
        for n in list(self.item_widgets.keys()):
            self._sync_item_from_widgets(n)
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path:
            return
        try:
            save_inventory(path, self.inv)
            messagebox.showinfo("OK", "Inventário salvo.")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao salvar: {e}")

    def on_load_inventory(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not path:
            return
        try:
            self.inv = load_inventory(path)
            self.map_w = int(self.inv["map"]["w"])
            self.map_h = int(self.inv["map"]["h"])
            self.chunk_size = int(self.inv.get("chunk_size", 4))
            self.map_w_var.set(str(self.map_w))
            self.map_h_var.set(str(self.map_h))
            self._populate_inventory_tabs()
            self._redraw()
            self._update_score()
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar: {e}")

    def on_save_layout(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path:
            return
        layout = {
            "map": {"w": self.map_w, "h": self.map_h},
            "chunk_size": self.chunk_size,
            "blocked": sorted([{"x": x, "y": y} for x,y in self.blocked], key=lambda d:(d["y"],d["x"])),
            "placements": [
                {"name": p.name, "x": p.x, "y": p.y, "w": p.w, "h": p.h}
                for p in self.placements
            ]
        }
        try:
            save_layout(path, layout)
            messagebox.showinfo("OK", "Layout salvo.")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao salvar: {e}")

    def on_load_layout(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not path:
            return
        try:
            layout = load_layout(path)
            self.map_w = int(layout["map"]["w"])
            self.map_h = int(layout["map"]["h"])
            self.chunk_size = int(layout.get("chunk_size", 4))
            self.map_w_var.set(str(self.map_w))
            self.map_h_var.set(str(self.map_h))
            self.blocked = set((int(d["x"]), int(d["y"])) for d in layout.get("blocked", []))
            self.placements = [Placement(p["name"], int(p["x"]), int(p["y"]), int(p["w"]), int(p["h"])) for p in layout.get("placements", [])]
            self._redraw()
            self._update_score()
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar: {e}")

    # -------------- Optimizer --------------

    def on_optimize(self):
        # sync all items from widgets
        for n in list(self.item_widgets.keys()):
            self._sync_item_from_widgets(n)

        # validate footprints
        missing = [n for n,info in self.inv["items"].items() if int(info.get("count",0) or 0) > 0 and (info.get("w") is None or info.get("h") is None)]
        if missing:
            messagebox.showwarning("Faltando tamanhos", "Defina o tamanho (1x1/2x2/4x4...) para:\n- " + "\n- ".join(sorted(missing)[:30]) + ("\n..." if len(missing) > 30 else ""))
            return

        if not messagebox.askyesno("Confirmar", "O optimizer vai montar um layout do zero usando seu inventário. Continuar?"):
            return

        # Heavy presets: higher restarts + steps increases the chance of reaching the global optimum.
        presets = {
            "Rápido": (6, 1500),
            "Normal": (12, 3500),
            "Insano": (30, 12000),
            "Absoluto": (70, 28000),
        }
        quality = (self.opt_quality.get() or "Normal").strip()
        restarts, steps = presets.get(quality, (12, 3500))

        seed_txt = (self.opt_seed.get() or "").strip()
        if seed_txt:
            try:
                seed = int(seed_txt)
            except ValueError:
                messagebox.showerror("Seed inválida", "Seed deve ser um número inteiro (ex: 0, 123, -1).")
                return
        else:
            seed = None if self.opt_random.get() else 0
        # Reserved free space inputs
        try:
            reserve_chunks = int((self.reserve_chunks_var.get() or "0").strip())
            reserve_2x2 = int((self.reserve_2x2_var.get() or "0").strip())
            reserve_1x1 = int((self.reserve_1x1_var.get() or "0").strip())
        except ValueError:
            messagebox.showerror("Reserva inválida", "Reserva deve ser número inteiro (ex: 0, 10).")
            return

        if reserve_chunks < 0 or reserve_2x2 < 0 or reserve_1x1 < 0:
            messagebox.showerror("Reserva inválida", "Reserva não pode ser negativa.")
            return


        self.progress.configure(value=0, maximum=100)
        self.progress.update_idletasks()

        def run_bg():
            try:
                def on_prog(r, t, best_score):
                    # rough progress
                    # r: 0..restarts-1 ; t: 0..steps
                    pct = (r + min(1.0, t/float(steps))) / float(restarts) * 100.0
                    self._set_progress_safe(pct, best_score)

                placements, score, warnings = optimize_layout(
                    inventory_items=self.inv["items"],
                    map_w=self.map_w,
                    map_h=self.map_h,
                    blocked_tiles=self.blocked,
                    spots=self.spots,
                    chunk_size=self.chunk_size,
                    restarts=restarts,
                    steps=steps,
                    seed=seed,
                    reserve_empty_chunks=reserve_chunks,
                    reserve_free_2x2=reserve_2x2,
                    reserve_free_1x1=reserve_1x1,
                    on_progress=on_prog
                )
                self.after(0, lambda: self._apply_optimized_result(placements, score, warnings))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Erro", f"Optimizer falhou: {e}"))

        threading.Thread(target=run_bg, daemon=True).start()

    def on_improve(self):
        """Continue optimizing starting from the CURRENT layout (10 independent attempts)."""
        # sync all items from widgets
        for n in list(self.item_widgets.keys()):
            self._sync_item_from_widgets(n)

        if not getattr(self, "placements", None):
            messagebox.showinfo("Nada para melhorar", "Monte um layout primeiro (manual ou 'Otimizar (meta)').")
            return

        # validate footprints (only for items that exist in inventory and are meant to be placeable)
        missing = [
            n
            for n, info in self.inv["items"].items()
            if int(info.get("count", 0) or 0) > 0 and (info.get("w") is None or info.get("h") is None)
        ]
        if missing:
            messagebox.showwarning(
                "Faltando tamanhos",
                "Defina o tamanho (1x1/2x2/4x4...) para:\n- "
                + "\n- ".join(sorted(missing)[:30])
                + ("\n..." if len(missing) > 30 else ""),
            )
            return

        attempts = 10
        presets = {
            "Rápido": 900,
            "Normal": 2500,
            "Insano": 9000,
            "Absoluto": 18000,
        }
        quality = (self.opt_quality.get() or "Normal").strip()
        steps = int(presets.get(quality, 2500))

        seed_txt = (self.opt_seed.get() or "").strip()
        if seed_txt:
            try:
                seed = int(seed_txt)
            except ValueError:
                messagebox.showerror("Seed inválida", "Seed deve ser um número inteiro (ex: 0, 123, -1).")
                return
        else:
            seed = None if self.opt_random.get() else 0

        # Reserved free space inputs
        try:
            reserve_chunks = int((self.reserve_chunks_var.get() or "0").strip())
            reserve_2x2 = int((self.reserve_2x2_var.get() or "0").strip())
            reserve_1x1 = int((self.reserve_1x1_var.get() or "0").strip())
        except ValueError:
            messagebox.showerror("Reserva inválida", "Reserva deve ser número inteiro (ex: 0, 10).")
            return

        if reserve_chunks < 0 or reserve_2x2 < 0 or reserve_1x1 < 0:
            messagebox.showerror("Reserva inválida", "Reserva não pode ser negativa.")
            return

        self.progress.configure(value=0, maximum=100)
        self.progress.update_idletasks()

        def run_bg():
            try:
                def on_prog(a, t, best_score):
                    # a: 0..attempts-1 ; t: 0..steps
                    pct = (a + min(1.0, t / float(steps))) / float(attempts) * 100.0
                    self._set_progress_safe(pct, best_score)

                placements, score = improve_layout(
                    inventory_items=self.inv["items"],
                    initial_placements=self.placements,
                    map_w=self.map_w,
                    map_h=self.map_h,
                    blocked_tiles=self.blocked,
                    spots=self.spots,
                    chunk_size=self.chunk_size,
                    attempts=attempts,
                    steps_per_attempt=steps,
                    seed=seed,
                    reserve_empty_chunks=reserve_chunks,
                    reserve_free_2x2=reserve_2x2,
                    reserve_free_1x1=reserve_1x1,
                    on_progress=on_prog,
                )
                self.after(0, lambda: self._apply_optimized_result(placements, score, []))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Erro", f"Melhoria falhou: {e}"))

        threading.Thread(target=run_bg, daemon=True).start()

    def _set_progress_safe(self, pct: float, best_score: float):
        def _update():
            self.progress.configure(value=max(0, min(100, pct)))
            self.score_var.set(f"Total buffs: {best_score:.1f}%")
        self.after(0, _update)

    def _apply_optimized_result(self, placements, score, warnings):
        self.placements = list(placements)
        self.progress.configure(value=100)
        self._redraw()
        self._update_score()
        if warnings:
            messagebox.showinfo("Aviso", "Itens ignorados (sem tamanho definido):\n- " + "\n- ".join(sorted(warnings)))

    # -------------- Canvas helpers --------------

    def _xy_to_tile(self, px: int, py: int) -> tuple[int,int]:
        x = px // self.tile_px
        y = py // self.tile_px
        return int(x), int(y)

    def _snap_to_valid(self, tx: int, ty: int, w: int, h: int) -> tuple[int,int]:
        """Snap (tx,ty) so the footprint stays inside the map and inside a single 4x4 chunk."""
        # Clamp inside the map first
        tx = max(0, min(int(tx), self.map_w - w))
        ty = max(0, min(int(ty), self.map_h - h))

        cs = self.chunk_size
        # If it would cross the chunk border, pull it back just enough to fit
        ox = (tx % cs) + w - cs
        if ox > 0:
            tx -= ox
        oy = (ty % cs) + h - cs
        if oy > 0:
            ty -= oy

        # Clamp again after snapping
        tx = max(0, min(int(tx), self.map_w - w))
        ty = max(0, min(int(ty), self.map_h - h))
        return tx, ty


    def _tile_to_xy(self, tx: int, ty: int) -> tuple[int,int]:
        return tx * self.tile_px, ty * self.tile_px

    def _find_placement_at(self, tx: int, ty: int):
        # returns index or None
        for i, p in enumerate(self.placements):
            if tx >= p.x and tx < p.x + p.w and ty >= p.y and ty < p.y + p.h:
                return i
        return None

    def on_canvas_right_click(self, evt):
        # quick toggle a single blocked tile (works even without Block Mode)
        tx, ty = self._xy_to_tile(evt.x, evt.y)
        if tx < 0 or ty < 0 or tx >= self.map_w or ty >= self.map_h:
            return
        if (tx, ty) in self.blocked:
            self.blocked.remove((tx, ty))
        else:
            # cannot block under an existing building
            if self._find_placement_at(tx, ty) is None:
                self.blocked.add((tx, ty))
        self._redraw()

    def on_canvas_click(self, evt):
        tx, ty = self._xy_to_tile(evt.x, evt.y)
        if tx < 0 or ty < 0 or tx >= self.map_w or ty >= self.map_h:
            return

        if self.block_mode.get():
            # toggle blocked tile if empty
            if self._find_placement_at(tx, ty) is None:
                if (tx, ty) in self.blocked:
                    self.blocked.remove((tx, ty))
                else:
                    self.blocked.add((tx, ty))
                self._redraw()
            return

        # try pick an existing placement to drag
        idx = self._find_placement_at(tx, ty)
        if idx is not None:
            self.drag_idx = idx
            p = self.placements[idx]
            self.drag_start = (p.x, p.y)
            return

        # place new
        if not self.selected_item_name:
            return
        info = self.inv["items"].get(self.selected_item_name, {})
        w = info.get("w"); h = info.get("h")
        if w is None or h is None:
            messagebox.showwarning("Faltando tamanho", f"Defina o tamanho para: {self.selected_item_name}")
            return
        w = int(w); h = int(h)
        tx, ty = self._snap_to_valid(tx, ty, w, h)
        if not is_valid_position(self.placements, None, tx, ty, w, h, self.map_w, self.map_h, self.blocked, self.chunk_size):
            return
        self.placements.append(Placement(self.selected_item_name, tx, ty, w, h))
        self._redraw()
        self._update_score()

    def on_canvas_drag(self, evt):
        if self.drag_idx is None:
            return
        tx, ty = self._xy_to_tile(evt.x, evt.y)
        p = self.placements[self.drag_idx]
        tx, ty = self._snap_to_valid(tx, ty, p.w, p.h)
        valid = is_valid_position(self.placements, self.drag_idx, tx, ty, p.w, p.h, self.map_w, self.map_h, self.blocked, self.chunk_size)
        # draw a preview rectangle
        self._redraw()
        x0, y0 = self._tile_to_xy(tx, ty)
        x1, y1 = self._tile_to_xy(tx + p.w, ty + p.h)
        outline = "#00aa00" if valid else "#cc0000"
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=outline, width=3)

    def on_canvas_release(self, evt):
        if self.drag_idx is None:
            return
        tx, ty = self._xy_to_tile(evt.x, evt.y)
        p = self.placements[self.drag_idx]
        tx, ty = self._snap_to_valid(tx, ty, p.w, p.h)
        if is_valid_position(self.placements, self.drag_idx, tx, ty, p.w, p.h, self.map_w, self.map_h, self.blocked, self.chunk_size):
            self.placements[self.drag_idx] = Placement(p.name, tx, ty, p.w, p.h)
            self._update_score()
        else:
            # revert
            self.placements[self.drag_idx] = Placement(p.name, self.drag_start[0], self.drag_start[1], p.w, p.h)
        self.drag_idx = None
        self.drag_start = None
        self._redraw()

    # -------------- Drawing & score --------------

    def _update_score(self):
        val = score_layout(self.placements, self.pair_bonus) if self.placements else 0.0
        self.score_var.set(f"Total buffs: {val:.1f}%")

    def _redraw(self):
        self.canvas.delete("all")
        w_px = self.map_w * self.tile_px
        h_px = self.map_h * self.tile_px
        self.canvas.configure(scrollregion=(0,0,w_px,h_px))

        # grid lines
        for x in range(self.map_w + 1):
            x0 = x * self.tile_px
            width = 3 if (x % self.chunk_size == 0) else 1
            color = "#bbb" if width == 1 else "#888"
            self.canvas.create_line(x0, 0, x0, h_px, fill=color, width=width)
        for y in range(self.map_h + 1):
            y0 = y * self.tile_px
            width = 3 if (y % self.chunk_size == 0) else 1
            color = "#bbb" if width == 1 else "#888"
            self.canvas.create_line(0, y0, w_px, y0, fill=color, width=width)

        # blocked tiles
        for (bx, by) in self.blocked:
            x0, y0 = bx * self.tile_px, by * self.tile_px
            self.canvas.create_rectangle(x0, y0, x0 + self.tile_px, y0 + self.tile_px, fill="#ff3b3b", outline="")

        # placements
        for p in self.placements:
            x0, y0 = p.x * self.tile_px, p.y * self.tile_px
            x1, y1 = (p.x + p.w) * self.tile_px, (p.y + p.h) * self.tile_px

            # Light tint by category (just for readability)
            cat = self.inv["items"].get(p.name, {}).get("category", "Buildings")
            fill = "#cfe8ff" if cat == "Environment" else ("#d9f7d9" if cat == "Facilities" else "#f4d3ff")

            self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#333", width=2)
            # label in center
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            self.canvas.create_text(cx, cy, text=p.name, font=("Segoe UI", 10, "bold"))

        # selection highlight (optional)
        if self.selected_item_name:
            # show in title bar
            self.title(f"{APP_TITLE}  |  Selecionado: {self.selected_item_name}")

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
