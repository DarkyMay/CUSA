# -*- coding: utf-8 -*-
"""
k_click_width.py —— 在本机（有界面）人工点选裂缝开口宽度，用于 F4：人工图像开口 vs 卡尺
依赖：pip install pillow matplotlib numpy
用法（Windows，在脚本所在目录打开 cmd / PowerShell）：
    python k_click_width.py                      # 默认只做 5 张 0° 图（x-1.jpg）
    python k_click_width.py --all                # 全部 20 张
    python k_click_width.py --dir "G:\\写论文\\...\\实拍图片"   # 换目录

操作：
  · 每个粉笔刻度点量一次宽度：在裂缝开口的一侧边缘左键点一下，再在正对面另一侧点一下（两点连线应垂直于裂缝走向）
  · 按粉笔编号顺序 1 → 2 → 3，每张图点满 3 对后自动保存并跳到下一张
  · 工具栏放大镜 / 平移可随意用；处于放大或平移模式时点击不会被记录（先按工具栏按钮退出，或按 o / p 切换）
  · 快捷键：u 撤销上一次点击   r 重做本张   n 点满前强制跳下一张（不足 3 对也保存）   q 退出（已保存的不丢）
  · 可选校核尺度：按 g 进入"网格模式"，在网格纸上沿同一条格线点两个相距整数格的交点，弹窗输入格数，即得该图局部 mm/px；
    不按 g 则直接使用服务器 F1 拟合得到的 s0
输出：同目录下 click_widths_<时间>.csv 和 click_widths_<时间>.txt（每次点击即写盘）
"""
import argparse, csv, datetime as dt, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

DEFAULT_DIR = r"G:\写论文\新论文\26论文7-大模型+RDD\新的\CUSA\返修\实拍图片"
# 服务器 F1 结果：各裂缝 0° 图的 mm/px（f_field_20261024_122148.txt）
S0 = {1: 0.2051, 2: 0.2102, 3: 0.2004, 4: 0.1990, 5: 0.1963}
CALIPER = {1: (6.2, 17.0, 8.0), 2: (4.6, 2.9, 5.8), 3: (5.3, 6.0, 11.0), 4: (4.8, 5.2, 6.8), 5: (5.6, 7.4, 8.4)}
TILT = {1: [0, 4.7, 9.8, 15.2], 2: [0, 5.7, 10.4, 15.6], 3: [0, 4.8, 10.6, 14.7], 4: [0, 5.5, 10.2, 15.5], 5: [0, 5.3, 10.4, 15.3]}
N_TICKS = 3


class Session:
    def __init__(self, files, out_csv, out_txt):
        self.files, self.i = files, 0
        self.out_csv, self.out_txt = out_csv, out_txt
        self.rows = []
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["file", "crack", "tilt_deg", "tick", "x1", "y1", "x2", "y2",
                                    "width_px", "mmpx_used", "mmpx_source", "width_mm", "caliper_mm", "ratio_img_over_caliper"])
        self.fig, self.ax = plt.subplots(figsize=(9, 12))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load()

    # ---------- 当前图状态 ----------
    def load(self):
        if self.i >= len(self.files):
            self.finish(); return
        p = self.files[self.i]
        self.crack = int(p.stem.split("-")[0]); self.k = int(p.stem.split("-")[1])
        self.tilt = TILT[self.crack][self.k - 1]
        self.img = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
        self.pending = None           # 第一次点击
        self.pairs = []               # [(x1,y1,x2,y2)]
        self.grid_mode, self.grid_pts = False, []
        self.mmpx, self.mmpx_src = S0[self.crack], "F1_s0"
        self.artists = []
        self.ax.clear(); self.ax.imshow(self.img); self.ax.set_axis_off()
        self.title(); self.fig.canvas.draw_idle()

    def title(self):
        c = CALIPER[self.crack]
        nxt = len(self.pairs) + 1
        s = (f"[{self.i+1}/{len(self.files)}] {self.files[self.i].name}  倾角 {self.tilt}°   卡尺: 1={c[0]}  2={c[1]}  3={c[2]} mm\n"
             f"mm/px={self.mmpx:.4f} ({self.mmpx_src})   ")
        if self.grid_mode:
            s += f"【网格模式】沿同一格线点两个交点 ({len(self.grid_pts)}/2)"
        elif nxt <= N_TICKS:
            s += f"请点刻度 {nxt} 的开口两侧边缘（{'第 2 点' if self.pending else '第 1 点'}）"
        else:
            s += "已点满，按 n 下一张"
        self.ax.set_title(s, fontsize=10)

    def draw_pt(self, x, y, color):
        r = max(8, self.img.shape[1] / 300)
        a = self.ax.add_patch(Circle((x, y), r, fill=False, color=color, lw=2)); self.artists.append(a)
        m, = self.ax.plot([x], [y], "+", color=color, ms=10, mew=1.5); self.artists.append(m)

    def draw_pair(self, x1, y1, x2, y2, idx):
        ln, = self.ax.plot([x1, x2], [y1, y2], "-", color="yellow", lw=1.5); self.artists.append(ln)
        t = self.ax.text(x2 + 15, y2, f"{idx}: {np.hypot(x2-x1, y2-y1):.1f}px", color="yellow", fontsize=9,
                         bbox=dict(facecolor="black", alpha=0.5, pad=1)); self.artists.append(t)

    # ---------- 事件 ----------
    def on_click(self, ev):
        if ev.inaxes != self.ax or ev.button != 1:
            print("点击未落在图像上或不是左键，忽略"); return
        tb = getattr(self.fig.canvas, "toolbar", None)
        mode = str(getattr(tb, "mode", "") or "") if tb is not None else ""
        if mode:   # 放大 / 平移模式下不记录
            self.ax.set_title(f"工具栏处于「{mode}」模式，点击不记录：再按一次工具栏按钮退出（或按键 o / p），再点", color="red", fontsize=11)
            self.fig.canvas.draw_idle(); print(f"工具栏模式 {mode}，忽略此次点击"); return
        x, y = float(ev.xdata), float(ev.ydata)
        print(f"点击 ({x:.0f},{y:.0f})  pending={'有' if self.pending else '无'}  已完成 {len(self.pairs)} 对")
        if self.grid_mode:
            self.grid_pts.append((x, y)); self.draw_pt(x, y, "cyan")
            if len(self.grid_pts) == 2:
                self.finish_grid()
        elif len(self.pairs) < N_TICKS:
            if self.pending is None:
                self.pending = (x, y); self.draw_pt(x, y, "red")
            else:
                x1, y1 = self.pending; self.pending = None
                self.pairs.append((x1, y1, x, y)); self.draw_pt(x, y, "red")
                self.draw_pair(x1, y1, x, y, len(self.pairs))
                if len(self.pairs) == N_TICKS:
                    self.save_current(); self.title(); self.fig.canvas.draw_idle()
                    plt.pause(0.6); self.i += 1; self.load(); return
        self.title(); self.fig.canvas.draw_idle()

    def finish_grid(self):
        (x1, y1), (x2, y2) = self.grid_pts
        d = np.hypot(x2 - x1, y2 - y1)
        try:
            import tkinter.simpledialog as sd
            n = sd.askinteger("网格", f"两点像素距离 {d:.1f} px，相距几格（每格 10 mm）？", minvalue=1, maxvalue=30)
        except Exception:
            n = None
        if n:
            self.mmpx, self.mmpx_src = 10.0 * n / d, f"grid_{n}cells"
        self.grid_mode, self.grid_pts = False, []

    def on_key(self, ev):
        k = ev.key
        if k == "q":
            self.finish()
        elif k == "n":
            if self.pairs: self.save_current()
            self.i += 1; self.load()
        elif k == "r":
            self.load()
        elif k == "g":
            self.grid_mode, self.grid_pts = True, []
        elif k == "u":
            if self.grid_mode and self.grid_pts:
                self.grid_pts.pop()
            elif self.pending is not None:
                self.pending = None
            elif self.pairs:
                self.pairs.pop()
            # 重绘
            for a in self.artists: a.remove()
            self.artists = []
            for j, (x1, y1, x2, y2) in enumerate(self.pairs, 1):
                self.draw_pt(x1, y1, "red"); self.draw_pt(x2, y2, "red"); self.draw_pair(x1, y1, x2, y2, j)
            for (x, y) in self.grid_pts: self.draw_pt(x, y, "cyan")
        self.title(); self.fig.canvas.draw_idle()

    # ---------- 落盘 ----------
    def save_current(self):
        p = self.files[self.i]
        with open(self.out_csv, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            for j, (x1, y1, x2, y2) in enumerate(self.pairs, 1):
                wpx = float(np.hypot(x2 - x1, y2 - y1)); wmm = wpx * self.mmpx
                cal = CALIPER[self.crack][j - 1]
                row = [p.name, self.crack, self.tilt, j, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                       round(wpx, 2), round(self.mmpx, 5), self.mmpx_src, round(wmm, 2), cal, round(wmm / cal, 3)]
                w.writerow(row); self.rows.append(row)
        print(f"已保存 {p.name}: " + "  ".join(f"t{r[3]} {r[8]:.1f}px={r[11]:.2f}mm (卡尺 {r[12]}, 比 {r[13]:.2f})" for r in self.rows if r[0] == p.name))

    def finish(self):
        rows = self.rows
        lines = [f"k_click_width.py  {dt.datetime.now():%Y%m%d_%H%M%S}", f"CSV: {self.out_csv}", "",
                 f"{'file':10s} {'tilt':>5s} tick  宽px   mm/px  人工mm  卡尺mm  人工/卡尺"]
        for r in rows:
            lines.append(f"{r[0]:10s} {r[2]:5.1f} {r[3]:4d} {r[8]:6.1f} {r[9]:7.4f} {r[11]:7.2f} {r[12]:7.1f} {r[13]:9.2f}")
        z = [r for r in rows if r[2] == 0]
        if z:
            ratio = np.array([r[13] for r in z]); img = np.array([r[11] for r in z]); cal = np.array([r[12] for r in z])
            lg = np.log(img / cal)
            rho = np.corrcoef(img, cal)[0, 1] if len(z) > 2 else float("nan")
            lines += ["", f"0° 图 {len(z)} 点：人工图像开口/卡尺 中值 {np.median(ratio):.2f}  均值 {ratio.mean():.2f}  "
                          f"log 偏差 b={lg.mean():+.3f} σ={lg.std(ddof=1) if len(z)>1 else 0:.3f}  ρ={rho:.3f}  MAE {np.abs(img-cal).mean():.2f} mm",
                      "判读：比值≈1 → 图像能看到卡尺量的开口，SAM3 的 −70% 是欠分割；比值≪1 → 卡尺开口在图像里本就不可见，属定义差异。"]
        txt = "\n".join(lines); print("\n" + txt)
        Path(self.out_txt).write_text(txt, encoding="utf-8")
        plt.close(self.fig); sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--all", action="store_true", help="全部 20 张；默认只做 0° 的 x-1.jpg")
    a = ap.parse_args()
    d = Path(a.dir)
    files = sorted({p.resolve() for p in list(d.glob("*.jpg")) + list(d.glob("*.JPG")) + list(d.glob("*.jpeg"))},
                   key=lambda p: tuple(int(t) for t in p.stem.split("-")))
    if not a.all:
        files = [p for p in files if p.stem.endswith("-1")]
    if not files:
        print(f"目录里没找到 x-k.jpg：{d}"); sys.exit(1)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"共 {len(files)} 张：", [p.name for p in files])
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    sess = Session(files, d / f"click_widths_{ts}.csv", d / f"click_widths_{ts}.txt")   # 必须持有引用：mpl 回调是弱引用
    sess.fig._sess = sess
    plt.show()


if __name__ == "__main__":
    main()
