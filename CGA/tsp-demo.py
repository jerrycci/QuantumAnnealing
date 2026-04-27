# tsp_gui_map.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point, LineString
from shapely.affinity import translate
import numpy as np
import itertools
import time
import os

# Optional quantum packages (pyqubo + neal)
try:
    from pyqubo import Array
    import neal
    HAS_PYQUBO = True
except Exception:
    HAS_PYQUBO = False

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 範例門市 (name, lat, lon) — 可替換成實際內湖資料
stores = [
    ("長鴻門市", 25.08493, 121.59253),
    ("成功門市", 25.06865, 121.58695),
    ("成湖門市", 25.08022, 121.59382),
    ("馨瑩門市", 25.08302, 121.56471),
    ("金湖門市", 25.08239, 121.55933),
    ("里昂門市", 25.08011, 121.56844),
    ("港墘門市", 25.07845, 121.57532),
    ("瑞光門市", 25.07987, 121.56123),
    ("文德門市", 25.08276, 121.57789),
    ("康寧門市", 25.08512, 121.59034),
    ("湖光門市", 25.08645, 121.58876),
    ("麗山門市", 25.08723, 121.59112),
    ("民權門市", 25.08098, 121.58234),
    ("洲子門市", 25.07567, 121.56189),
    ("石潭門市", 25.07345, 121.56987),
    ("新湖門市", 25.07789, 121.56543),
]

# 內湖區 Ubike 租借站點
stations = [
    ("內湖高工", 25.08285, 121.59045),
    ("港墘捷運站", 25.07845, 121.57532),
    ("麗山國中", 25.08723, 121.59112),
    ("文德捷運站", 25.08276, 121.57789),
    ("民權隧道口", 25.08098, 121.58234),
    ("內湖國小", 25.08412, 121.58623),
    ("康寧國中", 25.08512, 121.59034),
    ("湖光市場", 25.08645, 121.58876),
    ("瑞光路口", 25.07987, 121.56123),
    ("洲子街口", 25.07567, 121.56189),
    ("石潭路口", 25.07345, 121.56987),
    ("新湖一路", 25.07789, 121.56543),
    ("成功路口", 25.06865, 121.58695),
    ("金湖路口", 25.08239, 121.55933),
    ("內湖行政中心", 25.08022, 121.59382),
    ("內湖運動中心", 25.08302, 121.56471),
]

def haversine_m(lat1, lon1, lat2, lon2):
    """Return distance (meters) between two lat/lon points using haversine formula."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def on_closing():
    root.destroy()
    root.quit()

# Tkinter UI
root = tk.Tk()
#root.title("TSP Solver Demo - 內湖 7-11 (GeoPandas + contextily)")
root.title("內湖 Ubike Station Route Planning")
root.protocol("WM_DELETE_WINDOW", on_closing)

frame_left = ttk.Frame(root, padding=6)
frame_left.pack(side=tk.LEFT, fill=tk.Y)
frame_right = ttk.Frame(root)
frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Matplotlib figure
fig, ax = plt.subplots(figsize=(8,8))
canvas = FigureCanvasTkAgg(fig, master=frame_right)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# checkbox for stations
var_dict = {}
for name, lat, lon in stations:
    var = tk.BooleanVar(value=True)
    cb = ttk.Checkbutton(frame_left, text=name, variable=var,
                         command=lambda: draw_points_only())
    cb.pack(anchor="w")
    var_dict[name] = (var, lat, lon)


method_vars = {
    "CGA": tk.BooleanVar(value=True),
    "BruteForce": tk.BooleanVar(value=False)
}

ttk.Label(frame_left, text="選擇求解方法:").pack(anchor="w", pady=(8,0))
for m, v in method_vars.items():
    cb = ttk.Checkbutton(frame_left, text=m, variable=v)
    cb.pack(anchor="w")

status_var = tk.StringVar(value="尚未求解")
status_label = ttk.Label(frame_left, textvariable=status_var, wraplength=220)
status_label.pack(anchor="w", pady=(8,0))

# ---------- TSP solvers ----------

from compal_solver import compal_solver as solver
import dimod
import numpy as np
import os, math
from datetime import datetime
from pyqubo import Array, Constraint

def solve_tsp_compal_solver(coords, timeout=5):
    """
    Solve TSP using compal_solver.Quantix_GA
    Uses Euclidean distance (same as tsp-cga.py).
    coords: list of (lat, lon)
    """
    n = len(coords)
    x = Array.create("x", (n, n), "BINARY")

    # --- Constraints (same as tsp-cga.py) ---
    time_const = 0.0
    for i in range(n):
        time_const += Constraint((sum(x[i, j] for j in range(n)) - 1)**2, label=f"time{i}")

    city_const = 0.0
    for j in range(n):
        city_const += Constraint((sum(x[i, j] for i in range(n)) - 1)**2, label=f"city{j}")

    # --- Distance term (Euclidean) ---
    def euclidean(i, j):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[j]
        return ((lat1 - lat2)**2 + (lon1 - lon2)**2) ** 0.5

    distance = 0.0
    for i in range(n):
        for j in range(n):
            d_ij = euclidean(i, j)
            for k in range(n):
                distance += d_ij * x[k, i] * x[(k+1) % n, j]

    # --- Hamiltonian ---
    A = 1.2
    H = distance + A * (time_const + city_const)
    model = H.compile()
    qubo, offset = model.to_qubo(index_label=True)
    variables = model.variables
    nvars = len(variables)

    # --- dynamic scaling (same as tsp-cga.py) ---
    abs_key = max(qubo, key=lambda y: abs(float(qubo[y])))
    abs_value = abs(float(qubo[abs_key]))
    order_upper = 13 - len(str(round(abs_value)))

    len_key = max(qubo, key=lambda y: len(str(abs(float(qubo[y]))).split(".")[-1]))
    len_value = len(str(abs(float(qubo[len_key]))).split(".")[-1])
    N = min(len_value, order_upper)
    if N < 0:
        N = 0

    # --- write integer QUBO file ---
    if not os.path.exists("qubo_int"):
        os.makedirs("qubo_int")
    file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    qubo_int_path = f"qubo_int/{file_name}_qubo_int.txt"
    with open(qubo_int_path, "w") as f:
        print(nvars, offset, file=f)
        for (i, j), value in qubo.items():
            print(i, j, round(float(value) * (10**N)), file=f)

    # --- run GA ---
    ga = solver.Quantix_GA(qubo_int_path)
    result, energy, count, timeout_flag = ga.run(batch_factor=10, main_factor=0.2, run_time=timeout)

    result = np.array(result)
    if result.size == 0:
        raise RuntimeError("GA returned empty result array.")

    if result.ndim == 1:
        inferred_count = result.size // nvars
        result = result.reshape((inferred_count, nvars))
        count = inferred_count
    elif result.ndim == 2:
        count = result.shape[0]
    else:
        raise RuntimeError(f"Unexpected GA result ndim={result.ndim}")

    # --- build Q matrix ---
    Q = np.zeros((nvars, nvars))
    for (i, j), val in qubo.items():
        Q[i, j] = float(val)

    sample_list, energy_list = [], []
    for c in range(count):
        vec = np.asarray(result[c, :], dtype=np.uint8)
        sample = {variables[i]: int(vec[i]) for i in range(nvars)}
        sample_list.append(sample)
        b = vec.reshape((nvars, 1))
        e_val = float((b.T @ Q @ b)[0, 0])
        energy_list.append(e_val)

    sampleset = dimod.SampleSet.from_samples(sample_list, dimod.BINARY, energy_list)
    decoded = model.decode_sampleset(sampleset)

    def decode_to_path(sol):
        path = [None] * n
        for i in range(n):
            for j in range(n):
                if sol.array("x", (i, j)) == 1:
                    path[i] = j
        return path

    best = min(decoded, key=lambda d: d.energy)
    path = decode_to_path(best)

    if None in path or len(set(path)) != n:
        assigned = [p for p in path if p is not None]
        missing = [i for i in range(n) if i not in assigned]
        path = [p if p is not None else missing.pop(0) for p in path]

    # --- Distance term (Haversine in meters) ---
    def distance_m(i, j):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[j]
        return haversine_m(lat1, lon1, lat2, lon2)

    # cost = Euclidean total length
    #cost = sum(euclidean(path[i], path[(i+1) % n]) for i in range(n))
    cost = sum(distance_m(path[i], path[(i+1) % n]) for i in range(n))
    return path, cost


def solve_tsp_bruteforce(coords, max_n=10):
    """
    Brute force TSP solver using haversine distance (meters).
    Precomputes distance matrix for efficiency.
    """
    n = len(coords)
    if n > max_n:
        raise ValueError(f"BruteForce method takes too long to solve route > {max_n} stations (your problme have {n} stations).")

    # --- Precompute pairwise distance matrix ---
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                dist_matrix[i, j] = haversine_m(lat1, lon1, lat2, lon2)

    # --- Brute force permutations ---
    best_path = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(n)):
        cost = sum(dist_matrix[perm[i], perm[(i + 1) % n]] for i in range(n))
        if cost < best_cost:
            best_cost = cost
            best_path = perm

    return list(best_path), best_cost

def solve_tsp_bruteforce_old(coords, max_n=10):
    """
    Brute force TSP solver using Euclidean distance.
    coords: list of (lat, lon)
    """
    n = len(coords)
    if n > max_n:
        raise ValueError(f"BruteForce only supports up to {max_n} points (got {n}).")

    #def euclidean(i, j):
    #    lat1, lon1 = coords[i]
    #    lat2, lon2 = coords[j]
    #    return ((lat1 - lat2)**2 + (lon1 - lon2)**2) ** 0.5

    def distance_m(i, j):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[j]
        return haversine_m(lat1, lon1, lat2, lon2)

    best_path = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(n)):
        cost = sum(distance_m(perm[i], perm[(i + 1) % n]) for i in range(n))
        if cost < best_cost:
            best_cost = cost
            best_path = perm
    return list(best_path), best_cost


# ---------- Map drawing ----------
DEFAULT_ZOOM = 16
basemap_loaded = False
basemap_extent = None

def clear_routes():
    """清除舊的路徑 (保持底圖與門市點不變)"""
    for artist in getattr(ax, "route_artists", []):
        artist.remove()
    ax.route_artists = []

def draw_points_only():
    """更新選取/未選取的門市，不清掉背景地圖，同時清除舊路徑"""
    global basemap_loaded, basemap_extent

    # 初次載入底圖
    if not basemap_loaded:
        all_pts = [(name, lat, lon) for name,(var,lat,lon) in var_dict.items()]
        gdf_pts = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for _, lat, lon in all_pts], crs="EPSG:4326"
        ).to_crs(epsg=3857)
        xmin, ymin, xmax, ymax = gdf_pts.total_bounds
        pad_x = (xmax - xmin) * 0.2
        pad_y = (ymax - ymin) * 0.2
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=DEFAULT_ZOOM)
            basemap_loaded = True
            basemap_extent = ax.axis()
        except Exception as e:
            print("[警告] 無法載入底圖:", e)

    if basemap_extent:
        ax.axis(basemap_extent)

    # 清除舊的點、標籤、路徑
    for artist in getattr(ax, "point_artists", []):
        artist.remove()
    for artist in getattr(ax, "text_artists", []):
        artist.remove()
    clear_routes()
    ax.point_artists, ax.text_artists = [], []

    # clear legend
    ax.legend_.remove() if hasattr(ax, "legend_") and ax.legend_ else None

    # 畫新的門市點
    selected = {name for name,(var,lat,lon) in var_dict.items() if var.get()}
    all_pts = [(name, lat, lon) for name,(var,lat,lon) in var_dict.items()]
    gdf_pts = gpd.GeoDataFrame(
        {"name": [n for n,_,_ in all_pts]},
        geometry=[Point(lon,lat) for _,lat,lon in all_pts], crs="EPSG:4326"
    ).to_crs(epsg=3857)

    for x,y,label in zip(gdf_pts.geometry.x, gdf_pts.geometry.y, gdf_pts["name"]):
        if label in selected:
            p, = ax.plot(x, y, "ro", markersize=8, zorder=3)
        else:
            p, = ax.plot(x, y, "ro", markersize=8, mfc="none", zorder=3)
        t = ax.text(x, y, " "+label, fontsize=9, ha="left", va="bottom", zorder=4)
        ax.point_artists.append(p)
        ax.text_artists.append(t)

    canvas.draw()

def draw_route(selected, method, path, cost, t, br_offset_m=60):
    """
    Draw one method's route on the existing basemap.
    - selected: list of (name, lat, lon) in same order used to build dist matrix
    - method: string, e.g. "SA" or "BruteForce"
    - path: list of indices (route order)
    - cost: numeric
    - t: elapsed seconds (numeric)
    - br_offset_m: meter offset applied when method == "BruteForce"
    """
    if path is None:
        return

    # ensure we have a route_artists container
    if not hasattr(ax, "route_artists"):
        ax.route_artists = []

    # build lon/lat coords from selected indices (close the tour)
    coords_lonlat = [(selected[i][2], selected[i][1]) for i in path] + [(selected[path[0]][2], selected[path[0]][1])]
    line = LineString(coords_lonlat)

    # transform to web-mercator for plotting
    gdf_line = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326")
    gdf_line_3857 = gdf_line.to_crs(epsg=3857)
    geom = gdf_line_3857.geometry.iloc[0]

    # apply small meter offset to BruteForce to avoid overlap
    if method == "BruteForce" and br_offset_m:
        geom = translate(geom, xoff=br_offset_m, yoff=br_offset_m)

    # extract coords and plot as Line2D so we can remove later
    x, y = geom.xy
    colors = {"CGA": "green", "BruteForce": "red"}
    (line_artist,) = ax.plot(x, y,
                             linewidth=2.5,
                             color=colors.get(method, "blue"),
                             label=f"{method}: {cost:.1f} m, {t:.2f}s",
                             zorder=2)
    ax.route_artists.append(line_artist)

    # refresh legend and canvas
    try:
        ax.legend()
    except Exception:
        pass
    canvas.draw_idle()

# ---------- Solve action ----------
import multiprocessing as mp

def run_bruteforce_async(coords, selected):
    def worker(conn, coords):
        try:
            t0 = time.time()
            path, cost = solve_tsp_bruteforce(coords)
            t1 = time.time()
            conn.send((path, cost, t1 - t0, None))
        except Exception as e:
            conn.send((None, None, None, str(e)))
        finally:
            conn.close()

    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=worker, args=(child_conn, coords))
    p.daemon = True
    p.start()

    def check():
        if parent_conn.poll():
            path, cost, elapsed, error = parent_conn.recv()
            if error:
                status_var.set(status_var.get() + f"\nBruteForce 失敗: {error}")
            else:
                draw_route(selected, "BruteForce", path, cost, elapsed)
                status_var.set(status_var.get() + f"\nBruteForce: cost={cost:.4f}, time={elapsed:.2f}s")#, path={path}")
            p.join()
        else:
            root.after(100, check)  # check again after 100ms

    root.after(100, check)


import threading

def solve_tsp_action():
    selected = [(name, lat, lon) for name, (var, lat, lon) in var_dict.items() if var.get()]
    if len(selected) < 3:
        messagebox.showwarning("提醒", "請至少選擇三個門市")
        return

    # Clear old routes, redraw points only
    status_var.set("求解中...")
    root.update_idletasks()
    draw_points_only()

    coords = [(lat, lon) for name, lat, lon in selected]

    def run_cga():
        try:
            t0 = time.time()
            path, cost = solve_tsp_compal_solver(coords)
            t1 = time.time()
            root.after(0, lambda: draw_route(selected, "CGA", path, cost, t1 - t0))
            root.after(0, lambda: status_var.set(
                status_var.get() + f"\nCGA: cost={cost:.4f} (meters, haversine), time={t1-t0:.2f}s")#, path={path}")
            )
        except Exception as e:
            root.after(0, lambda: status_var.set(status_var.get() + f"\nCGA 失敗: {e}"))


    # Launch threads concurrently
    if method_vars["BruteForce"].get():
        run_bruteforce_async(coords, selected)
    if method_vars["CGA"].get():
        threading.Thread(target=run_cga, daemon=True).start()


# button
btn = ttk.Button(frame_left, text="開始求解", command=solve_tsp_action)
btn.pack(pady=10)

# initial draw
draw_points_only()

root.mainloop()
os._exit(0)