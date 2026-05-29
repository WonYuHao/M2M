import os
from collections import deque
import cv2
import numpy as np
from tqdm import tqdm

# ====================== 配置参数 ======================
MIN_WHITE_CC_PIXELS = 100

USE_VOID_REACHABILITY_FILL = True
VOID_SEED_MIN_DT = 2.5
VOID_SEED_FRAC_OF_MAX = 0.5
VOID_THICK_PATH_MIN_DT = 2.0
VOID_FILL_IF_DT_LEQ = 4.5

MAX_BLACK_VOID_AREA_TO_FILL = 120
MAX_CRACK_VOID_DEPTH_PX = 4.0
CRACK_VOID_MAX_CIRCULARITY = 0.42

EDGE_BARRIER_METHOD = "canny"
EDGE_CANNY_LOW = 150
EDGE_CANNY_HIGH = 200
EDGE_SOBEL_THRESH = 35.0
EDGE_BARRIER_DILATE_ITER = 0

OVERLAY_ALPHA = 0.3

HAZELNUT_EDGE_PARAMS = {
    "use_clahe": True,
    "canny_low": 20,
    "canny_high": 40,
    "flood_border_sides": "ltrb",
}

TOOTHBRUSH_EDGE_PARAMS = {
    "use_clahe": True,
    "canny_low": 10,
    "canny_high": 40,
    "flood_border_sides": "lr",
}

# ====================== 图像处理算法核心 ======================
def preprocess_gray_for_grid(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def preprocess_gray_for_object_edge(bgr, use_clahe=False):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def compute_edge_map(
    gray,
    method=EDGE_BARRIER_METHOD,
    canny_low=EDGE_CANNY_LOW,
    canny_high=EDGE_CANNY_HIGH,
    sobel_thresh=EDGE_SOBEL_THRESH,
    dilate_iter=EDGE_BARRIER_DILATE_ITER,
):
    if method == "canny":
        edges = cv2.Canny(gray, canny_low, canny_high)
    elif method == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edges = (mag >= float(sobel_thresh)).astype(np.uint8) * 255
    else:
        raise ValueError(f"unknown EDGE_BARRIER_METHOD: {method}")

    if dilate_iter and dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, k, iterations=int(dilate_iter))
    return edges

def extract_mask_ori(gray):
    raw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=17, C=7
    )
    mask = cv2.bitwise_not(raw)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    mask_ori = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_WHITE_CC_PIXELS:
            mask_ori[labels == i] = 255
    return mask_ori

def merge_bound_binary(mask_ori, edge_binary):
    merge_bound = cv2.bitwise_or(mask_ori, edge_binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merge_bound)
    mask_ori = np.zeros_like(merge_bound)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_WHITE_CC_PIXELS:
            mask_ori[labels == i] = 255
    return mask_ori

def _void_component_circularity(region_bool):
    m = (region_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    area = abs(cv2.contourArea(c))
    perim = cv2.arcLength(c, True)
    if perim < 1e-6:
        return 0.0
    return float((4.0 * np.pi * area) / (perim * perim))

def fill_void_reachability(mask_gray, seed_min_dt, seed_frac_of_max, thick_path_min_dt, fill_if_dt_leq):
    wire = (mask_gray >= 128).astype(np.uint8) * 255
    dt = cv2.distanceTransform(255 - wire, cv2.DIST_L2, 5).astype(np.float32)
    h, w = mask_gray.shape
    black_fg = np.where(mask_gray == 0, 255, 0).astype(np.uint8)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(black_fg)
    out = mask_gray.copy()

    for i in range(1, n_lab):
        comp = labels == i
        max_dt = float(dt[comp].max())
        if max_dt < 1e-6:
            continue

        seed_thresh = max(float(seed_min_dt), float(seed_frac_of_max) * max_dt)
        seed = comp & (dt >= seed_thresh)
        if not seed.any():
            seed = comp & (dt >= max_dt * 0.5)

        reachable = np.zeros((h, w), dtype=bool)
        q = deque()
        ys, xs = np.where(seed)
        for y, x in zip(ys, xs):
            reachable[y, x] = True
            q.append((y, x))

        while q:
            y, x = q.popleft()
            tp = float(dt[y, x])
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if not comp[ny, nx] or reachable[ny, nx]:
                    continue
                tn = float(dt[ny, nx])
                if min(tp, tn) < thick_path_min_dt:
                    continue
                reachable[ny, nx] = True
                q.append((ny, nx))

        unreachable = comp & (~reachable)
        tr = np.zeros((h, w), dtype=bool)
        tr[1:, :] |= reachable[:-1, :]
        tr[:-1, :] |= reachable[1:, :]
        tr[:, 1:] |= reachable[:, :-1]
        tr[:, :-1] |= reachable[:, 1:]
        neighbor_of_reachable_void = comp & tr
        fill = unreachable & (dt <= float(fill_if_dt_leq)) & (~neighbor_of_reachable_void)
        out[fill] = 255
    return out

def fill_crack_like_black_voids(mask_gray, max_area, max_depth_px, max_circularity):
    wire = (mask_gray >= 128).astype(np.uint8) * 255
    dt = cv2.distanceTransform(255 - wire, cv2.DIST_L2, 5)

    black_fg = np.where(mask_gray == 0, 255, 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(black_fg)
    out = mask_gray.copy()
    for i in range(1, n):
        region = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        max_d = float(dt[region].max())
        circ = _void_component_circularity(region)

        by_area = max_area > 0 and area <= max_area
        by_crack_shape = (
            max_depth_px > 0 and max_d <= max_depth_px and circ < max_circularity
        )
        if by_area or by_crack_shape:
            out[region] = 255
    return out

def fill_narrow_voids_on_mask(bound_binary):
    m = bound_binary.copy()
    if USE_VOID_REACHABILITY_FILL:
        m = fill_void_reachability(
            m, VOID_SEED_MIN_DT, VOID_SEED_FRAC_OF_MAX, VOID_THICK_PATH_MIN_DT, VOID_FILL_IF_DT_LEQ
        )
    if MAX_BLACK_VOID_AREA_TO_FILL > 0 or MAX_CRACK_VOID_DEPTH_PX > 0:
        m = fill_crack_like_black_voids(
            m, MAX_BLACK_VOID_AREA_TO_FILL, MAX_CRACK_VOID_DEPTH_PX, CRACK_VOID_MAX_CIRCULARITY
        )
    return m

def _parse_flood_border_sides(params):
    raw = params.get("flood_border_sides", "ltrb")
    if isinstance(raw, (list, tuple, set)):
        chars = {str(x).lower()[0] for x in raw if str(x)}
    else:
        chars = {c for c in str(raw).lower() if c in "ltrb"}
    if not chars:
        chars = set("ltrb")
    return chars

def generate_mask_edge_closed_fill(image_bgr, params):
    gray = preprocess_gray_for_object_edge(image_bgr, params.get("use_clahe", False))
    edges = cv2.Canny(gray, int(params["canny_low"]), int(params["canny_high"]))
    
    kernel_size = 3 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    strong_barrier = cv2.dilate(edges, kernel, iterations=1)

    h, w = gray.shape
    cv2.rectangle(strong_barrier, (0, 0), (w - 1, h - 1), 255, 1)

    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    free = (strong_barrier == 0)

    def try_seed(y, x):
        if 0 <= y < h and 0 <= x < w and free[y, x] and not visited[y, x]:
            visited[y, x] = True
            q.append((y, x))

    test_offset = 5
    sides = _parse_flood_border_sides(params)
    if "l" in sides:
        for y in range(test_offset, h - test_offset): try_seed(y, test_offset)
    if "r" in sides:
        for y in range(test_offset, h - test_offset): try_seed(y, w - 1 - test_offset)
    if "t" in sides:
        for x in range(test_offset, w - test_offset): try_seed(test_offset, x)
    if "b" in sides:
        for x in range(test_offset, w - test_offset): try_seed(h - 1 - test_offset, x)

    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if free[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))

    mask = (~visited).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        out = mask
    else:
        best_i = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        out = np.zeros((h, w), dtype=np.uint8)
        out[labels == best_i] = 255

    return out, edges

# ==================== 1. 辅助函数：提取并保存掩码 ====================
def geometric_inference(src_path, dst_path, category):
    """
    读取源图像，使用几何学方法生成掩码并保存到目标路径。
    """
    original_img = cv2.imread(src_path)
    if original_img is None:
        return

    try:
        if category == "grid":
            gray = preprocess_gray_for_grid(original_img)
            edge = compute_edge_map(gray)
            mask_ori = extract_mask_ori(gray)
            bound = merge_bound_binary(mask_ori, edge)
            mask_final = fill_narrow_voids_on_mask(bound)
        elif category == "hazelnut":
            mask_final, _ = generate_mask_edge_closed_fill(original_img, HAZELNUT_EDGE_PARAMS)
        elif category == "toothbrush":
            mask_final, _ = generate_mask_edge_closed_fill(original_img, TOOTHBRUSH_EDGE_PARAMS)
        else:
            raise ValueError(f"未实现几何分割的类别: {category}")
            
        cv2.imwrite(dst_path, mask_final)

    except Exception as e:
        print(f"\n[!] 处理 {src_path} 时出错: {e}")
        # 兜底：处理失败时生成全白掩码
        h, w = original_img.shape[:2]
        fb = np.ones((h, w), dtype=np.uint8) * 255
        cv2.imwrite(dst_path, fb)


# ==================== 2. 核心构建逻辑 ====================
def build_custom_dataset(
    original_mvtec_root,
    new_dataset_root,
    categories=()
):
    if not categories:
        print("[-] categories 为空，未处理任何数据（请通过 seg_all.py 传入类别列表）。")
        return

    for category in categories:
        print(f"\n[*] 正在处理类别: {category} (Geometric)")
        orig_cat_dir = os.path.join(original_mvtec_root, category)    
        orig_test_dir = os.path.join(orig_cat_dir, 'test')
        orig_train_dir = os.path.join(orig_cat_dir, 'train', 'good')
        
        new_cat_dir = os.path.join(new_dataset_root, category)
        new_test_dir = os.path.join(new_cat_dir, 'test')
        new_good_dir = os.path.join(new_cat_dir, 'good')
        
        os.makedirs(new_test_dir, exist_ok=True)
        os.makedirs(new_good_dir, exist_ok=True)

        # ---------------------------------------------------------
        # 模块 A: 处理 test 路径
        # ---------------------------------------------------------
        if os.path.exists(orig_test_dir):
            for defect_type in os.listdir(orig_test_dir):
                orig_defect_dir = os.path.join(orig_test_dir, defect_type)
                
                if not os.path.isdir(orig_defect_dir):
                    continue
                    
                new_defect_dir = os.path.join(new_test_dir, defect_type)
                os.makedirs(new_defect_dir, exist_ok=True)

                for img_name in tqdm(os.listdir(orig_defect_dir), desc=f"test - {defect_type}"):
                    orig_img_path = os.path.join(orig_defect_dir, img_name)
                    img_path = os.path.join(new_defect_dir, img_name) 
                    
                    geometric_inference(orig_img_path, img_path, category)

        # ---------------------------------------------------------
        # 模块 B: 处理 train/good 路径
        # ---------------------------------------------------------
        if os.path.exists(orig_train_dir):
            for img_name in tqdm(os.listdir(orig_train_dir), desc="train/good"):
                orig_img_path = os.path.join(orig_train_dir, img_name)
                img_path = os.path.join(new_good_dir, img_name) 
                
                geometric_inference(orig_img_path, img_path, category)


if __name__ == "__main__":
    print("请通过项目根目录的 seg_all.py 配置类别与方法后运行；本模块不维护类别列表。")