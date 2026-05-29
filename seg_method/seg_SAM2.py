import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import SAM



def sam2_inference(src_path, dst_path, model):
    """
    读取源图像，使用 SAM2 生成掩码并保存到目标路径。
    """
    original_img = cv2.imread(src_path)
    if original_img is None:
        return

    h, w = original_img.shape[:2]

    margin_x, margin_y = int(w * 0.05), int(h * 0.05)
    box_prompt = [margin_x, margin_y, w - margin_x, h - margin_y]

    results = model.predict(source=original_img, bboxes=[box_prompt], verbose=False)

    if results and results[0].masks is not None:
        mask_data = results[0].masks.data[0].cpu().numpy()
        if mask_data.shape != (h, w):
            mask_data = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_data * 255).astype(np.uint8)
    else:
        binary_mask = np.ones((h, w), dtype=np.uint8) * 255

    cv2.imwrite(dst_path, binary_mask)


def build_custom_dataset(
    original_mvtec_root,
    new_dataset_root,
    model_type="/d242/wyh/model/SAM/sam2_b.pt",
    categories=()
):
    if not categories:
        print("[-] categories 为空，未处理任何数据（请通过 seg_all.py 传入类别列表）。")
        return

    print(f"[*] 正在初始化 SAM 2 模型 (权重文件: {model_type})...")
    model = SAM(model_type)

    for category in categories:
        print(f"\n[*] 正在处理类别: {category}")
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
                
                # 过滤掉非文件夹的文件
                if not os.path.isdir(orig_defect_dir):
                    continue
                    
                new_defect_dir = os.path.join(new_test_dir, defect_type)
                os.makedirs(new_defect_dir, exist_ok=True)

                # 使用 tqdm 添加进度条
                for img_name in tqdm(os.listdir(orig_defect_dir), desc=f"test - {defect_type}"):
                    orig_img_path = os.path.join(orig_defect_dir, img_name)
                    img_path = os.path.join(new_defect_dir, img_name) 
                    
                    sam2_inference(orig_img_path, img_path, model)

        # ---------------------------------------------------------
        # 模块 B: 处理 train/good 路径
        # ---------------------------------------------------------
        if os.path.exists(orig_train_dir):
            # MVTec 的 train/good 下没有更深层级的子文件夹，直接遍历图片
            for img_name in tqdm(os.listdir(orig_train_dir), desc="train/good"):
                orig_img_path = os.path.join(orig_train_dir, img_name)
                img_path = os.path.join(new_good_dir, img_name) 
                
                sam2_inference(orig_img_path, img_path, model)


if __name__ == "__main__":
    print("请通过项目根目录的 seg_all.py 配置类别与方法后运行；本模块不维护类别列表。")