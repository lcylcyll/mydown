# -*- coding: utf-8 -*- # 指定编码
import os
import time
# import argparse # No longer needed for menu
import cv2
import numpy as np
import requests
import shutil
from datetime import datetime
import csv # 导入 csv 模块，方便处理索引
import traceback # For better error reporting
import glob # <--- IMPORTED FOR FILE SEARCHING

# ====================
# 配置区（按需修改）
# ====================
CONFIG = {
    "download_list_file": "download_list.txt",  # 下载列表文件路径
    "template_dir": "templates",      # 模板存储目录
    "template_index": "templates_index.csv",  # 模板索引文件 (使用 .csv)
    "output_dir": "processed",        # 处理结果目录
    "download_dir": "downloads",      # 下载存储目录
    "archive_dir": "processed_originals", # 处理完成的原始文件移动目录 - Not used in current process_images
    "timeout": 15,                    # 下载超时(秒) - 稍增加
    "max_retries": 3,                 # 最大重试次数
    "retry_delay": 2,                 # 重试等待基数(秒)
    "match_threshold": 0.65,          # 模板匹配相似度阈值 (可根据实际情况调整)
    "rotation_range": 15,             # 旋转校正范围（±15度）
    "rotation_step": 1,               # 旋转步长（1度）
    "wait_timeout": 0                 # cv2.waitKey(0) 表示无限等待用户按键
}

# 全局变量用于自定义 ROI 选择
drawing = False # 标记是否正在绘制矩形
ix, iy = -1, -1 # 鼠标按下时的初始坐标
roi_coords = None # 存储最终选择的 ROI 坐标 (x, y, w, h)
img_copy = None # 存储当前显示图像的副本，用于绘制

def mouse_callback(event, x, y, flags, param):
    """鼠标事件回调函数，用于自定义 ROI 选择"""
    global drawing, ix, iy, roi_coords, img_copy

    if img_copy is None: # 防止 img_copy 未设置时出错
        return

    window_title = '选择ROI区域（自定义）' # 窗口标题

    if event == cv2.EVENT_LBUTTONDOWN: # 鼠标左键按下
        drawing = True
        ix, iy = x, y
        # print(f"鼠标按下: ({x}, {y})") # 调试信息

    elif event == cv2.EVENT_MOUSEMOVE: # 鼠标移动
        if drawing:
            img_temp = img_copy.copy() # 在副本上绘制，避免覆盖
            # 绘制绿色细矩形框作为预览
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 1)
            cv2.imshow(window_title, img_temp)

    elif event == cv2.EVENT_LBUTTONUP: # 鼠标左键松开
        drawing = False
        # 确保坐标顺序正确 (左上角 -> 右下角)
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        w = x2 - x1 # 计算宽度
        h = y2 - y1 # 计算高度
        if w > 0 and h > 0: # 确保选区有效
            roi_coords = (x1, y1, w, h) # 存储 ROI 坐标 (相对于显示图像)
            # print(f"鼠标松开，选择区域: {roi_coords}") # 调试信息
            img_temp = img_copy.copy()
            # 在副本上绘制最终的、稍粗的绿色矩形
            cv2.rectangle(img_temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(window_title, img_temp)
        else:
             # print("选择区域无效 (宽度或高度为0)") # 调试信息
             roi_coords = None # 重置无效选择


def custom_select_roi(window_name, img):
    """
    自定义 ROI 选择函数。
    Args:
        window_name (str): 显示窗口的名称。
        img (numpy.ndarray): 需要选择 ROI 的图像 (通常是缩放后的用于显示的图像)。
    Returns:
        tuple: (x, y, w, h) 坐标元组，相对于传入的 img。如果取消或无效则返回 (0, 0, 0, 0)。
    """
    global drawing, ix, iy, roi_coords, img_copy
    drawing = False # 重置绘制状态
    ix, iy = -1, -1 # 重置初始坐标
    roi_coords = None # 重置 ROI 结果
    img_copy = img.copy() # 使用传入的图像创建副本

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # 确保窗口存在且可调整
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- 自定义ROI选择 ---")
    print("用鼠标左键拖动选择区域.")
    print("按 Enter/Space 确认选择.")
    print("按 C 取消选择并跳过此图.")
    print("按 ESC 退出程序.")
    print("----------------------")

    while True:
        if not drawing and roi_coords:
             img_display = img_copy.copy()
             x,y,w,h = roi_coords
             cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
             cv2.imshow(window_name, img_display)
        elif not drawing:
             cv2.imshow(window_name, img_copy)

        key = cv2.waitKey(20) & 0xFF

        if key == 13 or key == 32:
            if roi_coords is not None:
                x, y, w, h = roi_coords
                if w > 0 and h > 0:
                    print(f"确认ROI (显示坐标): {roi_coords}")
                    cv2.destroyWindow(window_name)
                    return roi_coords
                else:
                    print("警告: 选择的ROI无效 (宽或高为0). 请重新选择或按 C/ESC.")
                    roi_coords = None
            else:
                print("未选择有效区域，请拖动鼠标选择，或按 C 键跳过，或按 ESC 退出")

        elif key == ord('c'):
            print("取消选择 (C)")
            cv2.destroyWindow(window_name)
            return (0, 0, 0, 0)

        elif key == 27:
            print("用户按 ESC 键，退出程序")
            cv2.destroyWindow(window_name)
            exit(0)

# ====================
# 核心功能函数 (这些都需要保留)
# ====================
def read_download_list(file_path):
    """
    读取下载列表文件 (download_list.txt)。
    每行格式：URL,目标文件名 (逗号分隔, 支持中文逗号)
    跳过空行和 '#' 开头的注释行。
    进行基本的 URL 和文件名检查与清理。
    Returns:
        list: 包含 (url, safe_filename) 元组的列表。
    """
    download_list = [] # 初始化空列表
    if not os.path.exists(file_path):
        print(f"错误：下载列表文件未找到: {file_path}")
        return download_list
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip() # 去除首尾空白
                if not line or line.startswith('#'): # 跳过空行和注释
                    continue
                # 使用逗号作为主要分隔符，兼容中文逗号
                parts = [p.strip() for p in line.replace('，', ',').split(',')]
                if len(parts) >= 2 and parts[0] and parts[1]: # 确保至少有 URL 和 文件名，且不为空
                    url = parts[0]
                    filename_raw = parts[1]
                    # 基本 URL 格式检查 (可选但推荐)
                    if not url.startswith(('http://', 'https://')):
                         print(f"警告：第 {line_num} 行 URL 格式可能不正确: {url}，已跳过")
                         continue
                    # 清理文件名，移除或替换非法字符
                    safe_name = filename_raw.translate(str.maketrans('/\\:*?"<>|', '_________'))
                    safe_name = safe_name.replace("\0", "") # 移除 null 字符

                    # 如果文件名没有扩展名，则添加默认扩展名 (可选)
                    base_filename = os.path.basename(safe_name)
                    if '.' not in base_filename:
                         try:
                             path_from_url = requests.utils.urlparse(url).path
                             ext_from_url = os.path.splitext(path_from_url)[1]
                             if ext_from_url and len(ext_from_url) <= 5: # 简单的扩展名检查
                                 safe_name += ext_from_url
                                 #print(f"信息：第 {line_num} 行文件名 '{filename_raw}' 无扩展名，从 URL 推断并添加 '{ext_from_url}' -> '{safe_name}'")
                             else:
                                 raise ValueError("No valid extension in URL")
                         except Exception:
                             default_ext = '.jpg'
                             safe_name += default_ext # 无法推断则用默认
                             print(f"警告：第 {line_num} 行文件名 '{filename_raw}' 无扩展名且无法从URL推断，已添加 {default_ext} -> '{safe_name}'")

                    download_list.append((url, safe_name)) # 添加到列表
                else:
                    print(f"警告：第 {line_num} 行格式错误或缺少内容，已跳过: '{line}'")
        #print(f"成功读取 {len(download_list)} 条有效下载记录从 '{file_path}'")
    except Exception as e:
        print(f"读取下载列表 '{file_path}' 时出错：{str(e)}")
        traceback.print_exc()
    return download_list

def download_images(items_to_download):
    """
    根据下载列表下载图片，包含重试机制和简单的文件大小检查。
    Args:
        items_to_download (list): 包含 (url, filename) 元组的列表。
    """
    os.makedirs(CONFIG["download_dir"], exist_ok=True) # 确保下载目录存在
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    total_items = len(items_to_download)

    print(f"\n--- 开始下载 {total_items} 个文件 ---")

    for i, (url, filename) in enumerate(items_to_download):
        save_path = os.path.join(CONFIG["download_dir"], filename) # 构造完整保存路径

        print(f"\n[{i+1}/{total_items}] 处理下载: {filename}")

        # 检查文件是否已存在
        if os.path.exists(save_path):
            try:
                if os.path.getsize(save_path) < 1024: # 小于 1KB 认为可能不完整
                    print(f"  - 文件已存在但大小可疑 (<1KB)，尝试重新下载: {filename}")
                    os.remove(save_path)
                else:
                    print(f"  - 文件已存在，跳过下载：{filename}")
                    skipped_count += 1
                    continue
            except OSError as e:
                 print(f"  - 检查已存在文件时出错: {e}, 跳过下载。")
                 skipped_count += 1
                 continue

        success = False
        last_error = "未知错误"
        for attempt in range(CONFIG["max_retries"]):
            try:
                print(f"  - 尝试下载 ({attempt+1}/{CONFIG['max_retries']}): {url}")
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(url, timeout=CONFIG["timeout"], headers=headers, stream=True)
                response.raise_for_status()

                content_type = response.headers.get('content-type')
                is_image = content_type and content_type.lower().startswith('image/')

                if content_type and not is_image:
                    print(f"  警告: URL 返回的 Content-Type 可能不是图片 ({content_type}), 但仍继续下载。")

                block_size = 1024 * 8
                start_time = time.time()
                downloaded_size = 0

                with open(save_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        f.write(data)
                        downloaded_size += len(data)

                download_time = time.time() - start_time
                if os.path.exists(save_path):
                    file_size_kb = os.path.getsize(save_path) / 1024
                else:
                    raise IOError("文件下载后未找到，可能写入失败。")

                if file_size_kb < 1:
                    print(f"  警告: 下载的文件大小很小 ({file_size_kb:.2f} KB)。可能不是预期的图片，但仍保留。")

                print(f"  ✓ 下载成功: {filename} ({file_size_kb:.2f} KB, {download_time:.2f}s)")
                success = True
                downloaded_count += 1
                break

            except requests.exceptions.Timeout:
                last_error = "请求超时"
            except requests.exceptions.SSLError:
                last_error = "SSL 错误"
                print(f"  ! SSL 错误下载 {url}.")
            except requests.exceptions.RequestException as e:
                last_error = f"网络或请求错误: {str(e)}"
            except ValueError as e:
                last_error = str(e)
            except IOError as e:
                 last_error = f"文件读写错误: {str(e)}"
            except Exception as e:
                last_error = f"下载时发生意外错误: {type(e).__name__}: {str(e)}"
                traceback.print_exc()

            if os.path.exists(save_path) and not success:
                try:
                    os.remove(save_path)
                    print(f"  - 已清理部分下载或损坏的文件: {filename}")
                except OSError as remove_err:
                    print(f"  - 清理失败文件时出错: {remove_err}")

            if attempt < CONFIG["max_retries"] - 1:
                wait = CONFIG["retry_delay"] * (2 ** attempt)
                print(f"  × 下载失败 ({last_error})，{wait:.1f}秒后重试...")
                time.sleep(wait)
            else:
                 print(f"  ! 最终下载失败: {filename}\n    错误: {last_error}\n    URL: {url}")
                 failed_count += 1
                 if os.path.exists(save_path):
                      try: os.remove(save_path)
                      except OSError: pass

    print("-" * 30)
    print(f"下载总结: 成功 {downloaded_count}, 已存在/跳过 {skipped_count}, 失败 {failed_count}")
    print("-" * 30)

def select_roi_interactive(img_path):
    """
    交互式选择ROI区域，包含旋转调整阶段和 ROI 选择阶段。
    返回: (roi, angle)
           roi: (x, y, w, h) 坐标元组 (相对于最终旋转后图像的 *原始* 尺寸)，如果跳过或取消则为 None。
           angle: 用户选择的最终旋转角度 (度)。
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"! 无法读取图像: {img_path}")
            return None, 0.0
    except Exception as e:
        print(f"! 读取图像时出错 {img_path}: {e}")
        traceback.print_exc()
        return None, 0.0

    current_img = img.copy()
    angle = 0.0

    # --- 旋转调整阶段 ---
    WINDOW_NAME_ROTATE = '旋转调整 (R/T: 旋转, Q: 完成, S: 跳过, Esc: 退出)'
    cv2.namedWindow(WINDOW_NAME_ROTATE, cv2.WINDOW_NORMAL)
    print("\n--- 旋转调整 ---")
    print("按 R 逆时针旋转 / T 顺时针旋转 (步长 5 度)")
    print("按 Q 完成旋转，进入ROI选择")
    print("按 S 跳过此图片")
    print("按 Esc 退出程序")
    print("-----------------")

    keep_rotating = True
    while keep_rotating:
        h, w = current_img.shape[:2]
        screen_height, screen_width = 700, 1000
        scale_rot = 1.0
        if w > 0 and h > 0:
            scale_rot = min(screen_width / w, screen_height / h, 1.0)

        display_width_rot = max(1, int(w * scale_rot))
        display_height_rot = max(1, int(h * scale_rot))

        try:
             cv2.resizeWindow(WINDOW_NAME_ROTATE, display_width_rot, display_height_rot)
             display_img_rot = cv2.resize(current_img, (display_width_rot, display_height_rot), interpolation=cv2.INTER_AREA)
             cv2.putText(display_img_rot, f'Angle: {angle:.1f}', (10, max(20, display_height_rot - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
             cv2.imshow(WINDOW_NAME_ROTATE, display_img_rot)
        except cv2.error as e:
            print(f"Error during rotation display setup: {e}")
            cv2.destroyWindow(WINDOW_NAME_ROTATE)
            return None, angle

        key = cv2.waitKey(CONFIG['wait_timeout']) & 0xFF

        rotate_step = 5.0
        needs_rotation_update = False
        if key == ord('r'):
            angle = (angle - rotate_step) % 360
            needs_rotation_update = True
        elif key == ord('t'):
            angle = (angle + rotate_step) % 360
            needs_rotation_update = True
        elif key == ord('q'):
            print(f"旋转完成，最终角度: {angle:.1f} 度")
            keep_rotating = False
        elif key == ord('s'):
            print("用户按 S 键，跳过当前图片")
            cv2.destroyWindow(WINDOW_NAME_ROTATE)
            return None, angle
        elif key == 27:
            print("用户按 ESC 键，退出程序")
            cv2.destroyWindow(WINDOW_NAME_ROTATE)
            exit(0)
        else:
            continue

        if needs_rotation_update:
            h_orig, w_orig = img.shape[:2]
            if w_orig > 0 and h_orig > 0:
                center = (w_orig // 2, h_orig // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                try:
                    current_img = cv2.warpAffine(img, matrix, (w_orig, h_orig), borderMode=cv2.BORDER_REPLICATE)
                except cv2.error as e:
                    print(f"Error during image rotation: {e}")
                    current_img = img.copy()
                    angle = 0.0
            else:
                print("Warning: Original image has zero dimensions, cannot rotate.")
                keep_rotating = False

    cv2.destroyWindow(WINDOW_NAME_ROTATE)

    # --- ROI 选择阶段 ---
    final_rotated_img = current_img
    roi_final_coords = None

    while True:
        print("\n--- ROI 选择方式 ---")
        print("1. OpenCV 内建选择器 (拖动鼠标, Enter/Space确认)")
        print("2. 自定义鼠标选择器 (拖动鼠标, Enter/Space确认)")
        print("3. 手动输入坐标 (M)")
        print("S: 跳过此图片 (不生成模板)")
        print("Esc: 退出程序")
        print("----------------------")
        choice = input("请选择 ROI 获取方式 (1/2/3/S/Esc): ").strip().lower()

        h_roi_img, w_roi_img = final_rotated_img.shape[:2]
        if h_roi_img <= 0 or w_roi_img <= 0:
            print("! 错误：旋转后的图像尺寸无效，无法选择ROI。")
            return None, angle

        screen_height_roi, screen_width_roi = 700, 1000
        scale_roi = 1.0
        if w_roi_img > 0 and h_roi_img > 0: # Avoid division by zero
            scale_roi = min(screen_width_roi / w_roi_img, screen_height_roi / h_roi_img, 1.0)
        display_width_roi = max(1, int(w_roi_img * scale_roi))
        display_height_roi = max(1, int(h_roi_img * scale_roi))

        try:
             display_img_roi = cv2.resize(final_rotated_img, (display_width_roi, display_height_roi), interpolation=cv2.INTER_AREA)
        except cv2.error as e:
             print(f"! 错误: 调整图像大小以选择ROI时失败: {e}")
             return None, angle

        roi_scaled = None

        if choice == '1':
            WINDOW_NAME_ROI_CV = '选择ROI (OpenCV 内建 - 拖动, Enter/Space确认, C取消)'
            cv2.namedWindow(WINDOW_NAME_ROI_CV, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME_ROI_CV, display_width_roi, display_height_roi)
            print("提示: 在弹出的窗口中拖动选择区域, 按 Enter/Space 确认, 按 C 取消.")
            try:
                # Ensure display_img_roi is valid before passing to selectROI
                if display_img_roi is None or display_img_roi.size == 0:
                     print("! 错误: 无法创建用于ROI选择的显示图像。")
                     cv2.destroyWindow(WINDOW_NAME_ROI_CV)
                     return None, angle # Indicate failure
                roi_scaled = cv2.selectROI(WINDOW_NAME_ROI_CV, display_img_roi, showCrosshair=False, fromCenter=False)
            except Exception as e:
                 print(f"! 调用 OpenCV selectROI 时出错: {e}")
                 roi_scaled = (0, 0, 0, 0)
            cv2.destroyWindow(WINDOW_NAME_ROI_CV)
            if roi_scaled == (0, 0, 0, 0) or not roi_scaled or roi_scaled[2] <= 0 or roi_scaled[3] <= 0:
                print("用户取消选择或选择无效 (OpenCV)。")
                roi_scaled = None

        elif choice == '2':
            WINDOW_NAME_ROI_CUSTOM = '选择ROI区域（自定义）'
            roi_scaled = custom_select_roi(WINDOW_NAME_ROI_CUSTOM, display_img_roi)
            if roi_scaled == (0, 0, 0, 0) or not roi_scaled or roi_scaled[2] <= 0 or roi_scaled[3] <= 0:
                print("用户取消选择或选择无效 (自定义)。")
                roi_scaled = None

        elif choice == 'm':
            print(f"当前图像尺寸 (旋转后): 宽={w_roi_img}, 高={h_roi_img}")
            try:
                x_m_str = input("请输入 ROI 的 x 坐标 (左上角): ")
                y_m_str = input("请输入 ROI 的 y 坐标 (左上角): ")
                w_m_str = input("请输入 ROI 的宽度 w: ")
                h_m_str = input("请输入 ROI 的高度 h: ")
                x_m = int(x_m_str)
                y_m = int(y_m_str)
                w_m = int(w_m_str)
                h_m = int(h_m_str)
                if x_m < 0 or y_m < 0 or w_m <= 0 or h_m <= 0 or x_m + w_m > w_roi_img or y_m + h_m > h_roi_img:
                    print("错误：输入的 ROI 坐标无效或超出图像边界，请重试。")
                    continue
                else:
                    roi_final_coords = (x_m, y_m, w_m, h_m)
                    break
            except ValueError:
                print("错误：请输入有效的整数坐标，请重试。")
                continue
            except Exception as e:
                 print(f"输入时发生错误: {e}")
                 continue

        elif choice == 's':
            print("用户按 S 键，跳过当前图片 (不生成模板)")
            return None, angle

        elif choice == 'esc':
             print("用户按 ESC 键，退出程序")
             exit(0)
        elif choice == '':
             print("无效选择 (直接按回车)，请选择 1, 2, 3, S 或 Esc.")
             continue
        else:
            print("无效选择，请重新输入。")
            continue

        if roi_scaled:
            x_s, y_s, w_s, h_s = roi_scaled
            if scale_roi > 1e-6:
                x = int(x_s / scale_roi)
                y = int(y_s / scale_roi)
                w = int(w_s / scale_roi)
                h = int(h_s / scale_roi)
            else:
                print("! 错误: 显示缩放比例为零，无法转换ROI坐标。")
                continue

            x = max(0, x)
            y = max(0, y)
            if x + w > w_roi_img:
                w = w_roi_img - x
            if y + h > h_roi_img:
                h = h_roi_img - y

            if w > 0 and h > 0:
                 roi_final_coords = (x, y, w, h)
                 print(f"最终 ROI (原始尺寸坐标): {roi_final_coords}")
                 break
            else:
                 print("警告: ROI 转换或修正后无效 (宽或高<=0)。请重试。")

    return roi_final_coords, angle

def read_template_index(index_path):
    """
    读取模板索引 CSV 文件到字典中。
    字典格式: { "source_filename": [row_data_list], ... }
    """
    template_index = {} # 初始化空字典
    if not os.path.exists(index_path):
        print(f"信息：模板索引文件未找到: {index_path}，将创建新的索引。")
        return template_index # 返回空字典
    try:
        with open(index_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None) # 读取表头
            expected_header = ["SourceFilename", "TemplateName", "ROI_X", "ROI_Y", "ROI_W", "ROI_H", "GenerationAngle"]
            line_num = 1
            if not header or header != expected_header:
                 print(f"警告: 模板索引 '{index_path}' 的表头格式不符合预期或缺失。")
                 print(f"  预期: {expected_header}")
                 print(f"  找到: {header}")
                 print(f"  将尝试在没有有效表头的情况下读取。")
                 f.seek(0)
                 line_num = 0

            for row in reader:
                 line_num += 1
                 if len(row) >= len(expected_header):
                     filename = row[0].strip()
                     if not filename:
                          #print(f"警告: 索引文件第 {line_num} 行缺少源文件名，已跳过。")
                          continue
                     try:
                         if row[1].strip().upper() == "NONE":
                             roi_vals = list(map(int, row[2:6]))
                             angle_val = float(row[6])
                             if any(v != 0 for v in roi_vals) or abs(angle_val) > 1e-6:
                                #print(f"警告: 索引文件第 {line_num} 行 TemplateName 是 NONE 但 ROI/Angle 不为零。")
                                pass # Allow this inconsistency
                         else:
                             list(map(int, row[2:6]))
                             float(row[6])
                     except ValueError:
                          print(f"警告: 索引文件第 {line_num} 行 ROI 或角度格式错误，已跳过: {row}")
                          continue
                     except IndexError:
                          print(f"警告: 索引文件第 {line_num} 行缺少列，已跳过: {row}")
                          continue

                     #if filename in template_index:
                         #print(f"警告: 索引文件第 {line_num} 行发现重复的源文件名 '{filename}'。将使用此最新条目。")
                     template_index[filename] = row
                 else:
                     print(f"警告: 索引文件第 {line_num} 行格式不完整，已跳过: {row}")
        #print(f"成功加载 {len(template_index)} 条模板索引记录从 '{index_path}'。")
    except Exception as e:
        print(f"读取模板索引 '{index_path}' 时发生严重错误: {e}。将视为无现有索引。")
        traceback.print_exc()
        return {}
    return template_index

def write_template_index(index_path, template_index_dict):
    """
    将内存中的模板索引字典写入 CSV 文件 (覆盖写入)。
    """
    temp_index_path = index_path + ".tmp"
    try:
        with open(temp_index_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ["SourceFilename", "TemplateName", "ROI_X", "ROI_Y", "ROI_W", "ROI_H", "GenerationAngle"]
            writer.writerow(header)
            sorted_filenames = sorted(template_index_dict.keys())
            for filename in sorted_filenames:
                writer.writerow(template_index_dict[filename])
        # Atomic move/replace
        shutil.move(temp_index_path, index_path)
        #print(f"模板索引已成功写入/更新到: {index_path}")
    except Exception as e:
        print(f"写入模板索引 '{index_path}' 时发生错误: {e}")
        traceback.print_exc()
        if os.path.exists(temp_index_path):
             try:
                 os.remove(temp_index_path)
                 print(f"已删除临时索引文件: {temp_index_path}")
             except Exception as e_rem:
                 print(f"删除临时索引文件失败: {e_rem}")

# ==========================================
# HELPER FUNCTION for template deletion
# ==========================================
def _delete_all_templates_for_file(filename, template_dir):
    """Finds and deletes all template files associated with a given filename."""
    base_name = os.path.splitext(filename)[0]
    # Search for common image extensions used for templates
    # Use glob patterns that match base and numbered versions
    search_patterns = [
        os.path.join(template_dir, f"template_{base_name}.png"),
        os.path.join(template_dir, f"template_{base_name}_*.png"),
        os.path.join(template_dir, f"template_{base_name}.jpg"),
        os.path.join(template_dir, f"template_{base_name}_*.jpg"),
        os.path.join(template_dir, f"template_{base_name}.jpeg"),
        os.path.join(template_dir, f"template_{base_name}_*.jpeg"),
    ]

    files_to_delete = set() # Use a set to avoid duplicates
    for pattern in search_patterns:
        # Use glob to find files matching the pattern
        # Note: glob itself handles the wildcard '*' correctly
        found_files = glob.glob(pattern)
        files_to_delete.update(found_files)

    deleted_count = 0
    failed_count = 0
    if not files_to_delete:
        print(f"  - 未找到与 '{filename}' 关联的模板文件进行删除。")
        return deleted_count, failed_count

    print(f"  - 准备删除以下与 '{filename}' 关联的模板文件:")
    for f_path in sorted(list(files_to_delete)): # Sort for consistent output
        print(f"    - {os.path.basename(f_path)}")
        try:
            os.remove(f_path)
            deleted_count += 1
        except Exception as e:
            print(f"  ! 删除模板文件失败: {f_path} - {e}")
            failed_count += 1
    print(f"  - 模板文件删除完成 (成功: {deleted_count}, 失败: {failed_count})")
    return deleted_count, failed_count

# =======================================
# MODIFIED: generate_templates function
# =======================================
def generate_templates(items_to_process):
    """
    为下载列表中的图片生成/更新模板文件。
    交互式操作：添加新模板、删除所有模板、标记为无模板、保留或跳过。
    Args:
        items_to_process (list): 包含 (url, filename) 元组的列表。
    """
    os.makedirs(CONFIG["template_dir"], exist_ok=True)
    index_file = CONFIG["template_index"]

    template_index_data = read_template_index(index_file)
    original_index_count = len(template_index_data)

    # Counters
    added_count = 0      # 新添加的模板文件数量 (via R)
    deleted_files_count = 0 # 通过 D 或 N 删除的文件总数
    none_marked_count = 0 # 标记为无模板的图片数量 (via N or D)
    skipped_count = 0   # 本次跳过或保留的数量 (via S or K)
    source_missing_skipped_count = 0 # 因源文件丢失而跳过的数量
    deleted_index_missing_source_count = 0 # 因源文件丢失删除的索引条目数
    error_count = 0     # 处理时发生错误

    total_items = len(items_to_process)
    print(f"\n--- 开始生成/管理 {total_items} 个文件的模板 ---")

    for i, (url, filename) in enumerate(items_to_process):
        img_path = os.path.join(CONFIG["download_dir"], filename)
        base_name = os.path.splitext(filename)[0] # Get base name early

        print(f"\n[{i+1}/{total_items}] 处理模板: {filename}")

        # --- 检查源图片是否存在 ---
        if not os.path.exists(img_path):
            print(f"  - 源图片文件不存在，跳过模板处理: {img_path}")
            source_missing_skipped_count += 1
            if filename in template_index_data:
                print(f"  - 从索引中移除丢失文件的记录: {filename}")
                template_index_data.pop(filename, None)
                deleted_index_missing_source_count += 1
            continue

        # --- 文件存在，检查现有记录并与用户交互 ---
        user_action = None
        was_existing_in_index = filename in template_index_data

        if was_existing_in_index:
            try:
                existing_info = template_index_data[filename]
                template_name_in_index = existing_info[1]
                if template_name_in_index == "NONE":
                     print(f"  - 文件 '{filename}' 当前在索引中标记为 [无模板 (NONE)]。")
                else:
                    roi_in_index = f"({','.join(map(str, existing_info[2:6]))})"
                    angle_in_index = existing_info[6]
                    print(f"  - 文件 '{filename}' 在索引中的记录 (指向最后操作的模板):") # Clarify index meaning
                    print(f"    模板名: {template_name_in_index}, ROI: {roi_in_index}, 角度: {angle_in_index}")
            except (IndexError, KeyError):
                print(f"  - 文件 '{filename}' 的现有索引记录格式错误，将视为无记录。")
                was_existing_in_index = False

        else:
             print(f"  - 文件 '{filename}' 无索引记录。")

        # New Prompt with updated options
        prompt = (f"  请选择操作: [R]添加新模板, [D]删除所有模板, [K]保留现有, [N]标记为无模板, [S]本次跳过? (默认: K 保留): ")
        choice = input(prompt).strip().lower()

        if choice == 'r':
            user_action = 'add_new'
        elif choice == 'd':
            user_action = 'delete_all'
        elif choice == 'n':
            user_action = 'mark_none'
        elif choice == 's':
            user_action = 'skip'
        else:
            user_action = 'keep'
            if choice != 'k' and choice != '':
                 print("  - 无效输入，默认保留现有状态。")

        # --- 根据用户选择执行操作 ---
        if user_action == 'skip' or user_action == 'keep':
            print(f"  - {'跳过' if user_action == 'skip' else '保留现有状态'} 文件: {filename}")
            skipped_count += 1
            continue

        elif user_action == 'delete_all' or user_action == 'mark_none':
            action_desc = "删除所有模板并标记为无模板" if user_action == 'mark_none' else "删除所有关联模板"
            print(f"  - 执行操作: {action_desc} for '{filename}'")
            deleted_now, failed_now = _delete_all_templates_for_file(filename, CONFIG["template_dir"])
            deleted_files_count += deleted_now
            if failed_now > 0: error_count += failed_now

            # Update index to NONE
            template_index_data[filename] = [filename, "NONE", 0, 0, 0, 0, "0.00"]
            print(f"  - 索引已更新为 'NONE' for {filename}")
            if user_action == 'mark_none' or user_action == 'delete_all':
                 # Count only once if marked as none, regardless of previous state
                 # Check if it wasn't already NONE to count as a change
                 if was_existing_in_index and existing_info[1] != "NONE":
                     none_marked_count += 1
                 elif not was_existing_in_index: # If it was new and marked none
                     none_marked_count += 1


        elif user_action == 'add_new':
            print(f"  - 开始为 '{filename}' 添加新模板...")
            result = select_roi_interactive(img_path)

            if result is None:
                print(f"  ! 获取 ROI 时发生意外错误，跳过: {filename}")
                error_count += 1
                continue

            roi_coords_final, generation_angle = result

            if roi_coords_final is None or roi_coords_final == (0,0,0,0):
                print(f"  - 用户在选择 ROI 时取消或跳过，未添加新模板。")
                skipped_count += 1
                continue

            x, y, w, h = roi_coords_final
            print(f"  - 已选择有效ROI (原始尺寸坐标): x={x}, y={y}, w={w}, h={h}, 角度={generation_angle:.1f}°")

            # --- 确定新模板的文件名 ---
            next_num = 1
            new_template_name = ""
            new_template_path = ""

            template_name_base_png = f"template_{base_name}.png"
            template_path_base_png = os.path.join(CONFIG["template_dir"], template_name_base_png)
            template_name_base_jpg = f"template_{base_name}.jpg" # Check common alternatives too
            template_path_base_jpg = os.path.join(CONFIG["template_dir"], template_name_base_jpg)
            template_name_base_jpeg = f"template_{base_name}.jpeg"
            template_path_base_jpeg = os.path.join(CONFIG["template_dir"], template_name_base_jpeg)

            # Use PNG as the primary target format
            if not os.path.exists(template_path_base_png) and \
               not os.path.exists(template_path_base_jpg) and \
               not os.path.exists(template_path_base_jpeg):
                new_template_name = template_name_base_png
                new_template_path = template_path_base_png
                print(f"  - 将创建第一个模板: {new_template_name}")
            else:
                # Base exists (any format), find the next available number _n for PNG
                while True:
                    potential_name = f"template_{base_name}_{next_num}.png"
                    potential_path = os.path.join(CONFIG["template_dir"], potential_name)
                    # Also check for numbered jpg/jpeg just in case to avoid naming clash sense
                    potential_path_jpg = os.path.join(CONFIG["template_dir"], f"template_{base_name}_{next_num}.jpg")
                    potential_path_jpeg = os.path.join(CONFIG["template_dir"], f"template_{base_name}_{next_num}.jpeg")

                    if not os.path.exists(potential_path) and \
                       not os.path.exists(potential_path_jpg) and \
                       not os.path.exists(potential_path_jpeg):
                        new_template_name = potential_name
                        new_template_path = potential_path
                        print(f"  - 将创建下一个编号模板: {new_template_name}")
                        break
                    next_num += 1
                    if next_num > 999: # Safety break
                         print("  ! 无法找到下一个可用的模板编号 (已达 999)，无法添加。")
                         error_count += 1
                         new_template_path = None
                         break

            if not new_template_path:
                continue

            # --- 裁剪模板 ---
            template_img = None
            try:
                img_orig_for_crop = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img_orig_for_crop is None: raise ValueError("无法重新读取原始图像进行裁剪")

                rotated_img_for_crop = img_orig_for_crop
                if abs(generation_angle) > 1e-6:
                     h_orig_crop, w_orig_crop = img_orig_for_crop.shape[:2]
                     if w_orig_crop > 0 and h_orig_crop > 0:
                         center_crop = (w_orig_crop // 2, h_orig_crop // 2)
                         matrix_crop = cv2.getRotationMatrix2D(center_crop, generation_angle, 1.0)
                         rotated_img_for_crop = cv2.warpAffine(img_orig_for_crop, matrix_crop, (w_orig_crop, h_orig_crop), borderMode=cv2.BORDER_REPLICATE)
                     else: raise ValueError("原始图像尺寸无效")

                h_rot_crop, w_rot_crop = rotated_img_for_crop.shape[:2]
                x_c = max(0, x); y_c = max(0, y)
                w_c = min(w, w_rot_crop - x_c); h_c = min(h, h_rot_crop - y_c)
                if w_c <= 0 or h_c <= 0: raise ValueError(f"ROI裁剪后尺寸无效 ({w_c}x{h_c})")

                template_img = rotated_img_for_crop[y_c : y_c + h_c, x_c : x_c + w_c]
                if template_img is None or template_img.size == 0: raise ValueError("裁剪模板结果为空")

            except Exception as crop_e:
                 print(f"  × 裁剪模板时出错: {crop_e}")
                 error_count += 1
                 continue

            # --- 保存新模板文件 ---
            try:
                # Save as PNG
                cv2.imwrite(new_template_path, template_img)
                template_index_data[filename] = [filename, new_template_name, x, y, w, h, f"{generation_angle:.2f}"]
                print(f"  ✓ 新模板已添加并保存: {new_template_name}")
                print(f"  - 索引已更新以反映最新添加的模板信息。")
                added_count += 1
            except Exception as e:
                print(f"  × 保存新模板失败: {new_template_name} - {str(e)}")
                error_count += 1
                if os.path.exists(new_template_path):
                    try: os.remove(new_template_path)
                    except: pass
                continue

    # --- Loop End ---
    write_template_index(index_file, template_index_data)

    print("-" * 30)
    final_index_count = len(template_index_data)
    print(f"模板生成/管理总结:")
    print(f"  - 新添加模板文件: {added_count}")
    print(f"  - 标记为无模板 (N/D操作): {none_marked_count}")
    print(f"  - 通过 D/N 操作删除的模板文件总数: {deleted_files_count}")
    print(f"  - 本次跳过/保留现有状态: {skipped_count}")
    print(f"  - 因源文件丢失跳过: {source_missing_skipped_count}")
    print(f"  - 因源文件丢失删除索引条目: {deleted_index_missing_source_count}")
    print(f"  - 处理时发生错误: {error_count}")
    print(f"  - 最终索引中记录数: {final_index_count} (原始: {original_index_count})")
    print(f"模板索引已更新: {index_file}")
    print("-" * 30)

# =======================================
# Crop with Multiple Templates function (No changes needed here)
# =======================================
def crop_with_multiple_templates(img_path, output_path, template_paths):
    """
    Attempts to crop an image using a list of templates. Stops on the first successful match.
    Args:
        img_path (str): Path to the source image.
        output_path (str): Path to save the cropped result.
        template_paths (list): A list of potential template file paths to try, in order.
    Returns:
        tuple: (success, message)
    """
    try:
        img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_color is None: return False, f"无法读取主图像: {os.path.basename(img_path)}"
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        h_img_orig, w_img_orig = img_gray.shape[:2]
        if h_img_orig <= 0 or w_img_orig <= 0: return False, f"主图像尺寸无效: {w_img_orig}x{h_img_orig}"
        center_orig = (w_img_orig / 2, h_img_orig / 2)

        for template_path in template_paths:
            try:
                template_color = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
                if template_color is None:
                    print(f"  ! 警告: 无法读取模板 {os.path.basename(template_path)}，跳过此模板")
                    continue
                template_gray = None; mask = None
                if template_color.ndim == 2: template_gray = template_color
                elif template_color.ndim == 3: template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
                elif template_color.ndim == 4:
                    template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGRA2GRAY)
                    alpha = template_color[:, :, 3]
                    _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
                else:
                    print(f"  ! 警告: 无法处理模板颜色格式 {os.path.basename(template_path)} (维度: {template_color.ndim})，跳过")
                    continue
                template_h, template_w = template_gray.shape[:2]
                if template_h == 0 or template_w == 0:
                    print(f"  ! 警告: 模板尺寸无效 {os.path.basename(template_path)} ({template_w}x{template_h})，跳过")
                    continue
            except Exception as e:
                 print(f"  ! 警告: 加载/处理模板 {os.path.basename(template_path)} 时出错: {e}，跳过")
                 continue

            print(f"  - 尝试匹配模板: {os.path.basename(template_path)}")
            best_match_info = {'val': -1.0, 'loc': None, 'angle': 0.0, 'rotated_img_color': None}
            match_threshold = CONFIG['match_threshold']
            step = CONFIG["rotation_step"]; range_ = CONFIG["rotation_range"]
            angles_to_try = np.linspace(-range_, range_, int(2 * range_ / step) + 1).tolist()
            if 0.0 not in angles_to_try: angles_to_try.extend([0.0]); angles_to_try.sort()

            for angle in angles_to_try:
                angle = round(angle, 2)
                try:
                    rot_matrix = cv2.getRotationMatrix2D(center_orig, angle, 1.0)
                    abs_cos = abs(rot_matrix[0, 0]); abs_sin = abs(rot_matrix[0, 1])
                    new_w = int(h_img_orig * abs_sin + w_img_orig * abs_cos)
                    new_h = int(h_img_orig * abs_cos + w_img_orig * abs_sin)
                    rot_matrix[0, 2] += (new_w / 2) - center_orig[0]
                    rot_matrix[1, 2] += (new_h / 2) - center_orig[1]
                    rotated_gray = cv2.warpAffine(img_gray, rot_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                except cv2.error: continue
                if template_h > rotated_gray.shape[0] or template_w > rotated_gray.shape[1]: continue

                method = cv2.TM_CCOEFF_NORMED
                try:
                    result = cv2.matchTemplate(rotated_gray, template_gray, method, mask=mask)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                except cv2.error: continue

                if max_val > best_match_info['val']:
                    best_match_info['val'] = max_val; best_match_info['loc'] = max_loc; best_match_info['angle'] = angle
                    try:
                        rotated_color = cv2.warpAffine(img_color, rot_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                        best_match_info['rotated_img_color'] = rotated_color
                    except cv2.error as e:
                         print(f"  ! 旋转彩色图时出错 (角度 {angle:.1f}°): {e}")
                         best_match_info['val'] = -1.0; continue

            best_val = best_match_info['val']; best_loc = best_match_info['loc']; best_rotated_color_img = best_match_info['rotated_img_color']
            print(f"    > 模板 '{os.path.basename(template_path)}' 最佳匹配分: {best_val:.4f} (阈值: {match_threshold})")

            if best_val >= match_threshold and best_loc is not None and best_rotated_color_img is not None:
                print(f"  ✓ 找到成功匹配! 使用模板: {os.path.basename(template_path)}")
                match_x, match_y = best_loc
                x_min = match_x; y_min = match_y; x_max = match_x + template_w; y_max = match_y + template_h
                h_rot, w_rot = best_rotated_color_img.shape[:2]
                x_min_c = max(0, x_min); y_min_c = max(0, y_min)
                x_max_c = min(w_rot, x_max); y_max_c = min(h_rot, y_max)
                crop_w = x_max_c - x_min_c; crop_h = y_max_c - y_min_c

                if crop_w <= 0 or crop_h <= 0:
                    print(f"  ! 裁剪区域无效 (w={crop_w}, h={crop_h}) for template {os.path.basename(template_path)}. 尝试下一个...")
                    continue
                cropped_img = best_rotated_color_img[y_min_c:y_max_c, x_min_c:x_max_c]
                if cropped_img is None or cropped_img.size == 0:
                    print(f"  ! 裁剪结果为空 (模板: {os.path.basename(template_path)}). 尝试下一个...")
                    continue
                try:
                    cv2.imwrite(output_path, cropped_img)
                    return True, f"成功使用模板 '{os.path.basename(template_path)}' 裁剪 (匹配度 {best_val:.4f})"
                except Exception as save_e:
                    print(f"  ! 保存裁剪结果失败 (模板: {os.path.basename(template_path)}): {save_e}")
                    continue
        return False, "所有尝试的模板均未成功匹配、裁剪或保存"
    except Exception as e:
        print(f"! 处理 '{os.path.basename(img_path)}' 时发生严重错误 (crop_with_multiple_templates): {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, f"处理时发生严重异常: {type(e).__name__}"


# ========================================
# process_images function (No changes needed here)
# ========================================
def process_images(items_to_process):
    """
    处理下载列表中的所有图片 (强制处理，覆盖输出目录):
    - Searches for templates like `template_{base_filename}*.png/jpg/jpeg`.
    - Tries matching each found template in order using `crop_with_multiple_templates`.
    - If any template matches successfully, saves the crop and stops for that image.
    - If no template matches, copies the original picture to output_dir (covering).
    - Deletes the original file from the downloads directory after processing.
    Args:
        items_to_process (list): 包含 (url, filename) 元组的列表。
    """
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["template_dir"], exist_ok=True)

    processed_count = 0; cropped_count = 0; copied_count = 0
    skipped_source_missing_count = 0; failed_processing_count = 0
    deleted_count = 0; delete_failed_count = 0
    total_items = len(items_to_process)
    print(f"\n--- 开始处理 {total_items} 个图片 (目标: '{CONFIG['output_dir']}', 将覆盖现有输出) ---")

    for i, (url, filename) in enumerate(items_to_process):
        img_path = os.path.join(CONFIG["download_dir"], filename)
        output_path = os.path.join(CONFIG["output_dir"], filename)
        base_name = os.path.splitext(filename)[0]
        print(f"\n[{i+1}/{total_items}] 处理图片: {filename}")
        output_generated = False

        if not os.path.exists(img_path):
            print(f"  ! 跳过处理：未找到源图片文件 {img_path}")
            skipped_source_missing_count += 1; continue

        potential_template_paths = []
        try:
            search_dir = CONFIG["template_dir"]
            patterns = [f"template_{base_name}*.png", f"template_{base_name}*.jpg", f"template_{base_name}*.jpeg"]
            found_templates = set()
            for pattern in patterns:
                found_templates.update(glob.glob(os.path.join(search_dir, pattern)))

            main_templates = {
                'png': os.path.join(search_dir, f"template_{base_name}.png"),
                'jpg': os.path.join(search_dir, f"template_{base_name}.jpg"),
                'jpeg': os.path.join(search_dir, f"template_{base_name}.jpeg")
            }
            main_template_to_prioritize = None
            for ext in ['png', 'jpg', 'jpeg']: # Prioritize order
                 if main_templates[ext] in found_templates:
                     main_template_to_prioritize = main_templates[ext]; break

            sorted_templates = sorted(list(found_templates))
            if main_template_to_prioritize:
                 if main_template_to_prioritize in sorted_templates:
                     sorted_templates.remove(main_template_to_prioritize)
                 potential_template_paths = [main_template_to_prioritize] + sorted_templates
            else: potential_template_paths = sorted_templates
        except Exception as e: print(f"  ! 搜索模板时出错: {e}")

        crop_success = False; crop_message = "没有找到模板文件"
        if potential_template_paths:
            print(f"  - 找到 {len(potential_template_paths)} 个潜在模板，将按顺序尝试:")
            for tpath in potential_template_paths: print(f"    - {os.path.basename(tpath)}")
            crop_success, crop_message = crop_with_multiple_templates(img_path, output_path, potential_template_paths)
            if crop_success:
                output_generated = True; processed_count += 1; cropped_count += 1
                print(f"  ✓ 处理完成 (已裁剪覆盖/创建输出): {filename} ({crop_message})")
            else: print(f"  ! 所有模板尝试失败: {crop_message}")
        else: print(f"  - 未找到匹配 'template_{base_name}*.png/jpg/jpeg' 的模板文件。")

        if not crop_success:
            print(f"  - (后备操作) 尝试直接复制原始文件到输出目录 (覆盖)...")
            try:
                if os.path.exists(output_path): os.remove(output_path)
                shutil.copy(img_path, output_path)
                output_generated = True; processed_count += 1; copied_count += 1
                print(f"  ✓ 复制成功 (已覆盖/创建输出): {filename}")
            except Exception as e:
                failed_processing_count += 1; output_generated = False
                print(f"  × 后备复制原始文件失败: {e}")

        if os.path.exists(img_path):
            try:
                os.remove(img_path); deleted_count += 1
                if output_generated: print(f"  - 已删除下载目录中的原始文件: {filename}")
                else: print(f"  - (注意) 处理失败，但仍尝试删除下载目录中的原始文件: {filename}")
            except Exception as e:
                delete_failed_count += 1
                print(f"  ! 删除下载目录原始文件失败: {filename} - {e}")

    print("-" * 30)
    print(f"图片处理总结:")
    print(f"  - 成功生成/覆盖输出文件总数: {processed_count}")
    print(f"    - 通过模板裁剪生成: {cropped_count}")
    print(f"    - 通过复制原始图片生成: {copied_count}")
    print(f"  - 跳过 (源文件缺失): {skipped_source_missing_count}")
    print(f"  - 处理失败 (裁剪和后备复制均失败): {failed_processing_count}")
    print(f"  - 下载目录原始文件删除成功: {deleted_count}")
    if delete_failed_count > 0: print(f"  - 下载目录原始文件删除失败: {delete_failed_count}")
    print(f"最终处理结果保存在: {os.path.abspath(CONFIG['output_dir'])}")
    print(f"下载目录 '{CONFIG['download_dir']}' 中的文件已被尝试清理。")
    print("-" * 30)


# ================================================
# Main Program Entry Point
# ================================================
if __name__ == "__main__":
    # --- Print Header ---
    print("=" * 50)
    print("=== 自动化图片处理系统 (菜单模式) ===")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"OpenCV 版本: {cv2.__version__}")
    print(f"配置:")
    for key, value in CONFIG.items():
        if key.endswith("_dir") or key.endswith("_file") or key.endswith("_index"):
             try: print(f"  - {key}: {os.path.abspath(str(value))}")
             except Exception: print(f"  - {key}: {value}")
        else: print(f"  - {key}: {value}")
    print("-" * 50)

    # --- Read download list ---
    download_items = []
    try:
        print("\n--- 正在读取下载列表 ---")
        download_list_file = CONFIG['download_list_file']
        download_items = read_download_list(download_list_file) # Definition MUST exist above
        if not download_items: print(f"警告：下载列表 '{download_list_file}' 为空或读取失败。")
        else: print(f"已成功加载 {len(download_items)} 条记录。")
        print("-" * 30)
    except NameError as ne:
        print(f"\n! 严重错误: 函数 '{ne.name}' 未定义。请确保脚本是完整的。")
        traceback.print_exc(); exit(1)
    except Exception as e:
        print(f"\n! 读取下载列表时发生严重错误: {type(e).__name__}: {str(e)}")
        traceback.print_exc(); exit(1)

    # --- Main Menu Loop ---
    while True:
        print("\n--- 主菜单 ---")
        print("1. 下载图片 (根据列表)")
        print("2. 生成/管理模板 (交互式: 添加, 删除, 标记None)") # Updated menu text
        print("3. 处理图片 (查找多模板, 强制处理下载目录, 覆盖输出并清理下载)")
        print("4. 下载并处理图片 (选项1 + 选项3)")
        print("5. 退出程序")
        print("--------------")
        choice = input("请输入选项数字: ").strip()

        try:
            if choice == '1':
                print("\n=== 选项 1: 下载图片 ===")
                if not download_items: print("警告/错误：下载列表为空或未成功加载，无法执行下载。")
                else: download_images(download_items)
                print("-" * 30); print("下载流程结束。")

            elif choice == '2':
                print("\n=== 选项 2: 生成/管理模板 ===")
                if not download_items: print("错误：下载列表为空或未成功加载，无法确定要为哪些文件管理模板。")
                elif not os.path.isdir(CONFIG["download_dir"]): print(f"错误: 下载目录 '{CONFIG['download_dir']}' 不存在。请先运行选项 1 下载图片。")
                else: generate_templates(download_items) # Call the MODIFIED function
                print("-" * 30); print("模板生成/管理流程结束。")

            elif choice == '3':
                print("\n=== 选项 3: 处理图片 (多模板, 强制) ===")
                if not download_items: print("错误：下载列表为空或未成功加载，无法确定要处理哪些图片。")
                elif not os.path.isdir(CONFIG["download_dir"]):
                     print(f"警告: 下载目录 '{CONFIG['download_dir']}' 不存在。处理将跳过所有文件。")
                     process_images(download_items)
                else: process_images(download_items)
                print("-" * 30); print("图片处理流程结束。")

            elif choice == '4':
                 print("\n=== 选项 4: 下载并处理图片 ===")
                 if not download_items: print("错误：下载列表为空或未成功加载，无法执行此流程。")
                 else:
                     print("\n--- 第 1 步: 下载图片 ---")
                     download_images(download_items)
                     print("-" * 30); print("下载流程结束。")
                     print("\n--- 第 2 步: 处理图片 (多模板, 强制) ---")
                     if not os.path.isdir(CONFIG["download_dir"]):
                          print(f"警告: 下载目录 '{CONFIG['download_dir']}' 在下载后仍不存在或为空。处理步骤将跳过。")
                          process_images(download_items)
                     else: process_images(download_items)
                     print("-" * 30); print("图片处理流程结束。")
                 print("\n=== 选项 4 流程结束 ===")

            elif choice == '5': print("退出程序..."); break
            else: print("无效选项，请输入 1, 2, 3, 4 或 5。")

        except KeyboardInterrupt: print("\n! 用户中断操作。返回主菜单。"); time.sleep(1)
        except Exception as e:
             print(f"\n! 在执行选项 '{choice}' 时发生未捕获的严重错误: {type(e).__name__}: {str(e)}")
             traceback.print_exc(); print("操作可能未完成。返回主菜单。"); time.sleep(2)

    # --- End of Program ---
    print("=" * 50); print("程序已退出."); cv2.destroyAllWindows()