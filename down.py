# -*- coding: utf-8 -*- # 指定编码
import os
import time
import cv2
import numpy as np
import requests
import shutil
from datetime import datetime
import csv # 导入 csv 模块，方便处理索引
import traceback # For better error reporting
import glob # 用于文件搜索

# ====================
# 配置区（按需修改）
# ====================
CONFIG = {
    "download_list_file": "download_list.txt",  # 下载列表文件路径
    "template_dir": "templates",      # 模板存储目录
    "template_index": "templates_index.csv",  # 模板索引文件 (使用 .csv)
    "output_dir": "processed",        # 处理结果目录
    "download_dir": "downloads",      # 下载存储目录
    "archive_dir": "processed_originals", # 处理完成的原始文件移动目录 - 未在当前流程中使用
    "timeout": 15,                    # 下载超时(秒)
    "max_retries": 3,                 # 最大重试次数
    "retry_delay": 2,                 # 重试等待基数(秒)
    # --- 模板匹配和旋转配置 ---
    "match_threshold": 0.2,          # 模板匹配相似度阈值 (!!! 重要：根据"不用要求那么准确"的需求，你可能需要降低此值, e.g., 0.6 or 0.55 !!!)
    "rotation_range": 15,             # 旋转校正搜索范围（±15度）- 确保覆盖你的图像倾斜范围
    "rotation_step": 1,               # 旋转搜索步长（1度）- 越小越准但越慢
    # --------------------------
    "wait_timeout": 0                 # cv2.waitKey(0) 表示无限等待用户按键
}

# ====================
# 全局变量 (用于自定义ROI)
# ====================
drawing = False # 标记是否正在绘制矩形
ix, iy = -1, -1 # 鼠标按下时的初始坐标
roi_coords = None # 存储最终选择的 ROI 坐标 (x, y, w, h)
img_copy = None # 存储当前显示图像的副本，用于绘制

# ====================
# 鼠标回调函数 (自定义ROI)
# ====================
def mouse_callback(event, x, y, flags, param):
    """鼠标事件回调函数，用于自定义 ROI 选择"""
    global drawing, ix, iy, roi_coords, img_copy

    if img_copy is None: # 防止 img_copy 未设置时出错
        return

    window_title = '选择ROI区域（自定义）' # 窗口标题

    if event == cv2.EVENT_LBUTTONDOWN: # 鼠标左键按下
        drawing = True
        ix, iy = x, y

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
            img_temp = img_copy.copy()
            # 在副本上绘制最终的、稍粗的绿色矩形
            cv2.rectangle(img_temp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(window_title, img_temp)
        else:
             roi_coords = None # 重置无效选择

# ====================
# 自定义ROI选择函数
# ====================
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
        # 如果不在绘制且已有有效选区，显示带选框的图
        if not drawing and roi_coords:
             img_display = img_copy.copy()
             x,y,w,h = roi_coords
             cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
             cv2.imshow(window_name, img_display)
        # 否则显示原图 (或上次绘制的预览，如果鼠标刚松开)
        elif not drawing:
             cv2.imshow(window_name, img_copy)

        key = cv2.waitKey(20) & 0xFF # 等待按键

        # 确认选择 (Enter 或 Space)
        if key == 13 or key == 32:
            if roi_coords is not None:
                x, y, w, h = roi_coords
                if w > 0 and h > 0:
                    print(f"确认ROI (显示坐标): {roi_coords}")
                    cv2.destroyWindow(window_name)
                    return roi_coords
                else:
                    print("警告: 选择的ROI无效 (宽或高为0). 请重新选择或按 C/ESC.")
                    roi_coords = None # 重置无效选择
            else:
                print("未选择有效区域，请拖动鼠标选择，或按 C 键跳过，或按 ESC 退出")

        # 取消选择 (C)
        elif key == ord('c'):
            print("取消选择 (C)")
            cv2.destroyWindow(window_name)
            return (0, 0, 0, 0)

        # 退出程序 (ESC)
        elif key == 27:
            print("用户按 ESC 键，退出程序")
            cv2.destroyWindow(window_name)
            exit(0)

# ====================
# 读取下载列表函数 (!!! 这个是之前缺失的 !!!)
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
                    # 保留点号 '.' 用于扩展名
                    invalid_chars = '/\\:*?"<>|'
                    safe_name = filename_raw
                    for char in invalid_chars:
                        safe_name = safe_name.replace(char, '_')
                    safe_name = safe_name.replace("\0", "") # 移除 null 字符

                    # 如果文件名没有扩展名，则尝试从 URL 推断或添加默认扩展名
                    base_filename = os.path.basename(safe_name)
                    if '.' not in base_filename:
                         try:
                             # 尝试从 URL 路径获取扩展名
                             path_from_url = requests.utils.urlparse(url).path
                             ext_from_url = os.path.splitext(path_from_url)[1]
                             # 简单的扩展名检查 (例如 .jpg, .png)
                             if ext_from_url and len(ext_from_url) > 1 and len(ext_from_url) <= 5:
                                 safe_name += ext_from_url
                             else:
                                 raise ValueError("No valid extension in URL path")
                         except Exception:
                             # 如果无法从 URL 推断，使用默认扩展名
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

# ====================
# 下载图片函数
# ====================
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

        # 检查文件是否已存在且大小是否合理
        if os.path.exists(save_path):
            try:
                # 小于 1KB 认为可能不完整，尝试重新下载
                if os.path.getsize(save_path) < 1024:
                    print(f"  - 文件已存在但大小可疑 (<1KB)，尝试重新下载: {filename}")
                    os.remove(save_path) # 删除旧文件
                else:
                    # 文件存在且大小看起来正常，跳过
                    print(f"  - 文件已存在，跳过下载：{filename}")
                    skipped_count += 1
                    continue
            except OSError as e:
                 print(f"  - 检查/删除已存在文件时出错: {e}, 跳过下载。")
                 skipped_count += 1
                 continue

        # --- 开始下载尝试 ---
        success = False
        last_error = "未知错误"
        for attempt in range(CONFIG["max_retries"]):
            try:
                print(f"  - 尝试下载 ({attempt+1}/{CONFIG['max_retries']}): {url}")
                # 设置 User-Agent 模拟浏览器
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                # 使用 stream=True 进行流式下载，适合大文件
                response = requests.get(url, timeout=CONFIG["timeout"], headers=headers, stream=True)
                response.raise_for_status() # 检查 HTTP 错误状态码

                # 检查 Content-Type 是否像图片
                content_type = response.headers.get('content-type')
                is_image = content_type and content_type.lower().startswith('image/')
                if content_type and not is_image:
                    print(f"  警告: URL 返回的 Content-Type 可能不是图片 ({content_type}), 但仍继续下载。")

                block_size = 1024 * 8 # 每次读取 8KB
                start_time = time.time()
                downloaded_size = 0

                # 以二进制写入模式打开文件
                with open(save_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        f.write(data)
                        downloaded_size += len(data)

                download_time = time.time() - start_time
                # 再次检查文件是否存在及大小
                if os.path.exists(save_path):
                    file_size_kb = os.path.getsize(save_path) / 1024
                else:
                    # 如果下载后文件不见了，抛出错误
                    raise IOError("文件下载后未找到，可能写入失败。")

                # 对下载后文件大小的额外检查
                if file_size_kb < 1:
                    print(f"  警告: 下载的文件大小很小 ({file_size_kb:.2f} KB)。可能不是预期的图片，但仍保留。")

                print(f"  ✓ 下载成功: {filename} ({file_size_kb:.2f} KB, {download_time:.2f}s)")
                success = True
                downloaded_count += 1
                break # 成功则跳出重试循环

            except requests.exceptions.Timeout:
                last_error = "请求超时"
            except requests.exceptions.SSLError as ssl_err:
                last_error = f"SSL 错误: {ssl_err}"
                print(f"  ! SSL 错误下载 {url}.")
            except requests.exceptions.RequestException as req_e:
                last_error = f"网络或请求错误: {str(req_e)}"
            except ValueError as val_e: # 可能来自 URL 解析等
                last_error = str(val_e)
            except IOError as io_e: # 文件读写错误
                 last_error = f"文件读写错误: {str(io_e)}"
            except Exception as e: # 捕获其他所有意外错误
                last_error = f"下载时发生意外错误: {type(e).__name__}: {str(e)}"
                traceback.print_exc() # 打印详细的回溯信息

            # 如果下载失败，尝试删除可能不完整的文件
            if os.path.exists(save_path) and not success:
                try:
                    os.remove(save_path)
                    print(f"  - 已清理部分下载或损坏的文件: {filename}")
                except OSError as remove_err:
                    print(f"  - 清理失败文件时出错: {remove_err}")

            # 如果还有重试次数，等待后重试
            if attempt < CONFIG["max_retries"] - 1:
                wait = CONFIG["retry_delay"] * (2 ** attempt) # 指数退避等待
                print(f"  × 下载失败 ({last_error})，{wait:.1f}秒后重试...")
                time.sleep(wait)
            else:
                 # 所有重试次数用尽，标记为最终失败
                 print(f"  ! 最终下载失败: {filename}\n    错误: {last_error}\n    URL: {url}")
                 failed_count += 1
                 # 确保失败的文件最终被删除
                 if os.path.exists(save_path):
                      try: os.remove(save_path)
                      except OSError: pass

    # --- 下载结束，打印总结 ---
    print("-" * 30)
    print(f"下载总结: 成功 {downloaded_count}, 已存在/跳过 {skipped_count}, 失败 {failed_count}")
    print("-" * 30)

# ====================
# 交互式选择ROI和角度函数
# ====================
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

    current_img = img.copy() # 用于旋转调整显示的副本
    angle = 0.0 # 初始角度

    # --- 旋转调整阶段 ---
    WINDOW_NAME_ROTATE = '旋转调整 (R/T: 旋转, Q: 完成, S: 跳过, Esc: 退出)'
    cv2.namedWindow(WINDOW_NAME_ROTATE, cv2.WINDOW_NORMAL) # 可调整大小的窗口
    print("\n--- 旋转调整 ---")
    print("按 R 逆时针旋转 / T 顺时针旋转 (步长 5 度)")
    print("按 Q 完成旋转，进入ROI选择")
    print("按 S 跳过此图片")
    print("按 Esc 退出程序")
    print("-----------------")

    keep_rotating = True
    while keep_rotating:
        h, w = current_img.shape[:2]
        # 计算缩放比例以适应屏幕 (限制最大尺寸)
        screen_height, screen_width = 700, 1000
        scale_rot = 1.0
        if w > 0 and h > 0:
            scale_rot = min(screen_width / w, screen_height / h, 1.0) # 不放大，最多缩小

        display_width_rot = max(1, int(w * scale_rot)) # 保证至少为1像素
        display_height_rot = max(1, int(h * scale_rot))

        try:
             # 调整窗口大小并显示缩放后的图像
             cv2.resizeWindow(WINDOW_NAME_ROTATE, display_width_rot, display_height_rot)
             display_img_rot = cv2.resize(current_img, (display_width_rot, display_height_rot), interpolation=cv2.INTER_AREA)
             # 在图像上显示当前角度
             cv2.putText(display_img_rot, f'Angle: {angle:.1f}', (10, max(20, display_height_rot - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
             cv2.imshow(WINDOW_NAME_ROTATE, display_img_rot)
        except cv2.error as e:
            print(f"Error during rotation display setup: {e}")
            cv2.destroyWindow(WINDOW_NAME_ROTATE)
            return None, angle # 返回错误

        key = cv2.waitKey(CONFIG['wait_timeout']) & 0xFF # 等待按键

        rotate_step = 5.0 # 每次旋转5度
        needs_rotation_update = False
        if key == ord('r'): # 逆时针
            angle = (angle - rotate_step) % 360 # 保证角度在 0-360 之间 (或使用 -180 到 180)
            needs_rotation_update = True
        elif key == ord('t'): # 顺时针
            angle = (angle + rotate_step) % 360
            needs_rotation_update = True
        elif key == ord('q'): # 完成旋转
            print(f"旋转完成，最终角度: {angle:.1f} 度")
            keep_rotating = False
        elif key == ord('s'): # 跳过图片
            print("用户按 S 键，跳过当前图片")
            cv2.destroyWindow(WINDOW_NAME_ROTATE)
            return None, angle # 返回 None 表示跳过
        elif key == 27: # 退出程序 (ESC)
            print("用户按 ESC 键，退出程序")
            cv2.destroyWindow(WINDOW_NAME_ROTATE)
            exit(0)
        else:
            continue # 忽略其他按键

        # 如果需要更新旋转
        if needs_rotation_update:
            h_orig, w_orig = img.shape[:2]
            if w_orig > 0 and h_orig > 0:
                center = (w_orig // 2, h_orig // 2) # 图像中心
                # 获取旋转矩阵
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0) # 1.0 表示不缩放
                try:
                    # 应用仿射变换进行旋转 (使用原始图像 img)
                    current_img = cv2.warpAffine(img, matrix, (w_orig, h_orig), borderMode=cv2.BORDER_REPLICATE)
                except cv2.error as e:
                    print(f"Error during image rotation: {e}")
                    current_img = img.copy() # 旋转失败则恢复原状
                    angle = 0.0
            else:
                print("警告: 原始图像尺寸为零，无法旋转。")
                keep_rotating = False # 退出旋转循环

    cv2.destroyWindow(WINDOW_NAME_ROTATE) # 关闭旋转窗口

    # --- ROI 选择阶段 (使用最终旋转后的图像 `current_img`) ---
    final_rotated_img = current_img
    roi_final_coords = None # 存储最终相对于原始尺寸的ROI

    while True: # ROI 选择循环，直到获得有效ROI或用户跳过/退出
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
            return None, angle # 返回错误

        # 计算显示 ROI 选择窗口的缩放比例
        screen_height_roi, screen_width_roi = 700, 1000
        scale_roi = 1.0
        if w_roi_img > 0 and h_roi_img > 0: # 避免除以零
            scale_roi = min(screen_width_roi / w_roi_img, screen_height_roi / h_roi_img, 1.0)
        display_width_roi = max(1, int(w_roi_img * scale_roi))
        display_height_roi = max(1, int(h_roi_img * scale_roi))

        try:
             # 准备用于显示的缩放图像
             display_img_roi = cv2.resize(final_rotated_img, (display_width_roi, display_height_roi), interpolation=cv2.INTER_AREA)
        except cv2.error as e:
             print(f"! 错误: 调整图像大小以选择ROI时失败: {e}")
             return None, angle # 返回错误

        roi_scaled = None # 存储从显示图像上获取的 ROI 坐标

        if choice == '1': # 使用 OpenCV 内建 selectROI
            WINDOW_NAME_ROI_CV = '选择ROI (OpenCV 内建 - 拖动, Enter/Space确认, C取消)'
            cv2.namedWindow(WINDOW_NAME_ROI_CV, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME_ROI_CV, display_width_roi, display_height_roi)
            print("提示: 在弹出的窗口中拖动选择区域, 按 Enter/Space 确认, 按 C 取消.")
            try:
                # 确保传递给 selectROI 的图像有效
                if display_img_roi is None or display_img_roi.size == 0:
                     print("! 错误: 无法创建用于ROI选择的显示图像。")
                     cv2.destroyWindow(WINDOW_NAME_ROI_CV)
                     return None, angle # 指示失败
                # 调用 selectROI
                roi_scaled = cv2.selectROI(WINDOW_NAME_ROI_CV, display_img_roi, showCrosshair=False, fromCenter=False)
            except Exception as e:
                 print(f"! 调用 OpenCV selectROI 时出错: {e}")
                 roi_scaled = (0, 0, 0, 0) # 出错时视为无效选择
            cv2.destroyWindow(WINDOW_NAME_ROI_CV) # 关闭窗口
            # 检查 selectROI 返回结果是否有效 (用户按 C 或 ESC 会返回全0)
            if roi_scaled == (0, 0, 0, 0) or not roi_scaled or roi_scaled[2] <= 0 or roi_scaled[3] <= 0:
                print("用户取消选择或选择无效 (OpenCV)。")
                roi_scaled = None # 标记为无效

        elif choice == '2': # 使用自定义选择器
            WINDOW_NAME_ROI_CUSTOM = '选择ROI区域（自定义）'
            roi_scaled = custom_select_roi(WINDOW_NAME_ROI_CUSTOM, display_img_roi)
            # 检查自定义函数返回结果是否有效
            if roi_scaled == (0, 0, 0, 0) or not roi_scaled or roi_scaled[2] <= 0 or roi_scaled[3] <= 0:
                print("用户取消选择或选择无效 (自定义)。")
                roi_scaled = None # 标记为无效

        elif choice == 'm': # 手动输入坐标
            print(f"当前图像尺寸 (旋转后): 宽={w_roi_img}, 高={h_roi_img}")
            try:
                x_m_str = input("请输入 ROI 的 x 坐标 (左上角): ")
                y_m_str = input("请输入 ROI 的 y 坐标 (左上角): ")
                w_m_str = input("请输入 ROI 的宽度 w: ")
                h_m_str = input("请输入 ROI 的高度 h: ")
                # 将输入转换为整数
                x_m = int(x_m_str)
                y_m = int(y_m_str)
                w_m = int(w_m_str)
                h_m = int(h_m_str)
                # 检查坐标是否有效且在图像边界内
                if x_m < 0 or y_m < 0 or w_m <= 0 or h_m <= 0 or x_m + w_m > w_roi_img or y_m + h_m > h_roi_img:
                    print("错误：输入的 ROI 坐标无效或超出图像边界，请重试。")
                    continue # 让用户重新输入
                else:
                    # 手动输入的坐标是相对于最终旋转后图像的原始尺寸
                    roi_final_coords = (x_m, y_m, w_m, h_m)
                    break # 成功获取，跳出 ROI 选择循环
            except ValueError:
                print("错误：请输入有效的整数坐标，请重试。")
                continue # 让用户重新输入
            except Exception as e:
                 print(f"输入时发生错误: {e}")
                 continue # 让用户重新输入

        elif choice == 's': # 跳过此图片
            print("用户按 S 键，跳过当前图片 (不生成模板)")
            return None, angle # 返回 None 表示跳过

        elif choice == 'esc': # 退出程序
             print("用户按 ESC 键，退出程序")
             exit(0)
        elif choice == '': # 用户直接按回车
             print("无效选择 (直接按回车)，请选择 1, 2, 3, S 或 Esc.")
             continue # 让用户重新选择
        else: # 其他无效输入
            print("无效选择，请重新输入。")
            continue # 让用户重新选择

        # --- 如果通过 OpenCV 或 自定义选择器获取了 ROI ---
        if roi_scaled:
            # 将显示图像上的 ROI 坐标转换回原始旋转后图像的坐标
            x_s, y_s, w_s, h_s = roi_scaled
            if abs(scale_roi) > 1e-6: # 避免除以零
                x = int(x_s / scale_roi)
                y = int(y_s / scale_roi)
                w = int(w_s / scale_roi)
                h = int(h_s / scale_roi)
            else:
                print("! 错误: 显示缩放比例为零，无法转换ROI坐标。")
                continue # 让用户重新选择

            # 修正坐标确保在图像边界内
            x = max(0, x)
            y = max(0, y)
            # 如果计算出的宽度/高度超出边界，则截断
            if x + w > w_roi_img:
                w = w_roi_img - x
            if y + h > h_roi_img:
                h = h_roi_img - y

            # 确保最终计算的 ROI 有效
            if w > 0 and h > 0:
                 roi_final_coords = (x, y, w, h)
                 print(f"最终 ROI (原始尺寸坐标): {roi_final_coords}")
                 break # 获取有效 ROI，跳出循环
            else:
                 print("警告: ROI 转换或修正后无效 (宽或高<=0)。请重试。")
                 # 继续循环让用户重新选择

    # 返回最终确定的 ROI (可能是 None) 和 角度
    return roi_final_coords, angle

# ====================
# 读取模板索引函数
# ====================
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
            line_num = 1 # 行号从1开始（因为表头被读取了）
            # 检查表头是否符合预期
            if not header or header != expected_header:
                 print(f"警告: 模板索引 '{index_path}' 的表头格式不符合预期或缺失。")
                 print(f"  预期: {expected_header}")
                 print(f"  找到: {header}")
                 print(f"  将尝试在没有有效表头的情况下读取。")
                 # 如果表头不对，文件指针移回开头，从第一行开始读数据
                 f.seek(0)
                 line_num = 0 # 如果没有表头，行号从0开始计数数据行

            # 逐行读取数据
            for row in reader:
                 line_num += 1
                 # 检查行是否有足够的列
                 if len(row) >= len(expected_header):
                     filename = row[0].strip() # 源文件名
                     if not filename: # 跳过没有源文件名的行
                          #print(f"警告: 索引文件第 {line_num} 行缺少源文件名，已跳过。")
                          continue
                     try:
                         # 基本的数据类型检查 (不强制类型转换，只检查格式)
                         if row[1].strip().upper() == "NONE": # 如果标记为 NONE
                             # 允许 ROI 和角度为 0
                             list(map(int, row[2:6])) # 检查 ROI 是否为整数
                             float(row[6]) # 检查角度是否为浮点数
                         else:
                             # 对于有模板的记录，也进行检查
                             list(map(int, row[2:6]))
                             float(row[6])
                     except ValueError:
                          print(f"警告: 索引文件第 {line_num} 行 ROI 或角度格式错误，已跳过: {row}")
                          continue # 跳过格式错误的行
                     except IndexError:
                          print(f"警告: 索引文件第 {line_num} 行缺少列，已跳过: {row}")
                          continue # 跳过列数不足的行

                     # 使用字典存储，源文件名为键，整行数据为值
                     # 如果有重复的源文件名，后出现的会覆盖之前的
                     #if filename in template_index:
                         #print(f"警告: 索引文件第 {line_num} 行发现重复的源文件名 '{filename}'。将使用此最新条目。")
                     template_index[filename] = row
                 else:
                     print(f"警告: 索引文件第 {line_num} 行格式不完整，已跳过: {row}")
        #print(f"成功加载 {len(template_index)} 条模板索引记录从 '{index_path}'。")
    except Exception as e:
        print(f"读取模板索引 '{index_path}' 时发生严重错误: {e}。将视为无现有索引。")
        traceback.print_exc()
        return {} # 出错时返回空字典
    return template_index

# ====================
# 写入模板索引函数
# ====================
def write_template_index(index_path, template_index_dict):
    """
    将内存中的模板索引字典写入 CSV 文件 (覆盖写入)。
    使用临时文件和移动操作保证原子性。
    """
    temp_index_path = index_path + ".tmp" # 临时文件名
    try:
        with open(temp_index_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            header = ["SourceFilename", "TemplateName", "ROI_X", "ROI_Y", "ROI_W", "ROI_H", "GenerationAngle"]
            writer.writerow(header)
            # 按源文件名排序后写入数据
            sorted_filenames = sorted(template_index_dict.keys())
            for filename in sorted_filenames:
                writer.writerow(template_index_dict[filename])
        # 写入成功后，用临时文件覆盖原始文件
        shutil.move(temp_index_path, index_path)
        #print(f"模板索引已成功写入/更新到: {index_path}")
    except Exception as e:
        print(f"写入模板索引 '{index_path}' 时发生错误: {e}")
        traceback.print_exc()
        # 如果写入失败，尝试删除临时文件
        if os.path.exists(temp_index_path):
             try:
                 os.remove(temp_index_path)
                 print(f"已删除临时索引文件: {temp_index_path}")
             except Exception as e_rem:
                 print(f"删除临时索引文件失败: {e_rem}")

# ====================
# 辅助函数：删除指定文件的所有关联模板
# ====================
def _delete_all_templates_for_file(filename, template_dir):
    """根据源文件名查找并删除所有关联的模板文件 (png, jpg, jpeg)。"""
    base_name = os.path.splitext(filename)[0] # 获取不带扩展名的文件名
    # 构建 glob 搜索模式，匹配 template_{base_name}*.*
    search_patterns = [
        os.path.join(template_dir, f"template_{base_name}.png"),
        os.path.join(template_dir, f"template_{base_name}_*.png"),
        os.path.join(template_dir, f"template_{base_name}.jpg"),
        os.path.join(template_dir, f"template_{base_name}_*.jpg"),
        os.path.join(template_dir, f"template_{base_name}.jpeg"),
        os.path.join(template_dir, f"template_{base_name}_*.jpeg"),
    ]

    files_to_delete = set() # 使用集合避免重复删除
    for pattern in search_patterns:
        # 使用 glob 查找匹配的文件
        found_files = glob.glob(pattern)
        files_to_delete.update(found_files) # 添加到集合

    deleted_count = 0
    failed_count = 0
    if not files_to_delete:
        #print(f"  - 未找到与 '{filename}' 关联的模板文件进行删除。")
        return deleted_count, failed_count # 如果没找到，直接返回

    print(f"  - 准备删除以下与 '{filename}' 关联的模板文件:")
    # 排序后删除，方便查看日志
    for f_path in sorted(list(files_to_delete)):
        print(f"    - {os.path.basename(f_path)}")
        try:
            os.remove(f_path) # 删除文件
            deleted_count += 1
        except Exception as e:
            print(f"  ! 删除模板文件失败: {f_path} - {e}")
            failed_count += 1
    print(f"  - 模板文件删除完成 (成功: {deleted_count}, 失败: {failed_count})")
    return deleted_count, failed_count

# ====================
# 生成/管理模板函数 (交互式)
# ====================
def generate_templates(items_to_process):
    """
    为下载列表中的图片生成/更新模板文件。
    交互式操作：添加新模板、删除所有模板、标记为无模板、保留或跳过。
    Args:
        items_to_process (list): 包含 (url, filename) 元组的列表。
    """
    os.makedirs(CONFIG["template_dir"], exist_ok=True) # 确保模板目录存在
    index_file = CONFIG["template_index"] # 索引文件路径

    template_index_data = read_template_index(index_file) # 读取现有索引
    original_index_count = len(template_index_data) # 记录原始索引数量

    # --- 初始化计数器 ---
    added_count = 0      # 新添加的模板文件数量
    deleted_files_count = 0 # 通过 D 或 N 删除的关联模板文件总数
    none_marked_count = 0 # 标记为无模板的源图片数量
    skipped_count = 0   # 本次跳过或保留的数量
    source_missing_skipped_count = 0 # 因源文件丢失而跳过的数量
    deleted_index_missing_source_count = 0 # 因源文件丢失删除的索引条目数
    error_count = 0     # 处理时发生错误计数

    total_items = len(items_to_process)
    print(f"\n--- 开始生成/管理 {total_items} 个文件的模板 ---")

    # --- 遍历下载列表中的每个文件 ---
    for i, (url, filename) in enumerate(items_to_process):
        img_path = os.path.join(CONFIG["download_dir"], filename) # 源图片路径
        base_name = os.path.splitext(filename)[0] # 不带扩展名的文件名

        print(f"\n[{i+1}/{total_items}] 处理模板: {filename}")

        # --- 检查源图片是否存在 ---
        if not os.path.exists(img_path):
            print(f"  - 源图片文件不存在，跳过模板处理: {img_path}")
            source_missing_skipped_count += 1
            # 如果索引中有这个丢失文件的记录，也从索引中移除
            if filename in template_index_data:
                print(f"  - 从索引中移除丢失文件的记录: {filename}")
                template_index_data.pop(filename, None) # 使用 pop 安全移除
                deleted_index_missing_source_count += 1
            continue # 处理下一个文件

        # --- 源文件存在，检查索引并与用户交互 ---
        user_action = None # 用户选择的操作
        was_existing_in_index = filename in template_index_data # 标记此文件之前是否在索引中

        # 显示现有索引信息（如果存在）
        if was_existing_in_index:
            try:
                existing_info = template_index_data[filename]
                template_name_in_index = existing_info[1] # 获取记录中的模板名
                if template_name_in_index == "NONE":
                     print(f"  - 文件 '{filename}' 当前在索引中标记为 [无模板 (NONE)]。")
                else:
                    # 显示 ROI 和角度信息
                    roi_in_index = f"({','.join(map(str, existing_info[2:6]))})"
                    angle_in_index = existing_info[6]
                    print(f"  - 文件 '{filename}' 在索引中的记录 (指向最后操作的模板):")
                    print(f"    模板名: {template_name_in_index}, ROI: {roi_in_index}, 角度: {angle_in_index}")
            except (IndexError, KeyError, ValueError): # 捕获可能的格式错误
                print(f"  - 文件 '{filename}' 的现有索引记录格式错误，将视为无记录。")
                was_existing_in_index = False # 视为新记录处理

        else: # 如果文件不在索引中
             print(f"  - 文件 '{filename}' 无索引记录。")

        # --- 获取用户操作 ---
        prompt = (f"  请选择操作: [R]添加新模板, [D]删除所有模板, [N]标记为无模板, [K]保留现有, [S]本次跳过? (默认: K 保留): ")
        choice = input(prompt).strip().lower()

        if choice == 'r': user_action = 'add_new'
        elif choice == 'd': user_action = 'delete_all'
        elif choice == 'n': user_action = 'mark_none'
        elif choice == 's': user_action = 'skip'
        else: # 默认或无效输入都视为 'keep'
            user_action = 'keep'
            if choice != 'k' and choice != '':
                 print("  - 无效输入，默认保留现有状态。")

        # --- 根据用户选择执行操作 ---
        if user_action == 'skip' or user_action == 'keep':
            action_desc = '跳过' if user_action == 'skip' else '保留现有状态'
            print(f"  - {action_desc} 文件: {filename}")
            skipped_count += 1
            continue # 处理下一个文件

        elif user_action == 'delete_all' or user_action == 'mark_none':
            # 删除操作 ('d' 或 'n') 都需要删除现有模板文件
            action_desc = "标记为无模板 (并删除现有)" if user_action == 'mark_none' else "删除所有关联模板"
            print(f"  - 执行操作: {action_desc} for '{filename}'")
            # 调用辅助函数删除所有关联模板文件
            deleted_now, failed_now = _delete_all_templates_for_file(filename, CONFIG["template_dir"])
            deleted_files_count += deleted_now # 累加删除的文件数
            if failed_now > 0: error_count += failed_now # 累加删除失败数

            # 将索引中的记录更新为 NONE
            # 格式: [源文件名, "NONE", 0, 0, 0, 0, "0.0"]
            template_index_data[filename] = [filename, "NONE", 0, 0, 0, 0, "0.0"]
            print(f"  - 索引已更新为 'NONE' for {filename}")

            # 更新 "标记为无模板" 计数器
            # 只有当它之前不是 NONE 时，才算作一次新的标记
            if user_action == 'mark_none':
                if was_existing_in_index and existing_info[1] != "NONE":
                    none_marked_count += 1
                elif not was_existing_in_index: # 如果是新文件直接标记为 none
                    none_marked_count += 1
            # 如果是 delete_all 操作，我们不增加 none_marked_count

        elif user_action == 'add_new':
            # --- 添加新模板流程 ---
            print(f"  - 开始为 '{filename}' 添加新模板...")
            # 调用交互式 ROI 和角度选择函数
            result = select_roi_interactive(img_path)

            # 检查函数调用是否成功
            if result is None:
                print(f"  ! 获取 ROI 时发生意外错误，跳过: {filename}")
                error_count += 1
                continue # 处理下一个文件

            # 解包返回结果
            roi_coords_final, generation_angle = result

            # 检查用户是否取消或跳过了 ROI 选择
            if roi_coords_final is None or roi_coords_final == (0,0,0,0):
                print(f"  - 用户在选择 ROI 时取消或跳过，未添加新模板。")
                skipped_count += 1
                continue # 处理下一个文件

            # 确认获取到有效的 ROI 坐标
            x, y, w, h = roi_coords_final
            print(f"  - 已选择有效ROI (原始尺寸坐标): x={x}, y={y}, w={w}, h={h}, 角度={generation_angle:.1f}°")

            # --- 确定新模板的文件名 ---
            # 优先使用 template_{base_name}.png，如果存在则使用 _1, _2 ... 后缀
            next_num = 1
            new_template_name = ""
            new_template_path = ""

            # 基础模板名 (优先使用 png)
            template_name_base_png = f"template_{base_name}.png"
            template_path_base_png = os.path.join(CONFIG["template_dir"], template_name_base_png)

            # 检查基础模板名是否已存在 (也检查 jpg/jpeg 以防万一)
            template_path_base_jpg = os.path.join(CONFIG["template_dir"], f"template_{base_name}.jpg")
            template_path_base_jpeg = os.path.join(CONFIG["template_dir"], f"template_{base_name}.jpeg")

            if not os.path.exists(template_path_base_png) and \
               not os.path.exists(template_path_base_jpg) and \
               not os.path.exists(template_path_base_jpeg):
                # 如果基础名不存在，就使用它 (png 格式)
                new_template_name = template_name_base_png
                new_template_path = template_path_base_png
                print(f"  - 将创建第一个模板: {new_template_name}")
            else:
                # 如果基础名已存在，查找下一个可用的 _n 后缀 (png 格式)
                while True:
                    potential_name_png = f"template_{base_name}_{next_num}.png"
                    potential_path_png = os.path.join(CONFIG["template_dir"], potential_name_png)
                    # 同时检查同编号的 jpg/jpeg 是否存在，避免潜在冲突
                    potential_path_jpg = os.path.join(CONFIG["template_dir"], f"template_{base_name}_{next_num}.jpg")
                    potential_path_jpeg = os.path.join(CONFIG["template_dir"], f"template_{base_name}_{next_num}.jpeg")

                    if not os.path.exists(potential_path_png) and \
                       not os.path.exists(potential_path_jpg) and \
                       not os.path.exists(potential_path_jpeg):
                        # 找到可用编号
                        new_template_name = potential_name_png
                        new_template_path = potential_path_png
                        print(f"  - 将创建下一个编号模板: {new_template_name}")
                        break # 找到后跳出循环
                    next_num += 1
                    if next_num > 999: # 设置一个上限防止无限循环
                         print("  ! 无法找到下一个可用的模板编号 (已达 999)，无法添加。")
                         error_count += 1
                         new_template_path = None # 标记为无法生成路径
                         break # 跳出循环

            # 如果未能确定新模板路径，则跳过
            if not new_template_path:
                continue

            # --- 裁剪模板 ---
            template_img = None
            try:
                # 重新读取原始图像以保证裁剪质量
                img_orig_for_crop = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img_orig_for_crop is None: raise ValueError("无法重新读取原始图像进行裁剪")

                # 根据用户选择的角度旋转原始图像
                rotated_img_for_crop = img_orig_for_crop
                if abs(generation_angle) > 1e-6: # 如果角度不为 0
                     h_orig_crop, w_orig_crop = img_orig_for_crop.shape[:2]
                     if w_orig_crop > 0 and h_orig_crop > 0:
                         center_crop = (w_orig_crop // 2, h_orig_crop // 2)
                         matrix_crop = cv2.getRotationMatrix2D(center_crop, generation_angle, 1.0)
                         # 使用 BORDER_REPLICATE 填充边界
                         rotated_img_for_crop = cv2.warpAffine(img_orig_for_crop, matrix_crop, (w_orig_crop, h_orig_crop), borderMode=cv2.BORDER_REPLICATE)
                     else: raise ValueError("用于裁剪的原始图像尺寸无效")

                # 从旋转后的图像上裁剪 ROI
                h_rot_crop, w_rot_crop = rotated_img_for_crop.shape[:2]
                # 再次确保 ROI 坐标在旋转后图像内
                x_c = max(0, x); y_c = max(0, y)
                w_c = min(w, w_rot_crop - x_c); h_c = min(h, h_rot_crop - y_c)
                # 检查裁剪尺寸是否有效
                if w_c <= 0 or h_c <= 0: raise ValueError(f"ROI裁剪后尺寸无效 ({w_c}x{h_c})")

                # 执行裁剪
                template_img = rotated_img_for_crop[y_c : y_c + h_c, x_c : x_c + w_c]
                # 检查裁剪结果
                if template_img is None or template_img.size == 0: raise ValueError("裁剪模板结果为空")

            except Exception as crop_e:
                 print(f"  × 裁剪模板时出错: {crop_e}")
                 error_count += 1
                 continue # 跳过此文件

            # --- 保存新模板文件 (PNG 格式) ---
            try:
                # 保存裁剪出的模板图像
                cv2.imwrite(new_template_path, template_img)
                # 更新索引信息，指向这个新创建的模板
                template_index_data[filename] = [filename, new_template_name, x, y, w, h, f"{generation_angle:.2f}"] # 角度保留两位小数
                print(f"  ✓ 新模板已添加并保存: {new_template_name}")
                print(f"  - 索引已更新以反映最新添加的模板信息。")
                added_count += 1 # 增加添加计数
            except Exception as e:
                print(f"  × 保存新模板失败: {new_template_name} - {str(e)}")
                error_count += 1
                # 如果保存失败，尝试删除可能已创建的文件
                if os.path.exists(new_template_path):
                    try: os.remove(new_template_path)
                    except: pass
                continue # 跳过此文件

    # --- 循环结束，写入最终的索引文件 ---
    write_template_index(index_file, template_index_data)

    # --- 打印总结信息 ---
    print("-" * 30)
    final_index_count = len(template_index_data)
    print(f"模板生成/管理总结:")
    print(f"  - 新添加模板文件: {added_count}")
    print(f"  - 标记为无模板 (N操作): {none_marked_count}")
    print(f"  - 通过 D/N 操作删除的关联模板文件总数: {deleted_files_count}")
    print(f"  - 本次跳过/保留现有状态: {skipped_count}")
    print(f"  - 因源文件丢失跳过: {source_missing_skipped_count}")
    print(f"  - 因源文件丢失删除索引条目: {deleted_index_missing_source_count}")
    print(f"  - 处理时发生错误: {error_count}")
    print(f"  - 最终索引中记录数: {final_index_count} (原始: {original_index_count})")
    print(f"模板索引已更新: {index_file}")
    print("-" * 30)

# =======================================
# 核心处理函数：使用多个模板进行裁剪（包含旋转搜索）
# =======================================
def crop_with_multiple_templates(img_path, output_path, template_paths):
    """
    尝试使用一系列模板裁剪图像，找到第一个成功的匹配即停止。
    关键特性：此函数通过在匹配过程中搜索旋转角度来内在地处理旋转。
    Args:
        img_path (str): 源图像路径 (例如，可能旋转的 'n62.jpg')。
        output_path (str): 保存裁剪结果的路径。
        template_paths (list): 按顺序尝试的模板文件路径列表 (这些模板最好是手动生成时已校正旋转的)。
    Returns:
        tuple: (success, message)
    """
    try:
        # 读取源图像
        img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_color is None: return False, f"无法读取主图像: {os.path.basename(img_path)}"
        # 转换为灰度图用于匹配
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        h_img_orig, w_img_orig = img_gray.shape[:2]
        if h_img_orig <= 0 or w_img_orig <= 0: return False, f"主图像尺寸无效: {w_img_orig}x{h_img_orig}"
        # 计算原始图像中心点 (浮点数)
        center_orig = (w_img_orig / 2.0, h_img_orig / 2.0)

        # --- 遍历提供的模板路径 ---
        for template_path in template_paths:
            # --- 加载并准备当前模板 ---
            try:
                # 使用 IMREAD_UNCHANGED 以支持带透明通道的 PNG 模板
                template_color = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
                if template_color is None:
                    print(f"  ! 警告: 无法读取模板 {os.path.basename(template_path)}，跳过此模板")
                    continue

                # 准备灰度模板和可选的掩码 (用于带透明度的模板)
                template_gray = None; mask = None
                if template_color.ndim == 2: # 模板本身是灰度图
                    template_gray = template_color
                elif template_color.ndim == 3: # BGR 彩色模板
                    template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
                elif template_color.ndim == 4: # BGRA 带透明通道模板
                    template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGRA2GRAY)
                    # 从 Alpha 通道创建掩码
                    alpha = template_color[:, :, 3]
                    _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
                else: # 不支持的维度
                    print(f"  ! 警告: 无法处理模板颜色格式 {os.path.basename(template_path)} (维度: {template_color.ndim})，跳过")
                    continue

                # 检查模板尺寸是否有效
                template_h, template_w = template_gray.shape[:2]
                if template_h <= 0 or template_w <= 0:
                    print(f"  ! 警告: 模板尺寸无效 {os.path.basename(template_path)} ({template_w}x{template_h})，跳过")
                    continue
            except Exception as e:
                 print(f"  ! 警告: 加载/处理模板 {os.path.basename(template_path)} 时出错: {e}，跳过")
                 continue

            print(f"  - 尝试匹配模板: {os.path.basename(template_path)}")
            # 初始化当前模板的最佳匹配信息
            best_match_info = {'val': -1.0, 'loc': None, 'angle': 0.0, 'rotated_img_color': None}
            match_threshold = CONFIG['match_threshold'] # 获取配置的阈值
            step = CONFIG["rotation_step"]; range_ = CONFIG["rotation_range"]
            # 生成要尝试的角度列表
            angles_to_try = np.linspace(-range_, range_, int(2 * range_ / step) + 1).tolist()
            if 0.0 not in angles_to_try: angles_to_try.extend([0.0]) # 确保包含0度
            angles_to_try.sort() # 按顺序尝试角度

            # --- 针对当前模板，遍历不同的旋转角度来旋转 *输入图像* ---
            for angle in angles_to_try:
                angle = round(angle, 2) # 保留两位小数
                rotated_gray = None
                rotated_color = None
                try:
                    # 获取旋转矩阵 (围绕原始中心)
                    rot_matrix = cv2.getRotationMatrix2D(center_orig, angle, 1.0)

                    # 计算旋转后图像的新边界框大小，以避免裁剪
                    abs_cos = abs(rot_matrix[0, 0]); abs_sin = abs(rot_matrix[0, 1])
                    new_w = int(h_img_orig * abs_sin + w_img_orig * abs_cos)
                    new_h = int(h_img_orig * abs_cos + w_img_orig * abs_sin)

                    # 调整旋转矩阵的平移分量，使旋转后的图像完整显示在新边界框中心
                    rot_matrix[0, 2] += (new_w / 2.0) - center_orig[0]
                    rot_matrix[1, 2] += (new_h / 2.0) - center_orig[1]

                    # 旋转灰度输入图像
                    rotated_gray = cv2.warpAffine(img_gray, rot_matrix, (new_w, new_h),
                                                  flags=cv2.INTER_LINEAR, # 使用线性插值
                                                  borderMode=cv2.BORDER_REPLICATE) # 复制边界像素填充

                    # 同时旋转彩色输入图像，以备后续裁剪使用
                    rotated_color = cv2.warpAffine(img_color, rot_matrix, (new_w, new_h),
                                                   flags=cv2.INTER_LINEAR,
                                                   borderMode=cv2.BORDER_REPLICATE)

                except cv2.error as rot_e:
                    # 如果旋转失败，跳过此角度
                    # print(f"    ! 角度 {angle:.1f}° 旋转失败: {rot_e}") # 可选的调试信息
                    continue

                # 检查模板是否能放在旋转后的图像内
                if template_h > rotated_gray.shape[0] or template_w > rotated_gray.shape[1]:
                    # print(f"    ! 模板在角度 {angle:.1f}° 时超出旋转后图像边界，跳过此角度。") # 可选的调试信息
                    continue

                # --- 在当前角度旋转后的图像上执行模板匹配 ---
                method = cv2.TM_CCOEFF_NORMED # 使用归一化相关系数匹配方法
                try:
                    # 使用旋转后的灰度图、模板灰度图和可选的掩码进行匹配
                    result = cv2.matchTemplate(rotated_gray, template_gray, method, mask=mask)
                    # 找到匹配结果中的最大值及其位置
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                except cv2.error as match_e:
                    # print(f"    ! 角度 {angle:.1f}° 模板匹配失败: {match_e}") # 可选的调试信息
                    continue # 匹配失败则跳过此角度

                # --- 如果当前角度的匹配优于之前的最佳匹配，则更新信息 ---
                if max_val > best_match_info['val']:
                    best_match_info['val'] = max_val
                    best_match_info['loc'] = max_loc # 存储最佳匹配位置 (左上角)
                    best_match_info['angle'] = angle # 存储最佳匹配角度
                    best_match_info['rotated_img_color'] = rotated_color # 存储对应角度的旋转彩色图

            # --- 针对当前模板的所有角度尝试完毕 ---
            best_val = best_match_info['val']
            best_loc = best_match_info['loc']
            best_angle = best_match_info['angle']
            best_rotated_color_img = best_match_info['rotated_img_color']

            print(f"    > 模板 '{os.path.basename(template_path)}' 最佳匹配分: {best_val:.4f} (在角度 {best_angle:.1f}°, 阈值: {match_threshold})")

            # --- 检查最佳匹配得分是否达到阈值 ---
            if best_val >= match_threshold and best_loc is not None and best_rotated_color_img is not None:
                print(f"  ✓ 找到成功匹配! 使用模板: {os.path.basename(template_path)}")

                # --- 从最佳匹配角度的旋转彩色图上裁剪 ---
                match_x, match_y = best_loc # 最佳匹配的左上角坐标
                # 计算裁剪区域的右下角坐标
                x_min = match_x; y_min = match_y
                x_max = match_x + template_w; y_max = match_y + template_h

                # 获取最佳旋转彩色图的实际尺寸
                h_rot, w_rot = best_rotated_color_img.shape[:2]

                # 确保裁剪坐标在图像边界内 (钳位操作)
                x_min_c = max(0, x_min); y_min_c = max(0, y_min)
                x_max_c = min(w_rot, x_max); y_max_c = min(h_rot, y_max)

                # 计算最终的裁剪宽度和高度
                crop_w = x_max_c - x_min_c; crop_h = y_max_c - y_min_c

                # 再次检查裁剪尺寸是否有效
                if crop_w <= 0 or crop_h <= 0:
                    print(f"  ! 裁剪区域无效 (w={crop_w}, h={crop_h}) for template {os.path.basename(template_path)}. 尝试下一个...")
                    continue # 如果裁剪无效，尝试下一个模板

                # 执行裁剪
                cropped_img = best_rotated_color_img[y_min_c:y_max_c, x_min_c:x_max_c]

                # 检查裁剪结果是否为空
                if cropped_img is None or cropped_img.size == 0:
                    print(f"  ! 裁剪结果为空 (模板: {os.path.basename(template_path)}). 尝试下一个...")
                    continue # 如果裁剪结果为空，尝试下一个模板

                # --- 保存裁剪后的图像 ---
                try:
                    cv2.imwrite(output_path, cropped_img)
                    # 成功保存后返回 True 和成功信息
                    return True, f"成功使用模板 '{os.path.basename(template_path)}' 裁剪 (匹配度 {best_val:.4f} @ {best_angle:.1f}°)"
                except Exception as save_e:
                    print(f"  ! 保存裁剪结果失败 (模板: {os.path.basename(template_path)}): {save_e}")
                    # 即使保存失败，也尝试下一个模板（可能只是磁盘权限问题）
                    continue

            # 如果当前模板的最佳匹配得分低于阈值，则循环继续尝试下一个模板

        # --- 如果所有模板都尝试完毕，没有一个成功 ---
        return False, "所有尝试的模板均未成功匹配、裁剪或保存"

    # --- 捕获整个函数的意外错误 ---
    except Exception as e:
        print(f"! 处理 '{os.path.basename(img_path)}' 时发生严重错误 (crop_with_multiple_templates): {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, f"处理时发生严重异常: {type(e).__name__}"


# ====================
# 处理图片主函数
# ====================
def process_images(items_to_process):
    """
    处理下载列表中的所有图片 (强制处理，覆盖输出目录):
    - 搜索每个文件的关联模板 (`template_{base_filename}*.png/jpg/jpeg`)。
    - 调用 `crop_with_multiple_templates` 尝试用找到的模板进行匹配（该函数内部处理旋转）。
    - 如果任何模板成功匹配并裁剪，保存结果。
    - 如果所有模板都失败或没有找到模板，则将原始图片复制到输出目录（覆盖）。
    - 处理完成后，删除下载目录中的原始文件。
    Args:
        items_to_process (list): 包含 (url, filename) 元组的列表。
    """
    os.makedirs(CONFIG["output_dir"], exist_ok=True) # 确保输出目录存在
    os.makedirs(CONFIG["template_dir"], exist_ok=True) # 确保模板目录存在

    # --- 初始化计数器 ---
    processed_count = 0 # 成功生成输出文件的总数
    cropped_count = 0   # 通过裁剪生成的数量
    copied_count = 0    # 通过复制生成的数量
    skipped_source_missing_count = 0 # 因源文件丢失跳过的数量
    failed_processing_count = 0 # 处理失败（裁剪和复制都失败）的数量
    deleted_count = 0   # 成功删除的下载文件数
    delete_failed_count = 0 # 删除下载文件失败的数量

    total_items = len(items_to_process)
    print(f"\n--- 开始处理 {total_items} 个图片 (目标: '{CONFIG['output_dir']}', 将覆盖现有输出) ---")

    # --- 遍历要处理的每个图片 ---
    for i, (url, filename) in enumerate(items_to_process):
        img_path = os.path.join(CONFIG["download_dir"], filename) # 源文件路径
        output_path = os.path.join(CONFIG["output_dir"], filename) # 输出文件路径 (同名)
        base_name = os.path.splitext(filename)[0] # 文件基本名
        print(f"\n[{i+1}/{total_items}] 处理图片: {filename}")
        output_generated = False # 标记是否成功生成了输出文件

        # --- 检查源文件是否存在 ---
        if not os.path.exists(img_path):
            print(f"  ! 跳过处理：未找到源图片文件 {img_path}")
            skipped_source_missing_count += 1
            continue # 处理下一个文件

        # --- 查找关联的模板文件 ---
        potential_template_paths = []
        try:
            search_dir = CONFIG["template_dir"]
            # 定义搜索模式
            patterns = [f"template_{base_name}*.png", f"template_{base_name}*.jpg", f"template_{base_name}*.jpeg"]
            found_templates = set() # 使用集合去重
            for pattern in patterns:
                found_templates.update(glob.glob(os.path.join(search_dir, pattern)))

            # --- 对找到的模板进行排序，优先使用基础模板 ---
            main_templates = {
                'png': os.path.join(search_dir, f"template_{base_name}.png"),
                'jpg': os.path.join(search_dir, f"template_{base_name}.jpg"),
                'jpeg': os.path.join(search_dir, f"template_{base_name}.jpeg")
            }
            main_template_to_prioritize = None
            # 按 png > jpg > jpeg 顺序检查基础模板是否存在
            for ext in ['png', 'jpg', 'jpeg']:
                 if main_templates[ext] in found_templates:
                     main_template_to_prioritize = main_templates[ext]
                     break # 找到第一个就停止

            # 对所有找到的模板按字母排序（通常会将 _1, _2 等排在后面）
            sorted_templates = sorted(list(found_templates))

            # 构建最终尝试列表：优先的基础模板 + 其他排序后的模板
            if main_template_to_prioritize:
                 if main_template_to_prioritize in sorted_templates:
                     sorted_templates.remove(main_template_to_prioritize) # 避免重复
                 potential_template_paths = [main_template_to_prioritize] + sorted_templates
            else: # 如果没有基础模板，就直接用排序后的列表
                 potential_template_paths = sorted_templates

        except Exception as e:
            print(f"  ! 搜索模板时出错: {e}")
            # 出错也继续，后续会走复制流程

        # --- 尝试使用模板裁剪 ---
        crop_success = False
        crop_message = "没有找到模板文件" # 默认失败原因

        if potential_template_paths: # 如果找到了模板
            print(f"  - 找到 {len(potential_template_paths)} 个潜在模板，将按以下顺序尝试:")
            for tpath in potential_template_paths: print(f"    - {os.path.basename(tpath)}")

            # 调用核心裁剪函数（该函数内部处理旋转）
            crop_success, crop_message = crop_with_multiple_templates(img_path, output_path, potential_template_paths)

            if crop_success:
                output_generated = True # 标记成功生成
                processed_count += 1
                cropped_count += 1
                print(f"  ✓ 处理完成 (已裁剪覆盖/创建输出): {filename} ({crop_message})")
            else:
                # crop_message 会包含失败原因
                print(f"  ! 所有模板尝试失败。")
        else: # 如果没有找到模板文件
            print(f"  - 未找到匹配 'template_{base_name}*.png/jpg/jpeg' 的模板文件。")

        # --- 如果裁剪失败或无模板，执行后备操作：复制原始文件 ---
        if not crop_success:
            print(f"  - (后备操作) 尝试直接复制原始文件到输出目录 (覆盖)...")
            try:
                # 确保目标目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # 如果输出文件已存在，先删除再复制（实现覆盖）
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.copy(img_path, output_path) # 复制文件
                output_generated = True # 标记成功生成
                processed_count += 1
                copied_count += 1
                print(f"  ✓ 复制成功 (已覆盖/创建输出): {filename}")
            except Exception as e:
                failed_processing_count += 1 # 计入处理失败
                output_generated = False # 标记未生成
                print(f"  × 后备复制原始文件失败: {e}")

        # --- 清理：删除下载目录中的原始文件 ---
        # 无论处理是否成功，都尝试删除下载目录中的源文件
        if os.path.exists(img_path):
            try:
                os.remove(img_path) # 删除文件
                deleted_count += 1
                # 根据处理结果显示不同提示
                if output_generated:
                    print(f"  - 已删除下载目录中的原始文件: {filename}")
                else:
                    print(f"  - (注意) 处理失败，但仍尝试删除下载目录中的原始文件: {filename}")
            except Exception as e:
                delete_failed_count += 1
                print(f"  ! 删除下载目录原始文件失败: {filename} - {e}")
        # --- 单个文件处理结束 ---

    # --- 所有文件处理完毕，打印总结 ---
    print("-" * 30)
    print(f"图片处理总结:")
    print(f"  - 成功生成/覆盖输出文件总数: {processed_count}")
    print(f"    - 通过模板裁剪生成: {cropped_count}")
    print(f"    - 通过复制原始图片生成: {copied_count}")
    print(f"  - 跳过 (源文件缺失): {skipped_source_missing_count}")
    print(f"  - 处理失败 (裁剪和后备复制均失败): {failed_processing_count}")
    print(f"  - 下载目录原始文件删除成功: {deleted_count}")
    if delete_failed_count > 0:
        print(f"  - 下载目录原始文件删除失败: {delete_failed_count}")
    print(f"最终处理结果保存在: {os.path.abspath(CONFIG['output_dir'])}")
    print(f"下载目录 '{CONFIG['download_dir']}' 中的文件已被尝试清理。")
    print("-" * 30)


# ================================================
# 主程序入口点
# ================================================
if __name__ == "__main__":
    # --- 打印程序标题和配置信息 ---
    print("=" * 50)
    print("=== 自动化图片处理系统 (菜单模式) ===")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"OpenCV 版本: {cv2.__version__}")
    print(f"配置:")
    for key, value in CONFIG.items():
        # 对路径配置尝试显示绝对路径
        if key.endswith("_dir") or key.endswith("_file") or key.endswith("_index"):
             try: print(f"  - {key}: {os.path.abspath(str(value))}")
             except Exception: print(f"  - {key}: {value}") # 出错则显示原始值
        else: print(f"  - {key}: {value}")
    print("-" * 50)

    # --- 读取下载列表 ---
    download_items = []
    try:
        print("\n--- 正在读取下载列表 ---")
        download_list_file = CONFIG['download_list_file']
        # 调用读取列表函数 (现在应该能找到了)
        download_items = read_download_list(download_list_file)
        if not download_items: print(f"警告：下载列表 '{download_list_file}' 为空或读取失败。")
        else: print(f"已成功加载 {len(download_items)} 条记录。")
        print("-" * 30)
    except NameError as ne: # 以防万一函数名还是错了
        print(f"\n! 严重错误: 函数 '{ne.name}' 未定义。请确保脚本是完整的。")
        traceback.print_exc(); exit(1)
    except Exception as e: # 捕获读取列表时的其他错误
        print(f"\n! 读取下载列表时发生严重错误: {type(e).__name__}: {str(e)}")
        traceback.print_exc(); exit(1)

    # --- 主菜单循环 ---
    while True:
        print("\n--- 主菜单 ---")
        print("1. 下载图片 (根据列表)")
        print("2. 生成/管理模板 (交互式: 添加, 删除, 标记None)")
        print("3. 处理图片 (自动旋转匹配+裁剪, 强制处理下载目录, 覆盖输出并清理下载)") # 菜单说明更清晰
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
                print("\n=== 选项 2: 生成/管理模板 (交互式) ===")
                if not download_items: print("错误：下载列表为空或未成功加载，无法确定要为哪些文件管理模板。")
                elif not os.path.isdir(CONFIG["download_dir"]): print(f"错误: 下载目录 '{CONFIG['download_dir']}' 不存在。请先运行选项 1 下载图片。")
                else: generate_templates(download_items) # 调用交互式模板生成/管理函数
                print("-" * 30); print("模板生成/管理流程结束。")

            elif choice == '3':
                print("\n=== 选项 3: 处理图片 (自动旋转匹配+裁剪, 强制) ===")
                if not download_items: print("错误：下载列表为空或未成功加载，无法确定要处理哪些图片。")
                elif not os.path.isdir(CONFIG["download_dir"]):
                     print(f"警告: 下载目录 '{CONFIG['download_dir']}' 不存在。处理将跳过所有文件。")
                     process_images(download_items) # 调用处理函数（内部处理旋转）
                else: process_images(download_items)
                print("-" * 30); print("图片处理流程结束。")

            elif choice == '4':
                 print("\n=== 选项 4: 下载并处理图片 ===")
                 if not download_items: print("错误：下载列表为空或未成功加载，无法执行此流程。")
                 else:
                     # --- 第 1 步: 下载 ---
                     print("\n--- 第 1 步: 下载图片 ---")
                     download_images(download_items)
                     print("-" * 30); print("下载流程结束。")
                     # --- 第 2 步: 处理 ---
                     print("\n--- 第 2 步: 处理图片 (自动旋转匹配+裁剪, 强制) ---")
                     # 检查下载目录是否存在且非空
                     if not os.path.isdir(CONFIG["download_dir"]) or not os.listdir(CONFIG["download_dir"]):
                          print(f"警告: 下载目录 '{CONFIG['download_dir']}' 在下载后不存在或为空。处理步骤可能无法找到文件。")
                     # 无论如何都调用处理函数，它内部会处理文件不存在的情况
                     process_images(download_items)
                     print("-" * 30); print("图片处理流程结束。")
                 print("\n=== 选项 4 流程结束 ===")

            elif choice == '5':
                print("退出程序...")
                break # 跳出 while 循环
            else:
                print("无效选项，请输入 1, 2, 3, 4 或 5。")

        except KeyboardInterrupt: # 捕获 Ctrl+C 中断
            print("\n! 用户中断操作。返回主菜单。")
            time.sleep(1)
        except Exception as e: # 捕获执行选项时的其他所有错误
             print(f"\n! 在执行选项 '{choice}' 时发生未捕获的严重错误: {type(e).__name__}: {str(e)}")
             traceback.print_exc() # 打印详细错误信息
             print("操作可能未完成。返回主菜单。")
             time.sleep(2)

    # --- 程序结束 ---
    print("=" * 50)
    print("程序已退出.")
    cv2.destroyAllWindows() # 关闭所有 OpenCV 窗口