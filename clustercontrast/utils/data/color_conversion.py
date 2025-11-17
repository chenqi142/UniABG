import cv2
import numpy as np
import os
import pdb


def bgr_to_rgb_float(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

def rgb_float_to_bgr_uint8(image_rgb_float):
    image_bgr_float = cv2.cvtColor(image_rgb_float, cv2.COLOR_RGB2BGR)
    return np.clip(image_bgr_float, 0, 255).astype(np.uint8)

# Matrizen aus dem Paper von Reinhard et al. (2001)
RGB_TO_LMS_MATRIX = np.array([
    [0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]
], dtype=np.float32)
LMS_TO_RGB_MATRIX = np.array([
    [ 4.4679, -3.5873,  0.1193], [-1.2186,  2.3809, -0.1624], [ 0.0497, -0.2439,  1.2045]
], dtype=np.float32)
LOG_LMS_TO_LAB_PRE_MATRIX = np.array([
    [1, 1, 1], [1, 1, -2], [1, -1, 0]
], dtype=np.float32)
LOG_LMS_TO_LAB_DIAG_MATRIX = np.array([
    [1/np.sqrt(3), 0, 0], [0, 1/np.sqrt(6), 0], [0, 0, 1/np.sqrt(2)]
], dtype=np.float32)
LOG_LMS_TO_LAB_MATRIX = np.dot(LOG_LMS_TO_LAB_DIAG_MATRIX, LOG_LMS_TO_LAB_PRE_MATRIX)
LAB_TO_LOG_LMS_INV_PRE_MATRIX = np.linalg.inv(LOG_LMS_TO_LAB_PRE_MATRIX)
LAB_TO_LOG_LMS_INV_DIAG_MATRIX = np.linalg.inv(LOG_LMS_TO_LAB_DIAG_MATRIX)
LAB_TO_LOG_LMS_MATRIX = np.dot(LAB_TO_LOG_LMS_INV_PRE_MATRIX, LAB_TO_LOG_LMS_INV_DIAG_MATRIX)
EPSILON = 1e-7

def convert_rgb_to_lab_reinhard(image_rgb_float):
    image_rgb_float = image_rgb_float.astype(np.float32)
    height, width, channels = image_rgb_float.shape
    pixels_rgb = image_rgb_float.reshape(-1, 3)
    pixels_lms = np.dot(pixels_rgb, RGB_TO_LMS_MATRIX.T)
    pixels_lms[pixels_lms <= 0] = EPSILON
    pixels_log_lms = np.log10(pixels_lms)
    pixels_lab = np.dot(pixels_log_lms, LOG_LMS_TO_LAB_MATRIX.T)
    return pixels_lab.reshape((height, width, channels))

def convert_lab_reinhard_to_rgb(image_lab_float):
    height, width, channels = image_lab_float.shape
    pixels_lab = image_lab_float.reshape(-1, 3)
    pixels_log_lms = np.dot(pixels_lab, LAB_TO_LOG_LMS_MATRIX.T)
    pixels_lms = np.power(10.0, pixels_log_lms)
    pixels_rgb = np.dot(pixels_lms, LMS_TO_RGB_MATRIX.T)
    return pixels_rgb.reshape((height, width, channels))
# --- 结束 Reinhard 核心函数 ---

def calculate_global_lab_stats(dataset):
    """
    计算给定图像路径列表中所有图像的全局平均lαβ均值和标准差。
    返回: (avg_mean_l, avg_std_l, avg_mean_a, avg_std_a, avg_mean_b, avg_std_b)
    """
    all_means_l, all_stds_l = [], []
    all_means_a, all_stds_a = [], []
    all_means_b, all_stds_b = [], []

    print(f"开始计算 {len(dataset)} 张图像的全局lαβ统计...")
    for i, (img_path, _) in enumerate(dataset):
        if (i+1) % 50 == 0:
            print(f"  处理第 {i+1}/{len(dataset)} 张图像...")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue

        img_rgb_float = bgr_to_rgb_float(img_bgr)
        img_lab = convert_rgb_to_lab_reinhard(img_rgb_float)

        mean_l, std_l = cv2.meanStdDev(img_lab[:,:,0])
        mean_a, std_a = cv2.meanStdDev(img_lab[:,:,1])
        mean_b, std_b = cv2.meanStdDev(img_lab[:,:,2])

        all_means_l.append(mean_l[0,0])
        all_stds_l.append(std_l[0,0])
        all_means_a.append(mean_a[0,0])
        all_stds_a.append(std_a[0,0])
        all_means_b.append(mean_b[0,0])
        all_stds_b.append(std_b[0,0])

    if not all_means_l: # 如果没有成功处理任何图像
        raise ValueError("未能从图像路径中计算任何统计数据。")

    # 计算所有图像统计数据的平均值
    avg_stats = (
        np.mean(all_means_l), np.mean(all_stds_l),
        np.mean(all_means_a), np.mean(all_stds_a),
        np.mean(all_means_b), np.mean(all_stds_b)
    )
    print("全局lαβ统计计算完成。")
    return avg_stats


def apply_color_transfer_to_drone(drone_img_bgr, global_satellite_lab_stats):
    """
    将无人机图像的颜色转换为全局卫星图像风格。
    drone_img_bgr: 单张无人机图像 (OpenCV BGR格式)
    global_satellite_lab_stats: 预计算的全局卫星lαβ统计数据元组
                                (μ_s_l, σ_s_l, μ_s_a, σ_s_a, μ_s_b, σ_s_b)
    返回: 色彩迁移后的无人机图像 (OpenCV BGR uint8格式)
    """
    drone_rgb_float = bgr_to_rgb_float(drone_img_bgr)
    drone_lab = convert_rgb_to_lab_reinhard(drone_rgb_float)

    # 动态计算当前无人机图像的lαβ统计
    mean_d_l, std_d_l = cv2.meanStdDev(drone_lab[:,:,0])
    mean_d_a, std_d_a = cv2.meanStdDev(drone_lab[:,:,1])
    mean_d_b, std_d_b = cv2.meanStdDev(drone_lab[:,:,2])

    stats_drone_current = np.array([
        mean_d_l[0,0], mean_d_a[0,0], mean_d_b[0,0],
        std_d_l[0,0],  std_d_a[0,0],  std_d_b[0,0]
    ])

    # 从全局卫星统计中解包
    mu_s_l, sigma_s_l, mu_s_a, sigma_s_a, mu_s_b, sigma_s_b = global_satellite_lab_stats

    # 分离无人机图像的lαβ通道
    l_drone, a_drone, b_drone = cv2.split(drone_lab)

    # 1. 中心化无人机通道 (使用其自身的均值)
    l_drone_centered = l_drone - stats_drone_current[0]
    a_drone_centered = a_drone - stats_drone_current[1]
    b_drone_centered = b_drone - stats_drone_current[2]

    # 2. 缩放无人机通道 (使用其自身的标准差和卫星图像的标准差)
    l_drone_scaled = (sigma_s_l / (stats_drone_current[3] + EPSILON)) * l_drone_centered
    a_drone_scaled = (sigma_s_a / (stats_drone_current[4] + EPSILON)) * a_drone_centered
    b_drone_scaled = (sigma_s_b / (stats_drone_current[5] + EPSILON)) * b_drone_centered

    # 3. 添加卫星图像的均值
    l_final = l_drone_scaled + mu_s_l
    a_final = a_drone_scaled + mu_s_a
    b_final = b_drone_scaled + mu_s_b

    transformed_drone_lab = cv2.merge([l_final, a_final, b_final])
    transformed_drone_rgb_float = convert_lab_reinhard_to_rgb(transformed_drone_lab)
    transformed_drone_bgr_uint8 = rgb_float_to_bgr_uint8(transformed_drone_rgb_float)

    return transformed_drone_bgr_uint8

def get_all_image_paths(base_path):
    """
    从给定路径返回所有图片的路径。
    :param base_path: 基础路径，例如 '/drone'
    :return: 包含所有图片路径的列表
    """
    image_paths = []
    # 遍历基础路径下的所有文件夹和文件
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # 检查文件扩展名是否为常见的图片格式
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                # 构造完整的文件路径
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    return image_paths

if __name__ == "__main__":
    image_paths = get_all_image_paths("/data0/chenqi_data/University-Release/train/satellite")
    print(f"找到 {len(image_paths)} 张图片")
    # 定义全局卫星图像的统计信息
    # 这些值通常是从卫星图像数据集中计算得到的
    global_satellite_lab_stats = calculate_global_lab_stats(image_paths)

    # 测试图像路径
    drone_image_path = "/data0/chenqi_data/University-Release/train/drone/1306/image-07.jpeg"
    output_image_path = "output_image_1306.jpg"

    # 读取无人机图像
    drone_img_bgr = cv2.imread(drone_image_path)
    if drone_img_bgr is None:
        print(f"无法读取图像 {drone_image_path}")
        exit()

    # 应用色彩迁移算法
    transformed_drone_bgr_uint8 = apply_color_transfer_to_drone(drone_img_bgr, global_satellite_lab_stats)

    # 保存转换后的图像
    cv2.imwrite(output_image_path, transformed_drone_bgr_uint8)
    print(f"转换后的图像已保存到 {output_image_path}")