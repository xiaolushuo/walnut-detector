#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os

def advanced_walnut_detection():
    """高级核桃检测算法"""
    
    output_file = 'detection_results.txt'
    
    with open(output_file, 'w') as f:
        f.write("=== Advanced Walnut Detection Results ===\n")
        f.write(f"Testing image: test_output.jpg\n")
        f.flush()
        
        # 读取图片
        image_path = 'test_output.jpg'
        image = cv2.imread(image_path)
        
        if image is None:
            f.write("ERROR: Cannot read image file\n")
            f.write("Please make sure test_output.jpg is in the same directory\n")
            f.flush()
            return
        
        f.write(f"Image loaded successfully. Size: {image.shape}\n")
        f.flush()
        
        # 方法1: 基于纹理和颜色的分割
        f.write("\n--- Method 1: Texture-Based Segmentation ---\n")
        f.flush()
        
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 使用K-means聚类进行颜色分割
        pixels = lab.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # 尝试不同的聚类数量
        for k in [3, 5, 8, 11]:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 重构分割后的图像
            centers = np.uint8(centers)
            segmented = centers[labels.flatten()]
            segmented = segmented.reshape(lab.shape)
            
            # 转换回BGR用于显示
            segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_LAB2BGR)
            
            # 分析每个聚类
            unique_labels = np.unique(labels)
            f.write(f"K={k} clusters:\n")
            
            for i, label in enumerate(unique_labels):
                mask = (labels == label).reshape(lab.shape[:2])
                area = np.sum(mask)
                
                if area > 100:  # 过滤小区域
                    # 计算该区域的统计信息
                    region_pixels = image[mask]
                    color_mean = np.mean(region_pixels, axis=0)
                    f.write(f"  Cluster {i}: Area={area}, Mean color=({color_mean[0]:.1f}, {color_mean[1]:.1f}, {color_mean[2]:.1f})\n")
            
            f.flush()
        
        # 方法2: 基于局部极值点的检测
        f.write("\n--- Method 2: Local Extrema Detection ---\n")
        f.flush()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 寻找局部最小值（暗点，可能是核桃）
        kernel_size = 15
        min_filtered = cv2.erode(blurred, np.ones((kernel_size, kernel_size), np.uint8))
        
        # 比较原图和最小值滤波后的图
        local_minima = (blurred == min_filtered)
        
        # 获取局部最小值的坐标
        min_coords = np.where(local_minima)
        min_points = list(zip(min_coords[1], min_coords[0]))
        
        f.write(f"Found {len(min_points)} local minima points\n")
        f.flush()
        
        # 过滤和聚类这些点
        if len(min_points) > 0:
            # 使用OpenCV的k-means聚类
            points = np.array(min_points, dtype=np.float32)
            
            # 设置聚类参数
            k = min(11, len(points))  # 最多11个聚类
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            
            # 执行k-means聚类
            _, labels, centers = cv2.kmeans(points, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            f.write(f"K-means found {n_clusters} clusters\n")
            f.flush()
            
            # 计算每个聚类的中心和点数
            cluster_centers = []
            for i in range(n_clusters):
                cluster_points = points[labels.flatten() == i]
                center = centers[i]
                cluster_centers.append(center)
                f.write(f"  Cluster {i}: Center=({center[0]:.1f}, {center[1]:.1f}), Points={len(cluster_points)}\n")
            
            f.flush()
            
            # 在图片上标记这些中心
            marked_image = image.copy()
            for i, center in enumerate(cluster_centers):
                cv2.circle(marked_image, (int(center[0]), int(center[1])), 8, (0, 255, 0), -1)
                cv2.putText(marked_image, str(i+1), (int(center[0])+10, int(center[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imwrite('local_extrema_detection.jpg', marked_image)
            f.write("Local extrema detection result saved\n")
            f.flush()
            
            extrema_count = len(cluster_centers)
        else:
            extrema_count = 0
        
        # 方法3: 基于模板匹配的检测
        f.write("\n--- Method 3: Template Matching ---\n")
        f.flush()
        
        # 创建一个简单的圆形模板
        template_size = 20
        template = np.zeros((template_size, template_size), dtype=np.uint8)
        cv2.circle(template, (template_size//2, template_size//2), template_size//2-2, 255, -1)
        
        # 在灰度图上进行模板匹配
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        
        # 设置阈值
        threshold = 0.6
        locations = np.where(result >= threshold)
        
        f.write(f"Template matching found {len(locations[0])} potential matches\n")
        f.flush()
        
        # 初始化模板匹配结果
        template_centers = []
        
        # 聚类匹配结果
        if len(locations[0]) > 0:
            points = list(zip(locations[1], locations[0]))
            
            # 使用简单的距离聚类
            clustered_points = []
            for point in points:
                is_duplicate = False
                for cluster in clustered_points:
                    for cluster_point in cluster:
                        distance = np.sqrt((point[0] - cluster_point[0])**2 + (point[1] - cluster_point[1])**2)
                        if distance < 15:
                            cluster.append(point)
                            is_duplicate = True
                            break
                    if is_duplicate:
                        break
                if not is_duplicate:
                    clustered_points.append([point])
            
            # 计算每个聚类的中心
            for cluster in clustered_points:
                if len(cluster) >= 2:  # 至少2个匹配点
                    center = np.mean(cluster, axis=0)
                    template_centers.append(center)
                    f.write(f"  Template cluster: Center=({center[0]:.1f}, {center[1]:.1f}), Matches={len(cluster)}\n")
            
            f.flush()
            
            # 在图片上标记模板匹配结果
            template_marked = image.copy()
            for i, center in enumerate(template_centers):
                cv2.circle(template_marked, (int(center[0]), int(center[1])), 10, (255, 0, 0), 2)
                cv2.putText(template_marked, str(i+1), (int(center[0])+15, int(center[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imwrite('template_matching_detection.jpg', template_marked)
            f.write("Template matching result saved\n")
            f.flush()
        
        template_count = len(template_centers)
        
        # 方法4: 综合所有检测结果
        f.write("\n--- Method 4: Comprehensive Analysis ---\n")
        f.flush()
        
        # 收集所有检测到的候选位置
        all_candidates = []
        
        # 添加局部极值检测结果
        for center in cluster_centers:
            all_candidates.append(('extrema', center))
        
        # 添加模板匹配结果
        for center in template_centers:
            all_candidates.append(('template', center))
        
        # 去重和聚类
        unique_candidates = []
        for candidate_type, candidate in all_candidates:
            is_duplicate = False
            for existing_type, existing in unique_candidates:
                distance = np.sqrt((candidate[0] - existing[0])**2 + (candidate[1] - existing[1])**2)
                if distance < 25:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_candidates.append((candidate_type, candidate))
        
        comprehensive_count = len(unique_candidates)
        f.write(f"Comprehensive detection: {comprehensive_count} candidates\n")
        f.flush()
        
        # 创建最终可视化
        f.write("\n--- Creating Final Visualization ---\n")
        f.flush()
        
        # 创建对比图
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(20, 10))
        
        # 原始图片
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 3, 1)
        plt.imshow(original_rgb)
        plt.title('Original Image')
        plt.axis('off')
        
        # 局部极值检测结果
        if extrema_count > 0:
            extrema_rgb = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 3, 2)
            plt.imshow(extrema_rgb)
            plt.title(f'Local Extrema: {extrema_count}')
            plt.axis('off')
        
        # 模板匹配结果
        if template_count > 0:
            template_rgb = cv2.cvtColor(template_marked, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 3, 3)
            plt.imshow(template_rgb)
            plt.title(f'Template Matching: {template_count}')
            plt.axis('off')
        
        # 综合结果
        comprehensive_image = image.copy()
        for i, (candidate_type, candidate) in enumerate(unique_candidates):
            color = (0, 255, 0) if candidate_type == 'extrema' else (255, 0, 0)
            cv2.circle(comprehensive_image, (int(candidate[0]), int(candidate[1])), 12, color, 3)
            cv2.putText(comprehensive_image, str(i+1), (int(candidate[0])+15, int(candidate[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(comprehensive_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Comprehensive: {comprehensive_count}')
        plt.axis('off')
        
        # 灰度图
        plt.subplot(2, 3, 5)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        # 边缘图
        edges = cv2.Canny(gray, 50, 150)
        plt.subplot(2, 3, 6)
        plt.imshow(edges, cmap='gray')
        plt.title('Edges')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('advanced_walnut_detection.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        cv2.imwrite('comprehensive_detection.jpg', comprehensive_image)
        f.write("Final visualization saved\n")
        f.flush()
        
        # 最终结果
        f.write("\n=== FINAL RESULTS ===\n")
        f.write(f"Local extrema detection: {extrema_count} candidates\n")
        f.write(f"Template matching: {template_count} candidates\n")
        f.write(f"Comprehensive detection: {comprehensive_count} candidates\n")
        
        # 根据综合分析给出最终估计
        final_count = comprehensive_count
        f.write(f"\nFinal estimated walnut count: {final_count}\n")
        f.write("=== Advanced Detection Complete ===\n")
        f.flush()
        
        # 打印结果到控制台
        print(f"\n=== 核桃检测结果 ===")
        print(f"图片文件: {image_path}")
        print(f"检测到的核桃数量: {final_count}")
        print(f"局部极值检测: {extrema_count} 个")
        print(f"模板匹配检测: {template_count} 个")
        print(f"综合检测结果: {comprehensive_count} 个")
        print(f"生成的文件:")
        print(f"  - detection_results.txt (详细日志)")
        print(f"  - advanced_walnut_detection.png (综合对比图)")
        print(f"  - comprehensive_detection.jpg (最终检测结果)")
        print(f"  - local_extrema_detection.jpg (局部极值检测结果)")

def main():
    """主函数"""
    print("=== 高级核桃检测程序 ===")
    print("正在检测图片中的核桃...")
    
    # 检查图片文件是否存在
    if not os.path.exists('test_output.jpg'):
        print("错误: 找不到 test_output.jpg 文件")
        print("请确保核桃图片文件在当前目录中")
        return
    
    try:
        advanced_walnut_detection()
        print("\n检测完成！")
    except Exception as e:
        print(f"检测过程中出现错误: {e}")
        print("请检查是否安装了所需的依赖包")

if __name__ == "__main__":
    main()