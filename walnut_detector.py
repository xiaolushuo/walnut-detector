#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os

def walnut_detector():
    """基于局部极值检测的核桃识别程序"""
    
    print("=== 核桃识别检测程序 ===")
    print("正在使用局部极值检测方法识别核桃...")
    
    # 输出文件
    output_file = 'detection_results.txt'
    
    with open(output_file, 'w') as f:
        f.write("=== Walnut Detection Results ===\n")
        f.write("Method: Local Extrema Detection with K-means Clustering\n")
        f.flush()
        
        # 读取图片
        image_path = 'test_output.jpg'
        image = cv2.imread(image_path)
        
        if image is None:
            error_msg = f"错误: 找不到图片文件 {image_path}"
            print(error_msg)
            f.write(f"ERROR: {error_msg}\n")
            f.write("请确保核桃图片文件在当前目录中\n")
            f.flush()
            return
        
        print(f"✅ 图片加载成功: {image.shape}")
        f.write(f"Image loaded successfully. Size: {image.shape}\n")
        f.flush()
        
        # 步骤1: 图像预处理
        print("🔍 步骤1: 图像预处理...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 步骤2: 局部极值检测
        print("🔍 步骤2: 局部极值检测...")
        kernel_size = 15
        min_filtered = cv2.erode(blurred, np.ones((kernel_size, kernel_size), np.uint8))
        local_minima = (blurred == min_filtered)
        
        # 获取局部最小值的坐标
        min_coords = np.where(local_minima)
        min_points = list(zip(min_coords[1], min_coords[0]))
        
        print(f"✅ 找到 {len(min_points)} 个局部极值点")
        f.write(f"Found {len(min_points)} local minima points\n")
        f.flush()
        
        # 步骤3: K-means聚类
        print("🔍 步骤3: K-means聚类分析...")
        if len(min_points) > 0:
            # 使用OpenCV的k-means聚类
            points = np.array(min_points, dtype=np.float32)
            
            # 设置聚类参数 - 最多11个聚类（对应核桃数量）
            k = min(11, len(points))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            
            # 执行k-means聚类
            _, labels, centers = cv2.kmeans(points, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            print(f"✅ K-means聚类完成，发现 {n_clusters} 个聚类")
            f.write(f"K-means found {n_clusters} clusters\n")
            f.flush()
            
            # 分析每个聚类
            cluster_centers = []
            for i in range(n_clusters):
                cluster_points = points[labels.flatten() == i]
                center = centers[i]
                cluster_centers.append(center)
                
                # 计算聚类的统计信息
                points_count = len(cluster_points)
                f.write(f"  Cluster {i+1}: Center=({center[0]:.1f}, {center[1]:.1f}), Points={points_count}\n")
            
            f.flush()
            
            # 步骤4: 可视化结果
            print("🔍 步骤4: 生成可视化结果...")
            
            # 创建标记图片
            marked_image = image.copy()
            for i, center in enumerate(cluster_centers):
                # 在核桃中心画绿色圆点
                cv2.circle(marked_image, (int(center[0]), int(center[1])), 8, (0, 255, 0), -1)
                # 添加编号
                cv2.putText(marked_image, str(i+1), (int(center[0])+10, int(center[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 保存结果图片
            cv2.imwrite('walnut_detection_result.jpg', marked_image)
            print("✅ 检测结果已保存: walnut_detection_result.jpg")
            f.write("Detection result image saved\n")
            f.flush()
            
            # 步骤5: 创建对比图
            print("🔍 步骤5: 创建对比分析图...")
            try:
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(15, 5))
                
                # 原始图片
                original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.subplot(1, 3, 1)
                plt.imshow(original_rgb)
                plt.title('Original Image')
                plt.axis('off')
                
                # 灰度图
                plt.subplot(1, 3, 2)
                plt.imshow(gray, cmap='gray')
                plt.title('Grayscale')
                plt.axis('off')
                
                # 检测结果
                result_rgb = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)
                plt.subplot(1, 3, 3)
                plt.imshow(result_rgb)
                plt.title(f'Detection Result: {n_clusters} walnuts')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig('walnut_detection_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                print("✅ 对比图已保存: walnut_detection_comparison.png")
                f.write("Comparison image saved\n")
                f.flush()
                
            except ImportError:
                print("⚠️  警告: matplotlib未安装，跳过对比图生成")
                f.write("Warning: matplotlib not available, skipping comparison image\n")
                f.flush()
            
            # 最终结果
            walnut_count = n_clusters
            print(f"\n🎉 检测完成！")
            print(f"📊 检测结果: {walnut_count} 个核桃")
            
            f.write("\n=== FINAL RESULTS ===\n")
            f.write(f"Detection method: Local Extrema Detection\n")
            f.write(f"Total local minima points: {len(min_points)}\n")
            f.write(f"K-means clusters: {n_clusters}\n")
            f.write(f"Estimated walnut count: {walnut_count}\n")
            f.write("=== Detection Complete ===\n")
            f.flush()
            
            # 输出文件列表
            print(f"\n📁 生成的文件:")
            print(f"  - detection_results.txt (详细日志)")
            print(f"  - walnut_detection_result.jpg (检测结果)")
            print(f"  - walnut_detection_comparison.png (对比图)")
            
            return walnut_count
            
        else:
            error_msg = "未找到局部极值点"
            print(f"❌ {error_msg}")
            f.write(f"ERROR: {error_msg}\n")
            f.flush()
            return 0

def main():
    """主函数"""
    try:
        # 检查图片文件是否存在
        if not os.path.exists('test_output.jpg'):
            print("❌ 错误: 找不到 test_output.jpg 文件")
            print("请确保核桃图片文件在当前目录中")
            return
        
        # 运行检测程序
        count = walnut_detector()
        
        if count > 0:
            print(f"\n✅ 成功检测到 {count} 个核桃！")
        else:
            print(f"\n❌ 检测失败")
            
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        print("请检查是否安装了所需的依赖包")
        print("运行: pip install opencv-python numpy matplotlib")

if __name__ == "__main__":
    main()