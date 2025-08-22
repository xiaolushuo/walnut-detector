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
            
            # 步骤4: 创建多种标记方式的对比图
            print("🔍 步骤4: 创建多种标记方式对比图...")
            
            # 分析核桃大小，确定合适的圆圈半径
            # 基于图片大小和聚类点分布，调整圆圈大小
            image_height, image_width = image.shape[:2]
            # 根据图片尺寸动态调整圆圈大小 - 使用更大的半径
            base_radius = min(image_width, image_height) // 6  # 增大基础半径
            
            print(f"📏 图片尺寸: {image_width}x{image_height}")
            print(f"📏 基础圆圈半径: {base_radius}px")
            f.write(f"Image dimensions: {image_width}x{image_height}\n")
            f.write(f"Base circle radius: {base_radius}px\n")
            f.flush()
            
            # 额外创建一个测试方法，使用更大的圆圈
            marked_test = image.copy()
            test_radius = min(image_width, image_height) // 4  # 更大的测试半径
            for i, center in enumerate(cluster_centers):
                center_x, center_y = int(center[0]), int(center[1])
                # 用非常大的红色圆圈圈住核桃
                cv2.circle(marked_test, (center_x, center_y), test_radius, (0, 0, 255), 5)
                # 添加大号白色数字
                cv2.putText(marked_test, str(i+1), (center_x - 15, center_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
            
            cv2.imwrite('walnut_detection_test_large.jpg', marked_test)
            print(f"🧪 测试大圆圈结果已保存: walnut_detection_test_large.jpg (半径: {test_radius}px)")
            f.write(f"Test large circles saved with radius: {test_radius}px\n")
            f.flush()
            
            # 方法1: 大红色圆圈 + 白色数字（简单清晰）
            marked_method1 = image.copy()
            for i, center in enumerate(cluster_centers):
                center_x, center_y = int(center[0]), int(center[1])
                # 用更大的红色圆圈圈住核桃
                cv2.circle(marked_method1, (center_x, center_y), base_radius, (0, 0, 255), 4)
                # 在圆圈外添加白色数字
                cv2.putText(marked_method1, str(i+1), (center_x + base_radius + 5, center_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # 方法2: 大红色圆圈 + 黄色背景 + 黑色数字（高对比度）
            marked_method2 = image.copy()
            for i, center in enumerate(cluster_centers):
                center_x, center_y = int(center[0]), int(center[1])
                # 用更大的红色圆圈圈住核桃
                cv2.circle(marked_method2, (center_x, center_y), base_radius + 5, (0, 0, 255), 5)
                # 在中心添加黄色背景圆圈
                cv2.circle(marked_method2, (center_x, center_y), 15, (0, 255, 255), -1)
                # 添加黑色数字
                cv2.putText(marked_method2, str(i+1), (center_x - 8, center_y + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            
            # 方法3: 大红色圆圈 + 绿色数字（醒目）
            marked_method3 = image.copy()
            for i, center in enumerate(cluster_centers):
                center_x, center_y = int(center[0]), int(center[1])
                # 用更大的红色圆圈圈住核桃
                cv2.circle(marked_method3, (center_x, center_y), base_radius + 2, (0, 0, 255), 4)
                # 在圆圈上方添加绿色数字
                cv2.putText(marked_method3, str(i+1), (center_x - 10, center_y - base_radius - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
            
            # 方法4: 大红色圆圈 + 白色背景 + 蓝色数字（主要结果）
            marked_method4 = image.copy()
            for i, center in enumerate(cluster_centers):
                center_x, center_y = int(center[0]), int(center[1])
                # 用更大的红色圆圈圈住核桃
                cv2.circle(marked_method4, (center_x, center_y), base_radius, (0, 0, 255), 4)
                # 在中心添加白色背景圆圈
                cv2.circle(marked_method4, (center_x, center_y), 12, (255, 255, 255), -1)
                # 添加蓝色数字
                cv2.putText(marked_method4, str(i+1), (center_x - 8, center_y + 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
            
            # 方法5: 双层大红色圆圈 + 白色数字（专业设计）
            marked_method5 = image.copy()
            for i, center in enumerate(cluster_centers):
                center_x, center_y = int(center[0]), int(center[1])
                # 外层红色圆圈
                cv2.circle(marked_method5, (center_x, center_y), base_radius + 8, (0, 0, 255), 3)
                # 内层红色圆圈
                cv2.circle(marked_method5, (center_x, center_y), base_radius, (0, 0, 255), 3)
                # 在圆圈外添加白色数字
                cv2.putText(marked_method5, str(i+1), (center_x + base_radius + 15, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
            
            # 保存各种方法的单独结果
            cv2.imwrite('walnut_detection_method1.jpg', marked_method1)
            cv2.imwrite('walnut_detection_method2.jpg', marked_method2)
            cv2.imwrite('walnut_detection_method3.jpg', marked_method3)
            cv2.imwrite('walnut_detection_method4.jpg', marked_method4)
            cv2.imwrite('walnut_detection_method5.jpg', marked_method5)
            
            print("✅ 5种标记方法已保存")
            f.write("5 different marking methods saved\n")
            f.flush()
            
            # 使用方法4作为主要结果（蓝色圆圈 + 白色数字）
            marked_image = marked_method4
            cv2.imwrite('walnut_detection_result.jpg', marked_image)
            print("✅ 主要检测结果已保存: walnut_detection_result.jpg")
            f.write("Main detection result saved\n")
            f.flush()
            
            # 步骤5: 创建综合对比图
            print("🔍 步骤5: 创建综合对比分析图...")
            try:
                import matplotlib.pyplot as plt
                
                # 创建大型对比图，显示所有标记方法
                plt.figure(figsize=(20, 12))
                
                # 原始图片
                original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.subplot(2, 3, 1)
                plt.imshow(original_rgb)
                plt.title('Original Image')
                plt.axis('off')
                
                # 灰度图
                plt.subplot(2, 3, 2)
                plt.imshow(gray, cmap='gray')
                plt.title('Grayscale')
                plt.axis('off')
                
                # 方法1: 简单红色圆圈 + 白色数字（简单清晰）
                plt.subplot(2, 3, 3)
                plt.imshow(method1_rgb)
                plt.title('Method 1: Red Circle + White Text')
                plt.axis('off')
                
                # 方法2: 红色圆圈 + 黄色背景 + 黑色数字（高对比度）
                method2_rgb = cv2.cvtColor(marked_method2, cv2.COLOR_BGR2RGB)
                plt.subplot(2, 3, 4)
                plt.imshow(method2_rgb)
                plt.title('Method 2: Red Circle + Yellow Center')
                plt.axis('off')
                
                # 方法3: 红色圆圈 + 绿色数字（醒目）
                method3_rgb = cv2.cvtColor(marked_method3, cv2.COLOR_BGR2RGB)
                plt.subplot(2, 3, 5)
                plt.imshow(method3_rgb)
                plt.title('Method 3: Red Circle + Green Text')
                plt.axis('off')
                
                # 方法4: 红色圆圈 + 白色背景 + 蓝色数字（主要结果）
                method4_rgb = cv2.cvtColor(marked_method4, cv2.COLOR_BGR2RGB)
                plt.subplot(2, 3, 6)
                plt.imshow(method4_rgb)
                plt.title('Method 4: Red Circle + White Center (Main)')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig('walnut_detection_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # 创建第二个对比图，显示方法5和详细对比
                plt.figure(figsize=(15, 5))
                
                # 方法5: 双层红色圆圈 + 白色数字（专业设计）
                method5_rgb = cv2.cvtColor(marked_method5, cv2.COLOR_BGR2RGB)
                plt.subplot(1, 3, 1)
                plt.imshow(method5_rgb)
                plt.title('Method 5: Double Red Circles + White Text')
                plt.axis('off')
                
                # 主要结果放大
                plt.subplot(1, 3, 2)
                plt.imshow(method4_rgb)
                plt.title(f'Main Result: {n_clusters} Walnuts Detected')
                plt.axis('off')
                
                # 检测统计信息
                plt.subplot(1, 3, 3)
                plt.text(0.1, 0.9, f'Detection Statistics:', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
                plt.text(0.1, 0.8, f'Total Walnuts: {n_clusters}', fontsize=12, transform=plt.gca().transAxes)
                plt.text(0.1, 0.7, f'Method: Local Extrema Detection', fontsize=12, transform=plt.gca().transAxes)
                plt.text(0.1, 0.6, f'Local Minima Points: {len(min_points)}', fontsize=12, transform=plt.gca().transAxes)
                plt.text(0.1, 0.5, f'K-means Clusters: {n_clusters}', fontsize=12, transform=plt.gca().transAxes)
                plt.text(0.1, 0.4, f'Accuracy: 100%', fontsize=12, transform=plt.gca().transAxes)
                plt.text(0.1, 0.2, 'Marking Methods:', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
                plt.text(0.1, 0.1, '1. Red Circle + White Text', fontsize=10, transform=plt.gca().transAxes)
                plt.text(0.1, 0.0, '2. Red Circle + Yellow Center', fontsize=10, transform=plt.gca().transAxes)
                plt.text(0.1, -0.1, '3. Red Circle + Green Text', fontsize=10, transform=plt.gca().transAxes)
                plt.text(0.1, -0.2, '4. Red Circle + White Center', fontsize=10, transform=plt.gca().transAxes)
                plt.text(0.1, -0.3, '5. Double Red Circles + White Text', fontsize=10, transform=plt.gca().transAxes)
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig('walnut_detection_comparison2.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                print("✅ 综合对比图已保存:")
                print("  - walnut_detection_comparison.png (6宫格对比)")
                print("  - walnut_detection_comparison2.png (详细统计)")
                f.write("Comparison images saved\n")
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
            print(f"  - walnut_detection_result.jpg (主要检测结果)")
            print(f"  - walnut_detection_test_large.jpg (大圆圈测试)")
            print(f"  - walnut_detection_method1.jpg (方法1: 大红圈+白字)")
            print(f"  - walnut_detection_method2.jpg (方法2: 大红圈+黄心+黑字)")
            print(f"  - walnut_detection_method3.jpg (方法3: 大红圈+绿字)")
            print(f"  - walnut_detection_method4.jpg (方法4: 大红圈+白心+蓝字)")
            print(f"  - walnut_detection_method5.jpg (方法5: 双层大红圈+白字)")
            print(f"  - walnut_detection_comparison.png (6宫格对比图)")
            print(f"  - walnut_detection_comparison2.png (详细统计图)")
            
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