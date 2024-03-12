import cv2
import os


if __name__ == '__main__':
    # 视频保存设置
    image_folder = 'demo'
    video_name = 'video.avi'
    # 读取并排序图像文件
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda img_name: int(img_name.split('.')[0][4:]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    # 创建视频写入器
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    cv2_images = []  # 初始化图像列表

    for image in images:
        cv2_images.append(cv2.imread(os.path.join(image_folder, image)))

    for img in cv2_images:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    print("成功保存视频到主文件夹")
    
