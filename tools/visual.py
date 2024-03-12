from det3d.core.bbox.box_np_ops import center_to_corner_box3d
import open3d as o3d  # 一个开源库，用于处理3D数据和进行3D可视化
import argparse
import pickle  # 序列化和反序列化Python对象结构
import ast

# import debugpy
# debugpy.listen(11123)
# print("wait for debugger")
# debugpy.wait_for_client()
# print("debugger attach")

def label2color(label):  # 根据检测对象的类别标签返回对应的颜色
    colors = [[204/255, 0, 0], [52/255, 101/255, 164/255],
    [245/255, 121/255, 0], [115/255, 210/255, 22/255]]
    return colors[label]

def corners_to_lines(qs, color=[204/255, 0, 0]):  # 将 3D 边界框的角点转换为线条集合以便于在 open3d 中绘制。边界框通过它的 8 个角点定义，函数将这些点连接成线条。
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    idx = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7)]
    cl = [color for i in range(12)]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(qs),
        lines=o3d.utility.Vector2iVector(idx),
    )
    line_set.colors = o3d.utility.Vector3dVector(cl)
    
    return line_set

def plot_boxes(boxes, score_thresh):  # 接受检测框和一个分数阈值作为输入，如果检测框的分数高于这个阈值，函数会将检测框转换为'open3d'中的线条集合，并将他们添加到可视化对象列表中
    visuals =[] 
    num_det = boxes['scores'].shape[0]
    for i in range(num_det):
        score = boxes['scores'][i]
        if score < score_thresh:
            continue 

        box = boxes['boxes'][i:i+1]
        label = boxes['classes'][i]
        corner = center_to_corner_box3d(box[:, :3], box[:, 3:6], box[:, -1])[0].tolist()
        color = label2color(label)
        visuals.append(corners_to_lines(corner, color))
    return visuals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--path', help='path to visualization file', type=str)
    parser.add_argument('--thresh', help='visualization threshold', type=float, default=0.3)
    args = parser.parse_args()

    with open(args.path, 'rb') as f:
        data_dicts = pickle.load(f)
        data_dicts = ast.literal_eval(data_dicts)

    for data in data_dicts:
        points = data['points']  # 包含了点云数据，每个点为一个 [x, y, z] 坐标。
        detections = data['detections']  # 包含了检测结果，scores（检测得分）和其他与检测相关的信息。

        pcd = o3d.geometry.PointCloud()  # 创建一个 open3d.geometry.PointCloud 对象来存储点云数据
        # pcd = o3d.geometry.PointCloud()
        # 使用 o3d.utility.Vector3dVector 将 points 转换成适合 open3d 使用的格式，并设置为点云对象的 points 属性。
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # 绘制检测框，收集所有要可视化的对象，首先添加点云对象
        visual = [pcd]
        num_dets = detections['scores'].shape[0]
        visual += plot_boxes(detections, args.thresh)

        # 可视化
        # 将点云和所有边界框绘制到一个可视化窗口中，以便用户查看
        o3d.visualization.draw_geometries(visual)
