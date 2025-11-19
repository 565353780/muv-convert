import numpy as np
import open3d as o3d


def random_color():
    return np.random.rand(3)


def vis_faces_edges(
    face_pnts,           # (N, 32, 32, 3)
    edge_pnts,           # (M, 32, 3)
    edge_corner_pnts,    # (M, 2, 3)
):
    geoms = []

    N = face_pnts.shape[0]
    M = edge_pnts.shape[0]

    # ---------------------
    # 1) visualize faces
    # ---------------------
    for i in range(N):
        pts = face_pnts[i].reshape(-1, 3)

        # create point cloud for the face
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)

        color = random_color()
        pc.colors = o3d.utility.Vector3dVector(
            np.tile(color, (pts.shape[0], 1))
        )

        geoms.append(pc)

    # ---------------------
    # 2) visualize edges
    # ---------------------
    for eid in range(M):
        pts = edge_pnts[eid]  # (32, 3)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts)

        # each consecutive pair makes a line segment
        lines = [[i, i+1] for i in range(len(pts)-1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)

        color = random_color()
        line_set.colors = o3d.utility.Vector3dVector(
            np.tile(color, (len(lines), 1))
        )

        geoms.append(line_set)

    # ---------------------
    # 3) visualize edge corners
    # ---------------------
    for eid in range(M):
        corners = edge_corner_pnts[eid]  # (2, 3)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(corners)

        color = np.array([1, 0, 0])  # red for corners
        pc.colors = o3d.utility.Vector3dVector(
            np.tile(color, (2, 1))
        )

        geoms.append(pc)

    o3d.visualization.draw_geometries(geoms)
    return True


def vis_faces_edges_list(shape_data_list):
    """
    可视化多个形状的面、边和顶点，所有形状在同一个窗口中显示

    Args:
        shape_data_list: 形状数据列表，每个元素包含 'type' 和 'data' 字段
                        data 字段需要包含 'surf_wcs', 'edge_wcs', 'corner_wcs'

    Returns:
        bool: 是否成功可视化
    """
    if not shape_data_list or len(shape_data_list) == 0:
        print('[WARN] shape_data_list is empty, nothing to visualize')
        return False

    geoms = []
    total_faces = 0
    total_edges = 0

    print(f'Visualizing {len(shape_data_list)} shape(s) in one window...')

    for shape_idx, shape_data in enumerate(shape_data_list):
        shape_type = shape_data.get('type', 'Unknown')
        data = shape_data['data']

        face_pnts = data.get('surf_gcs', np.array([]))
        edge_pnts = data.get('edge_gcs', np.array([]))
        edge_corner_pnts = data.get('corner_gcs', np.array([]))

        # 检查数据是否为空
        if face_pnts.size == 0 and edge_pnts.size == 0:
            print(f'[WARN] Shape {shape_idx} ({shape_type}) has no geometry data, skipping')
            continue

        N = face_pnts.shape[0] if face_pnts.size > 0 else 0
        M = edge_pnts.shape[0] if edge_pnts.size > 0 else 0

        print(f'  Shape {shape_idx} ({shape_type}): {N} faces, {M} edges')
        total_faces += N
        total_edges += M

        # ---------------------
        # 1) visualize faces
        # ---------------------
        for i in range(N):
            pts = face_pnts[i].reshape(-1, 3)

            # create point cloud for the face
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts)

            color = random_color()
            pc.colors = o3d.utility.Vector3dVector(
                np.tile(color, (pts.shape[0], 1))
            )

            geoms.append(pc)

        # ---------------------
        # 2) visualize edges
        # ---------------------
        for eid in range(M):
            pts = edge_pnts[eid]  # (32, 3)

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(pts)

            # each consecutive pair makes a line segment
            lines = [[i, i+1] for i in range(len(pts)-1)]
            line_set.lines = o3d.utility.Vector2iVector(lines)

            color = random_color()
            line_set.colors = o3d.utility.Vector3dVector(
                np.tile(color, (len(lines), 1))
            )

            geoms.append(line_set)

        # ---------------------
        # 3) visualize edge corners
        # ---------------------
        if edge_corner_pnts.size > 0:
            for eid in range(M):
                corners = edge_corner_pnts[eid]  # (2, 3)

                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(corners)

                color = np.array([1, 0, 0])  # red for corners
                pc.colors = o3d.utility.Vector3dVector(
                    np.tile(color, (2, 1))
                )

                geoms.append(pc)

    if len(geoms) == 0:
        print('[WARN] No geometry to visualize')
        return False

    print(f'Total: {total_faces} faces, {total_edges} edges')
    print('Opening visualization window...')
    o3d.visualization.draw_geometries(geoms)
    return True
