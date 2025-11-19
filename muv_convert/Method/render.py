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
