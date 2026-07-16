import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pyvista as pv
from plotly.subplots import make_subplots

from polymorpheo.utils import polylines_2d_3d

pio.renderers.default = "browser"


def plot_img(img):
    plt.imshow(img, origin="lower")
    plt.axis("off")
    plt.gca().set_aspect("equal")


def plot_contour(
    contour,
    col=None,
    cmap="tab20",
    linewidth=1,
    markersize=6,
    xlim=[-1.3, 1.3],
    ylim=[-1.3, 1.3],
    scal_normals=0.1,
):
    pts, simps, normals, labs = contour
    npts, ndims = pts.shape

    if col is None:
        cmap_cols = np.array(plt.get_cmap(cmap).colors)
        if labs is None:
            col = cmap_cols[0, :]
            col_pts = np.stack([np.array(col)] * npts, axis=0)
        else:
            col_pts = cmap_cols[labs, :]
        col_simps = [0.5] * 3
    else:
        col_simps = (1 + np.array(col)) / 2
        col_pts = np.stack([np.array(col)] * npts, axis=0)

    if simps is not None:
        xs = []
        ys = []
        for a, b in simps:
            xs.extend([pts[a, 0], pts[b, 0], np.nan])
            ys.extend([pts[a, 1], pts[b, 1], np.nan])
        plt.plot(xs, ys, "-", color=col_simps, linewidth=linewidth, zorder=-2)

    if normals is not None:
        normals = scal_normals * normals
        for i in range(npts):
            plt.plot(
                [pts[i, 0], pts[i, 0] + normals[i, 0]],
                [pts[i, 1], pts[i, 1] + normals[i, 1]],
                color=col_simps,
            )

    plt.scatter(pts[1:-1, 0], pts[1:-1, 1], markersize, color=col_pts[1:-1, :])
    plt.scatter(pts[0, 0], pts[0, 1], 4 * markersize, color=col_pts[0, :], marker="^")
    plt.scatter(pts[-1, 0], pts[-1, 1], 3 * markersize, color=col_pts[-1, :], marker="s")

    plt.axis("off")
    plt.gca().set_aspect("equal")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)


def plot_opts(
    opts,
    col=[1, 0, 0],
    linewidth=1,
    markersize=6,
    xlim=[-2, 2],
    ylim=[-2, 2],
    scal_normals=0.1,
):
    col_pts = np.array(col)
    col_simps = (1 + col_pts) / 2

    plt.plot(opts[:, 0], opts[:, 1], color=col_simps)
    plt.scatter(opts[1:-1, 0], opts[1:-1, 1], markersize, color=col_pts)
    plt.scatter(opts[0, 0], opts[0, 1], 4 * markersize, color=col_pts, marker="^")
    plt.scatter(opts[-1, 0], opts[-1, 1], 3 * markersize, color=col_pts, marker="s")

    plt.axis("off")
    plt.gca().set_aspect("equal")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)


def plot_disp(
    disp,
    pts,
    col=[1, 0, 0],
    linewidth=1,
    markersize=3,
    xlim=[-2, 2],
    ylim=[-2, 2],
    scal_normals=0.1,
):
    col_pts = np.array(col)
    col_vecs = (1 + col_pts) / 2

    plt.quiver(pts[:, 0], pts[:, 1], disp[:, 0], disp[:, 1], color=col_vecs)

    plt.axis("off")
    plt.gca().set_aspect("equal")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)


def plot_path(path, n1, n2):
    mat = np.zeros((n1, n2))
    for p in path:
        mat[p[0], p[1]] = 1
    plt.imshow(mat)
    plt.axis("off")


def plot_graph_conn(nodes_list, links_list, col=[0, 0, 1]):
    n = len(nodes_list)
    col_nodes = np.array(col)
    col_links = (1 + col_nodes) / 2
    plt.figure(figsize=(6, 1))

    for i in range(n):
        if i < n - 1:
            links = links_list[i]
            for j in range(len(links)):
                plt.plot([i, i + 1], links[j], color=col_links)

        nodes = nodes_list[i]
        for j in range(len(nodes)):
            plt.plot(i, nodes[j], ".", markersize=3, color=col_nodes)

    plt.ylim([-0.5, 1.5])
    plt.xlim([-1, n])
    plt.show()


def boxplot(y, x, col=[0, 0, 1], w=0.5, lw=2):
    whis = (0, 100)
    bp = plt.boxplot(y, positions=[x], widths=[w], whis=whis, showfliers=False, patch_artist=True)
    for a in bp.keys():
        plt.setp(bp[a], color=col)
    for a in bp["boxes"]:
        a.set_facecolor((*col, 0.5))
    for b in ["whiskers", "caps", "boxes", "medians"]:
        for a in bp[b]:
            a.set_linewidth(lw)
    plt.plot(x, np.mean(y), "*", color=col)
    return bp


def plot_mesh(pts, simps, lib="plotly"):
    dsimp = simps.shape[1]

    if lib == "plotly":
        fig = go.Figure()
        if dsimp == 3:
            fig.add_trace(
                go.Mesh3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    i=simps[:, 0],
                    j=simps[:, 1],
                    k=simps[:, 2],
                    opacity=1,
                    color="lightblue",
                )
            )
        elif dsimp == 2:
            fig.add_trace(
                go.Mesh3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    i=simps[:, 0],
                    j=simps[:, 1],
                    opacity=1,
                    color="lightblue",
                )
            )

        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=5, color="blue", opacity=1),
            )
        )
        fig.update_layout(scene=dict(aspectmode="data"))
        fig.show(auto_open=True)

    elif lib == "pyvista":
        simps_pv = np.hstack([[dsimp, *tri] for tri in simps])
        mesh = pv.PolyData(pts, simps_pv)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color="lightseagreen", show_edges=True)
        plotter.add_axes()
        plotter.show_grid()
        plotter.show()


def plot_obj(
    pts,
    simps=None,
    point_size=2,
    line_width=2,
    pts_col=(0, 0, 1),
    line_col=(0, 0, 0.5),
    face_col=(0.5, 0.5, 1),
    opacity=0.7,
    show_points=True,
    title=None,
    xlim=None,
    ylim=None,
    zlim=None,
    fig=None,
):
    def to_rgb_string(col):
        return f"rgb({int(col[0] * 255)}, {int(col[1] * 255)}, {int(col[2] * 255)})"

    pts_col = to_rgb_string(pts_col)
    line_col = to_rgb_string(line_col)
    face_col = to_rgb_string(face_col)

    pts = np.asarray(pts)
    n, d = pts.shape

    if d not in [2, 3]:
        raise ValueError(f"Only 2D and 3D supported, got {d}D")

    if fig is None:
        fig = go.Figure()

    if simps is not None:
        simps = np.asarray(simps)
        m, p = simps.shape

        if d == 2:
            if p == 2:
                for simplex in simps:
                    i, j = simplex
                    fig.add_trace(
                        go.Scatter(
                            x=[pts[i, 0], pts[j, 0]],
                            y=[pts[i, 1], pts[j, 1]],
                            mode="lines",
                            line=dict(color=line_col, width=line_width),
                        )
                    )

            elif p == 3:
                for simplex in simps:
                    i, j, k = simplex
                    fig.add_trace(
                        go.Scatter(
                            x=[pts[i, 0], pts[j, 0], pts[k, 0], pts[i, 0]],
                            y=[pts[i, 1], pts[j, 1], pts[k, 1], pts[i, 1]],
                            fill="toself",
                            fillcolor=face_col,
                            opacity=opacity,
                            line=dict(color=line_col, width=line_width),
                        )
                    )

        elif d == 3:
            if p == 2:
                x_lines, y_lines, z_lines = [], [], []
                for simplex in simps:
                    i, j = simplex
                    x_lines.extend([pts[i, 0], pts[j, 0], None])
                    y_lines.extend([pts[i, 1], pts[j, 1], None])
                    z_lines.extend([pts[i, 2], pts[j, 2], None])

                fig.add_trace(
                    go.Scatter3d(
                        x=x_lines,
                        y=y_lines,
                        z=z_lines,
                        mode="lines",
                        line=dict(color=line_col, width=line_width),
                    )
                )

            elif p == 3:
                fig.add_trace(
                    go.Mesh3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        i=simps[:, 0],
                        j=simps[:, 1],
                        k=simps[:, 2],
                        color=face_col,
                        opacity=opacity,
                    )
                )

            elif p == 4:
                edges_set = set()
                for simplex in simps:
                    for i in range(4):
                        for j in range(i + 1, 4):
                            edge = tuple(sorted([simplex[i], simplex[j]]))
                            edges_set.add(edge)

                x_lines, y_lines, z_lines = [], [], []
                for i, j in edges_set:
                    x_lines.extend([pts[i, 0], pts[j, 0], None])
                    y_lines.extend([pts[i, 1], pts[j, 1], None])
                    z_lines.extend([pts[i, 2], pts[j, 2], None])

                fig.add_trace(
                    go.Scatter3d(
                        x=x_lines,
                        y=y_lines,
                        z=z_lines,
                        mode="lines",
                        line=dict(color=line_col, width=line_width),
                    )
                )

    if show_points:
        if d == 2:
            fig.add_trace(
                go.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode="markers",
                    marker=dict(size=point_size, color=pts_col),
                )
            )
        elif d == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=point_size, color=pts_col),
                )
            )

    if d == 2:
        xaxis = dict(range=list(xlim)) if xlim is not None else None
        yaxis = dict(range=list(ylim)) if ylim is not None else None
        fig.update_layout(title=title, xaxis=xaxis, yaxis=yaxis)
    elif d == 3:
        scene = dict(aspectmode="data")
        if xlim is not None:
            scene["xaxis"] = dict(range=list(xlim))
        if ylim is not None:
            scene["yaxis"] = dict(range=list(ylim))
        if zlim is not None:
            scene["zaxis"] = dict(range=list(zlim))
        fig.update_layout(title=title, scene=scene)

    return fig


def plot_contour_stack(polylines_list, z_coords, labels=None):
    n = len(polylines_list)
    labels = labels or [""] * n

    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "scene"}] * n],
        subplot_titles=labels,
    )

    for col, polylines in enumerate(polylines_list, start=1):
        pts, simps = polylines_2d_3d(polylines, 2, z_coords)
        x_lines, y_lines, z_lines, c_lines = [], [], [], []
        for i, j in simps:
            z_mid = (pts[i, 2] + pts[j, 2]) / 2
            x_lines += [pts[i, 0], pts[j, 0], None]
            y_lines += [pts[i, 1], pts[j, 1], None]
            z_lines += [pts[i, 2], pts[j, 2], None]
            c_lines += [z_mid, z_mid, np.nan]

        fig.add_trace(
            go.Scatter3d(
                x=x_lines, y=y_lines, z=z_lines,
                mode="lines",
                line=dict(color=c_lines, colorscale="plasma", width=2),
                showlegend=False,
            ),
            row=1, col=col,
        )

    scene = dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z")
    fig.update_layout(**{f"scene{'' if i == 0 else i + 1}": scene for i in range(n)})
    fig.show()
