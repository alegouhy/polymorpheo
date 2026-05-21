    for idx, region in enumerate(region_transforms):
        t_matrix = np.asarray(region["transforms"], dtype=float)
        inv_t = np.linalg.inv(t_matrix)

        polygon = np.asarray(region["polygon"][0], dtype=float)
        polygon_h = np.hstack((polygon, np.ones((polygon.shape[0], 1))))
        raw_polygon = (inv_t @ polygon_h.T).T
        raw_polygon = polygon_h


        shp_polygon = Polygon(raw_polygon[:, :2] * crop_params["output_area2"][2]/1000)

        # plot the polygon to check it looks correct
        x, y = shp_polygon.exterior.xy
        plt.plot(x, y)
    # plot all new points checkpoints 2
