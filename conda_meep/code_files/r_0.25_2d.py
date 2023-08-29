from main_funcs import fdtd_meep_2d

x_dist = 4
y_dist = 5
dists = [x_dist, y_dist]

plm_depth = min(dists)/10

plate_pos_x = -x_dist/2+plm_depth
plate_pos_y = 0
plate_pos = [plate_pos_x,plate_pos_y]

radius = 0.25
plate_thickness = 0.3
wavelength = 0.532
beam_width = 10
eps = 1e6 + 1e6j


fdtd_meep_2d.run(dists, plate_pos,plate_thickness, radius, wavelength,beam_width,eps,plm_depth)
