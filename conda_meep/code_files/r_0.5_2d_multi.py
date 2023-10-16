from main_funcs import fdtd_meep_multipinhole_2d

x_dist = 18
y_dist = 12
dists = [x_dist, y_dist]

plm_depth = min(dists)/10

radius = 0.3
plate_thickness = 3
wavelength = 0.589
beam_width = 1
eps = 1 + 0.1j

pinhole_pos = [-2]
window_pos = [1]

plate_pos_x = -x_dist/2+plm_depth+plate_thickness/2
plate_pos_y = 0
plate_pos = [plate_pos_x,plate_pos_y]



fdtd_meep_multipinhole_2d.run(dists, plate_pos,plate_thickness, radius, wavelength,beam_width,eps,plm_depth,pinhole_pos,window_pos)
