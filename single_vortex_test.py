import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import tdgl
from tdgl.geometry import box, circle, ellipse
from pint import UnitRegistry
ureg=UnitRegistry()
#import typing


#units to be used throughout the simulation
units_length="nm"
units_field="mT"
units_current="uA"

#parameters of the superconducting material to create SC layer
xi=10 #10
lambda_depth=40 #80
d=1
gamma_phonon = 1#1
kappa_gl= lambda_depth/xi
lambda_eff_screening = lambda_depth**2/xi
sc_layer = tdgl.Layer(coherence_length=xi, london_lambda=lambda_depth, thickness=d, gamma=gamma_phonon)
print(f"kappa= {kappa_gl}, screening length= {lambda_eff_screening}{units_length}, lambda= {lambda_depth}{units_length}, coherence length={xi}{units_length}")

#parameters of environment
B_app=150 #60
epsilon_disorder = 1
print(f"Applied magnetic field: {B_app}{units_field}")

#outer geometry of the SC film
side_length_sc_film = 400 #1000
side_width_sc_film = side_length_sc_film
base_sc_film = box(width=side_length_sc_film, points=601)
sc_film= tdgl.Polygon("film", points=base_sc_film)#.resample(701)
#geometry of circular hole in the middle
hole_radius_factor = 5 #5
hole_radius = hole_radius_factor*xi
hole_center = (0,0)
hole_round = tdgl.Polygon("round hole", points=circle(radius=hole_radius, center=hole_center))
#notch geometry
notch_ellipse_big= 60 #length of notch
notch_ellipse_small= 20 #half of height of notch
y_displacement_notch = -0.3*side_length_sc_film
hole_notch= (
    tdgl.Polygon("notch", points=ellipse(a=notch_ellipse_big,b=notch_ellipse_small, points=401))
    .difference(tdgl.Polygon(points=box(width=notch_ellipse_big,height=2*notch_ellipse_small))
                .translate(dx=notch_ellipse_big/2))
                .translate(dx=0.5*side_length_sc_film, dy=y_displacement_notch)
)
#track geometry
tmp_coord_notch = hole_notch.bbox
notch_endpoint = (tmp_coord_notch[0][0], (tmp_coord_notch[0][1]+(tmp_coord_notch[1][1]-tmp_coord_notch[0][1])/2))
#track_length = np.linalg.norm(np.array(notch_endpoint)-np.array(hole_center))
track_length = np.sqrt(notch_endpoint[0]**2 + notch_endpoint[1]**2)
track_angle = np.arccos((hole_center[1]-notch_endpoint[1])/track_length)
track_angle= np.degrees(track_angle)
track_center=(0,0)
#track_center = (notch_endpoint[0]/2, notch_endpoint[1]+track_length/2)
track = tdgl.Polygon("track", points=box(width=0.3*xi, height=track_length, center=track_center)).rotate(degrees=track_angle,origin=track_center)#.union(hole_notch).resample(200)
track_notch_gap_y = notch_endpoint[1] - track.bbox[0][1]
track_notch_gap_x = notch_endpoint[0] - track.bbox[1][0]
track=track.translate(dx=track_notch_gap_x,dy=track_notch_gap_y).difference(hole_round).resample(801)
#track = box(width=xi, length=track_length).rotate(track_angle)
print(f"track box: {track.bbox}")
print(f"notch endpoint: {notch_endpoint}, track length: {track_length}, track angle: {track_angle}")
#output about simulation geometry
print(f"Dimensions of the SC film: {side_length_sc_film}x{side_length_sc_film}{units_length}, radius of the hole: {hole_radius}{units_length}")

#function for disorder_epsilon
def track_epsilon(r):
    if track.contains_points(r)==True or track.on_boundary(r)==True:
        epsilon=0.5
    else:
        epsilon=1.0
    return epsilon



#put SC material (tdgl.Layer) and geometry (tdgl.Polygon) together into a complete device (tdgl.Device)
sc_device= tdgl.Device("square with hole", layer=sc_layer, film=sc_film, holes=[hole_round, hole_notch], length_units=units_length)
#discretize the device into a mesh to be solved over
mesh_edge_factor = 1.0 #should be small compared to xi
sc_device.make_mesh(max_edge_length=mesh_edge_factor*xi, smooth=1) #smooth 1 or 100 had very little effect on how the mesh looks
fig, ax = sc_device.plot(mesh=True)


#theoretical calculations
PHI0 = 2.06783e-15*ureg.T*ureg.m**2
#critical fields
Bc_thermo=(PHI0/(2*np.sqrt(2)*np.pi*lambda_depth*ureg(units_length).to('m')*xi*ureg(units_length).to('m'))).to(units_field)
Bc_lower=Bc_thermo*np.log(kappa_gl)/(np.sqrt(2)*kappa_gl)
Bc_upper=np.sqrt(2)*kappa_gl*Bc_thermo
print(f"Bc(thermo)= {Bc_thermo}, Bc(lower)= {Bc_lower}, Bc(upper)= {Bc_upper}")
#nr of vortices
converted_B_app=B_app*ureg(units_field).to('T')
converted_width_film=side_width_sc_film*ureg(units_length).to('m')
converted_length_film=side_length_sc_film*ureg(units_length).to('m')
print(f"PHI0= {PHI0}, B_app_T: {converted_B_app}, film length: {converted_width_film}")
fluxoid_theoretic = (converted_B_app*converted_width_film*converted_length_film)/PHI0
print(f"Vortices that can enter: {fluxoid_theoretic}")


#solve TDGL
tdgl_options = tdgl.SolverOptions(skip_time=0, solve_time=100,monitor=False, monitor_update_interval=0.5, field_units=units_field, current_units=units_current)
solution_zero_current = tdgl.solve(sc_device, tdgl_options, applied_vector_potential=B_app, disorder_epsilon=track_epsilon)

fig, axes= solution_zero_current.plot_order_parameter(squared=False)
plt.suptitle(r"Order Parameter Plot, $B_{app}$="+f"{B_app}{units_field}\n$\kappa$={kappa_gl} ($\lambda$={lambda_depth}{units_length}, $\\xi$={xi}{units_length})")

london_box=box(width=(side_length_sc_film-lambda_depth))
xi_box=box(width=(side_length_sc_film-xi))
for ax in axes:
    ax.plot(*london_box.T)
    ax.plot(*xi_box.T)



#postprocessing
#Determine how many vortices over area
r_fluxoid_calc_surface = 2*hole_radius
center_fluxoid_calc_surface = (0,0)
fluxoid_calc_surface = circle(radius=r_fluxoid_calc_surface, center=center_fluxoid_calc_surface, points=201)
fluxoid_in_surface = solution_zero_current.polygon_fluxoid(fluxoid_calc_surface, with_units=False)
print(f"Fluxoid over outlined area: \n\t{fluxoid_in_surface} Phi_0 \n\tTotal fluxoid over outlined area: {sum(fluxoid_in_surface):.2f} Phi_0 \n")
#Determine how many vortices through the entire film area, incl hole
scale_sc_film_by = 0.99  #must be lower than 1 because fluxoid calculation area has to be smaller than sc_film
reduced_simulation_surface = sc_film.scale(xfact=scale_sc_film_by, yfact=scale_sc_film_by).points
fluxoid_in_simulation_area = solution_zero_current.polygon_fluxoid(reduced_simulation_surface, with_units=False)
print(f"Fluxoid over entire simulation area: \n\t{fluxoid_in_simulation_area} Phi_0 \n\tTotal fluxoid over entire simulation area: {sum(fluxoid_in_simulation_area):.2f} Phi_0 \n")

'''
#output and visualization of order parameter
fig, axes = solution_zero_current.plot_order_parameter()
plt.suptitle("Order Parameter Plot", fontsize=16)
subtitle = f"B_app = {B_app} {units_field}"
plt.title(subtitle, fontsize=12)

file_name=f"figure_plots/phi_order-B_{B_app}{units_field}_hole{hole_radius}{units_length}.png"
#plt.savefig(file_name)

#output and visualization of current density
fig, ax = solution_zero_current.plot_currents(min_stream_amp=0.075, vmin=0, vmax=10)

file_name=f"figure_plots/K_current-B_{B_app}{units_field}_hole{hole_radius}{units_length}.png"
plt.savefig(file_name)

#visualization of fluxoid calculation areas
for ax in axes:
    ax.plot(*fluxoid_calc_surface.T)
    ax.plot(*reduced_simulation_surface.T)
'''

plt.show()
