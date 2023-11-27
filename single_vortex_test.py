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
xi_coherence=10 #10
lambda_london=40 #80
d=1
gamma_scattering_gap = 1#1
kappa_gl= lambda_london/xi_coherence
lambda_eff_screening = lambda_london**2/xi_coherence
sc_layer = tdgl.Layer(coherence_length=xi_coherence, london_lambda=lambda_london, thickness=d, gamma=gamma_scattering_gap)
print(f"kappa= {kappa_gl}, screening length= {lambda_eff_screening}{units_length}, lambda= {lambda_london}{units_length}, coherence length={xi_coherence}{units_length}")

#parameters of environment
B_applied_field=145 #60
print(f"Applied magnetic field: {B_applied_field}{units_field}")

#outer geometry of the SC film
sc_film_length = 400 #1000
sc_film_width = sc_film_length

#notch geometry
notch_ellipse_big= 50 #length of notch, 60
notch_ellipse_small= 50 #half of height of notch, 20
notch_length= 60
notch_max_heigth= 40
notch_placement_y = 0*sc_film_length

def create_notch(notch_length, notch_max_heigth, notch_placement_y):
    hole_notch=tdgl.Polygon("notch test", points=ellipse(a=notch_length,b=notch_max_heigth/2))
    hole_notch=hole_notch.translate(dx=sc_film_width/2, dy=notch_placement_y)
    return hole_notch

notch_in_film1=create_notch(notch_length, notch_max_heigth, notch_placement_y)

#create a superconducting film with a notch
sc_film= tdgl.Polygon("film", points=box(width=sc_film_length)).difference(notch_in_film1).resample(400).buffer(0)

#geometry of circular hole in the middle
hole_radius = 2*xi_coherence
hole_center = (-50,-20)

def create_hole_round(hole_center, hole_radius):
    hole_round = tdgl.Polygon("round hole", points=circle(radius=hole_radius, center=hole_center))
    return hole_round

hole_round= create_hole_round(hole_center, hole_radius)

#track geometry
def create_track(track_width, notch_in_film, center_of_hole):
    #hole_notch=notch_in_film1.difference(tdgl.Polygon(points=box(width=notch_ellipse_big,height=2*notch_ellipse_small)).translate(dx=notch_ellipse_big/2))
    tmp_coord_notch = notch_in_film.bbox
    notch_endpoint = (tmp_coord_notch[0][0], (tmp_coord_notch[0][1]+(tmp_coord_notch[1][1]-tmp_coord_notch[0][1])/2))
    track_length = np.sqrt((notch_endpoint[0]-center_of_hole[0])**2 + (notch_endpoint[1]-center_of_hole[1])**2)
    adjacent_side_length = center_of_hole[1]-notch_endpoint[1]
    opposite_side_length = center_of_hole[0]-notch_endpoint[0]
    if adjacent_side_length==0 or opposite_side_length==0:
        track_angle=90
        track = tdgl.Polygon("track", points=box(width=track_width, height=track_length, center=center_of_hole, angle=track_angle))
        track_notch_gap_x = notch_endpoint[0] - track.bbox[1][0]
        track_notch_gap_y = 0
    else:
        tan_angle=opposite_side_length/adjacent_side_length
        #track_center=center_of_hole
        if tan_angle<0:
            track_angle = np.arccos(adjacent_side_length/track_length)
            track_angle= np.degrees(track_angle)
            track = tdgl.Polygon("track", points=box(width=track_width, height=track_length, center=center_of_hole, angle=track_angle))#.rotate(degrees=track_angle,origin=track_center)
            track_notch_gap_y = notch_endpoint[1] - track.bbox[0][1]
            track_notch_gap_x = notch_endpoint[0] - track.bbox[1][0]
        elif tan_angle>0:
            track_angle = np.arccos(abs(adjacent_side_length/track_length))
            track_angle= 180 - np.degrees(track_angle)
            track = tdgl.Polygon("track", points=box(width=track_width, height=track_length, center=center_of_hole, angle=track_angle))#.rotate(degrees=track_angle,origin=track_center)
            track_notch_gap_y = notch_endpoint[1] - track.bbox[1][1] #same bbox coordinates rotate too
            track_notch_gap_x = notch_endpoint[0] - track.bbox[1][0]

    track=track.translate(dx=track_notch_gap_x,dy=track_notch_gap_y).difference(hole_round).resample(800)
    print(f"track box: {track.bbox}")
    print(f"notch endpoint: {notch_endpoint}, track length: {track_length}, track angle: {track_angle}")
    return track

track_width = 0.3*xi_coherence
track=create_track(track_width, notch_in_film1, hole_center)

#track dependent disorder_epsilon
def track_epsilon(r):
    if track.contains_points(r)==True or track.on_boundary(r)==True:
        epsilon=0.5
    else:
        epsilon=1.0
    return epsilon


#output about simulation geometry
print(f"Dimensions of the SC film: {sc_film_length}x{sc_film_length}{units_length}")


#put SC material (tdgl.Layer) and geometry (tdgl.Polygon) together into a complete device (tdgl.Device)
sc_device= tdgl.Device("square with hole", layer=sc_layer, film=sc_film, holes=[hole_round], length_units=units_length)
#discretize the device into a mesh to be solved over
mesh_edge_factor = 1.0 #should be small comped to xi_coherence
sc_device.make_mesh(max_edge_length=mesh_edge_factor*xi_coherence, smooth=1) #smooth 1 or 100 had very little effect on how the mesh looks
fig, ax = sc_device.plot(mesh=True)
plt.show()

#theoretical calculations
PHI0 = 2.06783e-15*ureg.T*ureg.m**2
#critical fields
B_critical_thermo=(PHI0/(2*np.sqrt(2)*np.pi*lambda_london*ureg(units_length).to('m')*xi_coherence*ureg(units_length).to('m'))).to(units_field)
B_critical_lower=B_critical_thermo*np.log(kappa_gl)/(np.sqrt(2)*kappa_gl)
B_critical_upper=np.sqrt(2)*kappa_gl*B_critical_thermo
print(f"Bc(thermo)= {B_critical_thermo}, Bc(lower)= {B_critical_lower}, Bc(upper)= {B_critical_upper}")
#nr of vortices
converted_B_applied_field=B_applied_field*ureg(units_field).to('T')
converted_width_film=sc_film_width*ureg(units_length).to('m')
converted_length_film=sc_film_length*ureg(units_length).to('m')
print(f"PHI0= {PHI0}, B_applied_field_T: {converted_B_applied_field}, film length: {converted_width_film}")
fluxoid_theoretic = (converted_B_applied_field*converted_width_film*converted_length_film)/PHI0
print(f"Vortices that can enter: {fluxoid_theoretic}")


#solve TDGL
tdgl_options = tdgl.SolverOptions(skip_time=0, solve_time=20,monitor=True, monitor_update_interval=0.5, field_units=units_field, current_units=units_current)
solution_zero_current = tdgl.solve(sc_device, tdgl_options, applied_vector_potential=B_applied_field, disorder_epsilon=track_epsilon)

fig, axes= solution_zero_current.plot_order_parameter(squared=False)
plt.suptitle(r"Order Parameter Plot, $B_{app}$="+f"{B_applied_field}{units_field}\n$\kappa$={kappa_gl} ($\lambda$={lambda_london}{units_length}, $\\xi_coherence$={xi_coherence}{units_length})")

london_box=box(width=(sc_film_length-lambda_london))
xi_coherence_box=box(width=(sc_film_length-xi_coherence))
for ax in axes:
    ax.plot(*london_box.T)
    ax.plot(*xi_coherence_box.T)



#postprocessing
#Determine how many vortices over area
'''
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

'''
#output and visualization of order parameter
fig, axes = solution_zero_current.plot_order_parameter()
plt.suptitle("Order Parameter Plot", fontsize=16)
subtitle = f"B_applied_field = {B_applied_field} {units_field}"
plt.title(subtitle, fontsize=12)

file_name=f"figure_plots/phi_order-B_{B_applied_field}{units_field}_hole{hole_radius}{units_length}.png"
#plt.savefig(file_name)

#output and visualization of current density
fig, ax = solution_zero_current.plot_currents(min_stream_amp=0.075, vmin=0, vmax=10)

file_name=f"figure_plots/K_current-B_{B_applied_field}{units_field}_hole{hole_radius}{units_length}.png"
plt.savefig(file_name)

#visualization of fluxoid calculation areas
for ax in axes:
    ax.plot(*fluxoid_calc_surface.T)
    ax.plot(*reduced_simulation_surface.T)
'''

plt.show()
