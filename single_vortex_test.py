import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import h5py

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import tdgl
from tdgl.geometry import box, circle, ellipse
from tdgl.visualization import create_animation
from tdgl.solution import data
from pint import UnitRegistry
ureg=UnitRegistry()

#np.set_printoptions(threshold=sys.maxsize)
#units to be used throughout the simulation
units_length="nm"
units_field="mT"
units_current="uA"
PHI0 = 2.06783e-15*ureg.T*ureg.m**2

##############ENTER USER-DEFINED SIMULATION AND MATERIAL PARAMETERS HERE###################
ADJUST_TO_NONZERO_T = True
enter_xi_coherence = 10
enter_lambda_london = 80
enter_sigma_conductivity = 6.7e-3 #S/nm
enter_gamma_scattering = 1

enter_B_applied_field = 0

enter_film_thickness = 1
enter_film_length = 2000
enter_film_width = 1000

make_notch_on_vertical = True
enter_notch_length= 110
enter_notch_max_heigth= 110
enter_notch_orienation = "Right"
enter_notch_placement_y = 0*enter_film_length/2

make_hole = True
AUTO_HOLE_RADIUS = True
enter_hole_radius_factor = 2
enter_hole_radius = 60
enter_hole_center = (0,0)

make_track = False
AUTO_TRACK_WIDTH = False
enter_track_width_factor = 2.5
enter_track_width = 170
enter_track_epsilon = 0.4

MAKE_TERMINALS = True
enter_source_width = enter_film_width
enter_source_length = enter_film_width/100
enter_initialization_time = 0
CURRENT_MODE = "constant" #"constant" or "pulse"
enter_dc_source = 51.2421875
enter_dc_drain = -enter_dc_source
#enter_dc_background_factor = 0
enter_pulse_on_length = 27
enter_pulse_on_zero_length = 30
enter_pulse_off_length = 12
enter_pulse_off_zero_length = 15
enter_voltmeter_points = [(0, enter_film_length/2.5), (0, -enter_film_length/2.5)]

CONTINUE_SOLVING = False
enter_filename_previous_solution = None#"h5_on_wtrack170_y06.h5"
AUTO_MESH_EDGE = True
enter_max_edge_factor = 0.6
enter_max_edge_length = 15  #mesh element size, should be small compared to xi_coherence
enter_skip_time = 0
enter_solve_time = 2000
enter_save_every = 400
do_monitor = True
show_london_box=False
show_xi_coherence_box=False

MAKE_ANIMATIONS = False
enter_write_solution_results = None#"h5_hole2xi_AC_vortex_coming_back.h5"
enter_animation_input = enter_write_solution_results
enter_animation_output = "trap_bad_hole2xi_AC_vortex_coming_back.mp4"
enter_animation_quantities = ('order_parameter', 'phase', 'supercurrent', 'normal_current')
enter_fps = 15

print("---------------------------")
##############MAIN SIMULATION LOGIC STARTS BEYOND THIS POINT##############################


#parameters of the superconducting material and creation of the SC layer (tdgl.Layer)
xi_coherence= enter_xi_coherence #10
lambda_london= enter_lambda_london #80
kappa_gl= lambda_london/xi_coherence
sigma_conductivity=enter_sigma_conductivity
print(f"Material parameters at T=0: lambda={lambda_london}{units_length}, xi={xi_coherence}{units_length} (kappa= {kappa_gl}).")
B_critical_thermo=(PHI0/(2*np.sqrt(2)*np.pi*lambda_london*ureg(units_length).to('m')*xi_coherence*ureg(units_length).to('m'))).to(units_field)
print(f"Calculated with these values: B_c(thermo)= {B_critical_thermo:.4f}")

B_critical_lower=B_critical_thermo*np.log(kappa_gl)/(np.sqrt(2)*kappa_gl)
B_critical_upper=np.sqrt(2)*kappa_gl*B_critical_thermo
print(f"Calc from Bc: Bc(lower)= {B_critical_lower:.4f}, Bc(upper)= {B_critical_upper:.4f}")


def recalc_characteristic_lengths(xi_coherence_0, lambda_london_0, B_critical_thermo_0):
    T_rel=0.95  #T/T_c
    slope_lambda = 1/(2*np.sqrt(1-T_rel))
    lambda_london_Tc = lambda_london_0 * slope_lambda
    slope_xi = 1/(np.sqrt(1-T_rel))
    xi_coherence_Tc = xi_coherence_0*slope_xi
    slope_Bc = 2*(1-T_rel)
    B_critical_Tc = B_critical_thermo_0*slope_Bc
    return lambda_london_Tc, xi_coherence_Tc, B_critical_Tc

if ADJUST_TO_NONZERO_T == True:
    lambda_london, xi_coherence, B_critical_thermo = recalc_characteristic_lengths(xi_coherence, lambda_london, B_critical_thermo)
    kappa_gl= lambda_london/xi_coherence
    print(f"Temperature corrections applied, new values: lambda= {lambda_london:.4f}, xi={xi_coherence:.4f} (kappa= {kappa_gl}), Bc(thermo)={B_critical_thermo:.4f}")
    B_critical_lower=B_critical_thermo*np.log(kappa_gl)/(np.sqrt(2)*kappa_gl)
    B_critical_upper=np.sqrt(2)*kappa_gl*B_critical_thermo
    print(f"Recalculated critical fields: Bc(lower)= {B_critical_lower:.4f}, Bc(upper)= {B_critical_upper:.4f} \n---------------------------")
else:
    print("Not applying temperature corrections \n---------------------------")


film_thickness = enter_film_thickness
gamma_scattering_gap = enter_gamma_scattering#1
lambda_eff_screening = lambda_london**2/xi_coherence
sc_layer = tdgl.Layer(coherence_length=xi_coherence, london_lambda=lambda_london, thickness=film_thickness, gamma=gamma_scattering_gap, conductivity=sigma_conductivity)


#outer geometry of the SC film including domain size, notch, hole, track (tdgl.Polygon)
sc_film_length = enter_film_length #1000
sc_film_width = enter_film_width

sc_film= tdgl.Polygon("film", points=box(width=sc_film_width, height=sc_film_length))
print(f"Superconducting layer created, dimensions: {sc_film_length}x{sc_film_length}x{film_thickness} {units_length}")

#notch geometry
notch_length= enter_notch_length
notch_max_heigth= enter_notch_max_heigth
notch_placement_y = enter_notch_placement_y

notch_orientation = enter_notch_orienation
def create_notch(notch_length, notch_max_heigth, notch_placement_y):
    hole_notch=tdgl.Polygon("notch test", points=ellipse(a=notch_length,b=notch_max_heigth/2))
    notch_placement_x = sc_film_width/2
    if notch_orientation == "Left":
        notch_placement_x = (-1)*notch_placement_x
    hole_notch=hole_notch.translate(dx=notch_placement_x, dy=notch_placement_y)
    print(f"Making notch in y={notch_placement_y} ({notch_orientation} edge)")
    return hole_notch


#create a notch and  add it to film
if make_notch_on_vertical == True:
    notch_in_film1=create_notch(notch_length, notch_max_heigth, notch_placement_y)
    sc_film= sc_film.difference(notch_in_film1).resample(400).buffer(0)
else:
    print("No notch in SC sheet")

#geometry of circular hole
hole_center = enter_hole_center
if AUTO_HOLE_RADIUS == True:
    hole_radius = enter_hole_radius_factor * xi_coherence
else:
    hole_radius = enter_hole_radius

def create_hole_round(hole_center, hole_radius):
    hole_round = tdgl.Polygon("round hole", points=circle(radius=hole_radius, center=hole_center))
    print(f"Hole created at {hole_center}, r= {hole_radius}")
    return hole_round

if make_hole== True:
    hole_round= create_hole_round(hole_center, hole_radius)
    holes_in_film= [hole_round]
    print("Hole added to SC sheet")
else:
    holes_in_film=[]
    print("No holes added to SC sheet")

#track geometry
def create_track(track_width, notch_in_film, center_of_hole):
    #hole_notch=notch_in_film1.difference(tdgl.Polygon(points=box(width=notch_ellipse_big,height=2*notch_ellipse_small)).translate(dx=notch_ellipse_big/2))
    tmp_coord_notch = notch_in_film.bbox
    #notch_endpoint = (tmp_coord_notch[0][0], (tmp_coord_notch[0][1]+(tmp_coord_notch[1][1]-tmp_coord_notch[0][1])/2)) #original
    if notch_placement_y<0:
    	notch_endpoint = ((tmp_coord_notch[0][0]+(tmp_coord_notch[1][0]-tmp_coord_notch[0][0])/4), (tmp_coord_notch[0][1])) #works for notch below hole
    elif notch_placement_y==0:
        notch_endpoint = (tmp_coord_notch[0][0], (tmp_coord_notch[0][1]+(tmp_coord_notch[1][1]-tmp_coord_notch[0][1])/2)) #original
    elif notch_placement_y>0:
        notch_endpoint = ((tmp_coord_notch[0][0]+(tmp_coord_notch[1][0]-tmp_coord_notch[0][0])/4), (tmp_coord_notch[1][1])) #works for notch above hole

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

    track=track.translate(dx=track_notch_gap_x+50,dy=track_notch_gap_y-30).resample(800)#.difference(hole_round).resample(800) #OBS! Re-add difference if needed
    print(f"Track between hole center and {notch_endpoint}, length: {track_length}, angle: {track_angle}")
    return track

if make_track==True:
    if not (make_hole==True and make_notch_on_vertical==True):
        print("WARNING! Can't create track because hole, notch or both are missing. Constant epsilon_disorder will be set.")
        def track_epsilon(r):
            epsilon=1
            return epsilon

    elif make_hole==True:
        if AUTO_TRACK_WIDTH == True:
            track_width=enter_track_width_factor*xi_coherence
        else:
            track_width = enter_track_width
        track=create_track(track_width, notch_in_film1, hole_center)
        #track dependent disorder_epsilon
        #track_epsilon=enter_track_epsilon
        print(f"Track width={track_width}, epsilon in track= {enter_track_epsilon}")

        def track_epsilon(r):
            track_epsilon.track_points = track.points #to be able to access track coordinates from saved solution later
            if track.contains_points(r)==True or track.on_boundary(r)==True:
                epsilon=enter_track_epsilon
            else:
                epsilon=1.0
            return epsilon
else:
    print("No track on SC sheet")
    def track_epsilon(r):
        epsilon=1
        return epsilon

print("---------------------------")

#parameters of environment
B_applied_field= enter_B_applied_field #60
print(f"Applied magnetic field: {B_applied_field}{units_field}")

#nr of vortices that should enter given this field
if B_applied_field!=0:
    converted_B_applied_field=B_applied_field*ureg(units_field).to('T')
    converted_width_film=sc_film_width*ureg(units_length).to('m')
    converted_length_film=sc_film_length*ureg(units_length).to('m')
    #print(f"PHI0= {PHI0}, B_applied_field_T: {converted_B_applied_field}, film length: {converted_width_film}")
    fluxoid_theoretic = (converted_B_applied_field*converted_width_film*converted_length_film)/PHI0
    #print(f"Vortices that can enter: {fluxoid_theoretic}")

#External current terminals
if MAKE_TERMINALS== True:
    source_width=enter_source_width
    source_length=enter_source_length
    terminal_source = tdgl.Polygon("source", points=box(source_width, source_length)).translate(dy=1*sc_film_length/2)
    terminal_drain = terminal_source.scale(yfact=-1).set_name("drain")
    ncurrent_terminals= [terminal_source, terminal_drain]
    print(f"Two {source_width}x{source_length} current terminals added")
    #supplied_ncurrent = dict(source=enter_dc_source, drain=enter_dc_drain)
    dc_source=enter_dc_source
    dc_drain=enter_dc_drain
    pulse_on_length = enter_pulse_on_length
    pulse_on_zero_length = enter_pulse_on_zero_length
    pulse_off_length = enter_pulse_off_length
    pulse_off_zero_length = enter_pulse_off_zero_length
    #dc_background_factor = enter_dc_background_factor
    print(f"Supplied source current: {dc_source}{units_current} (drain current = {dc_drain}{units_current}). DC pulse length: {pulse_on_length} (dimensionless)")

    initialization_time = enter_initialization_time

    
    def dc_pulse(time):
        #Constant curret mode will keep supplying constant current as long as simulation goes on
        if CURRENT_MODE=="constant":
            supplied_ncurrent = dict(source=dc_source, drain=dc_drain)
        
        #Pulse current mode will supply current only for a specified time and then set current to zero
        elif CURRENT_MODE=="pulse":
            if time<=initialization_time:
                supplied_ncurrent= dict(source=0, drain=0) 

            elif time>initialization_time and time<=(pulse_on_length+initialization_time):
                supplied_ncurrent = dict(source=dc_source, drain=dc_drain)

            elif time>(pulse_on_length+initialization_time) and time<=(pulse_on_length+initialization_time+pulse_on_zero_length):
                supplied_ncurrent = dict(source=0, drain=0)

            elif time>(pulse_on_length+initialization_time+pulse_on_zero_length) and time<=(pulse_on_length+initialization_time+pulse_on_zero_length+pulse_off_length):
                supplied_ncurrent = dict(source=-1*dc_source, drain=-1*dc_drain)

            elif time>(pulse_on_length+initialization_time+pulse_on_zero_length+pulse_off_length) and time<=(pulse_on_length+initialization_time+pulse_on_zero_length+pulse_off_length+pulse_off_zero_length):
                supplied_ncurrent = dict(source=0, drain=0)
            else:
                supplied_ncurrent = dict(source=0, drain=0)

        else:
            supplied_ncurrent = dict(source=0, drain=0)
            
        return supplied_ncurrent
else:
    ncurrent_terminals = []
    supplied_ncurrent = None
    print("No current terminals added")

#put SC material (tdgl.Layer) and geometry (tdgl.Polygon) together into a complete device (tdgl.Device)
sc_device= tdgl.Device("square with hole", layer=sc_layer, film=sc_film, holes=holes_in_film, terminals=ncurrent_terminals, probe_points=enter_voltmeter_points, length_units=units_length)
tau0_time = sc_device.tau0()
print(f"tau0 factor for time in seconds: {tau0_time}")


#discretize the device into a mesh to be solved over
if AUTO_MESH_EDGE == True:
    mesh_edge_length = enter_max_edge_factor * xi_coherence
else:
    mesh_edge_length = enter_max_edge_length
print(f"Max length of mesh edge set to {mesh_edge_length}")
print("---------------------------")
sc_device.make_mesh(max_edge_length=mesh_edge_length, smooth=1) #smooth 1 or 100 had very little effect on how the mesh looks

#plot the device before solving
#fig, ax = sc_device.plot(mesh=True, legend=True)
#fig, ax= sc_device.draw()
#plt.show()

#solve TDGL
tdgl_options = tdgl.SolverOptions(skip_time=enter_skip_time, solve_time=enter_solve_time, monitor=do_monitor, monitor_update_interval=0.5, save_every=enter_save_every, output_file=enter_write_solution_results, field_units=units_field, current_units=units_current)

if CONTINUE_SOLVING== True:
    solution_previous= tdgl.Solution.from_hdf5(enter_filename_previous_solution)
else:
    solution_previous= None

solution_zero_current = tdgl.solve(sc_device, tdgl_options, applied_vector_potential=B_applied_field, disorder_epsilon=track_epsilon, terminal_currents=dc_pulse, seed_solution=solution_previous)

fig, axes= solution_zero_current.plot_order_parameter(squared=False)
plt.suptitle(f"Order Parameter Plot after T={enter_solve_time}, "+ r"$B_{app}$="+f"{B_applied_field}{units_field}\n$\kappa$={kappa_gl} ($\lambda$={lambda_london:.4f}{units_length}, $\\xi$={xi_coherence:.4f}{units_length})")

#visualization of fluxoid calculation areas
r_fluxoid_calc_surface = 1.2*hole_radius
center_fluxoid_calc_surface = hole_center
fluxoid_calc_surface = circle(radius=r_fluxoid_calc_surface, center=center_fluxoid_calc_surface, points=201)
fluxoid_in_surface = solution_zero_current.polygon_fluxoid(fluxoid_calc_surface, with_units=False) #fluxoid in hole
print(f"Fluxoid in trap area: \n\t{fluxoid_in_surface} Phi_0 \n\tTotal fluxoid in trap area: {sum(fluxoid_in_surface):.2f} Phi_0 \n")

fluxoid_calc_point_track= circle(radius=r_fluxoid_calc_surface, center=(250,-330))
fluxoid_in_track_point = solution_zero_current.polygon_fluxoid(fluxoid_calc_point_track, with_units=False) #fluxoid somewhere else
print(f"Fluxoid over other point: \n\t{fluxoid_in_track_point} Phi_0 \n\tTotal fluxoid over outlined other point area: {sum(fluxoid_in_track_point):.2f} Phi_0 \n")

for ax in axes:
    ax.plot(*fluxoid_calc_surface.T)
    ax.plot(*fluxoid_calc_point_track.T)

if show_london_box==True:
    london_box=box(width=(sc_film_length-lambda_london))
    for ax in axes:
        ax.plot(*london_box.T)
if show_xi_coherence_box==True:
    xi_coherence_box=box(width=(sc_film_length-xi_coherence))
    for ax in axes:
        ax.plot(*xi_coherence_box.T)
        
fig, ax = solution_zero_current.plot_currents(min_stream_amp=0.075, vmin=0, vmax=10)

plt.show()

if MAKE_ANIMATIONS==True:
    create_animation(input_file=enter_animation_input, output_file=enter_animation_output, quantities=enter_animation_quantities, fps=enter_fps, max_cols=2)

'''
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

#visualization of fluxoid calculation areas
for ax in axes:
    ax.plot(*fluxoid_calc_surface.T)
    #ax.plot(*reduced_simulation_surface.T)

file_name=f"figure_plots/phi_order-B_{B_applied_field}{units_field}_hole{hole_radius}{units_length}.png"
#plt.savefig(file_name)

#output and visualization of current density
fig, ax = solution_zero_current.plot_currents(min_stream_amp=0.075, vmin=0, vmax=10)


file_name=f"figure_plots/K_current-B_{B_applied_field}{units_field}_hole{hole_radius}{units_length}.png"
plt.savefig(file_name)
'''
