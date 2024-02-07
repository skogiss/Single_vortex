import h5py
import cmath
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tdgl.visualization import create_animation
from tdgl.geometry import box, circle, ellipse
import numpy as np
import tdgl
from tdgl.solution import data
from pint import UnitRegistry
ureg=UnitRegistry()

#units to be used throughout the simulation
units_length="nm"
units_field="mT"
units_current="uA"
PHI0 = 2.06783e-15*ureg.T*ureg.m**2


#File to be post-processed and some parameters of that simulation
RESULTS_FILE= "h5_vvelocity_J_35uA_save25.h5"

solution_xi = 44.7214
solution_film_width = 1000
solution_notch_length = 110

MAKE_ANIMATIONS_FROM_INPUT = False
animation_file_name = None
animation_fps = 15

MAKE_VORTEX_MEASURE_CONTOURS = True
vortex_contour_radius = 0.5*solution_xi
DISPLAY_ONE_SOLVE_STEP = True
load_frame_nr = 14

CALC_VORTEX_DYNAMICS_IN_CONTOURS = True
output_write_file = "fluxoid_t_output.txt"
range_for_flux_calc = range(7,15) #range of frames in the solution to calc over

#Create animation from input file
if MAKE_ANIMATIONS_FROM_INPUT==True:
    create_animation(input_file=RESULTS_FILE, output_file=animation_file_name, quantities=('order_parameter', 'phase', 'supercurrent', 'normal_current'), fps=animation_fps, max_cols=2)

#not used - extracting track coordinates of the solution
'''
solution_for_vt_calc = tdgl.Solution.from_hdf5(RESULTS_FILE)
all_geometry_pts= (solution_for_vt_calc.device).points
track_pts = solution_for_vt_calc.disorder_epsilon.track_points
solution_for_vt_calc.load_tdgl_data(solve_step=61)

order_parameter_extracted = solution_for_vt_calc.interp_order_parameter(positions=track_pts)
print(f"solve step: {solution_for_vt_calc.solve_step}")
print(f"{order_parameter_extracted}")
print(f"{np.angle(order_parameter_extracted)}") #phases only

imported_dynamics= imported_solution.dynamics(dt=1600, mu=None, theta=1, screening_iterations=None)
'''

#Load TDGL solution from input file and its actual times
imported_solution_dynamics = tdgl.Solution.from_hdf5(RESULTS_FILE) 
actual_simulation_time = imported_solution_dynamics.times #times in tau0



#Function for contour creation
def setup_fluxoid_measure_stations():
    
    left_point_dx = -solution_film_width/2+solution_xi*3
    leftcenter_point_dx= -190
    right_point_dx=solution_film_width/2-(solution_notch_length+solution_xi*3)
    print(f"Center dx for rightmost point: dx={right_point_dx}, leftcenter dx={leftcenter_point_dx}, for leftmost point: dx={left_point_dx}")
    dx_C1_C2 = right_point_dx
    dx_C2_C3 = abs(0 - leftcenter_point_dx)
    dx_C3_C4 = abs(leftcenter_point_dx - left_point_dx)
    dx_C4_edge = vortex_contour_radius

    #dx_between_contours = np.array([dx_C1_C2, dx_C2_C3, dx_C3_C4, dx_C4_edge]) #right to left
    #print(f"dx between contours: {dx_between_contours}")

    fluxoid_measure_stations = {
        #name: (circle radius, circle center)
        "C1": (vortex_contour_radius, (right_point_dx, 0)), #rightmost
        "C2": (vortex_contour_radius, (0, 0)), #center
        "C3": (vortex_contour_radius, (leftcenter_point_dx, 0)), #left of center
        "C4": (vortex_contour_radius, (left_point_dx, 0)) #leftmost
    }
    return fluxoid_measure_stations

if MAKE_VORTEX_MEASURE_CONTOURS==True:
    fluxoid_measure_contours=setup_fluxoid_measure_stations()

if DISPLAY_ONE_SOLVE_STEP == True:
    imported_solution_dynamics.load_tdgl_data(solve_step=load_frame_nr)
    fig, (ax1, ax2)= imported_solution_dynamics.plot_order_parameter()

    if MAKE_VORTEX_MEASURE_CONTOURS==True:
        for name, (radius, center) in fluxoid_measure_contours.items():
            polygon = circle(radius, center=center, points=201)
            ax1.plot(*polygon.T, label=f"{name} {center}")
            ax2.plot(*polygon.T, label=f"{name} {center}")
        ax1.legend(framealpha=1)

    plt.show()

if CALC_VORTEX_DYNAMICS_IN_CONTOURS==True and MAKE_VORTEX_MEASURE_CONTOURS==True:
#simulation_frame_range = range(1,1) #frames, not actual time. User-defined range
    outputwrite = open(output_write_file, "w")
    print(f"Writing from {RESULTS_FILE} to {outputwrite}...")
    outputwrite.write("frame, time_tau0, time_ps, fluxoid_C1, fluxoid_C2, fluxoid_C3, fluxoid_C4 \n")
    for t_step in range_for_flux_calc:
        imported_solution_dynamics.load_tdgl_data(solve_step=t_step)
        outputwrite.write(f"{t_step}, {actual_simulation_time[t_step]}, {0.26942298611961255*actual_simulation_time[t_step]}")
        for name, (radius, center) in fluxoid_measure_contours.items():
            polygon = circle(radius, center=center, points=201)
            fluxoid_in_contour_at_step = imported_solution_dynamics.polygon_fluxoid(polygon, with_units=False)
            outputwrite.write(f", {sum(fluxoid_in_contour_at_step)}")
            #print(f"{name}:\n\t{fluxoid_in_contour_at_step} Phi_0\n\tTotal fluxoid: {sum(fluxoid_in_contour_at_step)} Phi_0\n")
        outputwrite.write(" \n")
        #outputwrite.write(f"{t_step}, {actual_simulation_time[t_step]}, {sum(fluxoid_at_step)}, {sum(fluxoid_left)}\n") #, {sum(fluxoid_right)}\n")
    print("Done!")
    outputwrite.close()



