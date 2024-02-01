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

RESULTS_FILE= "h5_vortex_init_37_5uA_87tau.h5"
STEP_NR = 9600
#from single_vortex_test.py import

'''
enter_animation_output = "vortex_movement_for_postprod_comparison.mp4"
enter_animation_quantities = ('order_parameter', 'phase', 'supercurrent', 'normal_current')
enter_fps = 30
'''
#create_animation(input_file=RESULTS_FILE, output_file="deleteme.gif", quantities=('order_parameter', 'phase', 'supercurrent', 'normal_current'), fps=15, max_cols=2)

some_track_points= np.array([[ 2.47934647e+02,  1.07269298e+02],
 [ 2.48895035e+02,  1.07854901e+02],
 [ 2.49855423e+02,  1.08440503e+02],
 [ 2.50815811e+02,  1.09026106e+02],
 [ 2.51776199e+02,  1.09611708e+02],
 [ 2.52736587e+02,  1.10197310e+02],
 [ 2.53696975e+02,  1.10782913e+02],
 [ 2.54657363e+02,  1.11368515e+02],
 [ 2.55617751e+02,  1.11954118e+02],
 [ 2.56578139e+02,  1.12539720e+02]
])
#load solution file and underlying device
'''
solution_for_vt_calc = tdgl.Solution.from_hdf5(RESULTS_FILE)
all_geometry_pts= (solution_for_vt_calc.device).points
track_pts = solution_for_vt_calc.disorder_epsilon.track_points
solution_for_vt_calc.load_tdgl_data(solve_step=61)
'''
solution_xi = 44.7214
solution_film_width = 1000
solution_notch_length = 110

imported_solution_dynamics = tdgl.Solution.from_hdf5(RESULTS_FILE) 
actual_simulation_time = imported_solution_dynamics.times #times in tau0
simulation_frame_range = range(36,48) #frames, not actual time. User-defined range
#chosen_solve_step = 294
#imported_solution_dynamics.load_tdgl_data(solve_step=chosen_solve_step)

outputwrite = open("fluxoid_t_output.txt", "w")

vortex_contour_radius = 3*solution_xi
left_point_dx = -solution_film_width/2+solution_xi*3
leftcenter_point_dx= -190
right_point_dx=solution_film_width/2-(solution_notch_length+solution_xi*3)
print(f"Center dx for rightmost point: dx={right_point_dx}, leftcenter dx={leftcenter_point_dx}, for leftmost point: dx={left_point_dx}")
dx_C1_C2 = right_point_dx
dx_C2_C3 = abs(0 - leftcenter_point_dx)
dx_C3_C4 = abs(leftcenter_point_dx - left_point_dx)
dx_C4_edge = vortex_contour_radius

dx_between_contours = np.array([dx_C1_C2, dx_C2_C3, dx_C3_C4, dx_C4_edge]) #right to left
print(f"dx between contours: {dx_between_contours}")

fluxoid_measure_contours = {
    # name: (circle radius, circle center)
    "C1": (vortex_contour_radius, (right_point_dx, 0)), #rightmost
    "C2": (vortex_contour_radius, (0, 0)), #center
    "C3": (vortex_contour_radius, (leftcenter_point_dx, 0)), #left of center
    "C4": (vortex_contour_radius, (left_point_dx, 0)) #leftmost
}


print(f"Writing to {outputwrite}...")
for t_step in simulation_frame_range:
    imported_solution_dynamics.load_tdgl_data(solve_step=t_step)
    print(f"step={t_step}, t={actual_simulation_time[t_step]}, ",end=" ")
    for name, (radius, center) in fluxoid_measure_contours.items():
        polygon = circle(radius, center=center, points=201)
        fluxoid_in_contour_at_step = imported_solution_dynamics.polygon_fluxoid(polygon, with_units=False)
        print(f"{name} fluxoid= {sum(fluxoid_in_contour_at_step)}, ",end=" ")
        #print(f"{name}:\n\t{fluxoid_in_contour_at_step} Phi_0\n\tTotal fluxoid: {sum(fluxoid_in_contour_at_step)} Phi_0\n")
    print(" ")
    #outputwrite.write(f"{t_step}, {actual_simulation_time[t_step]}, {sum(fluxoid_at_step)}, {sum(fluxoid_left)}\n") #, {sum(fluxoid_right)}\n")
print("Done!")
outputwrite.close()

fig, (ax1, ax2)= imported_solution_dynamics.plot_order_parameter()

#plt.suptitle(f"Order Parameter Plot")
for name, (radius, center) in fluxoid_measure_contours.items():
    polygon = circle(radius, center=center, points=201)
    ax1.plot(*polygon.T, label=f"{name} {center}")

    '''
    for ax in axes:
        ax.plot(*polygon.T, label=f"{name} {center}") #label that will be passed to plt.legend()
        ax.set_axis_off()
    '''



ax1.legend(framealpha=1)
plt.show()



'''
order_parameter_extracted = solution_for_vt_calc.interp_order_parameter(positions=track_pts)
print(f"solve step: {solution_for_vt_calc.solve_step}")
print(f"{order_parameter_extracted}")
print(f"{np.angle(order_parameter_extracted)}") #phases only
'''


#imported_dynamics= imported_solution.dynamics(dt=1600, mu=None, theta=1, screening_iterations=None)
