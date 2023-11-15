import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import tdgl
from tdgl.geometry import box, circle

#setting up the superconductor material
units_length="nm"

xi=10
depth_lambda=80
d=1
gamma_phonon = 1
gl_kappa= depth_lambda/xi
T_c= 18.2 #K
B_c1= 35 #35mT
B_c= 440 #mT
B_c2= 23e+1000 #mT


print(f"kappa= {gl_kappa}")

sc_layer = tdgl.Layer(coherence_length=xi, london_lambda=depth_lambda, thickness=d, gamma=gamma_phonon)

#setting up a square with circle hole(s)
length_side = 50
sc_film= tdgl.Polygon("film", points=box(length_side, length_side)).resample(401).buffer(0)
hole_round= tdgl.Polygon("round hole", points=circle(radius=0.5*xi))    #.translate(dy=length_side/4,dx=length_side/4)

sc_device= tdgl.Device("square with hole", layer=sc_layer, film=sc_film, holes=[hole_round], length_units=units_length)

#create a meshs
sc_device.make_mesh(max_edge_length=0.5*xi, smooth=100) #smooth 1 or 100 had very little effect on how the mesh looks
#fig, ax = sc_device.plot(mesh=True)

#solve TDGL for this geometry
B_app=B_c1*80
tdgl_options= tdgl.SolverOptions(solve_time=100,field_units="mT", current_units="uA")
solution_zero_current= tdgl.solve(sc_device, tdgl_options, applied_vector_potential=B_app)

fig, axes = solution_zero_current.plot_order_parameter(figsize=(5.5, 4))
fig, ax = solution_zero_current.plot_currents(min_stream_amp=0.075, vmin=0, vmax=10)


plt.show()
