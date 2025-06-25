from mwave import symbolic as ifr
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

# Set up symbols for symbolic computation
m, c, hbar, k, omega_r, delta, n, omega_m, T, Tp, t_traj = sp.symbols('m c hbar k omega_r delta n omega_m T Tp t_traj', real=True)
k_eff = 2*k

# Register constants
ifr.set_constants(m=m, c=c, hbar=hbar, t_traj=t_traj)

# Define unitary operators needed for an SCI
bs1 = ifr.Beamsplitter(n, 0, delta, k_eff)
bs2 = ifr.Beamsplitter(0, -n, delta - omega_m, k_eff)
bs2u = ifr.Beamsplitter(n, 2*n, delta + omega_m, k_eff)
fe1 = ifr.FreeEv(T)
fe2 = ifr.FreeEv(Tp)

# Create interferometer object, perform SCI sequence
ii = ifr.Interferometer()
ii.apply(bs1)
ii.apply(fe1)
ii.apply(bs1)
ii.apply(fe2)
ii.apply(bs2)
ii.apply(bs2u)
ii.apply(fe1)
ii.apply(bs2)
ii.apply(bs2u)
# Note that unitary operators can be applied using the matrix multiplication operator, i.e. "fe1 @ (bs1 @ ii)"

# Create a graphviz object of the tree
d_test = ii.generate_graph()

# Display the graph
d_test.view()

# Determine the intefering nodes, check that we have four interferences
inodes = ii.interfere()
print(f'Found {len(inodes)} interferences, which is {r"expected" if len(inodes) == 4 else r"not expected"}')

# Plot the trajectories (we only plot half of the nodes, as otherwise we will get overlapping trajectories)
subs = {m: 1, c: 1, hbar: 1, k: 1, omega_r: 1, delta: 4*3, n: 3, omega_m: 8*3, T: 5, Tp: 2}
tfinal = float(inodes[0][0].t.evalf(subs=subs))
t = np.linspace(0, tfinal, 1000)
plt.figure()
for node_pair in [inodes[0], inodes[2]]:
    fnc_traj = node_pair[0].get_trajectory(subs=subs)
    plt.plot(t, fnc_traj(t))
    fnc_traj = node_pair[1].get_trajectory(subs=subs)
    plt.plot(t, fnc_traj(t))
plt.show()

# Compute the differential phase
phi_A, phi_B, phi_C, phi_D = ii.phases()
print(sp.simplify(phi_A - phi_C).subs(k, sp.sqrt(2*m*omega_r/hbar)))

# Print out the total population, check it is equal to 1
pop = 0
for node in ii._clevel_nodes:
    pop += (node.get_amp())**2
print(sp.simplify(pop))