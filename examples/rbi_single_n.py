import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
from mwave import symbolic as ifr

# Enable nice printing
sp.init_printing()

# Set up symbols for symbolic computation
k, c, hbar, m, T_spacing, T, Tp, delta, traj_t, omega_r, omega_m = sp.symbols('k c hbar m T_spacing T Tp delta traj_t omega_r omega_m', real=True)
k_eff = 2*k

# Register constants
ifr.set_constants(m=m, c=c, hbar=hbar, t_traj=traj_t)

# Define unitary operators needed for SCI
ddelta = 8*omega_m

nbragg = 4

# First class of beamsplitters/mirrors
bs1 = []
for i in range(nbragg):
    bs1.append(ifr.Beamsplitter(i, i+1, delta+ddelta*i, k_eff))

m1 = []
for i in range(nbragg):
    m1.append(ifr.Mirror(i, i+1, delta+ddelta*i, k_eff))

# Second class of beamsplitters/mirrors
bs2 = []
for i in range(nbragg):
    bs2.append(ifr.Beamsplitter(-i, -i-1, delta-ddelta*(i+1), k_eff))

m2 = []
for i in range(nbragg):
    m2.append(ifr.Mirror(-i, -i-1, delta-ddelta*(i+1), k_eff))

# Free evolutions
feTs = ifr.FreeEv(T_spacing)
feT = ifr.FreeEv(T)
feTp = ifr.FreeEv(Tp)


def apply_bs(ii, invert=False, mirrors=False):
    if invert:
        for i in range(nbragg-1, 0, -1):
            ii.apply(m1[i] if mirrors else bs1[i])
            ii.apply(feTs)
        ii.apply(bs1[0])
    else:
        ii.apply(bs1[0])
        for i in range(1, nbragg):
            ii.apply(feTs)
            ii.apply(m1[i] if mirrors else bs1[i])

def apply_bs2(ii, invert=False, mirrors=False):
    if invert:
        for i in range(nbragg-1, 0, -1):
            ii.apply(m2[i] if mirrors else bs2[i])
            ii.apply(feTs)
        ii.apply(bs2[0])
    else:
        ii.apply(bs2[0])
        for i in range(1, nbragg):
            ii.apply(feTs)
            ii.apply(m2[i] if mirrors else bs2[i])

# Create interferometer object, perform SCI sequence
mirrors=True

ii = ifr.Interferometer()
apply_bs(ii, mirrors=mirrors)
ii.apply(feT)
apply_bs(ii, invert=True, mirrors=mirrors)
ii.apply(feTp)
apply_bs2(ii, mirrors=mirrors)
ii.apply(feT)
apply_bs2(ii, invert=True, mirrors=mirrors)

# Get interfering nodes
inodes = ii.interfere(progress=False)

print(f'Found {len(inodes)} interfering nodes.')

subs = {k: 2, c: 1, hbar: 1, m: 1, T_spacing: 1, delta: 4*omega_r, omega_r: 1, T:4, Tp: 1}

plt.figure()
tfinal = float(inodes[0][0].t.evalf(subs=subs))
t = np.linspace(0,tfinal,1000)
for pair in inodes:
    n1, n2 = pair
    traj1 = sp.lambdify(traj_t, n1.get_trajectory(traj_t).evalf(subs=subs), 'numpy')
    traj2 = sp.lambdify(traj_t, n2.get_trajectory(traj_t).evalf(subs=subs), 'numpy')

    l, = plt.plot(t, traj1(t), alpha=0.5)
    plt.plot(t, traj2(t), linestyle='--', alpha=0.5)

plt.show()

# Compute the phase of each state
node1, node2 = inodes[0]
phase1 = node1.get_phase()
phase2 = node2.get_phase()

# Compute the phase difference between the two states, print
lphase = sp.simplify((phase1 - phase2).subs(k, sp.sqrt(2*m*omega_r/hbar)))
print("lower interferometer phase: %s" % str(lphase))