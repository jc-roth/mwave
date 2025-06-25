import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
from mwave import symbolic as ifr

# Enable nice printing
sp.init_printing()

# Set up symbols for symbolic computation
k, c, hbar, m, g, T_spacing, T, Tp, delta, traj_t, omega_r, omega_0 = sp.symbols('k c hbar m g T_spacing T Tp delta traj_t omega_r omega_0', real=True)
k_eff = 2*k

# Register constants
ifr.set_constants(m=m, c=c, hbar=hbar, t_traj=traj_t)

# For an n=2 SCI with single Bragg pulses we must run the following pulse sequences for beam splitters

# Pulse sequence 1: Split initial state into two, mirror one of the outputs to bragg order n
# 1       2     ...    n
# pi/2 -> pi -> ... -> pi

# Pulse sequence 2: Mirror state at momentum n down to 1, then pi/2 to bring to correct state and diffract initially undiffracted arm to 1, then mirror initially undiffracted arm up to n
# 1            n-1   n       n+1          n+n-1
# pi -> ... -> pi -> pi/2 -> pi -> ... -> pi

# Pulse sequence 3: pi/2 lower interferometer upper arm down (pi/2 necessary to avoid diffracting away the lower arm), pi pulse to -n. Simulaneously do the same for the upper interferometer lower arm
# 1       2            n
# pi/2 -> pi -> ... -> pi

# Pulse sequence 4: pi lower interferometer upper arm up to 0, pi/2 pulse to interfere 0 and -1 states. Simulaneously do the same for the upper interferometer lower arm
# 1            2     n
# pi -> ... -> pi -> pi/2

# Define unitary operators needed for SCI

nbragg = 2

bsdict = {}
mdict = {}

def get_bs(n0, nf):
    tid = (min(n0,nf), max(n0,nf))
    if tid in bsdict:
        return bsdict[tid]
    else:
        bsdict[tid] = ifr.Beamsplitter(min(n0,nf), max(n0,nf), 4*(n0+nf)*omega_0, k_eff)
        return bsdict[tid]

def get_m(n0, nf):
    tid = (min(n0,nf), max(n0,nf))
    if tid in mdict:
        return mdict[tid]
    else:
        mdict[tid] = ifr.Mirror(min(n0,nf), max(n0,nf), 4*(n0+nf)*omega_0, k_eff)
        return mdict[tid]

# Free evolutions
feTs = ifr.FreeEv(T_spacing)
feT = ifr.FreeEv(T)
feTp = ifr.FreeEv(Tp)

# Beamsplitter sequences
def pulse1(ii):
    ii.apply(get_bs(0, 1))
    for n in range(1, nbragg):
        ii.apply(feTs)
        ii.apply(get_m(n, n+1))

def pulse2(ii):
    for n in range(nbragg, 1, -1):
        ii.apply(get_m(n, n-1))
        ii.apply(feTs)
    ii.apply(get_bs(1, 0))
    for n in range(1, nbragg):
        ii.apply(feTs)
        ii.apply(get_m(n, n+1))
        
def pulse3(ii):
    ii.apply(get_bs(0, -1))
    ii.apply(get_bs(nbragg, nbragg+1))
    for n in range(1, nbragg):
        ii.apply(feTs)
        ii.apply(get_m(-n, -n-1))
        ii.apply(get_m(nbragg+n, nbragg+n+1))
        
def pulse4(ii):
    for n in range(nbragg, 1, -1):
        ii.apply(get_m(-n, -n+1))
        ii.apply(get_m(nbragg+n, nbragg+n-1))
        ii.apply(feTs)
    ii.apply(get_bs(0, -1))
    ii.apply(get_bs(nbragg, nbragg+1))

# Create interferometer object, perform SCI sequence

ii = ifr.Interferometer()
pulse1(ii)
ii.apply(feT)
pulse2(ii)
ii.apply(feTp)
pulse3(ii)
ii.apply(feT)
pulse4(ii)

# Get interfering nodes
inodes = ii.interfere(progress=False, subs={g: 0})

print(f'Found {len(inodes)} interfering nodes.')

subs = {k: 2, c: 1, hbar: 1, m: 1, T_spacing: 1, delta: 4*omega_r, omega_r: 1, T:5, Tp: 3, g: 1}

plt.figure()
tfinal = float(inodes[0][0].t.evalf(subs=subs))
t = np.linspace(0,tfinal,200)
for pair in inodes:
    n1, n2 = pair
    traj1 = sp.lambdify(traj_t, n1.get_trajectory(traj_t).evalf(subs=subs), 'numpy')
    traj2 = sp.lambdify(traj_t, n2.get_trajectory(traj_t).evalf(subs=subs), 'numpy')

    l, = plt.plot(t, traj1(t), alpha=0.5)
    plt.plot(t, traj2(t), linestyle='--', alpha=0.5)

plt.show()

# Compute the differential phase
phi_A, phi_B, phi_C, phi_D = ii.phases()
print(sp.simplify(phi_A - phi_C).subs(k, sp.sqrt(2*m*omega_r/hbar)))

interfering_ports, junk_ports, no_ports = ii.get_ports({nbragg+1: 'A', nbragg: 'B', 0: 'C', -1: 'D'})
print(ii.generate_code_outline(interfering_ports))

# Print out the total population, check it is equal to 1
pop = 0
for node in ii._clevel_nodes:
    pop += (node.get_amp())**2
print(sp.simplify(pop))