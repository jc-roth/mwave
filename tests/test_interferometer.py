from mwave import symbolic as ifr
from mwave import integrate as intgr
import sympy as sp
import numpy as np
import pytest

def _test_TreeNode_class(class_def):
    """Checks classes that inherit from TreeNode. This allows us to easily perform the same set of standard tests on classes that inherit from TreeNode."""

    tn_parent = class_def()
    tn_child1 = class_def()
    tn_child2 = class_def()

    # Check depths are correct
    assert tn_parent.depth == 0
    assert tn_child1.depth == 0
    assert tn_child2.depth == 0

    # Create some inheritance, check depths are correct
    tn_child1.set_parent(tn_parent)
    tn_child2.set_parent(tn_parent)

    # Check depths are correct
    assert tn_parent.depth == 0
    assert tn_child1.depth == 1
    assert tn_child2.depth == 1

    # Check parent was properly assigned
    assert tn_parent == tn_child1.parent
    assert tn_parent == tn_child2.parent

    # Create some more children, directly assigning them to child1
    tn_child1_1 = class_def(parent_node=tn_child1)
    tn_child1_2 = class_def(parent_node=tn_child1)

    # Check depths are correct
    assert tn_child1_1.depth == 2
    assert tn_child1_2.depth == 2

    # Check parent was properly assigned
    assert tn_child1 == tn_child1_1.parent
    assert tn_child1 == tn_child1_2.parent


def test_TreeNode():
    """Test the TreeNode class."""
    _test_TreeNode_class(ifr.TreeNode)

def test_InterferometerNode():
    """Test the InterferometerNode class."""
    _test_TreeNode_class(ifr.InterferometerNode)

def test_constants():
    ifr.reset_constants()
    m, c = sp.symbols('m c', real=True)
    ifr.set_constants(m=m,c=c,g=9.8)
    ifr._check_constants(['m','c','g'])
    with pytest.raises(ValueError):
        ifr._check_constants(['undefined'])
    assert ifr.constants.g == 9.8

def test_Beamsplitter():
    # Set symbols in interferometer module
    ifr.reset_constants()
    ifr.set_constants(m=1,hbar=1,c=1,t_traj=1)

    # Create InterferometerNode, apply beamsplitter
    n = ifr.InterferometerNode()
    u = ifr.Beamsplitter(0,3,4*3,1)
    u.apply(n)

    # Determine momentum states of children
    children_n_values = []
    for ckey in n.children:
        children_n_values.append(n.children[ckey].n)
    children_n_values = set(children_n_values)

    # Check the momentum states are expected
    assert 0 in children_n_values
    assert 3 in children_n_values

def test_interferometer():
    # Set up symbols for symbolic computation
    k, c, hbar, m, n, delta, t_traj, = sp.symbols('k c hbar m n delta t_traj', real=True)
    k_eff = 2*k

    # Set symbols in interferometer module
    ifr.reset_constants()
    ifr.set_constants(m=m,hbar=hbar,c=c,t_traj=t_traj)

    # Define a beamsplitter unitary
    bs1 = ifr.Beamsplitter(0, n, delta, k_eff)

    # Create an interferometer
    ii = ifr.Interferometer(init_node=ifr.InterferometerNode())

    # Check that we can apply a unitary to it
    bs1 @ (bs1 @ ii)

def test_SCI_interferometer():
    # Set up symbols for symbolic computation
    k, c, hbar, m, n, T, Tp, delta, omega_m, omega_r, t_traj, g, v0, Toffset = sp.symbols('k c hbar m n T T_p delta omega_m omega_r t_traj g v_0 T_offset', real=True)
    k_eff = 2*k

    # Set symbols in interferometer module
    ifr.reset_constants()
    ifr.set_constants(m=m,hbar=hbar,c=c,t_traj=t_traj)

    # Define unitary operators needed for SCI
    bs1 = ifr.Beamsplitter(0, n, delta, k_eff, include_half_pi_shift=False)
    bs2 = ifr.Beamsplitter(0, -n, delta - omega_m, k_eff, include_half_pi_shift=False)
    bs2u = ifr.Beamsplitter(n, 2*n, delta + omega_m, k_eff, include_half_pi_shift=False)
    fe1 = ifr.FreeEv(T)
    fe2 = ifr.FreeEv(Tp)

    # Create interferometer object, perform SCI sequence
    ii = ifr.Interferometer(init_node=ifr.InterferometerNode(v=v0))
    ii.apply(bs1)
    ii.apply(fe1)
    ii.apply(bs1)
    ii.apply(fe2)
    ii.apply(bs2)
    ii.apply(bs2u)
    ii.apply(fe1)
    ii.apply(bs2)
    ii.apply(bs2u)

    # Interfere, compute differential phase
    inodes = ii.interfere()

    # Assert that we have four interferances
    assert len(inodes) == 4

    # Compute the phase of the lower interferometer
    lphase = sp.simplify((inodes[0][0].get_phase() - inodes[0][1].get_phase()).subs(k, sp.sqrt(2*m*omega_r/hbar)))

    # Compute the phase of the upper interferometer
    uphase = sp.simplify((inodes[2][0].get_phase() - inodes[2][1].get_phase()).subs(k, sp.sqrt(2*m*omega_r/hbar)))

    # Compute the differential phase
    dphase = sp.simplify(uphase - lphase)

    # Compare the computed differential phase to the expected differential phase
    exp_dphase = 2*T*n*(-8*n*omega_r + omega_m)
    assert sp.simplify(dphase - exp_dphase) == 0

def test_SCI_diffraction_phase_calc():
    # Set up symbols for symbolic computation
    k, c, hbar, m, n, T, Tp, delta, omega_m, t_traj = sp.symbols('k c hbar m n T T_p delta omega_m t_traj', real=True)
    k_eff = 2*k
    
    ifr.reset_constants()
    ifr.set_constants(m=m,hbar=hbar,c=c,t_traj=t_traj)

    # Define a numeric unitary operator for single frequency Bragg
    class SingleFreqBragg(ifr.Beamsplitter):

        def gen_numeric(self, node, subs={}):
            delta = ifr.eval_sympy_var(self.delta, subs)
            n_init = ifr.eval_sympy_var(node.parent.n, subs)
            n_final = ifr.eval_sympy_var(node.n, subs)
            kvec, n0_idx, nf_idx = intgr.make_kvec(n_init, n_final)
            def fnc(omega, sigma, v, ratio):
                sol = intgr.gbragg(kvec, intgr.make_phi(kvec, n_init), 2*2.716*sigma, delta + 2*v, -2*np.pi*omega, sigma)
                return sol.y[nf_idx,-1]
            return fnc

    # Define a numeric unitary operator for multi frequency Bragg
    class MultiFreqBragg(ifr.Beamsplitter):

        def __init__(self, n1, n2, delta, k, omega_m):
            super().__init__(n1, n2, delta + omega_m, k)
            self.delta_numeric = delta
            self.omega_m = omega_m

        def gen_numeric(self, node, subs={}):
            delta = float(ifr.eval_sympy_var(self.delta_numeric, subs))
            omega_m = float(ifr.eval_sympy_var(self.omega_m, subs))
            n_init = ifr.eval_sympy_var(node.parent.n, subs)
            n_final = ifr.eval_sympy_var(node.n, subs)
            kvec, n0_idx, nf_idx = intgr.make_kvec(n_init, n_final)
            def fnc(omega, sigma, v, ratio):
                sol = intgr.gbragg(kvec, intgr.make_phi(kvec, n_init), 2*2.716*sigma, delta + 2*v, -2*np.pi*omega*ratio/4, sigma, omega_mod=omega_m)
                return sol.y[nf_idx,-1]
            return fnc

    # Define the operators needed for an SCI
    bs1 = SingleFreqBragg(n, 0, delta, k_eff)
    bs2 = MultiFreqBragg(0, -n, delta, k_eff, -omega_m)
    bs2u = MultiFreqBragg(n, 2*n, delta, k_eff, +omega_m)
    fe1 = ifr.FreeEv(T)
    fe2 = ifr.FreeEv(Tp)

    # Define the interferometer
    ii = ifr.Interferometer(init_node=ifr.InterferometerNode(n=n, v=2*hbar*k*n/m))
    ii.apply(bs1)
    ii.apply(fe1)
    ii.apply(bs1)
    ii.apply(fe2)
    ii.apply(bs2)
    ii.apply(bs2u)
    ii.apply(fe1)
    ii.apply(bs2)
    ii.apply(bs2u)

    # Determine the intefering nodes
    inodes = ii.interfere()

    # Define substitutions
    subs = {hbar:1, k:1, m: 1, n: 5, delta: 4*n, omega_m: 8*n, T:5, Tp: 2}

    # Write a filter function that ensures we don't double-count the effect of the multifrequency beamsplitters
    def filterfnc(node, unitary):
        return node.n == unitary.n1 or node.n == unitary.n2

    # Get the numeric phase calculation for each output
    uuu = inodes[1][0].gen_numeric_wf_func(subs, filter=filterfnc)
    udu = inodes[1][1].gen_numeric_wf_func(subs, filter=filterfnc)

    uud = inodes[0][0].gen_numeric_wf_func(subs, filter=filterfnc)
    udd = inodes[0][1].gen_numeric_wf_func(subs, filter=filterfnc)

    duu = inodes[3][0].gen_numeric_wf_func(subs, filter=filterfnc)
    ddu = inodes[3][1].gen_numeric_wf_func(subs, filter=filterfnc)

    dud = inodes[2][0].gen_numeric_wf_func(subs, filter=filterfnc)
    ddd = inodes[2][1].gen_numeric_wf_func(subs, filter=filterfnc)

    # Create a function to compute the differential phase
    def calc_diff_phase(omega, sigma, v, ratio):
        ph = -(np.angle(-uuu(omega, sigma, v, ratio)/udu(omega, sigma, v, ratio)) - np.angle(-duu(omega, sigma, v, ratio)/ddu(omega, sigma, v, ratio)))
        phh = -(np.angle(uud(omega, sigma, v, ratio)/udd(omega, sigma, v, ratio)) - np.angle(dud(omega, sigma, v, ratio)/ddd(omega, sigma, v, ratio)))
        return (ph+phh)/2

    assert np.allclose(calc_diff_phase(30,0.188,0.5,1), 2.057268200712416)