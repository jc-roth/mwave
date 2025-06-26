from abc import ABC, abstractmethod
import inspect
import numbers
import warnings
import sympy
import graphviz
import jinja2
from tqdm import tqdm
from importlib import resources
from mwave import templates

from types import SimpleNamespace

def reset_constants():
    global constants, constants_keys
    constants = SimpleNamespace()
    constants_keys = set()

reset_constants()

def set_constants(**kwargs):
    """Each key value pair provided by the user is stored in the constants object. The value must be either a number or a sympy symbol.

    If the user provides a key that has already been set a warning is thrown.
    
    .. code-block:: python

        from mwave import symbolic as ifr
        import sympy as sp

        m, c = sp.symbols('m c', real=True)
        ifr.set_constants(m=m,c=c,g=9.8)
        for const in ifr.constants_keys:
            print(f"The constant {const} is set to {getattr(ifr.constants, const)}")
        
    """
    # Loop over all provided keyword arguments    
    for key in kwargs:
        # Check that the provided arguments are either Sympy symbols or numbers
        if isinstance(kwargs[key], numbers.Number) or isinstance(kwargs[key], sympy.Symbol):
            # Warn if the constant already exists
            if key in constants_keys:
                warnings.warn(f'The {key} constant has already been defined!')
            # Set the attribute
            setattr(constants, key, kwargs[key])
            # Save the key to _constants_keys
            constants_keys.add(key)
        else: # Provided arguments are not numbers or Sympy symbols, raise error
            raise ValueError(f"The {key} constant must either be a number or a sympy symbol.")    

def _check_constants(lst):
    """Checks that the keys in the list are present in constants_keys. If any keys are missing an error is thrown.
    
    :param lst: A list of keys, for example :code:`['m', 'c']`"""
    for key in lst:
        if key not in constants_keys:
            raise ValueError(f'The constant {key} must be defined.')
        
def eval_sympy_var(var, subs):
    """Numerically evaluates the provided variable and returns the result. The provided variable can either be a python number or a sympy expression. If a sympy expression is provided evaluation is attempted using subs. If this fails an error is raised.
    
    This function is useful for evaluating variables that could either be a python number or a sympy expression.
    
    :param var: The variable to evaluate
    :param subs: The substitutions to make (via sympy).
    """
    # Check if we have a sympy expression
    if isinstance(var, sympy.Expr):
        # Try evaluating the sympy expression
        var_eval = var.evalf(subs=subs)
        # If we did not fully evaluate, raise error
        if not var_eval.is_number:
            raise ValueError(f'Could not fully evaluate the sympy expression into a number, found {var_eval}')
        # Success, return
        return var_eval
    elif isinstance(var, numbers.Number):
        # Already have a number, return
        return var
    else:
        # Error
        raise ValueError(f'Cannot evaluate {var}')

class TreeNode():
    """
    Basic implementation of a directed tree. Nodes are assigned a unique ID. Children are stored in a dictionary, with the ID of the child serving as the dictionary key.

    :param parent_node: The parent node of the new node. Must be a :code:`TreeNode`. Optional.

    For example, to create a tree where node a has child node b:

    .. code-block:: python

        from mwave.symbolic import TreeNode
        a = TreeNode()
        b = TreeNode(a)
        a.children

    """

    # Static ID variable used for assigning IDs
    _id = 0

    def __init__(self, parent_node=None):

        # Set ID, iterate static variable
        self._id = TreeNode._id
        TreeNode._id += 1
        
        # Create dictionary to store children
        self.children = {}

        # Set depth to zero, mark that no parent (yet)
        self.depth = 0
        self._has_parent = False

        # Set parent if a parent was provided
        if parent_node is not None:
            self.set_parent(parent_node)
        else:
            self.parent = None

    def set_parent(self, parent_node):
        """Sets the parent node to the provided node.
        
        :param parent_node: The parent node of the new node. Must be a :code:`TreeNode`.
        """

        if not isinstance(parent_node, TreeNode):
            raise ValueError("Parent must be of class TreeNode")
        
        if self._has_parent:
            raise ValueError("Cannot set parent of TreeNode that already has a parent")
        
        self.parent = parent_node
        self._has_parent = True
        self.depth = self.parent.depth + 1
        self.parent._add_child(self)

    def _add_child(self, child):
        """Adds the provided :code:`child` to the dictionary of children, using the child ID as the dictionary key."""

        if not isinstance(child, TreeNode):
            raise ValueError("child must be of class TreeNode")
        
        # Add the child to the children dictionary using the hashed ID as a key
        self.children[hash(child._id)] = child

class InterferometerNode(TreeNode):
    """
    Used to represent terms in the wavefunction of an interferometer. Inherits from :code:`TreeNode`. Each :code:`InterferometerNode` contains a reference to the unitary object that generated it and can optionally contain an analytic expression for the trajectory between its parent node and itself.
    
    Interferometers have a tree structure where paths often split into multiple paths, which is why a tree structure is appropriate for representing their wavefunctions.

    :param n: The momentum state of the node.
    :param v: The velocity of the node.
    :param z: The z position of the node.
    :param t: The final time that the node represents.
    :param x: The x position of the node.
    :param y: The y position of the node.
    :param parent_node: The parent of the new node. If provided the parent_node for :code:`n`, :code:`v`, :code:`z`, :code:`t`, :code:`x`, and :code:`y` overwrite the ones passed in by the user.
    :param unitary: The unitary generator that produced the node. Optional.
    """

    def __init__(self, n=0, v=0, z=0, t=0, x=0, y=0, parent_node=None, unitary=None):
        super().__init__(parent_node=parent_node)

        # Set properties
        if parent_node is None:
            self.n = n
            self.v = v
            self.z = z
            self.t = t
            self.x = x
            self.y = y
        else:
            self.n = parent_node.n
            self.v = parent_node.v
            self.z = parent_node.z
            self.t = parent_node.t
            self.x = parent_node.x
            self.y = parent_node.y

        # Set the phase difference between this node and its parent
        self.phase_diff = 0

        # Set the amplitude of the node
        self.amp_diff = 1

        # Set the unitary generator
        if unitary is not None:
            if not isinstance(unitary, Unitary):
                raise ValueError("unitary must be or inherit from the Unitary class.")
            self._generator = unitary
        else:
            self._generator = None

        # Set the trajectory to None
        self._trajectory = None

    def has_generator(self):
        """Returns true if a generator has been assigned for this node, false otherwise."""
        return self._generator is not None

    def get_generator(self):
        """Returns the unitary operator that generated the node."""
        return self._generator

    def set_phase_diff(self, phase):
        """Sets the phase of the state. This phase should be the phase difference between this state and its parent state.
        
        :param phase: The phase difference between this state and its parent state."""
        self.phase_diff = phase

    def set_amp_diff(self, amp):
        """Sets the amplitude of the state relative to the previous state."""
        self.amp_diff = amp

    def get_phase_diff(self):
        """Returns the phase difference between this state and its parent state."""
        return self.phase_diff

    def get_amp_diff(self):
        """Gets the amplitude of the state relative to the previous state."""
        return self.amp_diff
    
    def get_phase(self):
        """Returns the the phase of the state by summing the phase differences of all ancestor states togeather."""
        # Create a variable to track the phase, set it equal to the phase of this state
        phase = self.get_phase_diff()

        # Create a variable to track the current node
        node = self

        # Loop through all of the parent nodes, adding the phase of each
        while node.parent is not None:
            phase += node.parent.get_phase_diff()
            node = node.parent

        # Finally return
        return phase
    
    def get_amp(self):
        """Returns the the amplitude of the state by multiplying the amplitudes of all ancestor states togeather."""
        # Create a variable to track the amplitude, set it equal to the amplitude of this state
        amp = self.get_amp_diff()

        # Create a variable to track the current node
        node = self

        # Loop through all of the parent nodes, adding the phase of each
        while node.parent is not None:
            amp *= node.parent.get_amp_diff()
            node = node.parent

        # Finally return
        return amp
    
    def get_wf_term(self):
        """Returns the wavefunction of the state."""
        return self.amp_diff*sympy.exp(sympy.I*self.phase_diff)

    def gen_numeric_wf_func(self, subs={}, filter=None, extended=False):
        """Returns a function that numerically computes the wavefunction term represented by the node. For detailed example usage please see the Numerical calculation of the SCI diffraction phase example.
        
        :param subs: The substitutions to make to the node parameters in the sympy subsitution format (i.e. :code:`{m: 1, n: 3, delta: 4*3}`). This is required if any node parameters used by the :code:`get_numeric` function depend on a sympy expression.
        :param filter: A function used for filtering which numeric functions are included in the calculation. The function must accept two arguments of the form :code:`(node, unitary)`, which are instances of :code:`InterferometerNode` and :code:`Unitary`. If the function returns true then the numeric function is included in the calculation. If false the numeric function is omitted from the calculation.
        :param extended: If true a list of individual functions is returned along with the function that computes the numerical phase of the node. The return then has form :code:`(total_phase_fnc, [phase_fnc1, phase_fnc2, ...])`."""

        # Check if a filter function was provided, if it was perform filtering.
        apply_filtering = False
        if filter is not None:
            if not callable(filter) and len(inspect.signature(filter).parameters) != 2:
                raise ValueError("filter must be a function with signature (node, unitary)")
            apply_filtering = True

        # Climb up the tree until we reach the root
        cnode = self
        fncs = []
        while cnode.parent is not None:
            # Check if the current node was generated by a numeric operator
            if cnode.get_generator().is_numeric():
                # The node was generated by a numeric operator, next check to see if that operator is filtered out
                if not apply_filtering or (apply_filtering and filter(cnode, cnode.get_generator())):
                    # The numeric operator was not filtered out. Generate the function that describes the numeric operator
                    fncs.append(cnode.get_generator().gen_numeric(cnode, subs))
            cnode = cnode.parent

        # Check that all function calls take the same arguments
        fs = inspect.signature(fncs[0])
        for i in range(1, len(fncs)):
            if fs != inspect.signature(fncs[i]):
                raise ValueError('All functions returned by get_numeric must have the same call signature.')

        # Create a function which just multiplies the results of each numeric function togeather
        def fnc(*args):
            amp = 1
            for fnc in fncs:
                amp*=fnc(*args)
            return amp
        
        # Return
        if extended:
            return fnc, fncs
        return fnc

    def __str__(self):
        return f"{type(self._generator).__name__}, n={self.n}, z={self.z}"
    
    def get_partial_trajectory(self):
        """Returns the trajectory taken from the parent node to this node as a sympy expression where time is parameterized by :code:`constants.t_traj`. The second returned parameter is the final time at which the trajectory is valid (note that the initial time at which the trajectory is valid is not returned)."""
        if self._trajectory is None or self.parent is None:
            return (sympy.nan, constants.t_traj < self.t)
        else:
            return (self._trajectory.subs(constants.t_traj, constants.t_traj - self.parent.t) + self.parent.z, constants.t_traj < self.t)
    
    def get_trajectory(self, subs=None):
        """Returns a piecewise function describing the trajectory of the node and all of its ancestor nodes where time is parameterized by :code:`constants.t_traj`. If :code:`subs` is passed in then a function of the form :code:`fnc_trajectory(t)` that returns a numerical value will be returned. Otherwise a sympy expression will be returned."""
        traj_fncs = []
        cnode = self
        while cnode.parent is not None:
            traj_fncs.insert(0, cnode.get_partial_trajectory())
            cnode = cnode.parent
        traj_expr = sympy.Piecewise(*traj_fncs, (sympy.nan, True))
        
        if isinstance(subs, dict):
            return sympy.lambdify(constants.t_traj, traj_expr.evalf(subs=subs), 'numpy')
        else:
            return traj_expr
    
    def lineage(self):
        """Returns a string that describes the lineage of the node (i.e. the ancestor nodes it is derived from)."""
        node = self
        lineage_str = str(node)
        while node.parent is not None:
            lineage_str = str(node.parent) + ' => ' + lineage_str
            node = node.parent
        return lineage_str

class Interferometer():
    """
    Class representing an interferometer geometry. The user defines the interferometer geometry by applying unitary operations (i.e. beamsplitter, free evolution, etc.). Internally the class applies each unitary operation to nodes in a directed tree based on the :code:`InterferometerNode` class. These nodes should be thought of as representing terms in the waverfunction propagating through the interferometer.
    
    Each application of a unitary operator creates child nodes that depend on the type of unitary operator applied and the properties of their respective parent nodes. For example, a free evolution operator would create one child node for each node in the bottom layer of the tree with a position, phase, etc. dependent on the parent node. A beamsplitter operator would create two or more child nodes for each node in the bottom layer of the tree.
    
    The :code:`Interferometer` class also provides methods are provided to visualize the tree representing the interferometer, to plot trajectories of its arms, to compute the analytic phase of the interferometer, and to generate a code outline that can be used to perform numerical simulations with the specified geometry.

    :param init_node: The :code:`InterferometerNode` instance to use as the inital interferometer state. If not provided an :code:`InterferometerNode` is created with default parameters.

    For an example of how the :code:`Interferometer` class can be used see :ref:`sci_example`.
    """

    def __init__(self, init_node=None):
        # Set the initial node if provided
        self.init_node = init_node
        if self.init_node is None:
            self.init_node = InterferometerNode()
        
        # Set the depth
        self.depth = self.init_node.depth
        
        # Create a list to store the unitary operators applied to the interferometer
        self._unitaries = []

        # Create a list of nodes at the current depth
        self._clevel_nodes = [self.init_node,]

        # Track if the interfere function has been called
        self._called_interfere = False

    def apply(self, unitary):
        """Applies the provided unitary operation to the interferometer. The interferometer tree depth is increased by 1 to accomodate the states that occur after the unitary operation.

        Internally this is performed by looping over each :code:`InterferometerNode` at the current depth and passing it to the :code:`apply` function of the provided :code:`Unitary` object.
        
        :param unitary: The unitary operation to apply.
        """

        if not isinstance(unitary, Unitary):
            raise ValueError("unitary must be or inherit from the Unitary class.")
        
        # Save to the list of the applied unitiaries
        self._unitaries.append(unitary)

        # Loop over each node at the current depth, apply the unitary operator to it
        for node in self._clevel_nodes:
            unitary.apply(node)

            # Check that created child nodes have the correct properties
            for key in node.children:
                if not node.children[key].has_generator():
                    warnings.warn(f'Unitary {unitary} did not assign itself as the unitary attribute during the apply method.')

        # Increase the depth by one, refresh the nodes at the current level
        self.depth += 1
        self._refresh_clevel_nodes()
        
        # Set _called_interfere to false
        self._called_interfere = False
    
    def _refresh_clevel_nodes(self):
        """Loops over all nodes at the current level, makes a list of their children, and then updates the current level list to be the list of children."""
        next_level_nodes = []
        for node in self._clevel_nodes:
            for ckey in node.children:
                next_level_nodes.append(node.children[ckey])

        self._clevel_nodes = next_level_nodes

    def generate_graph(self):
        """Returns a :code:`graphviz.Digraph` object containing the nodes in the inteferometer tree.
        
        :returns: The :code:`graphviz.Digraph` object.
        """
        # Initialize graph
        d = graphviz.Digraph()

        # Create a recursive method for generating a graphviz tree
        def iterate_tree(node, parent, depth):

            # Stop adding to tree if we have reached the tree depth
            if depth == self.depth + 1:
                return
            
            # Add the current node to the tree, create an edge to the parent
            d.node(repr(node._id), label=f'n={node.n}')
            d.edge(repr(parent._id), repr(node._id))

            # Loop over each child, call iterate_tree
            for ckey in node.children:
                child = node.children[ckey]
                iterate_tree(child, node, depth + 1)

        # Add the initial node, start iterating over the tree for each child node
        d.node(repr(self.init_node._id), label=f'n={self.init_node.n}')
        if self.depth > 0:
            for ckey in self.init_node.children:
                    child = self.init_node.children[ckey]
                    iterate_tree(child, self.init_node, 1)

        # Return    
        return d
    
    def interfere(self, subs=None, progress=False, recompute=False):
        """Determines which nodes are interfering at the current depth and returns a list of tuples where each tuples contains two interfering nodes.
        
        Two nodes are defined as interfering if they have equal positions and are in equal momentum states. This means that if the interferometer does not perfectly close (i.e. in the case of a gravity gradient), the nodes will *not* be detected as interfering. To ensure that the function still works in this case the user can pass :code:`subs={gamma: 0}` (or similar), which sets the :code:`gamma` symbol to zero during the search for interfering nodes.
        
        :param subs: A dictionary of sympy symbols to substitute. This can be useful if you want to do something like remove dependence on the gravity gradient.
        :param progress: If true a progress bar will be displayed when computing interferences.
        :param recompute: If true interferences will be recomputed, even if they have been computed previously. Use this if you compute interferances at multiple stages of the interferometer.
        """

        # If recompute is true or we have not called interfere continue
        if self._called_interfere and not recompute:
            return self._interfering_nodes
        
        # Note that we have called interfere
        self._called_interfere = True

        # Make a list to store interfering nodes
        self._interfering_nodes = []

        # Get iterator
        itr = range(len(self._clevel_nodes))
        if progress:
            itr = tqdm(itr)

        # Nested loop over all node pairs
        for i in itr:
            for j in range(i+1, len(self._clevel_nodes)):

                # Get positions and momentums
                z1 = self._clevel_nodes[i].z
                z2 = self._clevel_nodes[j].z
                n1 = self._clevel_nodes[i].n
                n2 = self._clevel_nodes[j].n

                # Perform subsitutions if provided
                if subs is not None:
                    z1 = z1.subs(subs) if isinstance(z1, sympy.Expr) else z1
                    z2 = z2.subs(subs) if isinstance(z2, sympy.Expr) else z2
                    n1 = n1.subs(subs) if isinstance(n1, sympy.Expr) else n1
                    n2 = n2.subs(subs) if isinstance(n2, sympy.Expr) else n2

                # Check if the positions are equal
                same_pos = sympy.simplify(z1-z2) == 0

                # Check if the momentums are equal
                same_n = sympy.simplify(n1-n2) == 0

                # If both the position and momentums are equal save the nodes as interfering
                if same_pos and same_n:
                     self._interfering_nodes.append((self._clevel_nodes[i], self._clevel_nodes[j]))

        # Return
        return self._interfering_nodes

    def phases(self, include_separation_phase=True):
        """Computes the phase between all interfering node pairs. The user must have called :code:`interfere` prior to invoking this method. Returns a list of phases, where each element in the list corresponds to the phase between the node pair computed by the :code:`interfere` method.
        
        :param include_separation_phase: If :code:`True` the separation phase in the :code:`z` direction is computed and included.
        :returns: The list of phase differences between each interfering node pair."""
        # Check that intefere has been called, if not error
        if not self._called_interfere:
            raise RuntimeError('The interfere function must be called before the phases function.')
    
        # Check that there are interferences
        if len(self._interfering_nodes) == 0:
            warnings.warn('No interferences were found, returning an empty array')
            return []
        
        # Loop over each node pair and compute the phase
        phase_diffs = []
        for node_pair in self._interfering_nodes:
            dp = node_pair[0].get_phase() - node_pair[1].get_phase()
            if include_separation_phase:
                avg_momentum = (node_pair[0].v + node_pair[1].v)/2*constants.m
                delta_pos = node_pair[0].z - node_pair[1].z
                dp += -delta_pos*avg_momentum/constants.hbar
            phase_diffs.append(dp)

        # Return
        return phase_diffs
        
    def get_nodes(self):
        """Returns the list of the :code:`InterferometerNode` s at the bottom of the tree (i.e. the nodes produced by the latest call to :code:`apply`).
        
        :returns: The list of :code:`InterferometerNode`s."""
        return self._clevel_nodes

    def get_ports(self, nstate_port_map):
        """Returns a dictionary of nodes in each port. Ports are defined by the momentum states in the provided momentum state to port mapping :code:`nstate_port_map`. Ports are returned in the interfering port dictionary if there are determined to be interfering ports by the :code:`interfere()` method. Otherwise they are put in the junk port dictionary. Nodes that do not have a port mapping are put in the no port list.
        
        The user must have called :code:`interfere` prior to invoking this method.
        
        :param nstate_port_map: A dictionary mapping momentum states to port labels.
        :returns: A tuple containing :code:`(interfering_port_dict, junk_port_dict, no_port_list)`
        
        .. code-block:: python

            from mwave import symbolic as ifr
            import sympy as sp

            # Set up symbols for symbolic computation
            m, c, hbar, k, omega_r, delta, delta_bloch, n, N, omega_m, T, Tp, t_traj = sp.symbols('m c hbar k omega_r delta delta_bloch n N omega_m T Tp t_traj', real=True)
            k_eff = 2*k

            # Register constants
            ifr.set_constants(m=m, c=c, hbar=hbar, t_traj=t_traj)

            # Define unitary operators needed for an SCI
            bs1 = ifr.Beamsplitter(0, n, delta, k_eff)

            bo1 = ifr.Beamsplitter(n, n+N, delta_bloch, k_eff)
            bo2 = ifr.Beamsplitter(0, -N, delta_bloch, k_eff)

            bs2u = ifr.Beamsplitter(n+N, 2*n+N, delta + omega_m, k_eff)
            bs2d = ifr.Beamsplitter(0-N, -n-N, delta - omega_m, k_eff)

            fe1 = ifr.FreeEv(T)
            fe2 = ifr.FreeEv(Tp/2)

            # Create interferometer object, perform SCI sequence
            ii = ifr.Interferometer()
            ii.apply(bs1)
            ii.apply(fe1)
            ii.apply(bs1)
            ii.apply(fe2)
            ii.apply(bo1)
            ii.apply(bo2)
            ii.apply(fe2)
            ii.apply(bs2u)
            ii.apply(bs2d)
            ii.apply(fe1)
            ii.apply(bs2u)
            ii.apply(bs2d)

            # Interfere
            ii.interfere()

            # Assign nodes to output ports
            interfering_ports, junk_ports, no_ports = ii.get_ports({N + 2*n: 'A', N + n: 'B', -N: 'C', -N-n: 'D'})
        """
        # Check that intefere has been called, if not error
        if not self._called_interfere:
            raise RuntimeError('The interfere function must be called before the get_ports function.')
        
        # Determine the momentum states associated with output ports
        nstates = nstate_port_map.keys()
        
        # Define interfering and junk port dictionaries
        iports = {}
        jports = {}
        
        # Define a port-less array
        no_port = []
        
        # Loop over interfering nodes and put in iports
        for node in self._interfering_nodes:
            nn = sympy.simplify(node[0].n)
            if nn in nstates:
                port = nstate_port_map[nn]
                
                if port not in iports.keys():
                    iports[port] = []
                
                iports[port].append(node[0])
                iports[port].append(node[1])
            
        # Loop over all nodes, put in jports if not in iports
        for node in self._clevel_nodes:
            nn = sympy.simplify(node.n)
            if nn in nstates:
                port = nstate_port_map[nn]
                if node not in iports[port]:
                    
                    if port not in jports.keys():
                        jports[port] = []
                        
                    jports[port].append(node)
            else:
                no_port.append(node)

        # Return
        return iports, jports, no_port
    
    def generate_code_outline(self, iports):
        """This function generates an outline of python code that can be used to compute the population at all of the output ports of an arbitrary interferometer geometry. The code will not be immediately ready to run. The user must adapt the automatically generated functions for their own use.
        
        :param iports: A dictionary of interfering port populations to generate a numerical computation for.
        :returns: The code as a string.
        
        .. code-block:: python
        
            from mwave import symbolic as ifr
            import sympy as sp

            # Set up symbols for symbolic computation
            m, c, hbar, k, omega_r, delta, delta_bloch, n, N, omega_m, T, Tp, t_traj = sp.symbols('m c hbar k omega_r delta delta_bloch n N omega_m T Tp t_traj', real=True)
            k_eff = 2*k

            # Register constants
            ifr.set_constants(m=m, c=c, hbar=hbar, t_traj=t_traj)

            # Define unitary operators needed for an SCI
            bs1 = ifr.Beamsplitter(0, n, delta, k_eff)

            bo1 = ifr.Beamsplitter(n, n+N, delta_bloch, k_eff)
            bo2 = ifr.Beamsplitter(0, -N, delta_bloch, k_eff)

            bs2u = ifr.Beamsplitter(n+N, 2*n+N, delta + omega_m, k_eff)
            bs2d = ifr.Beamsplitter(0-N, -n-N, delta - omega_m, k_eff)

            fe1 = ifr.FreeEv(T)
            fe2 = ifr.FreeEv(Tp/2)

            # Create interferometer object, perform SCI sequence
            ii = ifr.Interferometer()
            ii.apply(bs1)
            ii.apply(fe1)
            ii.apply(bs1)
            ii.apply(fe2)
            ii.apply(bo1)
            ii.apply(bo2)
            ii.apply(fe2)
            ii.apply(bs2u)
            ii.apply(bs2d)
            ii.apply(fe1)
            ii.apply(bs2u)
            ii.apply(bs2d)

            # Interfere
            ii.interfere()

            # Assign nodes to output ports
            interfering_ports, junk_ports, no_ports = ii.get_ports({N + 2*n: 'A', N + n: 'B', -N: 'C', -N-n: 'D'})

            # Make code outline, export
            with open('example.py', 'w') as f:
                f.write(ii.generate_code_outline(interfering_ports))
                
        """
        
        # Set initial node
        cnode = self._clevel_nodes[0]

        DeltaTs = []
        nbs = 0

        # Construct the list of free evolution durations and number of beamsplitters
        while cnode.parent is not None:
            gen = cnode.get_generator()
        
            # If beamsplitter, increment nbs
            if type(gen) == Beamsplitter:
                n = cnode.parent.n
                if gen.n1 == n or gen.n2 == n:
                    nbs += 1
        
            # If free evolution append time
            elif type(gen) == FreeEv:
                DeltaTs.append(sympy.simplify(cnode.t - cnode.parent.t))
        
            # Update cnode to parent
            cnode = cnode.parent

        # Make a function that generates code to compute the wavefunction at each output port
        def _gen_wavefunction_calc_along_path(node):
            func_calls = []
            cnode = node

            # Loop over lineage, append to function call
            while cnode.parent is not None:
                gen = cnode.get_generator()
                if isinstance(gen, Beamsplitter):
                    n = cnode.parent.n
                    if gen.n1 == n or gen.n2 == n:
                        func_calls.append(f'bs({n},{cnode.n},*args%i)')
                cnode = cnode.parent
        
            # Reverse func_calls to be in time order
            func_calls.reverse()
        
            # Join into single string, add in argument indices
            func_call_combined = '*'.join(func_calls)
        
            # Add in argument indices, return
            arg_idxs = tuple(range(len(func_calls)))
            return func_call_combined % arg_idxs

        # Loop over each port, generate wavefunction calculation code
        wavefunc_calcs = []
        ports = list(iports.keys())
        ports.sort()
        for port in ports:
            wavefunc_calcs.append(f'port{port}_1 = {_gen_wavefunction_calc_along_path(iports[port][0])}')
            wavefunc_calcs.append(f'port{port}_2 = {_gen_wavefunction_calc_along_path(iports[port][1])}')
            wavefunc_calcs.append('')
        del wavefunc_calcs[-1]

        # Open up template, generate code
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(resources.files(templates)))
        templ = env.get_template('ifr_codegen.pytemplate')
        return templ.render(DeltaTs=DeltaTs, num_beamsplitters=nbs, wavefunc_calcs=wavefunc_calcs, ports=ports)

class Unitary(ABC):
    """Represents a unitary transformation (i.e. free evolution, beamsplitter, etc.). The unitary transformation is applied to the provided state (represented as a :code:`InterferometerNode`) via the :code:`apply` method. The transformed state is added as a child node(s) of the provided :code:`InterferometerNode`. The new child node represents the state *after the unitary transformation has been performed*. Note that the phase of any given node is only the phase imparted by the unitary operator that generated it.
    
    If applying the unitary operator requires a numerical calculation, this should be implemented by overriding the :code:`gen_numeric` method. An example of how to do this is in the example Numerical calculation of the SCI diffraction phase.
    """

    def is_numeric(self):
        """Returns true if the :code:`get_numeric` function has been implemented. This will return false unless a subclass implements the function."""
        return type(self).gen_numeric != Unitary.gen_numeric

    @abstractmethod
    def apply(self, node):
        """Applies the unitary operator to the provided node. All unitary operators must implement this method.
        
        :param node: The node to apply the unitary operator to."""
        pass

    def gen_numeric(self, node, args, subs):
        """Function that child classes of :py:class:`mwave.symbolic.Unitary` can override in order to include numerically calculated corrections in phase calculations. See the example :ref:`numercal_evaluation_sci` for more information on how to use this function."""
        raise NotImplementedError('apply_numeric is not implemented.')

    def __matmul__(self, obj):
        # Check that the other object is off type Interferometer
        if not isinstance(obj, Interferometer):
            raise ValueError('Matrix multiplication can only happen between unitary operators and Interferometer objects')
        
        # Apply unitary operator to the interferometer
        obj.apply(self)

        # Return the interferometer
        return obj

class FreeEv(Unitary):
    """Creates a unitary evolution operator for propagating a particle in free space.
    
    :param T: The time to propagate the particle for.
    :param gravity: The gravity to evolve the particle under.
    :param gravity_grad: The gravity gradient to evolve the particle under. If :code:`gravity=None` this variable has no effect."""

    def __init__(self, T, gravity=None, gravity_grad=None):
        super().__init__()

        # Check that the constants needed by this operator are defined
        _check_constants(['hbar', 'm', 't_traj'])
        
        self._T = T
        self._gravity = gravity
        self._gravity_grad = gravity_grad

    def apply(self, node):

        # Add a child to node. This node will contain the next state.
        next_node = InterferometerNode(parent_node=node, unitary=self)

        # Modify z
        next_node.z += self._T*node.v
        if self._gravity is not None:
            next_node.z += -self._gravity*self._T**2/2
            
            if self._gravity_grad is not None:
                next_node.z += (self._T**2*node.z/2+self._T**3*node.v/6-self._gravity*self._T**4/24)*self._gravity_grad

        # Modify t
        next_node.t += self._T
        
        # Modify v
        if self._gravity is not None:
            next_node.v += -self._gravity*self._T

            if self._gravity_grad is not None:
                next_node.v += self._T*(-self._gravity*self._T**2 + 3*self._T*node.v + 6*node.z)*self._gravity_grad/6

        # Set phase
        phase = constants.m*self._T*node.v**2/2/constants.hbar
        if self._gravity is not None:
            phase += self._gravity*constants.m*self._T*(self._gravity*self._T**2-3*(self._T*node.v+node.z))/3/constants.hbar
            if self._gravity_grad is not None:
                phase += self._gravity_grad*constants.m*self._T*(2*self._gravity**2*self._T**4-10*self._gravity*self._T**2*(self._T*node.v+2*node.z)+5*(2*self._T**2*node.v**2+6*self._T*node.v*node.z+3*node.z**2))/30/constants.hbar
        next_node.set_phase_diff(phase)

        # Store the trajectory
        next_node._trajectory = constants.t_traj*node.v
        if self._gravity is not None:
            next_node._trajectory += -self._gravity*constants.t_traj**2/2
            
            if self._gravity_grad is not None:
                next_node._trajectory += (constants.t_traj**2*node.z/2+constants.t_traj**3*node.v/6-self._gravity*constants.t_traj**4/24)*self._gravity_grad
        
        # Do not modify n
        #next_node.n

class Beamsplitter(Unitary):
    """Creates a unitary beamsplitter operator. The phase imparted by the beamsplitter is the theoretical phase obtained from a perfect matterwave beamsplitter.

    A beamsplitter is always resonant between two momentum states. These must be specified for the beamsplitter to work.
    
    :param n1: One of the two momentum states in units of :math:`2\\hbar k`.
    :param n2: The other momentum state in units of :math:`2\\hbar k`
    :param delta: The detuning between the two lasers
    :param k: The effective wavevector. In the case that the beamsplitter provides two photon kicks to the particle this should be :math:`2k_\\text{laser}`
    :param theta_transfer: Describes the amplitude :math:`c` in the diffracted and undiffracted states after the beamsplitter via :math:`c_\\text{diffracted}=\\cos(\\theta_\\text{transfer})` and :math:`c_\\text{undiffracted}=\\sin(\\theta_\\text{transfer})`
    :param include_half_pi_shift: If true then each diffracted node has :math:`\\pi/2` added to its phase. This phase shift happens for all beamsplitters, but is typically ignored in the phase calculation. You might want to include it if you want to calculate the population between output ports, as ignoring it causes the output port populations to be in phase instead of :math:`\\pi/2` out of phase.
    """

    def __init__(self, n1, n2, delta, k, theta_transfer=sympy.pi/4, include_half_pi_shift=True):
        super().__init__()

        # Check that the constants needed by this operator are defined
        _check_constants(['hbar','m'])
        
        self.n1 = n1
        self.n2 = n2
        self.delta = delta
        self._k = k
        self._v_recoil = constants.hbar*k/constants.m
        self.theta_transfer = theta_transfer
        self.include_half_pi_shift = include_half_pi_shift

    def apply(self, node):

        # Check to see if the node is resonant with the beamsplitter
        if node.n == self.n1:
            # Resonant with n1, create two child nodes
            s1 = InterferometerNode(parent_node=node, unitary=self)
            s2 = InterferometerNode(parent_node=node, unitary=self)

            # Update the phase of the undiffracted node
            s1.set_phase_diff(0)

            # Set the amplitude of the undiffracted node
            s1.set_amp_diff(sympy.sin(self.theta_transfer))

            # Update momentum, velocity, and phase of diffracted node
            s2.n = self.n2
            s2.v += (self.n2 - self.n1)*self._v_recoil
            add_phase = sympy.pi/2 if self.include_half_pi_shift else 0
            s2.set_phase_diff((self.n2 - self.n1)*(self._k*node.z - self.delta*node.t) + add_phase)

            # Set the amplitude of the diffracted node
            s2.set_amp_diff(sympy.cos(self.theta_transfer))
        elif node.n == self.n2:
            # Resonant with n2, create two child nodes
            s1 = InterferometerNode(parent_node=node, unitary=self)
            s2 = InterferometerNode(parent_node=node, unitary=self)

            # Update the phase of the undiffracted node
            s1.set_phase_diff(0)

            # Set the amplitude of the undiffracted node
            s1.set_amp_diff(sympy.sin(self.theta_transfer))

            # Update momentum, velocity, and phase of diffracted node
            s2.n = self.n1
            s2.v += (self.n1 - self.n2)*self._v_recoil
            add_phase = sympy.pi/2 if self.include_half_pi_shift else 0
            s2.set_phase_diff((self.n1 - self.n2)*(self._k*node.z - self.delta*node.t) + add_phase)

            # Set the amplitude of the diffracted node
            s2.set_amp_diff(sympy.cos(self.theta_transfer))
        else:
            # Not resonant, create child node which is identical to parent node
            s1 = InterferometerNode(parent_node=node, unitary=self)

class Mirror(Unitary):
    """Creates a unitary mirror operator. The phase imparted by the mirror is the theoretical phase obtained from a perfect matterwave mirror.

    Mirror always flips population perfectly between two momentum states.
    
    :param n1: One of the two momentum states in units of :math:`2\\hbar k`.
    :param n2: The other momentum state in units of :math:`2\\hbar k`
    :param delta: The detuning between the two lasers
    :param k: The effective wavevector. In the case that the beamsplitter provides two photon kicks to the particle this should be :math:`2k_\\text{laser}`"""

    def __init__(self, n1, n2, delta, k):
        super().__init__()

        # Check that the constants needed by this operator are defined
        _check_constants(['hbar','m'])

        self._n1 = n1
        self._n2 = n2
        self.delta = delta
        self._k = k
        self._v_recoil = constants.hbar*k/constants.m

    def apply(self, node):

        # Check to see if the node is resonant with the beamsplitter
        if node.n == self._n1:
            # Resonant with n1, create two child nodes
            s2 = InterferometerNode(parent_node=node, unitary=self)

            # Update momentum, velocity, and phase of diffracted node
            s2.n = self._n2
            s2.v += (self._n2 - self._n1)*self._v_recoil
            s2.set_phase_diff((self._n2 - self._n1)*(self._k*node.z - self.delta*node.t))
        elif node.n == self._n2:
            # Resonant with n2, create two child nodes
            s2 = InterferometerNode(parent_node=node, unitary=self)

            # Update momentum, velocity, and phase of diffracted node
            s2.n = self._n1
            s2.v += (self._n1 - self._n2)*self._v_recoil
            s2.set_phase_diff((self._n1 - self._n2)*(self._k*node.z - self.delta*node.t))
        else:
            # Not resonant, create child node which is identical to parent node
            s1 = InterferometerNode(parent_node=node, unitary=self)
