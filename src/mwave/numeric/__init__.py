import inspect
import numpy as np

class NumericBraggInterferometer:
    """
    A class for numerically simulating Bragg interferometers. This class allows for the definition of an interferometer geometry and the numerical propagation of wavefunctions through that geometry. See :doc:`examples/numeric_interferometer_usage` for example usage.
    """

    def __init__(self, kmin, kmax, distance, kpad = 10, x0=0, t0=0, k0=0):
        """
        :param kmin: The minimum momentum state to include in the simulation grid.
        :param kmax: The maximum momentum state to include in the simulation grid.
        :param distance: The number of momentum states to keep on either side of the Bragg diffraction orders.
        :param kpad: The padding to add to the momentum grid. Defaults to 10.
        :param x0: The initial position. Defaults to 0.
        :param t0: The initial time. Defaults to 0.
        :param k0: The initial momentum. Defaults to 0.
        """

        # Ensure kpad is even
        if kpad % 2 != 0:
            raise ValueError('kpad must be even')
        
        # Compute k-state vector
        k1 = np.min([kmin, kmax]) - kpad
        k2 = np.max([kmin, kmax]) + kpad
        self.kvec = np.arange(k1, k2+1, 2, dtype=np.float64)
        
        # Save distance argument
        self.distance = distance

        # Initialize a node, create a list of the nodes at the current level, and create a list to store operations in
        self.root = NumericTreeNode(self, x0, t0, k0)
        self.current_level = [self.root]
        self.operations = []

    def split(self, klattice):
        """
        Applies a Bragg diffraction splitting operation to all nodes at the current level.

        :param klattice: The lattice wavevector(s) for the Bragg pulse. Can be a single value or a list of values.
        """
        # Store operation in list
        self.operations.append(('split', [klattice]))

        # Apply operation to each node, update list of current nodes
        next_level = []
        for node in self.current_level:
            node.split(klattice, distance=self.distance)
            next_level += node.children
        self.current_level = next_level

    def propagate(self, t):
        """
        Propagates the state of all nodes at the current level for a time :code:`t`.

        :param t: The duration of the propagation.
        """
        # Store operation in list
        self.operations.append(('propagate', [t]))

        # Apply operation to each node, update list of current nodes
        next_level = []
        for node in self.current_level:
            node.propagate(t)
            next_level += node.children
        self.current_level = next_level

    def get_nodes(self, x_tolerance=None):
        """
        Returns a dictionary of nodes at the current level, organized by momentum and position.

        :param x_tolerance: Optional. If provided, nodes whose position differs from an existing bucket key by less than :code:`x_tolerance` are grouped into that bucket rather than creating a new one. When multiple existing keys are within tolerance, the closest one is chosen. Note that bucket keys anchor to the position of the first node assigned to them, so the resulting grouping can depend on the order in which nodes are processed.
        :returns: A dictionary where keys are momentum states, values are dictionaries of positions, and values of those dictionaries are lists of nodes at that momentum and position.
        """
        # Make a dictionary of momentum states. Values are dictionaries of positions. Values of those dictionaries are lists of nodes at that momentum and position.
        nodedict = {}
        for node in self.current_level:
            if node.k not in nodedict:
                nodedict[node.k] = {}

            if node.x not in nodedict[node.k]:
                # Check if another node is within tolerance
                if x_tolerance:
                    xs = np.array(list(nodedict[node.k].keys()))
                    diffs = np.abs(node.x - xs)
                    mask = diffs < x_tolerance
                    if np.sum(mask) > 0:
                        target_x = xs[mask][np.argmin(diffs[mask])]
                        nodedict[node.k][target_x].append(node)
                    # Create a new array with the node in it
                    else:
                        nodedict[node.k][node.x] = [node]
                # Create a new array with the node in it
                else:
                    nodedict[node.k][node.x] = [node]
            else:
                nodedict[node.k][node.x].append(node)
        return nodedict
    
    def compile(self, split_funcs, propagate_func, output_momentums, kvector_funcs=None,
                func_pop_init=None, func_wf_init=None, func_wf2_init=None, x_tolerance=None):
        """
        Compiles the interferometer into a single population-calculation function.

        Split and propagate functions are provided separately and interleaved automatically
        to match the operation sequence.

        :param split_funcs: A list of functions, one per split operation. Each takes :code:`(*comm_args, k_init, k_final, klattice, t, x)`.
        :param propagate_func: A single function used for all propagation steps. Takes :code:`(*comm_args, t, k)`.
        :param output_momentums: A list of output momentum values to compute populations for.
        :param kvector_funcs: Optional. A list of functions, one per split operation. Each takes :code:`(*comm_args)` and returns a scalar wavevector shift :code:`dk`.
        :param func_pop_init: Optional. A function that initializes the population accumulator. Takes :code:`(*comm_args)`. Defaults to :code:`np.zeros_like(comm_args[0])`.
        :param func_wf_init: Optional. A function that initializes the wavefunction amplitude. Takes :code:`(*comm_args)`. Defaults to :code:`np.ones_like(comm_args[0])`.
        :param func_wf2_init: Optional. A function that initializes the wavefunction accumulator. Takes :code:`(*comm_args)`. Defaults to :code:`np.zeros_like(comm_args[0])`.
        :param x_tolerance: Optional. If provided, nodes whose position differs from an existing bucket key by less than :code:`x_tolerance` are grouped into that bucket rather than creating a new one. When multiple existing keys are within tolerance, the closest one is chosen. Note that bucket keys anchor to the position of the first node assigned to them, so the resulting grouping can depend on the order in which nodes are processed.
        :returns: A function :code:`calc_pops(momentum, comm_args)` that computes the population at the given momentum.
        :raises ValueError: If the number of split_funcs or kvector_funcs does not match the number of split operations, or if function signatures are inconsistent.
        """
        # Count split operations
        n_splits = sum(1 for op_type, _ in self.operations if op_type == 'split')

        # Validate lengths
        if len(split_funcs) != n_splits:
            raise ValueError(f'Expected {n_splits} split functions, got {len(split_funcs)}')
        if kvector_funcs is not None and len(kvector_funcs) != n_splits:
            raise ValueError(f'Expected {n_splits} kvector functions, got {len(kvector_funcs)}')

        # Validate function signatures
        split_counts = [self._param_count(f) for f in split_funcs]
        prop_count = self._param_count(propagate_func)

        known_split_counts = [c for c in split_counts if c is not None]
        if known_split_counts and len(set(known_split_counts)) > 1:
            raise ValueError(f'split_funcs have inconsistent parameter counts: {split_counts}')

        if known_split_counts and prop_count is not None:
            expected_prop = known_split_counts[0] - 3
            if prop_count != expected_prop:
                raise ValueError(
                    f'propagate_func takes {prop_count} parameters, expected '
                    f'{expected_prop} (split functions take {known_split_counts[0]}, '
                    f'propagate should take 3 fewer)')

        if kvector_funcs is not None:
            for i, kvf in enumerate(kvector_funcs):
                kv_count = self._param_count(kvf)
                if kv_count is not None and known_split_counts:
                    expected_kv = known_split_counts[0] - 5
                    if kv_count != expected_kv:
                        raise ValueError(
                            f'kvector_funcs[{i}] takes {kv_count} parameters, expected '
                            f'{expected_kv} (split functions take {known_split_counts[0]}, '
                            f'kvector should take 5 fewer)')

        # Build interleaved operation_funcs list, bundling kvector_funcs with split_funcs
        operation_funcs = []
        split_idx = 0
        for op_type, _ in self.operations:
            if op_type == 'split':
                kvf = kvector_funcs[split_idx] if kvector_funcs is not None else None
                operation_funcs.append((split_funcs[split_idx], kvf))
                split_idx += 1
            elif op_type == 'propagate':
                operation_funcs.append((propagate_func, None))

        self.operation_funcs = operation_funcs

        # Default init functions
        if func_pop_init is None:
            func_pop_init = lambda *comm_args: np.zeros_like(comm_args[0])
        if func_wf_init is None:
            func_wf_init = lambda *comm_args: np.ones_like(comm_args[0], dtype=np.complex128)
        if func_wf2_init is None:
            func_wf2_init = lambda *comm_args: np.zeros_like(comm_args[0], dtype=np.complex128)

        # Build wavefunction closures for each output momentum
        nodedict = self.get_nodes(x_tolerance=x_tolerance)
        func_dict = {}
        for momentum in output_momentums:
            if momentum in nodedict:
                popfuncs = []
                for position in nodedict[momentum]:
                    wffuncs = []
                    for node in nodedict[momentum][position]:
                        wffuncs.append(node.get_wf_func(func_wf_init))
                    popfuncs.append(wffuncs)
                func_dict[momentum] = popfuncs

        # Return compiled population function
        def calc_pops(momentum, comm_args):
            pop = func_pop_init(*comm_args)
            for flst in func_dict[momentum]:
                wf = func_wf2_init(*comm_args)
                for f in flst:
                    wf += f(comm_args)
                pop += np.abs(wf)**2
            return pop

        return calc_pops

    def compile_numeric_sweep(self, *, envelope, tfinal, beam_profile,
                              multi_envelope=None, output_momentums=None,
                              tol=1e-6, internal_tol=1e-5):
        """Compiles a fast, batched phase-sweep evaluator for this interferometer.

        Unlike :meth:`compile`, which composes user-provided split/propagate
        functions, this method numerically propagates every unique Bragg pulse
        configuration using an adaptive DP5 integrator and then evaluates the
        resulting population vs. sweep-phase curve via a Fourier decomposition.

        :param envelope: Callable ``envelope(t) -> array`` giving the pulse
            temporal shape for single-frequency splits. Must vectorize over
            numpy array input. For a standard Gaussian,
            :func:`mwave.numeric.gaussian_envelope` returns a matching
            ``(envelope, tfinal)`` pair.
        :param tfinal: Total pulse duration in recoil time units. The integrator
            runs from ``t=0`` to ``t=tfinal``.
        :param beam_profile: Callable ``beam_profile(x, y, z) -> array`` giving
            the effective peak Rabi frequency at each atom's position. Inputs
            are per-atom position arrays (all the same shape); output must
            broadcast to that shape. For a standard transverse Gaussian beam,
            :func:`mwave.numeric.gaussian_beam` returns a ready-to-use callable.
        :param multi_envelope: Optional callable ``env(t) -> array`` used for
            multi-frequency splits. Required if the geometry contains any split
            whose lattice wavevector is a list/tuple/array. For the common
            two-tone Bragg pulse, wrap a single-freq envelope with
            :func:`mwave.numeric.multi_freq_envelope`.
        :param output_momentums: Iterable of output momenta to compute populations
            for. Defaults to ``(4*nbragg, 2*nbragg, 0, -2*nbragg)``.
        :param tol: External tolerance (reserved; not currently consulted by the
            fixed-schedule hot path).
        :param internal_tol: Tolerance used when computing the adaptive step
            schedule on the worst-case pilot trajectory.
        :returns: A callable ``sweep(cloud, cphases, injected_dphase=0.0)`` that
            returns a tuple of 1D population arrays, one per output momentum.
            ``cloud`` is a dict with keys ``x0, y0, z0, vx, vy, vz``.
        """
        from ._fastsweep import _NumericSweep
        return _NumericSweep(self, envelope=envelope, tfinal=tfinal,
                             beam_profile=beam_profile,
                             multi_envelope=multi_envelope,
                             output_momentums=output_momentums,
                             tol=tol, internal_tol=internal_tol)

    @staticmethod
    def _param_count(func):
        """Returns the number of positional parameters, or None if the function uses *args."""
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            return None
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                return None
        return sum(1 for p in sig.parameters.values()
                   if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                 inspect.Parameter.POSITIONAL_OR_KEYWORD))

class NumericTreeNode:
    """
    Basic implementation of a directed tree. Children are stored in a list.

    :param parent_node: The parent node of the new node. Must be a :code:`TreeNode`. Optional.
    """

    def __init__(self, interferometer, x, t, k, parent=None):
        """
        :param interferometer: The :code:`NumericBraggInterferometer` instance this node belongs to.
        :param x: The position of the node.
        :param t: The time of the node.
        :param k: The momentum of the node.
        :param parent: The parent node. Optional.
        """
        self.ifr = interferometer
        self.x = x
        self.t = t
        self.k = k
        self.parent = parent
        self.children = []

    def split(self, klattice, distance=4, filter_func=lambda kvec, k0, kf: True):
        """
        Splits the node based on Bragg diffraction, creating child nodes for the diffracted states.

        :param klattice: The lattice wavevector(s).
        :param distance: The number of momentum states to keep around the diffraction orders. Defaults to 4.
        :param filter_func: A function to filter which child nodes are created. Defaults to allowing all.
        """

        # Put klattice into a list if not already
        if not isinstance(klattice, (list, tuple, np.ndarray)):
            klattice = [klattice]

        # Determine the kstates to keep surrounding the initial state
        valid_kvec = np.arange(self.k-2*distance, self.k+2*(distance+1), 2)

        # Loop over each lattice
        for kl in klattice:
            
            # Find the value of k for the reflected state
            bragg_order = np.abs(kl - self.k)/2
            k_reflected = self.k + 4*bragg_order if self.k < kl else self.k - 4*bragg_order

            # Find the kstates around the reflected state
            sub_kvec_kreflected = np.arange(k_reflected-2*distance, k_reflected+2*(distance+1), 2)

            # Add to the valid_kvec
            valid_kvec = np.concatenate([valid_kvec, sub_kvec_kreflected])
        
        # Make a new child node for each new state
        for k in np.unique(valid_kvec):
            if k in self.ifr.kvec and filter_func(self.ifr.kvec, self.k, k):
                self.children.append(NumericTreeNode(self.ifr, x=self.x, t=self.t, k=k, parent=self))

    def propagate(self, T):
        """
        Propagates the node for a time :code:`T`, creating a single child node.

        :param T: The duration of propagation.
        """
        # Make a single child node with a position and time updated according to the propagation time
        self.children.append(NumericTreeNode(self.ifr, x=self.x + self.k*T, t=self.t + T, k=self.k, parent=self))

    def get_ancestry(self):
        """
        Returns a list of ancestor nodes from the root node to this node.

        :returns: A list of :code:`NumericTreeNode` objects.
        """
        # Returns a list of ancestory nodes nodes from this node to the root node
        path = []
        cnode = self
        while cnode.parent:
            path.append(cnode)
            cnode = cnode.parent
        path.append(cnode)
        path.reverse()
        return path
    
    def get_trajectory(self):
        """
        Returns the trajectory (time and position) of the node and its ancestors.

        :returns: A tuple :code:`(times, positions)` containing lists of times and positions.
        """
        # Returns the time and position of each note
        cnode = self
        times = []
        positions = []
        while cnode.parent:
            times.append(cnode.t)
            positions.append(cnode.x)
            cnode = cnode.parent
        times.append(cnode.t)
        positions.append(cnode.x)
        times.reverse()
        positions.reverse()
        return times, positions
    
    def get_wf_func(self, func_init):
        """
        Returns a function that computes the wavefunction at this node by tracing back through its ancestry and applying the operation functions.

        :param func_init: The function to compute the initial wavefunction.
        :returns: A function that computes the wavefunction.
        """

        # Get the ancestory nodes and the list of interferometer operations
        nodes = self.get_ancestry()
        ops = self.ifr.operations

        # Check that we have the proper number of nodes/operations
        if len(nodes) != len(ops) + 1:
            raise RuntimeError('Inconsistent number of nodes and operations')
        
        # Create a list to store function calls, loop over each operation and ancestor and determine function call
        func_calls = []
        for i in range(len(ops)):
            op_type, op_args = ops[i]
            op_func, kvfunc = self.ifr.operation_funcs[i]
            initial_node = nodes[i]
            final_node = nodes[i+1]

            if op_type == 'propagate':

                # Consistency checking
                if initial_node.k != final_node.k:
                    raise RuntimeError('initial and final k do not match!')

                # Extract needed parameters, construct function call
                k = initial_node.k
                t_init = initial_node.t
                t_final = final_node.t
                t = op_args[0]

                # Further consistency checking
                if not np.allclose(t_final - t_init, t):
                    raise RuntimeError('propagation time is inconsistent between nodes and operation')

                # Construct function call and store
                func_calls.append(('propagate', op_func, [t, k], 0, None))

            elif op_type == 'split':

                # Consistency checking
                if initial_node.t != final_node.t:
                    raise RuntimeError('initial and final t do not match!')
                if initial_node.x != final_node.x:
                    raise RuntimeError('initial and final x do not match!')

                # Extract needed parameters, construct function call
                x = initial_node.x
                t = initial_node.t
                k_init = initial_node.k
                k_final = final_node.k
                klattice = op_args[0]
                delta_k = k_final - k_init

                # Construct function call and store
                func_calls.append(('split', op_func, [k_init, k_final, klattice, t, x], delta_k, kvfunc))

        # Construct function that computes the wavefunction from each function call
        def calc_wf(comm_args):
            wf = func_init(*comm_args)
            k_extra = 0.0
            for func_call in func_calls:
                op_type, func, args, delta_k, kvector_func = func_call
                if op_type == 'split':
                    if kvector_func is not None:
                        dk = kvector_func(*comm_args)
                        k_extra += delta_k * dk
                    args2 = comm_args + args
                    wf *= func(*args2)
                elif op_type == 'propagate':
                    t, k = args
                    args2 = comm_args + [t, k + k_extra]
                    wf *= func(*args2)
            return wf

        # Return
        return calc_wf