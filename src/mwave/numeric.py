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

    def get_nodes(self):
        """
        Returns a dictionary of nodes at the current level, organized by momentum and position.

        :returns: A dictionary where keys are momentum states, values are dictionaries of positions, and values of those dictionaries are lists of nodes at that momentum and position.
        """
        # Make a dictionary of momentum states. Values are dictionaries of positions. Values of those dictionaries are lists of nodes at that momentum and position.
        nodedict = {}
        for node in self.current_level:
            if node.k not in nodedict:
                nodedict[node.k] = {}
            if node.x not in nodedict[node.k]:
                nodedict[node.k][node.x] = [node]
            else:
                nodedict[node.k][node.x].append(node)
        return nodedict
    
    def set_operation_funcs(self, funcs):
        """
        Sets the functions used to numerically evaluate the operations (split, propagate) performed on the interferometer.

        :param funcs: A list of functions corresponding to the operations performed on the interferometer, in order.
        """
        # Store the supplied function list in the NumericBraggInterferometer object
        self.operation_funcs = funcs

    def get_population_func(self, momentums, func_pop_init, func_wf_init, func_wf2_init):
        """
        Returns a function that computes the population for a given set of output momentums.

        :param momentums: A list of output momentums to compute the population for.
        :param func_pop_init: A function that computes the initial population.
        :param func_wf_init: A function that computes the initial wavefunction.
        :param func_wf2_init: A function that computes the initial wavefunction (secondary, used in calculation).
        :returns: A function that takes :code:`comm_args` and returns the total population at the specified momentums.
        """
        # Get dictionary of momentums
        nodedict = self.get_nodes()

        # Create dictionary to store function call lists
        func_dict = {}

        # Loop through and construct a function call for each output momentum
        for momentum in momentums:

            # If momentum is actually output continue
            if momentum in nodedict:
                
                # For each momentum loop over all output positions
                popfuncs = []
                for position in nodedict[momentum]:

                    # Within each position get the wavefunction evaluation function
                    wffuncs = []
                    for node in nodedict[momentum][position]:
                        wffuncs.append(node.get_wf_func(func_wf_init))

                    # Append all of these functions to the population function list
                    popfuncs.append(wffuncs)
                    
                func_dict[momentum] = popfuncs

        # Create functions
        def calc_pops(momentum, comm_args):
            pop = func_pop_init(*comm_args)
            for flst in func_dict[momentum]:
                wf = func_wf2_init(*comm_args)
                for f in flst:
                    wf += f(comm_args)
                pop += np.abs(wf)**2
            return pop

        # Return
        return calc_pops

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
            op_func = self.ifr.operation_funcs[i]
            initial_node = nodes[i]
            final_node = nodes[i+1]
            
            if op_type == 'propagate':

                # Consistency checking
                if initial_node.k != final_node.k:
                    raise RuntimeError('initial and final k do not match!')

                # Extract needed parameters, construct function call
                k = initial_node.k
                x_init = initial_node.x
                t_init = initial_node.t
                x_final = final_node.x
                t_final = final_node.t
                t = op_args[0]
                
                # Further consistency checking
                if not np.allclose(t_final - t_init, t):
                    raise RuntimeError('propagation time is inconsistent between nodes and operation')

                # Construct function call and store
                func_calls.append((op_func, [t, k]))
                
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
                
                # Construct function call and store
                func_calls.append((op_func, [k_init, k_final, klattice, t, x]))

        # Construct function that computes the wavefunction from each function call
        def calc_wf(comm_args):
            wf = func_init(*comm_args)
            for func_call in func_calls:
                func, args = func_call
                args2 = comm_args + args
                wf *= func(*args2)
            return wf

        # Return
        return calc_wf