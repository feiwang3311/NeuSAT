import os, time, pickle
import numpy as np
import scipy.sparse as sp
from MCTSminisat.minisat.gym.GymSolver import sat

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_PI(counts, tau):
    p = 1 / tau
    if p == 1.0:
        Pi = counts / np.sum(counts)
    elif p < 500:
        counts = counts / counts.sum()
        # do it in small steps to prevent underflow
        while p >= 10:
            counts = np.power(counts, 10)
            p = p / 10
            counts = counts / counts.sum()
        Pi = np.power(counts, p)
        Pi = Pi / np.sum(Pi)
    else:
        # assume that tau is infinitely small
        Pi = np.zeros(np.shape(counts), dtype = np.float32)
        Pi[np.argmax(counts)] = 1.0
    return Pi

class Pi_struct(object):
    """
        inner class used by MCT class. It forms a tree structure and cache states, Pi, and other values for self play
    """
    def __init__(self, size, level, file_no, tau, parent = None):
        """
            size:    the size of Pi, which is the same as nact 
            level:   the level is the number of steps to reach this node
            file_no: the index of this file in the file list (this is only used in error message)
            tau:     the function that returns a proper tau value given the current level
            parent:  the parent node of this Pi_struct (None for the root of this tree)
        """
        self.size = size
        self.level = level
        self.file_no = file_no
        self.tau = tau
        self.parent = parent
        self.state = None # state is None at construction time
        
        # stateful part of this node
        self.children = {}   # pointers to all children of this node (indexed by next_act. MAYBE added by branch_next)
        self.repeat = 0      # how many times this node has been played (incremented by add_counts function)
        self.total_steps = 0 # how many total_steps for this node's play (incremented by prop_up_steps func)
        self.Pi = np.zeros(self.size, dtype = np.float32) # this is an average of all Pis played (updated by add_counts)
        
    def add_state(self, state):
        """
            take the state of this Pi_struct, save it as sparse matrix, and also compute the isValid array
        """
        if self.state is not None: return
        self.isValid = np.reshape(np.any(state, axis = 0), [self.size,])
        state_2d = np.reshape(state, [-1, state.shape[1] * state.shape[2]])
        self.state = sp.csc_matrix(state_2d)
    
    def add_counts(self, counts):
        """
            take the counts (of MCTS simulation from this state), save Pi, and return the sampled move
        """
        assert counts.sum() == (counts * self.isValid).sum(), "count: " + str(counts) + \
        " is invalid: " + str(self.isValid) + " in file " + str(self.file_no)
        temp_Pi = get_PI(counts, self.tau(self.level)) 
        
        assert (self.isValid * temp_Pi).sum() > 0.999999, "Pi: " + str(temp_Pi) + \
        " is invalid: " + str(self.isValid) + " in file " + str(self.file_no)
        action = np.random.choice(range(self.size), 1, p = temp_Pi)[0]

        # before returining action, average temp_Pi into self.Pi. Increment self.repeat
        self.Pi = (self.Pi * self.repeat + temp_Pi) / (self.repeat + 1)
        self.repeat += 1
    
        return action

    def branch_next(self, next_action):
        """
            this function MAYBE initialize a child at the "next_action" branch and return that child
        """
        if next_action not in self.children: 
            self.children[next_action] = Pi_struct(self.size, self.level + 1, self.file_no, self.tau, parent = self)
        return self.children[next_action]

    def prop_up_steps(self, steps):
        """
            this function prop up steps for this play, starting from the last Pi_struct node
        """
        self.total_steps += steps
        if self.parent is not None:
            self.parent.prop_up_steps(steps)

class MCT(object):
    def __init__(self, file_path, file_no, max_clause1, max_var1, nrepeat, tau, resign = 1000000):
        """
            file_path:   the directory to files that are used for training
            file_no:     the file index that this object works on (each MCT only focus on one file problem)
            max_clause1: the max_clause that should be passed to the env object
            max_var1:    the max_var that should be passed to the env object
            nrepeat:     the number of repeats that we want to self_play with this file problem (suggest 100)
            tau:         the function that, given the current number of step, return a proper tau value
            resign:      the steps to declare terminate (to save computation for very louzy self_play)
        """
        self.env = sat(file_path, max_clause = max_clause1, max_var = max_var1) 
        self.file_no = file_no
        self.state = self.env.resetAt(file_no) 
        # IMPORTANT: all reset call should use the resetAt(file_no) function to make sure that it resets at the same file
        if self.state is None: # extreme case where the SAT problem is solved by simplification
            self.Pi_root = None
            self.phase = None
        else: # normal case: set up!
            self.Pi_current = self.Pi_root = Pi_struct(max_var1 * 2, 0, file_no, tau) 
            self.Pi_current.add_state(self.state)
            self.resign = resign 
            self.nrepeats = nrepeat # need to run so many repeat for this SAT problem
            self.working_repeat = 0 # this is the 0th run. (it is incremented everytime "isDone" or "resign")
            self.phase = False 
            # phase False is "initial and normal running" phase, 
            # phase True is "pause and return state" phase
            # pahse None is "the problem is finished" phase

    def get_state(self, pi_array, v_value):
        """
            main logic function:
            pi_array: the pi array evaluated by neural net (when phase is False, this paramete is not used)
            v_value:  the v value evaluated by neural net  (when phase is False, this paramete is not used)
            Return a state (3d numpy array) if paused for evaluation.
            Return None if this problem is simulated nrepeat times (all required repeat times are finished)
        """
        if self.phase is None: return None

        # loop for the simulation
        needEnv = True
        while needEnv or needSim:
            if needEnv:
                if not self.phase:
                    self.phase = True
                    return self.state
                else:
                    self.phase = False
            self.state, needEnv, needSim = self.env.simulate(softmax(pi_array), v_value)

        # after simulation, save counts and make a step
        next_act = self.Pi_current.add_counts(self.env.get_visit_count())
        isDone, self.state = self.env.step(next_act) 
        assert isDone or np.any(self.state), "Error: should be isDone or state is not empty"
        if isDone or self.Pi_current.level >= self.resign:
            # update the steps taken for this play
            self.Pi_current.prop_up_steps(self.Pi_current.level)
            self.working_repeat += 1
            if self.working_repeat >= self.nrepeats:
                self.phase = None
                return None
            self.Pi_current = self.Pi_root
            self.state = self.env.resetAt(self.file_no)
        else:
            self.Pi_current = self.Pi_current.branch_next(next_act)
            self.Pi_current.add_state(self.state)

        # route back to the start of the function
        return self.get_state(pi_array, v_value)

    def write_data_to_buffer(self, sl_Buffer):
        if self.Pi_root is None: return # there is nothing to write
        sl_Buffer.add_from_Pi_structs(self.Pi_root)
        return 
        # sub-routine to save in buffer
        def analyze_Pi_graph_dump(Pi_node, sl_Buffer):
            for act in Pi_node.children:
                analyze_Pi_graph_dump(Pi_node.children[act], sl_Buffer)
            av = Pi_node.total_steps / Pi_node.repeat 
            sl_Buffer.add(Pi_node.file_no, Pi_node.state, Pi_node.Pi, av, Pi_node.repeat)
        # call subroutine
        analyze_Pi_graph_dump(self.Pi_root, sl_Buffer)

    def report_performance(self):
        if self.Pi_root is None:
            return self.file_no, 1, 1
        return self.file_no, self.Pi_root.repeat, self.Pi_root.total_steps
