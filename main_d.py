import os, time, pickle
import numpy as np
import tensorflow as tf
from utils import discount_with_dones, Scheduler, make_path, find_trainable_variables
from models import model, model2, model3, load, save
import scipy.sparse as sp
from sl_buffer_d import slBuffer_allFile
from mct_d import MCT
from status import Status

"""
    this function builds the model that is used by all three functions below
"""
def build_model(args, scope):
    nh = args.max_clause
    nw = args.max_var
    nc = 2
    nact = nc * nw
    ob_shape = (None, nh, nw, nc * args.nstack)
    X = tf.placeholder(tf.float32, ob_shape)
    Y = tf.placeholder(tf.float32, (None, nact))
    Z = tf.placeholder(tf.float32, (None))

    p, v = model3(X, nact, scope)
    params = find_trainable_variables(scope)
    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=p))
        value_loss = tf.losses.mean_squared_error(labels = Z, predictions = v)
        lossL2 = tf.add_n([ tf.nn.l2_loss(vv) for vv in params ])
        loss = cross_entropy + value_loss + args.l2_coeff * lossL2

    return X, Y, Z, p, v, params, loss


"""
    c_act (exploration parameter of MCTS) and num_mcts (the full size of MCTS tree) are determined in minisat.core.Const.h
    NOTE: max_clause, max_var and nc are define in both here (in args for model) and in minisat.core.Const.h (for writing states). 
    They need to BE the same.
    nbatch is the degree of parallel for neural net
    nstack is the number of history for a state
"""
def self_play(args, built_model, status_track):
    # take out the parts that self_play need from the model
    X, _, _, p, v, params, _ = built_model

    # within a tensorflow session, run MCT objects with model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_dir = status_track.get_model_dir()
        if (args.save_dir is not None) and (model_dir is not None):
            sess.run(load(params, os.path.join(args.save_dir, model_dir)))
            print("loaded model {} at dir {} for selfplay".format(args.save_dir, model_dir))
        else:
            # this is the initial random parameter! let's save it in hard drive!
            ps = sess.run(params)
            model_dir = status_track.generate_new_model()
            save(ps, os.path.join(args.save_dir, model_dir))

        # initialize a list of MCT and run self_play
        MCTList = []
        for i in status_track.get_nbatch_index(args.nbatch, args.n_train_files):
            MCTList.append(MCT(args.train_path, i, args.max_clause, args.max_var, args.nrepeat, 
                tau = lambda x: 1.0 if x <= 30 else 0.0001, resign = 400)) 
        pi_matrix = np.zeros((args.nbatch, 2 * args.max_var), dtype = np.float32)
        v_array = np.zeros((args.nbatch,), dtype = np.float32)
        needMore = np.ones((args.nbatch,), dtype = np.bool)
        while True:
            states = []
            pi_v_index = 0
            for i in range(args.nbatch):
                if needMore[i]:
                    temp = MCTList[i].get_state(pi_matrix[pi_v_index], v_array[pi_v_index])
                    pi_v_index += 1
                    if temp is None:
                        needMore[i] = False
                    else:
                        states.append(temp)
            if not np.any(needMore):
                break
            pi_matrix, v_array = sess.run([p, v], feed_dict = {X: np.asarray(states, dtype = np.float32)})

        print("loop finished and save Pi graph to slBuffer")
        # bring sl_buffer to memory
        os.makedirs(args.dump_dir, exist_ok = True)
        dump_trace = os.path.join(args.dump_dir, args.dump_file)
        if os.path.isfile(dump_trace):
            with open(dump_trace, 'rb') as sl_file:
                sl_Buffer = pickle.load(sl_file)
        else:
            sl_Buffer = slBuffer_allFile(args.sl_buffer_size, args.train_path, args.n_train_files)
        # write in sl_buffer
        for i in range(args.nbatch):
            MCTList[i].write_data_to_buffer(sl_Buffer)
        # write sl_buffer back to disk
        with open(dump_trace, 'wb') as sl_file:
            pickle.dump(sl_Buffer, sl_file, -1)

"""
    this function does supervised training
"""
def super_train(args, built_model, status_track):
    # take out the parts that self_play needs from the model
    X, Y, Z, _, _, params, loss = built_model
    with tf.name_scope("train"):
        if args.which_cycle == 0: lr = 1e-2
        else: lr = 1e-3
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_dir = status_track.get_sl_starter()
        assert (args.save_dir is not None) and (model_dir is not None), "save_dir and model_dir needs to be specified for super_training"
        sess.run(load(params, os.path.join(args.save_dir, model_dir)))
        print("loaded model {} at dir {} as super_training starter".format(args.save_dir, model_dir))

        # data for supervised training
        dump_trace = os.path.join(args.dump_dir, args.dump_file)
        with open(dump_trace, 'rb') as sl_file:
            sl_Buffer = pickle.load(sl_file)

        # supervised training cycle
        for i in range(args.sl_num_steps + 1):
            batch = sl_Buffer.sample(args.sl_nbatch)
            feed_dict = { X: batch[0], Y: batch[1], Z: batch[2] }
            sess.run(train_step, feed_dict)
            if i > 0 and i % args.sl_ncheckpoint == 0:
                new_model_dir = status_track.generate_new_model()
                print("checkpoint model {}".format(new_model_dir))
                ps = sess.run(params)
                save(ps, os.path.join(args.save_dir, new_model_dir))

"""
    this function evaluates all unevaluated model, as indicated in the status_track object
"""
def model_ev(args, built_model, status_track, ev_testing = False):
    # there may be a few number of unevaluated models, and this function evaluate them all
    model_dir = status_track.which_model_to_evaluate()
    if model_dir is None: return

    # add this layer of indirection so that the function is fit for both evaluating training files and testing files
    if ev_testing:
        sat_path = args.test_path
        sat_num  = args.n_test_files
    else:
        sat_path = args.train_path
        sat_num  = args.n_train_files

    # take out the parts that self_play needs from the model
    X, _, _, p, v, params, _ = built_model

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # may run this multiple times because there maybe multiple models to evaluate
        while model_dir is not None:
            sess.run(load(params, os.path.join(args.save_dir, model_dir)))
            print("loaded model {} at dir {} for evaluation".format(args.save_dir, model_dir))

            MCTList = []
            for i in range(args.nbatch):
                # tau is small for testing, and evaluation only solve a problem once.
                MCTList.append(MCT(sat_path, i, args.max_clause, args.max_var, 1, tau = lambda x: 0.001, resign = 400))
            pi_matrix = np.zeros((args.nbatch, 2 * args.max_var), dtype = np.float32)
            v_array = np.zeros((args.nbatch,), dtype = np.float32)
            needMore = np.ones((args.nbatch,), dtype = np.bool)
            next_file_index = args.nbatch
            assert (next_file_index <= sat_num), "this is a convention"
            all_files_done = next_file_index == sat_num
            performance = np.zeros(sat_num)
            while True:
                states = []
                pi_v_index = 0
                for i in range(args.nbatch):
                    if needMore[i]:
                        temp = MCTList[i].get_state(pi_matrix[pi_v_index], v_array[pi_v_index])
                        pi_v_index += 1
                        while temp is None:
                            idx, rep, scr = MCTList[i].report_performance()
                            performance[idx] = scr / rep
                            if all_files_done:
                                break
                            MCTList[i] = MCT(sat_path, next_file_index, args.max_clause, args.max_var, 1, tau = lambda x: 0.001, resign = 400)
                            next_file_index += 1
                            if next_file_index >= sat_num:
                                all_files_done = True
                            temp = MCTList[i].get_state(pi_matrix[pi_v_index-1], v_array[pi_v_index-1]) # the pi and v are not used (for new MCT object)
                        if temp is None:
                            needMore[i] = False
                        else:
                            states.append(temp)
                if not np.any(needMore):
                    break
                pi_matrix, v_array = sess.run([p, v], feed_dict = {X: np.asarray(states, dtype = np.float32)})

            # write performance to the status_track
            print(performance)
            status_track.write_performance(performance)
            model_dir = status_track.which_model_to_evaluate()

def ev_ss(args, built_model, status_track, file_no):
    model_dir = status_track.get_model_dir()
    sat_path = args.train_path
    sat_num = args.n_train_files

    if model_dir is None: return
    X, _, _, p, v, params, _ = built_model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(load(params, os.path.join(args.save_dir, model_dir)))
        print("load model {} at dir {}".format(args.save_dir, model_dir))
        MCT58 = MCT(sat_path, file_no, args.max_clause, args.max_var, 1, tau = lambda x: 0.9, resign = 80)
        pi_matrix = np.zeros((1, 2 * args.max_var, ), dtype = np.float32)
        v_array = np.zeros([1,], dtype = np.float32)
        while True:
            temp = MCT58.get_state(pi_matrix[0], v_array[0])
            if temp is None: break
            states = []
            states.append(temp)
            pi_matrix, v_array = sess.run([p, v], feed_dict = {X: np.asarray(states, dtype = np.float32)})
        idx, rep, scr = MCT58.report_performance()
        print("performance is {}".format(scr / rep))

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save_dir', type = str, help='where is the model saved', default="GPO5/mix20_120")
    parser.add_argument('--best_model', type = int, help="the index of the best model (-1 for unknown)", default = -1)
    parser.add_argument('--status_file', type = str, help="which file keeps a record of the status", default = "status.pkl")
    parser.add_argument('--result_file', type = str, help="this file keeps the performance of models on testing files", default = "result.pkl")
    parser.add_argument('--dump_dir', type = str, help="where to save (state, Pi, num_step) for SL", default = "parameters/")
    parser.add_argument('--dump_file', type = str, help="what is the filename to save (state, Pi, num_step) for SL", default="sl.pkl")
    parser.add_argument('--train_path', type = str, help='where are training files', default="satProb/mix20_train32/")
    parser.add_argument('--test_path', type = str, help='where are test files', default="satProb/mix20_test200/")

    parser.add_argument('--max_clause', type = int, help="what is the max_clause", default=120)
    parser.add_argument('--max_var', type = int, help="what is the max_var", default=20)
    parser.add_argument('--sl_buffer_size', type = int, help="max size of sl buffer", default = 1000000)
    parser.add_argument('--nbatch', type = int, help="what is the batch size to use", default = 32)
    parser.add_argument('--nstack', type = int, help="how many layers of states to use", default = 1)
    parser.add_argument('--nrepeat', type = int, help="how many times to repeat a SAT problem", default= 100)
    parser.add_argument('--n_start', type = int, help="which file index to start with (for running)", default = 0)
    parser.add_argument('--n_train_files', type = int, help="total number of training files", default = 0) # calculated later 
    parser.add_argument('--n_test_files', type = int, help="total number of testing files", default = 0) # calculated later

    parser.add_argument('--l2_coeff', type = float, help="the coefficient for l2 regularization", default = 0.0001)
    parser.add_argument('--sl_num_steps', type = int, help="how many times to do supervised training", default = 64000)
    parser.add_argument('--sl_nbatch', type = int, help="what is the batch size for supervised training", default = 32)
    parser.add_argument('--sl_ncheckpoint', type = int, help="how often to checkpoint a supervised trained model", default =32000)
    parser.add_argument('--n_cycles', type = int, help="how many cycles of self_play -> super_train -> model_ev do we want to run", default = 2)
    parser.add_argument('--show_only', type = str, help="if only show the result", default="No")
    parser.add_argument('--which_cycle', type = int, help="which cycle are we in now", default = 0)

    args = parser.parse_args()
    args.n_train_files = len([f for f in os.listdir(args.train_path) if os.path.isfile(os.path.join(args.train_path, f))]) # total number of training files
    args.n_test_files  = len([f for f in os.listdir(args.test_path)  if os.path.isfile(os.path.join(args.test_path,  f))]) # total number of testing files
    args.dump_dir = args.save_dir # all files related to this project are saved in save_dir, so dump_dir is useless
    os.makedirs(args.save_dir, exist_ok = True)

    # start the status_track for these operations
    status_track = Status()
    if os.path.isfile(os.path.join(args.save_dir, args.status_file)):
        status_track.start_with(os.path.join(args.save_dir, args.status_file))
    else: # otherwise the initial values in Status object fits with the default values here;
        status_track.init_with(args.best_model, args.n_start, [], 0, os.path.join(args.save_dir, args.status_file), args)
    status_track.show_itself()

    # following code evaluates the performance of models on testing files
    result_track = Status()
    if os.path.isfile(os.path.join(args.save_dir, args.result_file)):
        result_track.start_with(os.path.join(args.save_dir, args.result_file))
    else: # otherwise initilize values in Status object with the "total model number" --> "length_hist field" of status_track
        result_track.init_with(-1, 0, [], 0, os.path.join(args.save_dir, args.result_file))

    if args.show_only == "Yes":
       status_track.show_itself()
       status_track.print_all_models_performance()
       result_track.show_itself()
       result_track.print_all_models_performance()
       return

    # build the model for all three functions
    built_model = build_model(args, scope = "mcts")

    # run a specific file that has bugs
#    ev_ss(args, built_model, status_track, 0)
#    return
#    model_ev(args, built_model, status_track)
#    status_track.show_itself()
#    return
#    result_track.set_same_length_hist(status_track)
#    model_ev(args, built_model, result_track, ev_testing = True)
#    result_track.show_itself()
#    return

    # run args.n_cycles number of iteration (self_play -> super_train -> model_ev)
    for i in range(args.n_cycles):
        args.which_cycle = i
        self_play(args, built_model, status_track)
        status_track.show_itself()
        super_train(args, built_model, status_track)
        status_track.show_itself()
        model_ev(args, built_model, status_track)
        result_track.set_same_length_hist(status_track)
        model_ev(args, built_model, result_track, ev_testing = True)
        status_track.show_itself()

    # print the performance of all models we have so far:
    status_track.print_all_models_performance()
    result_track.print_all_models_performance()
    return

import sys
if __name__ == '__main__':
    main()
