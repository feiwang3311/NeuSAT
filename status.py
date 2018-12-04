import pickle
import numpy as np

class Status(object):
	def __init__(self):
		"""
			create a Status object that keeps track of the status of MCTS RL algorithm
			NOTE: any changes to this Status object is updated in hardware by pickle dump
			NOTE: a new instance of Status can be created by reading from a pickle dump
		"""
		self.best_model = -1     # the index of the best model (if -1, means no model available)
		self.n_start = 0         # the index of the training file to start with for self_play
		self.ev_hist = []        # the list of evaluation history for all models
		self.length_hist = 0     # the length of models generated (or to evaluate)
		# NOTE: if length_hist is larger than len(ev_hist), it means some number of models have not been evaluated
		self.status_file = None  # the file name of this object
		self.args = None         # the arguments list about this status file

	def retrete(self):
		"""
			this function decrease length_hist and ev_hist to the best model
		"""
		self.length_hist = self.best_model + 1
		del self.ev_hist[self.length_hist :]
		self.write_to_disc(update_hist = True)

	def reset_ev(self, resetTo):
		"""
			this function reset ev_hist to resetTo
		"""
		self.ev_hist = self.ev_hist[:resetTo]
		self.write_to_disc(update_hist = True)

	def reset_n_start(self, n_start):
		"""
			this function reset n_start value
		"""
		self.n_start = n_start
		self.write_to_disc()

	def reset_best_model(self, best_model):
		"""
			this function reset best_model
		"""
		self.best_model = best_model		
		self.write_to_disc()

	def reset_length_hist(self, length_hist):
		"""
			this function reset length_hist (only used if the model is not saved completely)
		"""
		self.length_hist = length_hist
		self.write_to_disc()

	def reset_args(self, args):
		"""
			this function reset the args field
		"""
		self.args = args
		self.write_to_disc()

	def set_same_length_hist(self, other):
		""" 
			this function set the same length_hist has the "other"
		"""
		self.length_hist = other.length_hist
		self.write_to_disc() 

	def self_check(self):
		if self.best_model == -1: 
			assert len(self.ev_hist) == 0, "when self.best_model is -1, self.ev_hist should be empty"
		assert self.best_model <= len(self.ev_hist), "self.best_model should be less than or equal to len(self.ev_hist)"
		assert len(self.ev_hist) <= self.length_hist, "self.ev_hist should have length less than or equal to self.length_hist"

	def write_to_disc(self, update_hist = False):
		"""
			this method write the model to disc at self.status_file (every change should be updated) 
			TODO : optimize so that the vast content of ev_hist is not unnecessarily updated
		"""
		with open(self.status_file, "wb") as f:
			pickle.dump((self.best_model, self.n_start, self.length_hist, self.status_file, self.args), f)
		if update_hist:
			with open(self.status_file + ".hist", "wb") as d:
				pickle.dump(self.ev_hist, d)

	def start_with(self, status_file):
		"""
			this method fill the fields of Status object with information stored in a status pickle dump
		"""
		with open(status_file, "rb") as f:
			self.best_model, self.n_start, self.length_hist, self.status_file, self.args = pickle.load(f)
		with open(status_file + ".hist", "rb") as d:
			self.ev_hist = pickle.load(d)
		self.self_check()
	
	def init_with(self, best_model, n_start, ev_hist, length_hist, status_file, args = None):
		"""
			this method initialize the fields of Status object with given parameters 
		"""
		self.best_model = best_model
		self.n_start = n_start
		self.ev_hist = ev_hist
		self.length_hist = length_hist
		self.status_file = status_file
		self.args = args
		self.self_check()
		self.write_to_disc(update_hist = True)

	def get_model_dir(self):
		"""
			this method returns the dir of the best model as indicated by self.best_model (or None of best_model == -1)
		"""
		if self.best_model == -1:
			return None
		return "model-" + str(self.best_model)

	def get_nbatch_index(self, nbatch, ntotal):
		"""
			this method update n_start field, with the help of nbatch and ntotal, and return the correct indexes of batch
		"""
		indexes = np.asarray(range(self.n_start, self.n_start + nbatch)) % ntotal
		self.n_start += nbatch
		self.n_start %= ntotal
		self.write_to_disc()
		return indexes

	def get_sl_starter(self):
		"""
			this method returns the starting model of the supervised learning
		"""
		assert self.length_hist > 0, "at supervised training stage, there should exist at least one model"
		# NOTE: if recent two models are worse, instead of better, use the "best_model" as the starting model for sl 
		if self.length_hist - 1 <= self.best_model + 2:
			return "model-" + str(self.length_hist - 1)
		else:
			return "model-" + str(max(self.best_model, 0))
	
	def generate_new_model(self):
		"""
			this function is used by sl_train (or the initial phase of self_play) to put more models in hard drive
			it returns the new model dir name for this new model
		"""
		self.length_hist += 1
		if self.best_model == -1:
			assert self.length_hist == 1, "this should be the first model"
			self.best_model = 0
		self.write_to_disc()
		return "model-" + str(self.length_hist - 1)

	def which_model_to_evaluate(self):
		"""
			this function returns the dir name of the model to be evaluated
			it returns None if no such model dir exists
		"""
		index = len(self.ev_hist)
		if index < self.length_hist:
			return "model-" + str(index)
		else:
			return None

	def write_performance(self, performance):
		"""
			this function report the performance of the last evaluated model, then compare with the best model and possibally update it
		"""
		self.ev_hist.append(performance)
		if self.best_model == -1:
			assert len(self.ev_hist) == 1, "this must be the first model evaluated"
			self.best_model = 0
		else:
			if self.better_than(performance, self.ev_hist[self.best_model]):
				self.best_model = len(self.ev_hist) - 1
		self.write_to_disc(update_hist = True)

	def better_than(self, per1, per2):
		if (per1 <= per2).sum() >= per1.shape[0] * 0.95 and np.mean(per1) < np.mean(per2) * 0.99:
			return True
		if (per1 <= per2).sum() >= per1.shape[0] * 0.65 and np.mean(per1) < np.mean(per2) * 0.95:
			return True
		if (per1 <= per2).sum() >= per1.shape[0] * 0.50 and np.mean(per1) < np.mean(per2) * 0.90:
			return True
		return False

	def show_itself(self):
		"""
			this function print the information in this object
		"""
		print("best_model is {}".format(self.best_model))
		print("n_start is {}".format(self.n_start))
		print("ev_hist has length {}".format(len(self.ev_hist)))
		print("length_hist is {}".format(self.length_hist))
		print("status_file is {}".format(self.status_file))
		if self.args is not None:
			print("args is {}__{}__{}".format(self.args.save_dir, self.args.train_path, self.args.test_path))

	def print_all_models_performance(self):
		"""
			this function print the performance of all models (all average values in ev_hist)
		"""
		for i in range(len(self.ev_hist)):
			print(np.mean(self.ev_hist[i]), end = ", ")
		print("\n")

import sys
if __name__ == '__main__':
        print("in the main function of status")
        st = Status()
        st.start_with(sys.argv[1])
        st.show_itself()
        st.print_all_models_performance()
        if (len(sys.argv) == 5):
                st.reset_best_model(int(sys.argv[2]))
                st.reset_n_start(int(sys.argv[3]))
                st.reset_ev(int(sys.argv[4]))
                st.show_itself()
        if (len(sys.argv) == 6):
                st.reset_best_model(int(sys.argv[2]))
                st.reset_n_start(int(sys.argv[3]))
                st.reset_ev(int(sys.argv[4]))
                st.reset_length_hist(int(sys.argv[5]))
                st.show_itself()
