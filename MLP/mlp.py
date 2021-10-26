import copy
import numpy as np
import math
import random
from sklearn.model_selection import train_test_split

class MLP():
	def __init__(self,max_iterations=500,shuffle=True, step_size = 0.1, no_improvement_break = None, hidden_nodes = 5, porportion_train=0.8, initial_weights_zero = False, momentum=0.2):
		self.max_iterations = max_iterations
		self.no_improvement_break = no_improvement_break
		self.shuffle=shuffle
		self.step_size = step_size
		self.porportion_train = porportion_train
		self.initial_weights_zero = initial_weights_zero
		self.momentum = momentum
		self.epoch_number = 0
		self.rmse = []
		self.training_rmse = []
		self.best_rmse = None
		self.best_epoch = 0
		self.best_weights = None
		self.hidden_nodes = hidden_nodes
	def run_epoch(self):
		self.epoch_number += 1
		def get_z(list_of_numbers):
			output_list = []
			for number in list_of_numbers:
				output_list.append((1/(1+math.exp(-number))))
			return output_list

		if self.shuffle:
			random.shuffle(self.train_data)
		for entry in self.train_data:
			first_hidden_net_values = np.matmul(entry[0], self.weight_matrices[0])
			first_hidden_z_values = get_z(first_hidden_net_values) +[1] # add the bias weight
			output_net_values = np.matmul(first_hidden_z_values, self.weight_matrices[1])
			output_z_values = get_z(output_net_values)
			delta_outputs = []
			for output_z_value, output_correct_value in zip(output_z_values, entry[1]):
				delta_outputs.append((output_correct_value-output_z_value)*output_z_value*(1-output_z_value))
			first_hidden_deltas = []
			for weights, z_value in zip(self.weight_matrices[1][:-1], first_hidden_z_values[:-1]):
				tmp_value = 0
				for hidden_weight, delta_value in zip(weights, delta_outputs):
					tmp_value += hidden_weight*delta_value
				individual_delta = tmp_value*z_value*(1-z_value)
				first_hidden_deltas.append(individual_delta)

			# now calculate the new weights
			change_in_hidden_to_output = np.outer(first_hidden_z_values, delta_outputs)*self.step_size
			new_hidden_to_output = change_in_hidden_to_output+self.weight_matrices[1]+self.momentum*self.previous_change[1]
			self.weight_matrices[1] = new_hidden_to_output
			self.previous_change[1] = change_in_hidden_to_output
			change_in_input_to_hidden = np.outer(entry[0], first_hidden_deltas)*self.step_size
			new_input_to_hidden = change_in_input_to_hidden+self.weight_matrices[0]+self.momentum*self.previous_change[0]
			self.weight_matrices[0] = new_input_to_hidden
			self.previous_change[0] = change_in_input_to_hidden
		# Now that the epoch has run, find out the root mean squared error to determine if it needs to run again
		sse = 0
		for entry in self.test_data:
			first_hidden_net_values = np.matmul(entry[0], self.weight_matrices[0])
			first_hidden_z_values = get_z(first_hidden_net_values) +[1] # add the bias weight
			output_net_values = np.matmul(first_hidden_z_values, self.weight_matrices[1])
			output_z_values = get_z(output_net_values)
			for output_z_value, output_correct_value in zip(output_z_values, entry[1]):
				sse += (output_correct_value - output_z_value)**2
		mse = sse/(len(self.test_data)*len(self.test_data[0][1]))
		individual_rmse = math.sqrt(mse)
		self.rmse.append(individual_rmse)
		if not self.best_rmse or individual_rmse < self.best_rmse:
			self.best_rmse = individual_rmse
			self.best_epoch = copy.deepcopy(self.epoch_number)
			self.best_weights = copy.deepcopy(self.weight_matrices)
		# Save the training RMSE values as well. These will not be used as stopping criteria
		training_sse = 0
		for entry in self.train_data:
			first_hidden_net_values = np.matmul(entry[0], self.weight_matrices[0])
			first_hidden_z_values = get_z(first_hidden_net_values) +[1] # add the bias weight
			output_net_values = np.matmul(first_hidden_z_values, self.weight_matrices[1])
			output_z_values = get_z(output_net_values)
			for output_z_value, output_correct_value in zip(output_z_values, entry[1]):
				training_sse += (output_correct_value - output_z_value)**2
		training_mse = training_sse/(len(self.train_data)*len(self.train_data[0][1]))
		training_individual_rmse = math.sqrt(training_mse)
		self.training_rmse.append(training_individual_rmse)


	def fit(self, input_values, input_labels):
		label_to_matrix_dictionary = {}
		label_set = sorted(set(input_labels))
		self.label_set = label_set
		length = len(label_set)
		zero_vector = [0]*length
		for i, label in enumerate(label_set):
			tmp_vector = copy.deepcopy(zero_vector)
			tmp_vector[i] = 1
			label_to_matrix_dictionary[label] = tmp_vector

		values_train, values_test, labels_train, labels_test = train_test_split(input_values, input_labels, train_size=self.porportion_train)
		# Change out the labels to matrices
		label_matrices_train = []
		for entry in labels_train:
			label_matrices_train.append(label_to_matrix_dictionary[entry])
		label_matrices_test = []
		for entry in labels_test:
			label_matrices_test.append(label_to_matrix_dictionary[entry])
		# Now add a bias weight of 1 to each input entry and make them into tuples so that they can be shuffled together
		train_data = []
		for value_matrix, label_matrix in zip(values_train, label_matrices_train):
			train_data.append((np.append(value_matrix, [1]), label_matrix)) # Adding the 1 for bias
		test_data = []
		for value_matrix, label_matrix in zip(values_test, label_matrices_test):
			test_data.append((np.append(value_matrix, [1]), label_matrix)) # Adding the 1 for bias

		self.train_data = train_data
		self.test_data = test_data

		# Now build the weight matrices
		self.weight_matrices = []
		self.weight_matrices.append(np.random.rand(len(self.train_data[0][0]), self.hidden_nodes))
		self.weight_matrices.append(np.random.rand(self.hidden_nodes + 1, len(self.train_data[0][1])))
		# save the change in weights so that you can change the momentum
		# same shape as the weight_matrices, except everything needs to start as zero
		self.previous_change = copy.deepcopy(self.weight_matrices)
		for i in range(len(self.previous_change)):
			for j in range(len(self.previous_change[i])):
				for k in range(len(self.previous_change[i][j])):
					self.previous_change[i][j][k] = 0
		if self.initial_weights_zero:
			self.weight_matrices = copy.deepcopy(self.previous_change)
			# Just for BYU class, start everything with zero instead of the random values

		while self.epoch_number < self.max_iterations:
			if self.no_improvement_break and (self.epoch_number - self.no_improvement_break) >= self.best_epoch:
				# If it has a specified no improvement break number
				# And it has not improved within the last set number of epochs, stop running
				break
			self.run_epoch()


	def predict(self, test_values):
		def get_z(list_of_numbers):
			output_list = []
			for number in list_of_numbers:
				output_list.append((1/(1+math.exp(-number))))
			return output_list

		test_values_bias_added = []
		for entry in test_values:
			test_values_bias_added.append(np.append(entry,[1]))
		output_z_values = []
		for entry in test_values_bias_added:
			first_hidden_net_values = np.matmul(entry, self.weight_matrices[0])
			first_hidden_z_values = get_z(first_hidden_net_values) + [1] # add the bias weight
			output_net_values = np.matmul(first_hidden_z_values, self.weight_matrices[1])
			output_z_values.append(get_z(output_net_values))
		output_classes = []
		for individual_matrix in output_z_values:
			class_index_value = individual_matrix.index(max(individual_matrix))
			output_classes.append(self.label_set[class_index_value])
		return output_classes


	def confidence_scores(self, test_values):
		def get_z(list_of_numbers):
			output_list = []
			for number in list_of_numbers:
				output_list.append((1/(1+math.exp(-number))))
			return output_list

		test_values_bias_added = []
		for entry in test_values:
			test_values_bias_added.append(np.append(entry,[1]))
		output_z_values = []
		for entry in test_values_bias_added:
			first_hidden_net_values = np.matmul(entry, self.weight_matrices[0])
			first_hidden_z_values = get_z(first_hidden_net_values) + [1] # add the bias weight
			output_net_values = np.matmul(first_hidden_z_values, self.weight_matrices[1])
			output_z_values.append(get_z(output_net_values))
		self.output_z_values = output_z_values
		confidence_matrix = []
		for entry in output_z_values:
			individual_confidence_matrix = []
			sum_of_values = sum(entry)
			for individual_value in entry:
				individual_confidence_matrix.append(individual_value/sum_of_values)
			confidence_matrix.append(individual_confidence_matrix)
		return confidence_matrix
