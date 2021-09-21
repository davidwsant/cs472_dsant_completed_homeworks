import random 
class Perceptron():
	def __init__(self,max_iterations=500,shuffle=True, step_size = 1, no_improvement_break = 25):
		self.max_iterations = max_iterations
		self.no_improvement_break = no_improvement_break
		self.shuffle=shuffle
		self.step_size = step_size
		self.epoch_number = 0
		self.accuracies = []
		self.best_accuracy = 0
		self.best_epoch = 0

	def run_epoch(self):
		if self.shuffle:
			random.shuffle(self.modified_data)
		for entry in self.modified_data:
			target_value = entry[1]
			net_value = 0
			output_value = 0

			for input_value, weight in zip(entry[0], self.weights):
				net_value += input_value*weight
			if net_value >= 0: # I chose 0 as the threshold
				output_value = 1
			if output_value != target_value: # only update the weights if it got it wrong
				weight_change = []
				for input_value in entry[0]:
					weight_change.append(self.step_size*input_value*(target_value - output_value))
				new_weights = []
				for old_weight, difference in zip(self.weights, weight_change):
					new_weights.append(old_weight + difference)
				self.weights = new_weights
		# test accuracy at the end of each epoch
		number_correct = 0
		total_tests = len(self.modified_data)
		for entry in self.modified_data:
			target_value = entry[1]
			net_value = 0
			output_value = 0
			for input_value, weight in zip(entry[0], self.weights):
				net_value += input_value*weight
			if net_value >= 0:
				output_value = 1
			if output_value == target_value:
				number_correct += 1

		accuracy = number_correct/total_tests
		self.accuracies.append(accuracy)
		if accuracy > self.best_accuracy:
			self.best_accuracy = accuracy
			self.best_weights = self.weights
			self.best_epoch = self.epoch_number
		self.epoch_number += 1 # This will say how many epochs have run, but needs to update after updating best epoch

	def fit(self, train_values, train_labels):
		train_data = []
		for input_values, input_label in zip(train_values,train_labels):
			train_data.append((input_values, input_label))
		self.train_data = train_data
		self.modified_data = []
		length_weights = len(self.train_data[0][0])+1 # requires that all data be the same length
		self.weights = [0]*length_weights
		self.best_weights = [0]*length_weights # this will update each time accuracy gets better
		for entry in self.train_data:
			self.modified_data.append((entry[0]+[1], entry[1])) # adding the bias weight
		while self.epoch_number < self.max_iterations: # Only run if it has not reached the max iterations
			if (self.epoch_number - self.no_improvement_break) >= self.best_epoch: # If it has not improved within the last 10 epochs, stop running
				break
			self.run_epoch()

	def predict(self, test_values):
		self.test_values = test_values
		self.y_pred = []
		for entry in self.test_values:
			net_value = 0
			output_value = 0
			for individual_value, weight in zip(entry+[1], self.best_weights): # added the 1 for the bias weight
				net_value += individual_value*weight
			if net_value >= 0:
				output_value = 1
			self.y_pred.append(output_value)
		return self.y_pred

	def score(self, test_values, test_labels):
		self.test_values = test_values
		self.test_labels = test_labels
		self.y_pred = []
		for entry in self.test_values:
			net_value = 0
			output_value = 0
			for individual_value, weight in zip(entry+[1], self.best_weights): # added the 1 for the bias weight
				net_value += individual_value*weight
			if net_value >= 0:
				output_value = 1
			self.y_pred.append(output_value)
		correct = 0
		total = len(self.test_labels)
		for y_label, correct_label in zip(self.y_pred, self.test_labels):
			if y_label == correct_label:
				correct += 1
		accuracy = correct/total
		return accuracy

	def get_weights(self):
		return self.best_weights


def test_train_split(input_values, input_labels, porportion_train):
	paired_values_labels = []
	for values, labels in zip(input_values,input_labels):
		paired_values_labels.append((values, labels))
	random.shuffle(paired_values_labels)
	total_length = len(paired_values_labels)
	number_train = round(porportion_train*total_length)
	train_dataset = paired_values_labels[:number_train]
	test_dataset = paired_values_labels[number_train:]
	train_values = []
	train_labels = []
	for pair in train_dataset:
		train_values.append(pair[0])
		train_labels.append(pair[1])
	test_values = []
	test_labels = []
	for pair in test_dataset:
		test_values.append(pair[0])
		test_labels.append(pair[1])
	return train_values, train_labels, test_values, test_labels
