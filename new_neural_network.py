import math
from random import uniform


#Deals with the behaviour for each layer of the neural network
class Layer:
    def __init__(self, nodes_in, nodes_out):
        #Initialise variables
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out

        self.weights, self.biases = self.create_initial()
        self.cost_gradient_weights, self.cost_gradient_biases = self.create_initial()
        
    def create_initial(self):
        #Create random weights and biases
        min_value, max_value = -1, 1

        biases = [0 for _ in range(self.nodes_out)]
        weights = [uniform(min_value, max_value) / math.sqrt(self.nodes_in) for _ in range(self.nodes_in * self.nodes_out)]

        return weights, biases

    def apply_gradients(self, learn_rate):
        #Apply the gradient descent algorithm
        for i in range(len(self.weights)):
            gradient = self.cost_gradient_weights[i]
            self.weights[i] -= gradient * learn_rate
            self.cost_gradient_weights[i] = 0

        for i in range(len(self.biases)):
            gradient = self.cost_gradient_biases[i]
            self.biases[i] -= gradient * learn_rate
            self.cost_gradient_biases[i] = 0
        
    def update_gradients(self, node_values):
        #Update the stored gradients for each weight and bias
        for i in range(self.nodes_out):
            node_value = node_values[i]
            for x in range(self.nodes_in):
                der_cost_weight = self.inputs[x] * node_value
                self.cost_gradient_weights[i * self.nodes_in + x] += der_cost_weight

        for i in range(self.nodes_out):
            node_value = node_values[i]
            der_cost_bias = node_value
            self.cost_gradient_biases[i] += der_cost_bias

    def output_node_values(self, expected_outputs):
        #Calculate a partial derivative for gradient descent (cost with respect to weighted inputs for output nodes)
        node_values = []
        for i in range(len(expected_outputs)):
            cost_derivative = self.cost_derivative(self.activations[i], expected_outputs[i])
            activation_derivative = self.activation_derivative(self.weighted_inputs, i)
            node_values.append(cost_derivative * activation_derivative)

        return node_values

    def hidden_node_values(self, old_layer, old_node_values):
        #Calculate a partial derivative for gradient descent (cost with respect to weighted inputs for hidden nodes)
        values = []
        for i in range(self.nodes_out):
            new_value = 0
            for x in range(len(old_node_values)):
                weighted_input_der = old_layer.get_weight(i, x)
                new_value += weighted_input_der * old_node_values[x]

            new_value *= self.activation_derivative(self.weighted_inputs, i)
            values.append(new_value)

        return values

    def calculate_outputs(self, inputs):
        #Use weights and biases to determine the output for each neurone in the layer
        self.inputs = [i for i in inputs]
        self.weighted_inputs = [0 for _ in range(self.nodes_out)]

        for i in range(self.nodes_out):
            weighted_input = self.biases[i]

            for x in range(self.nodes_in):
                weighted_input += self.get_weight(x, i) * inputs[x]

            self.weighted_inputs[i] = weighted_input

        self.activations = [self.activation_function(self.weighted_inputs, i) for i in range(len(self.weighted_inputs))]
        return self.activations

    def get_weight(self, node_in, node_out):
        #Return the particular weight of a connection given its index
        index = node_out * self.nodes_in + node_in
        return self.weights[index]

    def activation_function(self, inputs, index):
        #The sigmoid activation function
        try:
            return 1 / (1 + math.exp(-inputs[index]))
        except OverflowError:
            return 0

    def activation_derivative(self, inputs, index):
        #The derivative of the sigmoid function
        activation = self.activation_function(inputs, index)
        return activation * (1 - activation)

    def cost_derivative(self, predicted_output, expected_output):
        #The derivative of the cost function
        return predicted_output - expected_output


#Deals with the behaviour for the entire neural network
class Neural_network:
    def __init__(self, learn_rate, batch_size, layer_sizes):
        #Initialise variables
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.layers = self.create_layers(layer_sizes)

    def create_layers(self, layer_sizes):
        #Create the appropriate number of layers and add them to an array
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

        return layers

    def classify(self, inputs):
        #Use one-hot encoding to return the predicted class
        outputs = self.calculate_output(inputs)
        max_value = max(outputs)
        predicted_class = outputs.index(max_value)

        return predicted_class, outputs

    def cost(self, predicted_outputs, expected_outputs):
        #Calculate the total cost of the network
        cost = 0
        for i in range(len(predicted_outputs)):
            cost += self.cost_function(predicted_outputs[i], expected_outputs[i])

        return cost

    def average_cost(self, inputs, expected_outputs):
        #Calculate the average cost of the network
        total_cost = 0
        for i in range(len(inputs)):
            total_cost += self.cost(inputs[i], expected_outputs[i])

        return total_cost / len(inputs)

    def calculate_output(self, inputs):
        #Use forward propagation to calculate the output for the entire network
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)

        return inputs

    def learn(self, training_inputs, training_outputs):
        #Use the training data to train the network
        for i, x in enumerate(training_inputs):
            self.update_layer_gradients(x, training_outputs[i])

        for i in self.layers:
            i.apply_gradients(self.learn_rate / len(training_inputs))

    def gradients_weights(self, layer, training_inputs, training_outputs, h, og_cost):
        # -------- Not actually used within the program. Just here for explanation -------- #
        #Inefficient way to calculate the cost landscape for the weights 
        for i in range(layer.current_nodes):
            for x in range(layer.prev_nodes):
                layer.weights[x][i] += h
                cost_change = self.average_cost(training_inputs, training_outputs) - og_cost
                layer.weights[x][i] -= h
                layer.cost_gradient_weights[x][i] = cost_change / h

    def gradients_biases(self, layer, training_inputs, training_outputs, h, og_cost):
        # -------- Not actually used within the program. Just here for explanation -------- #
        #Inefficient way to calculate the cost landscape for the biases 
        for i in range(layer.current_nodes):
            layer.biases[i] += h
            cost_change = self.average_cost(training_inputs, training_outputs) - og_cost
            layer.biases[i] -= h
            layer.cost_gradient_biases[i] = cost_change / h

    def update_layer_gradients(self, input, output):
        #Forward propagation to get overall cost
        next_inputs = input
        for layer in self.layers:
            next_inputs = layer.calculate_outputs(next_inputs)

        #Backpropagation to get individual costs
        output_layer = self.layers[-1]
        node_values = output_layer.output_node_values(output)
        output_layer.update_gradients(node_values)

        for i in reversed(range(len(self.layers) - 1)):
            hidden_layer = self.layers[i]
            node_values = hidden_layer.hidden_node_values(self.layers[i + 1], node_values)
            hidden_layer.update_gradients(node_values)

    #The cost function used (mean squared error)
    cost_function = lambda self, output, expected_output: 0.5 * (output - expected_output)**2


def mini_batch_descent(network, training_inputs, training_outputs):
    #Use mini batches instead of the full training data for efficiency
    for i in range(len(training_inputs) // network.batch_size):
        inputs = training_inputs[network.batch_size * i : network.batch_size * (i + 1)]
        outputs = training_outputs[network.batch_size * i : network.batch_size * (i + 1)]

        network.learn(inputs, outputs)