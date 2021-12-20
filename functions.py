# AFFINE FUNCTION

def Affine(passed_values, weights, bias):
    
    import numpy as np
    
    return(np.dot(weights, passed_values) + bias)


# RELU FUNCTION

def ReLU(passed_values, weights, bias):
    
    import numpy as np
    
    return(np.maximum(np.zeros(bias.size), Affine(passed_values, weights, bias)))


# GET ε BY THE USER.

# The user provides an ε value. The length of the noise interval centred on each pixel will be 2ε.
def get_epsilon():

    epsilon = ""
    is_a_number = False
    
    # Ask for epsilon until a number is provided. Then is_a_number becomes true and the loop ends.
    while is_a_number == False:
        
        epsilon = input("The length of the noise interval centred on each pixel will be 2ε. Please, write a value for ε: ")
        
        try:
            float(epsilon)
            is_a_number = True
        except:
            print("ε must be a number.")

    return(float(epsilon))


# HAVE THE USER SAY IN WHICH LAYER TO STOP AND USE THE CONCRETE BOUNDS WHEN CALCULATING NEW UPPER AND LOWER BOUNDS.

def get_stop():

    stop = ""
    is_a_number = False
    
    # Ask for stop until a number is provided. Then is_a_number becomes true and the loop ends.
    while is_a_number == False:
        
        stop = input("Please, write the number of the layer you want to stop and take the concrete bounds at when calculating" + \
                     " the new upper and lower bounds. If you want maximum precision and thus to select the concrete bounds at " + \
                     "the input layer, you can write 0. If an invalid number is provided, it will be truncated to make it integer.")
        
        try:
            float(stop)
            is_a_number = True
        except:
            print("You need to provide a number.")
    
    stop = float(stop)
    if ((stop - int(stop)) != 0.5): stop = int(stop)
    
    return(stop)


# CREATE DICTIONARIES FOR THE UPPER AND LOWER BOUNDS AND THE POLYHEDRAL CONSTRAINTS.

def create_constraints_dictionaries(weights, activations):
    
    import numpy as np
    
    upper_lower_bounds = {}
    poly_constraints = {}

    upper_lower_bounds["layer0"] = 0 # This layer will be updated by the function update_layer_0.

    for layer in weights:

        # For each neuron in each layer, an array with 2 floats (initialized with zeros) is saved in upper_lower_bounds.
        # The first float will be used for the upper bound and the second for the lower bound.
        upper_lower_bounds[layer] = np.zeros((len(weights[layer]), 2)) # len(weights[layer]) = number of neurons in a layer.
        # poly_constraints[layer] will be a list containing a number of nested lists equal to the number of neurons in that
        # layer.The first and second inner lists of each neuron are meant to store respectively the coefficients of the upper 
        # and those of the lower relational polyhedral constraints.
        poly_constraints[layer] = [ [[],[]] for neuron in range(len(weights[layer]))]

        # If the activation function at layer x is ReLU, expand the representation of the network in the 2 constraints
        # dictionaries by creating layerx.5 after layer x.
        if (activations.get(layer) == "ReLU"):
            upper_lower_bounds[layer + ".5"] = np.zeros((len(weights[layer]), 2))
            poly_constraints[layer + ".5"] = [ [[],[]] for neuron in range(len(weights[layer]))]
            
    return([upper_lower_bounds, poly_constraints])

            
# UPDATE LAYER0

# Fill in the values for the input layer (layer 0) in the dictionary of upper and lower bounds.
def update_layer_0(picturex, input_dic, upper_lower_bounds, epsilon):
    
    import numpy as np

    upper_lower_bounds["layer0"] = np.zeros((len(input_dic.get(picturex)), 2))

    # Iterate over the number of neurons in the input layer.
    for i in range(0, len(input_dic.get(picturex))):
        # Calculate and fill in the upper bound of the ith neuron.
        upper_lower_bounds["layer0"][i][0] = input_dic.get(picturex)[i] + epsilon
        if (upper_lower_bounds["layer0"][i][0] > 1):
            upper_lower_bounds["layer0"][i][0] = 1
        elif (upper_lower_bounds["layer0"][i][0] < 0):
            upper_lower_bounds["layer0"][i][0] = 0
        # Calculate and fill in the lower bound of the ith neuron.
        upper_lower_bounds["layer0"][i][1] = input_dic.get(picturex)[i] - epsilon
        if (upper_lower_bounds["layer0"][i][1] > 1):
            upper_lower_bounds["layer0"][i][1] = 1
        elif (upper_lower_bounds["layer0"][i][1] < 0):
            upper_lower_bounds["layer0"][i][1] = 0
        
# CHECK ACTIVATION FUNCTIONS.

# Verify that the neural network only makes use of ReLU and Affine functions.
def check_activation_functions(activations_list):
    
    i = 0
    
    # The loop stops with i = len(activations_list) - 1 if and only if all activation functions are either ReLU or Affine.
    while ((i < len(activations_list)) and (activations_list[i] == "ReLU" or activations_list[i] == "Affine")):        
        i = i + 1
      
    if (i != (len(activations_list))):
        print("Invalid network provided: all functions must be either ReLU or Affine functions.")
        
        
# CALCULATE THE NUMBER OF THE PREVIOUS LAYER.

def calculate_previous_layer_number(layer_number, upper_lower_bounds):
    
    if ("layer" + str(layer_number - 0.5)) in upper_lower_bounds:
        previous_layer_number = layer_number - 0.5
    elif ("layer" + str(int(layer_number - 0.5))) in upper_lower_bounds:
        previous_layer_number = int(layer_number - 0.5)
    else:
        previous_layer_number = layer_number - 1 
    
    return(previous_layer_number)


# WORK OUT THE LIST OF CORRECTLY CLASSIFIED PICTURES.

def get_correctly_classified_pictures(correctly_classified, output_column):
    
    j = 0
    correctly_classified_pictures = {}
    correctly_classified_pictures_list = []

    for i in correctly_classified:

        j = j + 1

        if (i == 1):
            correctly_classified_pictures["picture" + str(j)] = output_column[j - 1] # output_column[j - 1] is the correct output neuron
    
    return(correctly_classified_pictures)


# SUM TWO NUMPY ARRAYS (AND MULTIPLY ONE FOR A COEFFICIENT) AFTER MAKING SURE THEY ARE OF THE SAME SIZE. SEPARATELY SUM THE LAST ELEMENTS.

def sum_arrays(a, b, coefficient):
    
    import numpy as np
    from copy import deepcopy
    
    a = deepcopy(a)
    b = deepcopy(b)
    
    last = coefficient * b[-1] + a[-1]
    
    a[-1] = 0
    b[-1] = 0
    
    if len(a) < len(b):
        a.resize(len(b))
    elif len(a) > len(b):
        b.resize(len(a))
                
    c = a + coefficient * b
    c[-1] = last
    
    return(c)


# UPDATE UPPER AND LOWER BOUND.

# Use back substitution.     
def update_upper_lower(layer, neuron, poly_constraints, upper_lower_bounds, stop):
    
    import numpy as np
    
    # Initialize variables that are not local in the while loop.
    upper = np.zeros(1)
    lower = np.zeros(1)
    coefficients_upper = poly_constraints[layer][neuron][0]
    coefficients_lower = poly_constraints[layer][neuron][1]
    constant_term_upper = coefficients_upper[-1]
    constant_term_lower = coefficients_lower[-1]
    
    # Calculate the number of the layer.
    layer_number = float([(s) for s in layer.split("r") if s != "laye"][0])
    if ((layer_number - int(layer_number)) != 0.5): layer_number = int(layer_number)
    
    previous_layer_number = calculate_previous_layer_number(layer_number, upper_lower_bounds)
    
    while (previous_layer_number > stop):
                
        previous_layer_number = calculate_previous_layer_number(layer_number, upper_lower_bounds)  
        
        # Work on the upper bound.
        index_counter = -1
        
        for coefficient in coefficients_upper[:-1]: # [:-1] because the last element is the bias (or constant term).
        
            index_counter = index_counter + 1           
            
            # If coefficient > 0, use the upper polyhedral constraints of the neurons of the previous layer to upgrade upper.
            # If coefficient < 0, use the lower polyhedral constraints instead. If coefficient = 0, no action is required.
            if coefficient > 0:
                upper = sum_arrays(upper, np.array(poly_constraints["layer" + str(previous_layer_number)][index_counter][0]), coefficient)
                
            elif coefficient < 0:
                upper = sum_arrays(upper, np.array(poly_constraints["layer" + str(previous_layer_number)][index_counter][1]), coefficient)

        constant_term_upper = constant_term_upper + upper[-1]
        coefficients_upper = upper
        upper = np.zeros((len(upper)))
                    
        # Work on the lower bound.
        index_counter = -1
        
        for coefficient in coefficients_lower[:-1]: # [:-1] because the last element is the bias (or constant term).
        
            index_counter = index_counter + 1
        
            # If coefficient > 0, use the lower polyhedral constraints of the neurons of the previous layer to upgrade lower.
            # If coefficient < 0, use the upper polyhedral constraints instead. If coefficient = 0, no action is required.
            if coefficient > 0:
                lower = sum_arrays(lower, np.array(poly_constraints["layer" + str(previous_layer_number)][index_counter][1]), coefficient)
                
            elif coefficient < 0:
                lower = sum_arrays(lower, np.array(poly_constraints["layer" + str(previous_layer_number)][index_counter][0]), coefficient)
        
        constant_term_lower = constant_term_lower + lower[-1]
        coefficients_lower = lower
        lower = np.zeros((len(lower)))
            
        layer_number = previous_layer_number
        
        previous_layer_number = calculate_previous_layer_number(previous_layer_number, upper_lower_bounds)

        
    # Operations at layer 1 if complete back substitution is done, or at the layer following the stop layer otherwise.
    
    # Work out the upper bound.
    index_counter = -1
    upper = 0

#coefficients_upper = np.array(coefficients_upper)
#coefficients_upper.resize(len(np.array(poly_constraints["layer" + str(layer_number)][0][0])))
    for coefficient in coefficients_upper[:-1]:
        
        index_counter = index_counter + 1
        
        if coefficient > 0:
            upper = upper + coefficient * upper_lower_bounds["layer" + str(previous_layer_number)][index_counter][0]
            
        elif coefficient < 0:
            upper = upper + coefficient * upper_lower_bounds["layer" + str(previous_layer_number)][index_counter][1]
            
        upper_lower_bounds[layer][neuron][0] = upper + constant_term_upper
    
    # Work on the lower bound.
    index_counter = -1
    lower = 0
#coefficients_lower = np.array(coefficients_lower)
#coefficients_lower.resize(len(np.array(poly_constraints["layer" + str(layer_number)][0][1])))
    for coefficient in coefficients_lower[:-1]:
        
        index_counter = index_counter + 1
# fare solo se upper_lower_bounds["layer0"][index_counter][1] esiste (index_counter potrebbe essere troppo grande)
        if coefficient > 0:
            lower = lower + coefficient * upper_lower_bounds["layer" + str(previous_layer_number)][index_counter][1]
            
        elif coefficient < 0:
            lower = lower + coefficient * upper_lower_bounds["layer" + str(previous_layer_number)][index_counter][0]
            
        upper_lower_bounds[layer][neuron][1] = lower + constant_term_lower        
        

# UPDATE ALL CONSTRAINTS OF CURRENT LAYER (AND ALSO OF RELU INTERMEDIATE LAYER IF IT IS THE CASE).

def propagate(layer, weights, poly_constraints, biases, upper_lower_bounds, activations, stop):
    
    from copy import deepcopy
    
    for neuron in range(len(weights[layer])):
        
        # This part is done both if the activation function is Affine or ReLU, since ReLU starts with Affine.
        # Coefficients of the upper relational polyhedral constraint are updated.
        poly_constraints[layer][neuron][0].extend(((weights.get(layer))[neuron]).tolist())
        poly_constraints[layer][neuron][0].append((biases.get(layer))[neuron])
        # Coefficients of the lower relational polyhedral constraint are updated.
        poly_constraints[layer][neuron][1] = deepcopy(poly_constraints[layer][neuron][0])
        
        # Update upper and lower bounds of neuron.
        update_upper_lower(layer, neuron, poly_constraints, upper_lower_bounds, stop)
        
        # ReLU only part: update layer.5.
        if (activations.get(layer) == "ReLU"):
            
            # Upper bound <= 0.
            if ((upper_lower_bounds.get(layer))[neuron][0] <= 0):
                
                # Set upper and lower bounds to zero.
                upper_lower_bounds[layer + ".5"][neuron][0] = 0
                upper_lower_bounds[layer + ".5"][neuron][1] = 0
                
                # Set all coefficients and the bias in the relational polyhedral constraints to 0.
                # Length of layer.5 = length of layer + the number of neurons in layer, which is len(weights[layer]).
                poly_constraints[layer + ".5"][neuron][0] = [0] * (len(poly_constraints[layer]) + 1)
                poly_constraints[layer + ".5"][neuron][1] = deepcopy(poly_constraints[layer + ".5"][neuron][0])
                    
            # Lower bound >= 0.
            elif ((upper_lower_bounds.get(layer))[neuron][1] >= 0):

                # Being x the next neuron and j the previous, the polyhedral constraints of x are j<=x<=j.
                poly_constraints[layer + ".5"][neuron][0] = [0] * (len(poly_constraints[layer]) + 1)
                poly_constraints[layer + ".5"][neuron][0][neuron] = 1
                poly_constraints[layer + ".5"][neuron][1] = [0] * (len(poly_constraints[layer]) + 1)
                poly_constraints[layer + ".5"][neuron][1][neuron] = 1
                
                # Upper and lower bounds are the same as in the previous neuron.
                upper_lower_bounds[layer + ".5"][neuron][0] = upper_lower_bounds[layer][neuron][0]
                upper_lower_bounds[layer + ".5"][neuron][1] = upper_lower_bounds[layer][neuron][1]
                
            # Lower bound < 0 and upper bound > 0.
            else:
                
                # The naming is the same used in the paper.
                lamda = (upper_lower_bounds[layer][neuron][0]) / \
                        (upper_lower_bounds[layer][neuron][0] - upper_lower_bounds[layer][neuron][1])
                mu = - (upper_lower_bounds[layer][neuron][1] * upper_lower_bounds[layer][neuron][0]) / \
                     (upper_lower_bounds[layer][neuron][0] - upper_lower_bounds[layer][neuron][1])
                                                
                # Choose approximation with the least area.
                if (upper_lower_bounds[layer][neuron][0] <= - upper_lower_bounds[layer][neuron][1]):
                    
                    # The lower polyhedral constraint is set to 0, whereas the upper polyhedral constraint is lamda * x + mu.
                    poly_constraints[layer + ".5"][neuron][1] = [0] * (len(poly_constraints[layer]) + 1)
                    # The upper polyhedral constraint will contain all zeros, except the bias (mu) and the coefficient of the 
                    # previous neuron (lamda). Therefore, it starts by putting into poly_constraints[layer + ".5"][neuron][0] a number
                    # of zeros equal to the position of the neuron in its layer, minus 2.
                    poly_constraints[layer + ".5"][neuron][0].extend([0] * neuron)
                    # Append lamda, append zeros for all coefficients still missing and eventually append the bias (mu).
                    poly_constraints[layer + ".5"][neuron][0].append(lamda)
                    poly_constraints[layer + ".5"][neuron][0].extend([0] * (len(poly_constraints[layer]) + 1 - neuron - 2))
                    poly_constraints[layer + ".5"][neuron][0].append(mu)
                                                                 
                    # The lower bound is set to 0, while the upper bound is the same as the one of the previous neuron.
                    upper_lower_bounds[layer + ".5"][neuron][0] = upper_lower_bounds[layer][neuron][0]
                    upper_lower_bounds[layer + ".5"][neuron][1] = 0                                             
                                                                 
                else:
                    
                    # Being x the next neuron and j the previous, the lower polyhedral constraint of x is j<=x.
                    poly_constraints[layer + ".5"][neuron][1] = [0] * (len(poly_constraints[layer]) + 1)
                    poly_constraints[layer + ".5"][neuron][1][neuron] = 1
                    # The upper polyhedral constraint is lamda * x + mu. The code used and commented right above is reused.
                    poly_constraints[layer + ".5"][neuron][0].extend([0] * neuron)
                    poly_constraints[layer + ".5"][neuron][0].append(lamda)
                    poly_constraints[layer + ".5"][neuron][0].extend([0] * (len(poly_constraints[layer]) + 1 - neuron - 2))
                    poly_constraints[layer + ".5"][neuron][0].append(mu)
                    
                    # The upper and lower bounds are the same as the ones of the previous neuron.
                    upper_lower_bounds[layer + ".5"][neuron][0] = upper_lower_bounds[layer][neuron][0]
                    upper_lower_bounds[layer + ".5"][neuron][1] = upper_lower_bounds[layer][neuron][1]
                    

# CREATE NEW VARIABLES IN AN EXTRA LAYER AND APPLY THE ABSTRACT TRANSFORMER. 

# Each new variable will have as polyhedral constraints the correct neuron minus one of the other neurons. With 10 variables in the output
# layer, there would be 9 new variables in the last layer.
def create_extra_layer(upper_lower_bounds, poly_constraints, picture, correctly_classified_pictures, stop):
    
    from copy import deepcopy
    import numpy as np
    
    # Calculate the number of the output layer.
    output_layer_number = float([(s) for s in (list(upper_lower_bounds)[-1]).split("r") if s != "laye"][0])
    if ((output_layer_number - int(output_layer_number)) != 0.5): output_layer_number = int(output_layer_number)
        
    # Create the extra layer.
    extra_layer = "layer" + str(output_layer_number + 1)
    poly_constraints[extra_layer] = \
        [ [[],[]] for n in range(len(poly_constraints["layer" + str(output_layer_number)]) - 1)]
    upper_lower_bounds[extra_layer] = \
        np.zeros((len(poly_constraints["layer" + str(output_layer_number)]) - 1, 2))
        
    # Update the polyhedral constraints of the extra layer.
     
    # Iterate over all neurons of the extra layer.
    for e_i in range(len(upper_lower_bounds[extra_layer])):
                
        # Iterate over all neurons of the ouput layer.
        for o_i in range(len(upper_lower_bounds["layer" + str(output_layer_number)])):
            
            # In correspondence with the correctly classified neuron, the coefficient must be one.
            if (o_i == correctly_classified_pictures[picture]):
                poly_constraints[extra_layer][e_i][0].extend([1])
            # In correspondence with all other neurons, the coefficient is 0.
            else:
                poly_constraints[extra_layer][e_i][0].extend([0])
        
        # One of the coefficients must be -1.
        added_minus_one = False
        for j in range(e_i, len(poly_constraints[extra_layer][e_i][0])):
            if e_i == 0:                
                if ((poly_constraints[extra_layer][e_i][0][j] != 1) and (not added_minus_one)):
                    poly_constraints[extra_layer][e_i][0][j] = -1
                    added_minus_one = True
            elif ((poly_constraints[extra_layer][e_i - 1][0][j] != -1) and (poly_constraints[extra_layer][e_i][0][j] != 1) \
                  and (not added_minus_one)):
                poly_constraints[extra_layer][e_i][0][j] = -1
                added_minus_one = True
            
        # The constant term is 0.
        poly_constraints[extra_layer][e_i][0].extend([0])
        
        # The upper and lower polyhedral constraints are equal.
        poly_constraints[extra_layer][e_i][1] = deepcopy(poly_constraints[extra_layer][e_i][0])
        
        # Update upper and lower bounds.
        update_upper_lower(extra_layer, e_i, poly_constraints, upper_lower_bounds, stop)
        
    return(extra_layer)
    

# CHECK THE SPECIFICATION.

def check_specification(upper_lower_bounds, correctly_classified_pictures, picture, counter, correct_array, poly_constraints, stop):
    
    # Get lower bound of the correct neuron (lbcn).
    lbcn = upper_lower_bounds[list(upper_lower_bounds)[-1]][correctly_classified_pictures[picture]][1]
    # list(upper_lower_bounds)[-1] is the the last key of upper_lower_bounds.
    
    # If the lower bound of the correct neuron is greater than the upper bound of all other neurons in the output layer, then
    # the picture is classified correctly.
    
    correct = 0
    
    # Iterate over all neurons of the output layer.
    for neuron in upper_lower_bounds[list(upper_lower_bounds)[-1]]:
        
        if (lbcn > neuron[0]): # neuron[0] is the upper bound of neuron.
            correct = correct + 1 
            
    if (correct == len(upper_lower_bounds[list(upper_lower_bounds)[-1]])):
        correct_array[counter] = 1
        
    # Otherwise, create new variables in a last layer and apply the abstract transformer.
    
    else:        
        extra_layer = create_extra_layer(upper_lower_bounds, poly_constraints, picture, correctly_classified_pictures, stop)        

    correct = 0

    # If the lower bound of all neurons of the extra layer is greater than 0, the picture is classified correctly.
    for neuron in range(len(upper_lower_bounds[extra_layer])):
        
        if (upper_lower_bounds[extra_layer][neuron][1] > 0):
            correct = correct + 1
            
    if (correct == len(upper_lower_bounds[extra_layer])):
        correct_array[counter] = 1