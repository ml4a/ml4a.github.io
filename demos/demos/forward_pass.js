function demo(parent, width, height)
{
	// setup canvas
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');
	
	// settings
	var c1 = 'rgba(0,0,0,0.7)';
	var c2 = 'rgba(0,0,0,1.0)';
	var t1 = 1.5;
	var t2 = 3;	

	var settings = {
	    context: ctx,
	    width: 700, 
	    height: 480,
	    architecture: [3, 2, 1],
	    visible: [3, 2, 1],
	    neuronStyle: {
	        color: c2,
	        thickness: t1,
	        radius: 50,
	        labelSize: 32,
	        biasLabelSize: 16,
	    },
	    connectionStyle: {
	        color: c1,
	        arrowLen: 12,
	        arrowWidth: 5,
	        thickness: t1,
	        labelSize: 16,
	        labelLerp: 0.02
	    }
	};

	// simulate neural net
	function initialize_network(correct_values) {
	    a = settings.architecture;
	    input = [2.4,1.2,1.3];
	    if (correct_values) {
			weights = [[[0.1,0.9], [0.6,0.7], [0.9,-0.2]], [[0.5], [0.4]]];
			biases = [[0.0,0.0,0.0], [0.4,-0.5], [0.7]];
	    } else {
			weights = [[[0.9,0.8], [0.4,0.5], [0.2,0.7]], [[0.3], [0.9]]];
			biases = [[0.0,0.0,0.0], [0.2,0.6], [0.3]];
	    }
	    acts = [...Array(a.length).keys()].map(i => [...Array(a[i]).keys()].map(k => 0));
	    z = [...Array(a.length).keys()].map(i => [...Array(a[i]).keys()].map(k => 0));
	    acts[0] = input;

	    // forward pass
	    for (var l=1; l<a.length; l++) {
	        for (var n=0; n<acts[l].length; n++) {
	            z[l][n] = 0;
	            for (var n1=0; n1<acts[l-1].length; n1++) {
	                z[l][n] += weights[l-1][n1][n] * acts[l-1][n1];
	            }
	            z[l][n] += biases[l][n];
	            acts[l][n] = 1.0 / (1.0 + Math.exp(-z[l][n])); // sigmoid
	        }
	    }
	};

	// create visualization
	var idx = -1;
	var a, inputs, weights, biases, acts, z;
	var net = new NetworkVisualization(settings);
	net.setHeightBounds([0.07,0.93],1);
	initialize_network();

	// convenience function for steps below
	function set_all_weights_and_labels_visible(){
	    for (var l=1; l<a.length; l++) {
	        for (var n=0; n<a[l]; n++) {
	            net.setNeuronStyle({
	                biasLabelText: 'b = '+biases[l][n].toFixed(1),
	                biasLabelColor: c1}, l, n);
	        }
	    }
	    for (var l=0; l<a.length-1; l++) {
	        for (var n1=0; n1<weights[l].length; n1++) {        
	            for (var n2=0; n2<weights[l][n1].length; n2++) {
	                net.setConnectionStyle({labelText: 'w = '+weights[l][n1][n2].toFixed(1)}, l, n1, n2);
	            }
	        }
	    }
	};

	function display_hidden_calculation(l, n) {
	    net.setNeuronStyle({
            color: c1,
            thickness: t1});
        net.setConnectionStyle({
            labelText:'', 
            color: c1,
            thickness: t1});
        net.setNeuronStyle({
            labelText: acts[l][n].toFixed(2), 
            color: c2,
            thickness: t2}, l, n);
        for (var n2=0; n2<acts[l-1].length; n2++){
            net.setNeuronStyle({
                color: c2,
                thickness: t2}, l-1, n2);
            net.setConnectionStyle({
                labelText:'w = '+weights[l-1][n2][n].toFixed(1), 
                color: c2,
                thickness: t2}, l-1, n2, n);
        }
	};

	function display_everything() {
		net.setNeuronStyle({
            color: c1,
            thickness: t1});
		for (var l=0; l<a.length; l++) {
		    for (var n=0; n<a[l]; n++) {
		    	net.setNeuronStyle({
	                labelText: acts[l][n].toFixed(2), 
	                color: c2,
	                thickness: t2}, l, n);
		    	if (l>0) {
		            for (var n2=0; n2<acts[l-1].length; n2++){
		                net.setNeuronStyle({
		                    color: c2,
		                    thickness: t2}, l-1, n2);
		                net.setConnectionStyle({
		                    labelText:'w = '+weights[l-1][n2][n].toFixed(2), 
		                    color: c2,
		                    thickness: t2}, l-1, n2, n);
		            }
		        }
	        }
		}
	}


	function display_values_layer1() {
		net.setNeuronStyle({labelText:'', thickness: t1});
	    net.setConnectionStyle({thickness: t1});
        for (var n=0; n<a[1]; n++) {
            net.setNeuronStyle({
                biasLabelText: 'b = '+biases[1][n].toFixed(1),
                biasLabelColor: c1}, 1, n);
        }
        for (var n1=0; n1<weights[0].length; n1++) {        
            for (var n2=0; n2<weights[0][n1].length; n2++) {
                net.setConnectionStyle({labelText: 'w = '+weights[0][n1][n2].toFixed(1)}, 0, n1, n2);
            }
        }
    };

	function display_values_layer2() {
		net.setNeuronStyle({labelText:'', thickness: t1});
	    net.setConnectionStyle({thickness: t1});
        net.setNeuronStyle({
            biasLabelText: 'b = '+biases[2][0].toFixed(1),
            biasLabelColor: c1}, 2, 0);
        net.setConnectionStyle({labelText: 'w = '+weights[1][0][0].toFixed(1)}, 1, 0, 0);
        net.setConnectionStyle({labelText: 'w = '+weights[1][1][0].toFixed(1)}, 1, 1, 0);
    };

	// animation steps
	var steps = [];

	
	// segment 0 
	steps.push(function(){
	    set_text_panel('\
			Suppose we have a dataset with one point: \
			$$x=\\begin{bmatrix} 2.4 & 1.2 & 1.3 \\end{bmatrix} \\qquad y=0.854$$\
			We can attempt to fit a 3x2x1 neural network with sigmoid activation functions, as seen below.');
	});

	// segment 1 (1 step): annotate all net labels
	steps.push(function(){
	    initialize_network(false);
	    display_values_layer1();
	    set_text_panel('\
			Let\'s try a set of random weights and biases. For the first hidden layer, let\'s say our weights and biases are: \
			$$w=\\begin{bmatrix} 0.9 & 0.4 & 0.2 \\\\ 0.8 & 0.5 & 0.7 \\end{bmatrix} \\qquad b=\\begin{bmatrix} 0.2 & 0.6 \\end{bmatrix} $$');
	});

	steps.push(function(){
		display_values_layer2();
		set_text_panel('\
			And for the output layer, let\'s say the weights are: \
			$$ \
			w=\\begin{bmatrix} \
			0.3 & 0.9 \
			\\end{bmatrix} \
			$$\
			And we let the final output neuron\'s bias be $b=0.3$');
	});

	steps.push(function(){
		for (var n=0; n<acts[0].length; n++) {
	        net.setNeuronStyle({labelText: acts[0][n].toFixed(2)}, 0, n);
	    }
		set_text_panel('\
			Let\'s input our datapoint $x=\\begin{bmatrix}2.4 & 1.2 & 1.3\\end{bmatrix}$ and see what output the network gives us.');
	});

	steps.push(function(){
		display_hidden_calculation(1, 0);
		set_text_panel('\
			The output of the first (top-most in the graphic) hidden unit is: \
			$$ z = (0.9 \\cdot 2.4) + (0.4 \\cdot 1.2) + (0.2 \\cdot 1.3) + 0.2 = 3.1 $$ \
			$$ \\sigma(3.1) = 0.96 $$');
	});
	
	steps.push(function(){
		display_hidden_calculation(1, 1);
		set_text_panel('\
			The output of the second hidden unit is: \
			$$ z = (0.9 \\cdot 2.4) + (0.4 \\cdot 1.2) + (0.2 \\cdot 1.3) + 0.2 = 4.03 $$ \
			$$ \\sigma(4.03) = 0.98 $$');
	});
	
	steps.push(function(){
		display_hidden_calculation(2, 0);
		set_text_panel('\
			Now we take these and feed it into the output unit (which doesn\'t have an activation function): \
			$$ z = (0.3 \\cdot 0.96) + (0.9 \\cdot 0.98) + 0.3 = 1.47 $$ \
			$$ \\sigma(1.47) = 0.81 $$');
	});
	
	steps.push(function(){
		set_text_panel('\
			Not quite right (we wanted to get $0.854$)...we can measure our error with the mean squared error (MSE), which is the most common measurement for error in regression problems: \
			$$ \
			\\begin{aligned} \
			\\text{error} &= (0.81 - 0.854)^2 \\\\ \
			\\text{error} &= 0.106 \
			\\end{aligned} \
			$$');
	});
	
	steps.push(function(){
		initialize_network(true);
		display_values_layer1();
		display_values_layer2();		
		set_text_panel('\
			Now I\'ll magically give you the best set of weights. For the hidden layer: \
			$$ \
			w=\\begin{bmatrix} \
			0.1 & 0.6 & 0.9 \\\\ \
			0.9 & 0.7 & -0.2 \
			\\end{bmatrix} \\qquad b=\\begin{bmatrix}0.4 -0.5\\end{bmatrix}\
			$$ \
			And for the output layer: \
			$$ \
			w=\\begin{bmatrix} \
			0.5 & 0.4 \
			\\end{bmatrix} \\qquad b=0.7 \
			$$');
	});

	steps.push(function(){
		display_everything();
		set_text_panel('\
			Let\'s try this all again with these new weights: \
			$$ \
			\\begin{aligned} \
			\\text{hidden unit 1} &= \\text{sigmoid}((0.1 * 2.4) + (0.6 * 1.2) + (0.9 * 1.3)) = 0.975 \\\\ \
			\\text{hidden unit 2} &= \\text{sigmoid}((0.9 * 2.4) + (0.7 * 1.2) + (-0.2 * 1.3)) = 0.917 \\\\ \
			\\text{output} &= (0.5 * 0.975) + (0.4 * 0.917) = 0.854 \
			\\end{aligned} \
			$$');
	});

	steps.push(function(){
		set_text_panel('\
			Voilà! We got the answer we wanted - so the weights of the network effectively control what it outputs.');
	});

	/*
	// segment 4 (1 step): show everything
	steps.push(function(){
	    net.setNeuronStyle({thickness: t1});
	    net.setConnectionStyle({thickness: t1});
	    set_all_weights_and_labels_visible();
	});
	*/

	// control flow
	function prev() {
	    idx = (idx + steps.length - 1) % steps.length;
	    steps[idx]();
	    redraw();
	};

	function next() {
	    idx = (idx + 1) % steps.length;
	    steps[idx]();
	    redraw();
	};

	function redraw() {
		ctx.clearRect(0, 0, canvas.width, canvas.height);
	    net.draw(5, 5);
	}
	
	// add control panels
	add_control_panel_action('prev', prev);
	add_control_panel_action('next', next);

	// start
	next();	
};
