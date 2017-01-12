/*
 - make sure fake y != actual y
 - act layer in last layer?
 - reformat text

*/

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
	        labelText: ''
	    },
	    connectionStyle: {
	        color: c1,
	        arrowLen: 12,
	        arrowWidth: 5,
	        thickness: t1,
	        labelSize: 16,
	        labelLerp: 0.02,
	        labelText: ''
	    }
	};

	function forward_pass(use_correct_weights) {
		if (use_correct_weights) {
	    	weights = correct_weights
	    	biases = correct_biases
	    } else {
			weights = [[...Array(3).keys()].map(i => [...Array(2).keys()].map(i => Math.round(-10+20*Math.random())/10)), [...Array(2).keys()].map(i => [...Array(1).keys()].map(i => Math.round(-10+20*Math.random())/10))];
	    	biases = [[...Array(3).keys()].map(i => Math.round(-10+20*Math.random())/10), [...Array(2).keys()].map(i => Math.round(-10+20*Math.random())/10), [...Array(1).keys()].map(i => Math.round(-10+20*Math.random())/10)];
	    }
	    acts = [...Array(a.length).keys()].map(i => [...Array(a[i]).keys()].map(k => 0));
	    z = [...Array(a.length).keys()].map(i => [...Array(a[i]).keys()].map(k => 0));
	    acts[0] = input;
	    for (var l=1; l<a.length; l++) {
	        for (var n=0; n<acts[l].length; n++) {
	            z[l][n] = 0;
	            for (var n1=0; n1<acts[l-1].length; n1++) {
	                z[l][n] += weights[l-1][n1][n] * acts[l-1][n1];
	            }
	            z[l][n] += biases[l][n];
	            z[l][n] = Math.round(100.0*z[l][n])/100; // round
	            acts[l][n] = 1.0 / (1.0 + Math.exp(-z[l][n])); // sigmoid
	            acts[l][n] = Math.round(100.0*acts[l][n])/100; // round
	        }
	    }
	};

	// simulate neural net
	function initialize_network() {
	    a = settings.architecture;
	    input = [Math.round(30.0*Math.random())/10, Math.round(30*Math.random())/10, Math.round(30*Math.random())/10];
	    correct_weights = [[...Array(3).keys()].map(i => [...Array(2).keys()].map(i => Math.round(-10+20*Math.random())/10)), [...Array(2).keys()].map(i => [...Array(1).keys()].map(i => Math.round(-10+20*Math.random())/10))]
	    correct_biases = [[...Array(3).keys()].map(i => Math.round(-10+20*Math.random())/10), [...Array(2).keys()].map(i => Math.round(-10+20*Math.random())/10), [...Array(1).keys()].map(i => Math.round(-10+20*Math.random())/10)]
		forward_pass(true);
	    y_correct = Math.round(100.0*acts[acts.length-1][0])/100;
	};

	function get_mse(x1, x2) {
		return (x2-x1)*(x2-x1);
	};

	// create visualization
	var idx = -1;
	var a, inputs, correct_weights, correct_biases, y_correct;
	var weights, biases, acts, z;
	var net = new NetworkVisualization(settings);
	net.setHeightBounds([0.07,0.93],1);

	// initialize
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
		                    labelText:'w = '+weights[l-1][n2][n].toFixed(1), 
		                    color: c2,
		                    thickness: t2}, l-1, n2, n);
		            }
		        }
	        }
		}
	};

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
	steps.push({
		action: function(){
			acts[2][0] = y_correct;
			while (get_mse(acts[2][0], y_correct) < 0.1) {	// make sure to pick weights whose forward pass is actually sufficiently incorrect
				forward_pass(false);
			};
			set_text_panel(parent.description_panel_div, 'Suppose we have a dataset with one point: \
				$$x=\\begin{bmatrix} '+input[0].toFixed(1)+' & '+input[1].toFixed(1)+' & '+input[2].toFixed(1)+' \\end{bmatrix} \\qquad y='+y_correct.toFixed(2)+'$$\
				We can attempt to fit a 3x2x1 neural network with sigmoid activation functions, as seen below.', true);
		},
		draw: function() {},
	});

	// segment 1 (1 step): annotate all net labels
	steps.push({
		action: function(){
			set_text_panel(parent.description_panel_div, '\
				Let\'s try a set of random weights and biases. For the first hidden layer, let\'s say our weights and biases are: \
				$$w=\\begin{bmatrix} '+weights[0][0][0].toFixed(1)+' & '+weights[0][1][0].toFixed(1)+' & '+weights[0][2][0].toFixed(1)+' \\\\ '+weights[0][0][1].toFixed(1)+' & '+weights[0][1][1].toFixed(1)+' & '+weights[0][2][1].toFixed(1)+' \\end{bmatrix} \\qquad b=\\begin{bmatrix} '+biases[1][0].toFixed(1)+' & '+biases[1][1].toFixed(1)+' \\end{bmatrix} $$', true);
		},
		draw: function() {
		    display_values_layer1();
		}
	});

	steps.push({
		action: function() {
			set_text_panel(parent.description_panel_div, '\
				And for the output layer, let\'s say the weights are: \
				$$ \
				w=\\begin{bmatrix} \
				'+weights[1][0][0]+' & '+weights[1][1][0].toFixed(1)+' \
				\\end{bmatrix} \
				$$\
				And we let the final output neuron\'s bias be $b='+biases[2][0].toFixed(1)+'$', true);
		},
		draw: function() {
			display_values_layer2();
		}
	});

	steps.push({
		action: function() {
			set_text_panel(parent.description_panel_div, '\
				Let\'s input our datapoint $x=\\begin{bmatrix}'+input[0].toFixed(1)+' & '+input[1].toFixed(1)+' & '+input[2].toFixed(1)+'\\end{bmatrix}$ and see what output the network gives us.', true);
		},
		draw: function() {
			for (var n=0; n<acts[0].length; n++) {
		        net.setNeuronStyle({labelText: acts[0][n].toFixed(2)}, 0, n);
		    }
		}
	});

	steps.push({
		action: function() {
			set_text_panel(parent.description_panel_div, '\
				The output of the first (top-most in the graphic) hidden unit is: \
				$$ z = ('+weights[0][0][0].toFixed(1)+' \\cdot '+input[0].toFixed(1)+') + ('+weights[0][1][0].toFixed(1)+' \\cdot '+input[1].toFixed(1)+') + ('+weights[0][2][0].toFixed(1)+' \\cdot '+input[2].toFixed(1)+') + '+biases[1][0].toFixed(1)+' = '+z[1][0].toFixed(1)+' $$ \
				$$ \\sigma('+z[1][0].toFixed(1)+') = '+acts[1][0].toFixed(2)+' $$', true);
		},
		draw: function() {
			display_hidden_calculation(1, 0);
		}
	});
	
	steps.push({
		action: function() {
			set_text_panel(parent.description_panel_div, '\
				The output of the second hidden unit is: \
				$$ z = ('+weights[0][0][1].toFixed(1)+' \\cdot '+input[0].toFixed(1)+') + ('+weights[0][1][1].toFixed(1)+' \\cdot '+input[1].toFixed(1)+') + ('+weights[0][2][1].toFixed(1)+' \\cdot '+input[2].toFixed(1)+') + '+biases[1][1].toFixed(1)+' = '+z[1][1].toFixed(1)+' $$ \
				$$ \\sigma('+z[1][1].toFixed(1)+') = '+acts[1][1].toFixed(2)+' $$', true);
		},
		draw: function() {
			display_hidden_calculation(1, 1);
		}
	});
	
	steps.push({
		action: function() {
			set_text_panel(parent.description_panel_div, '\
				Now we take these and feed it into the output unit: \
				$$ z = ('+weights[1][0][0].toFixed(1)+' \\cdot '+acts[1][0].toFixed(2)+') + ('+weights[1][1][0].toFixed(1)+' \\cdot '+acts[1][1].toFixed(2)+') + '+biases[2][0].toFixed(1)+' = '+z[2][0].toFixed(1)+' $$ \
				$$ \\sigma('+z[2][0].toFixed(2)+') = '+acts[2][0].toFixed(2)+' $$', true);
		},
		draw: function() {
			display_hidden_calculation(2, 0);
		}
	});
	
	steps.push({
		action: function() {
			var error = get_mse(acts[2][0],y_correct);
			set_text_panel(parent.description_panel_div, '\
				Not quite right (we wanted to get $'+y_correct+'$)...we can measure our error with the mean squared error (MSE), which is the most common measurement for error in regression problems: \
				$$ \
				\\begin{aligned} \
				\\text{mse} &= ('+y_correct.toFixed(2)+' - '+acts[2][0].toFixed(2)+')^2 \\\\ \
				\\text{mse} &= '+error.toFixed(3)+' \
				\\end{aligned} \
				$$', true);
		},
		draw: function() {}
	});
	
	steps.push({
		action: function() {
			forward_pass(true);
			set_text_panel(parent.description_panel_div, '\
				Now I\'ll magically give you the best set of weights. For the hidden layer: \
				$$ \
				w=\\begin{bmatrix} \
				'+weights[0][0][0].toFixed(1)+' & '+weights[0][1][0].toFixed(1)+' & '+weights[0][2][0].toFixed(1)+' \\\\ \
				'+weights[0][0][1].toFixed(1)+' & '+weights[0][1][1].toFixed(1)+' & '+weights[0][2][1].toFixed(1)+' \
				\\end{bmatrix} \\qquad b=\\begin{bmatrix}'+biases[1][0]+'  '+biases[1][1]+'\\end{bmatrix}\
				$$ \
				And for the output layer: \
				$$ \
				w=\\begin{bmatrix} \
				'+weights[1][0][0].toFixed(1)+' & '+weights[1][1][0].toFixed(1)+' \
				\\end{bmatrix} \\qquad b='+biases[2][0].toFixed(1)+' \
				$$', true);
		},
		draw: function() {
			display_values_layer1();
			display_values_layer2();		
		}
	});

	steps.push({
		action: function() {
			set_text_panel(parent.description_panel_div, '\
				Let\'s try this all again with these new weights: \
				$$ \
				\\begin{aligned} \
				\\text{hidden unit 1} &= \\sigma(('+weights[0][0][0].toFixed(1)+' * '+input[0].toFixed(1)+') + ('+weights[0][1][0].toFixed(1)+' * '+input[1].toFixed(1)+') + ('+weights[0][2][0].toFixed(1)+' * '+input[2].toFixed(1)+') + '+biases[1][0].toFixed(1)+') = '+acts[1][0].toFixed(2)+' \\\\ \
				\\text{hidden unit 2} &= \\sigma(('+weights[0][0][1].toFixed(1)+' * '+input[0].toFixed(1)+') + ('+weights[0][1][1].toFixed(1)+' * '+input[1].toFixed(1)+') + ('+weights[0][2][1].toFixed(1)+' * '+input[2].toFixed(1)+') + '+biases[1][1].toFixed(1)+') = '+acts[1][1].toFixed(2)+' \\\\ \
				\\text{output} &= \\sigma(('+weights[1][0][0].toFixed(1)+' * '+acts[1][0].toFixed(2)+') + ('+weights[1][1][0].toFixed(2)+' * '+acts[1][1].toFixed(2)+') + '+biases[2][0].toFixed(1)+') = '+acts[2][0].toFixed(2)+' \
				\\end{aligned} \
				$$', true);
		},
		draw: function() {
			display_everything();
		}
	});

	steps.push({
		action: function() {
			set_text_panel(parent.description_panel_div, '\
				Voilà! We got the answer we wanted - so the weights of the network effectively control what it outputs.', true);
		},
		draw: function() {}
	});

	// control flow
	function prev() {
		if (idx > 0) {
		    idx--;
		    redraw();
		}
	};

	function next() {
		idx++;
		if (idx == steps.length) {
			idx = 0;
			initialize_network();
		}
	    redraw();
	};

	function redraw() {
		net.resetSettings();
		steps[idx].action();
		for (var i=0; i<=idx; i++) {
			steps[i].draw();
		}
	    //steps[idx]();
	    ctx.clearRect(0, 0, canvas.width, canvas.height);
	    net.draw(7, 10);
	};
	
	// add control panels
	set_control_panel_height(parent.description_panel_div, 120);
	add_control_panel_action('prev', prev);
	add_control_panel_action('next', next);

	// start
	next();	
};
