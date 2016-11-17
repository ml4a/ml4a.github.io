var demo = function(parent, width, height)
{

	var control_panel = document.createElement('div');
	control_panel.style = "width:"+width+"px; height:20px; background-color:#f0f; margin:auto; display:block; padding:3px; ";	
	control_panel.innerHTML  = '<b>Forward Pass Demo</b>';
	control_panel.innerHTML += '<span id="nextLink">next</span>';
	parent.appendChild(control_panel);
	
	
	$('#nextLink').click(function () {
	    console.log("I CLICKED NEXT")
	});
	
	
	var text_panel = document.createElement('div');
	text_panel.style = "width:"+width+"px; height:120px; background-color:#f00; margin:auto; display:block; padding:3px; ";	
	var status = "Lets try a set of random weights. For the first hidden layer, lets say our weights are:";
	status += ""
	status += "$$";
	status += "\\begin{bmatrix}";
	status += "0.9 & 0.4 & 0.2 \\\\";
	status += "0.8 & 0.5 & 0.7";
	status += "\\end{bmatrix}";
	status += "$$";
	status += ""
	status += "We'll use sigmoid activation functions for the hidden layer.'";
	
	
	
	
	
	var html = `
	Lets try a set of random weights. For the first hidden layer, lets say our weights are:

	$$
	\\begin{bmatrix}
	0.9 & 0.4 & 0.2 \\\\
	0.8 & 0.5 & 0.7
	\\end{bmatrix}
	$$

	We'll use sigmoid activation functions for the hidden layer.'
	`;
	
	
	//text_panel.innerHTML = html;
	parent.appendChild(text_panel);
	
	
	/*
	And for the output layer, let's say the weights are:

	$$
	\begin{bmatrix}
	0.3 & 0.9
	\end{bmatrix}
	$$

	
	
	
	
	
	Let's input our datapoint $[2.4, 1.2, 1.3]$ and see what output the network gives us.
	
	
	
	The output of the first (top-most in the graphic) hidden unit is:

	$$
	\text{sigmoid}((0.9 * 2.4) + (0.4 * 1.2) + (0.2 * 1.3)) = 0.962
	$$
	
	
	
	
	The output of the second hidden unit is:

	$$
	\text{sigmoid}((0.8 * 2.4) + (0.5 * 1.2) + (0.7 * 1.3)) = 0.990
	$$
	
	
	
	Now we take these and feed it into the output unit (which doesn't have an activation function):

	$$
	(0.3 * 0.962) + (0.9 * 0.990) = 1.180
	$$
	
	
	
	
	
	
	
	Not quite right (we wanted to get $$0.854$$)...we can measure our error with the mean squared error (MSE), which is the most common measurement for error in regression problems:

	$$
	\begin{aligned}
	\text{error} &= (1.180 - 0.854)^2 \\
	&= 0.106
	\end{aligned}
	$$
	
	
	
	
	
	
	
	
	Now I'll magically give you the best set of weights. For the hidden layer:

	$$
	\begin{bmatrix}
	0.1 & 0.6 & 0.9 \\
	0.9 & 0.7 & -0.2
	\end{bmatrix}
	$$

	And for the output layer:

	$$
	\begin{bmatrix}
	0.5 & 0.4
	\end{bmatrix}
	$$
	
	
	
	
	
	
	Let's try this all again with these new weights:

	$$
	\begin{aligned}
	\text{hidden unit 1} &= \text{sigmoid}((0.1 * 2.4) + (0.6 * 1.2) + (0.9 * 1.3)) = 0.975 \\
	\text{hidden unit 2} &= \text{sigmoid}((0.9 * 2.4) + (0.7 * 1.2) + (-0.2 * 1.3)) = 0.917 \\
	\text{output} &= (0.5 * 0.975) + (0.4 * 0.917) = 0.854
	\end{aligned}
	$$
	
	
	
	
	
	Voilà! We got the answer we wanted - so the weights of the network effectively control what it outputs.
	
	
	
	
	
	*/
	
	// setup canvas
	var canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	canvas.style = 'margin:auto; display:block;';
	parent.appendChild(canvas);
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
	        thickness: t2,
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
	function initialize_network() {
	    a = settings.architecture;
	    input = [...Array(a[0]).keys()].map(i => Math.random(1))
	    weights = [...Array(a.length-1).keys()].map(i => [...Array(a[i]).keys()].map(k => [...Array(a[i+1]).keys()].map(k => 2.0*(Math.random(1)-0.5))));
	    biases = [...Array(a.length).keys()].map(i => [...Array(a[i]).keys()].map(k => 2.0*(Math.random(1)-0.5)));
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
	                biasLabelText: 'b = '+biases[l][n].toFixed(2),
	                biasLabelColor: c1}, l, n);
	        }
	    }
	    for (var l=0; l<a.length-1; l++) {
	        for (var n1=0; n1<weights[l].length; n1++) {        
	            for (var n2=0; n2<weights[l][n1].length; n2++) {
	                net.setConnectionStyle({labelText: 'w = '+weights[l][n1][n2].toFixed(2)}, l, n1, n2);
	            }
	        }
	    }
	};

	// animation steps
	var steps = [];

	// segment 1 (1 step): annotate all net labels
	steps.push(function(){
	    initialize_network();
	    net.setNeuronStyle({labelText:'', thickness: t1});
	    net.setConnectionStyle({thickness: t1});
	    set_all_weights_and_labels_visible();
	});

	// segment 2 (1 step): label input neurons
	steps.push(function(){
	    for (var n=0; n<acts[0].length; n++) {
	        net.setNeuronStyle({labelText: acts[0][n].toFixed(2)}, 0, n);
	    }
	});

	// segment 3 (multiple steps): do computation for each neuron
	for (var l=1; l<a.length; l++) {
	    const l_ = l;
	    for (var n=0; n<a[l]; n++) {
	        const n_ = n;
	        steps.push(function(){
	            net.setNeuronStyle({
	                color: c1,
	                thickness: t1});
	            net.setConnectionStyle({
	                labelText:'', 
	                color: c1,
	                thickness: t1});
	            net.setNeuronStyle({
	                labelText: acts[l_][n_].toFixed(2), 
	                color: c2,
	                thickness: t2}, l_, n_);
	            for (var n2=0; n2<acts[l_-1].length; n2++){
	                net.setNeuronStyle({
	                    color: c2,
	                    thickness: t2}, l_-1, n2);
	                net.setConnectionStyle({
	                    labelText:'w = '+weights[l_-1][n2][n_].toFixed(2), 
	                    color: c2,
	                    thickness: t2}, l_-1, n2, n_);
	            }
	        });
	    }
	}

	// segment 4 (1 step): show everything
	steps.push(function(){
	    net.setNeuronStyle({thickness: t1});
	    net.setConnectionStyle({thickness: t1});
	    set_all_weights_and_labels_visible();
	});


	// control flow
	function next() {
	    idx = (idx + 1) % steps.length;
	    steps[idx]();
	    redraw();
	};

	function redraw() {
	    ctx.clearRect(0, 0, canvas.width, canvas.height);
	    net.draw(5, 5);
	}
	
	/*
	window.addEventListener("keydown", function(e) { 
	    if (e.keyCode == 49) {next();} 
	}, false);
	*/
	
	//canvas.addEventListener("mousemove", mouseMoved, false);
	next();	
};
