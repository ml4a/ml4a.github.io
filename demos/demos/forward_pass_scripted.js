var demo = function(parent, width, height)
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

	next();	
};
