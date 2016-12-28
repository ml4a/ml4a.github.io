// 1 layer demo
// - integrate network diagram
// - controls (learning rate, numTrain, restart)
// 2 layer demo
// - controls (learning rate, numTrain, restart, num_hidden_units)



var demo3 = function(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
	// parameters
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');

	var scale = 3.0;
	var margin = 2;
	var fontSize = 16;
	var to_draw_labels = true;
	
	var num_train = 10*500;
	var learning_rate = 0.001;
	
	function update_canvas(){
		ctx.fillStyle = 'rgba(255,255,255,1.0)';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		ctx.textAlign = 'center';
		ctx.fillStyle = 'rgba(0,0,0,1.0)';
		ctx.font = fontSize+'px Arial';
		for (var idx=0; idx<15; idx++) {
			var x = (dim + margin) * scale * (idx % 10);
			var y = ((dim + margin) * scale + to_draw_labels * (fontSize+5)) * Math.floor(idx / 10);
			net.draw_filter(ctx, 1, idx, x, y, scale);
			if (to_draw_labels) {
				ctx.fillText(classes[idx], x + (dim * scale / 2.0), y + (dim * scale + fontSize + 3));
			}
	    }
	};

	function train_individually() {
		update_canvas();
		setTimeout(function() {
			net.train(50, train_individually);   // when to stop?
		}, 1);
	};

	function train_all() {
	    net.train(num_train, update_canvas);
	};

	// setup network
	var data = new dataset('CIFAR');
	var classes = data.get_classes();
	var dim = data.get_dim();
	var nc = classes.length;
	var net = new convnet(data);
	net.add_layer({type:'fc', num_neurons:15, activation:'sigmoid'});
	net.add_layer({type:'softmax', num_classes:10});
	net.setup_trainer({method:'adadelta', learning_rate:learning_rate, batch_size:4, l2_decay:0.001});

	// start
	train_individually();	
}

var demo = function(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
	// parameters
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');

	var scale = 3.0;
	var margin = 2;
	var fontSize = 14;
	
	var num_train = 30000;
	var learning_rate = 0.002;

	var num_hidden_units = 8;
	var max_y = 50;

	var classes_view = [0, 1, 5, 6, 8, 9];
	
	function update_canvas(){
		ctx.fillStyle = 'rgba(255,255,255,1.0)';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		ctx.textAlign = 'center';
		ctx.fillStyle = 'rgba(0,0,0,1.0)';
		ctx.font = fontSize+'px Arial';
		
		for (var idx=0; idx<num_hidden_units; idx++) {
			var x = 20 + (dim + margin) * scale * idx;
			var y = 20;
			net.draw_filter(ctx, 1, idx, x, y, scale);
			ctx.fillText("filter "+(idx+1), x + (dim * scale / 2.0), y + (dim * scale + fontSize + 3));
	    }
	    
	    for (var c=0; c<classes_view.length; c++) {//nc; c++) {
	    	var y = 20 + (dim + margin) * scale + fontSize + 10 + (c + 0.5) * max_y;
	    	ctx.fillStyle = 'rgba(0,0,0,1.0)';
	    	ctx.fillText(classes[classes_view[c]], 5, y+fontSize*0.5);
	    	ctx.fillStyle = 'rgba(0,0,0,0.4)';
	    	ctx.fillRect(20, y-0.25, (dim + margin) * scale * num_hidden_units, 0.5);
			ctx.fillStyle = 'rgba(0,0,0,1.0)';
	    	var cw = net.get_net().layers[3].filters[classes_view[c]].w;
	    	var sum = cw.reduce(function(cw, b) { return cw + b; }, 0);
	    	var max = cw.reduce(function(cw, b) { return Math.max(cw, b); }, 0);
	    	var min = cw.reduce(function(cw, b) { return Math.min(cw, b); }, 0);
			for (var idx=0; idx<num_hidden_units; idx++) {
				var x = 20 + (dim + margin) * scale * idx;
				var val = -0.5 + 1.0 * (cw[idx] - min) / (max-min);
				ctx.fillRect(x-8+dim*scale*0.5, y, 16, max_y * val * -0.75); //max_y * val * 0.7);
		        ctx.stroke();
		    }
		}
	};

	function train_all() {
	    net.train(num_train, update_canvas);
	};

	// setup network
	//var data = new dataset('MNIST');
	var data = new dataset('MNIST');
	var net = new convnet(data);
	var classes = data.get_classes();
	var dim = data.get_dim();
	var nc = classes.length;

	net.add_layer({type:'fc', num_neurons:num_hidden_units, activation:'sigmoid'});
	net.add_layer({type:'softmax', num_classes:10});
	net.setup_trainer({method:'adadelta', learning_rate:learning_rate, batch_size:4, l2_decay:0.001});


	train_all();
	//train_individually();

	this.n = net;
	this.c = classes;	
}
