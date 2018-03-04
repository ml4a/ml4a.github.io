// CIFAR vs MNIST
// 1 layer | 2 layers (15 hidden units)
// view combinations for classes
// view only the weights

var demo = function(parent, width, height, datasetName_, num_hidden_layers_, snapshot) 
{
	// parameters
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');

	var scale = 3.0;
	var margin = 2;
	var fontSize = 14;
	var to_draw_labels = true;		
	var max_y = 50;
	var num_hidden_units = 10;	
	var classes_view = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
	var num_per = 25;

	// learning
	var num_train = 10000;
	var learning_rate = 0.002;
	
	// vars
	var datasetName, num_hidden_layers, data, classes, dim, nc, net, num_trained;
	var training, to_kill=false, kill_callback;
	

	function draw_1layer(){
		ctx.fillStyle = 'rgba(255,255,255,1.0)';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		ctx.textAlign = 'center';
		ctx.fillStyle = 'rgba(0,0,0,1.0)';
		ctx.font = fontSize+'px Arial';
		for (var idx=0; idx<nc; idx++) {
			var x = 25 + (dim + margin) * scale * (idx % 5);
			var y = 20 + ((dim + margin) * scale + to_draw_labels * (fontSize+5)) * Math.floor(idx / 5);
			net.draw_filter(ctx, 1, idx, x, y, scale);
			if (to_draw_labels) {
				ctx.fillText(classes[idx], x + (dim * scale / 2.0), y + (dim * scale + fontSize + 3));
			}
	    }
	};

	function draw_2layer(){
		ctx.fillStyle = 'rgba(255,255,255,1.0)';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		ctx.textAlign = 'center';
		ctx.fillStyle = 'rgba(0,0,0,1.0)';
		ctx.font = fontSize+'px Arial';
		
		for (var idx=0; idx<num_hidden_units; idx++) {
			var x = 60 + (dim + margin) * scale * idx;
			var y = 20;
			net.draw_filter(ctx, 1, idx, x, y, scale);
			ctx.fillText("filter "+(idx+1), x + (dim * scale / 2.0), y + (dim * scale + fontSize));
	    }
	    
	    for (var c=0; c<classes_view.length; c++) {//nc; c++) {
	    	var y = 20 + (dim + margin) * scale + fontSize + 10 + (c + 0.5) * max_y;
	    	ctx.fillStyle = 'rgba(0,0,0,1.0)';
	    	ctx.textAlign = 'right';
	    	ctx.fillText(classes[classes_view[c]], 55, y+fontSize*0.25);
	    	ctx.fillStyle = 'rgba(0,0,0,0.4)';
	    	ctx.fillRect(60, y-0.25, (dim + margin) * scale * num_hidden_units, 0.5);
			ctx.fillStyle = 'rgba(0,0,0,1.0)';
	    	var cw = net.get_net().layers[3].filters[classes_view[c]].w;
	    	var sum = cw.reduce(function(cw, b) { return cw + b; }, 0);
	    	var max = cw.reduce(function(cw, b) { return Math.max(cw, b); }, 0);
	    	var min = cw.reduce(function(cw, b) { return Math.min(cw, b); }, 0);
			for (var idx=0; idx<num_hidden_units; idx++) {
				var x = 60 + (dim + margin) * scale * idx;
				var val = -0.5 + 1.0 * (cw[idx] - min) / (max-min);
				ctx.fillRect(x-8+dim*scale*0.5, y, 16, max_y * val * -0.75); //max_y * val * 0.7);
		        ctx.stroke();
		    }
		}
	};

	function update_canvas(){
		if (num_hidden_layers > 0) {
			draw_2layer();
		} else {
			draw_1layer();
		}
	};
		
	function train_individually() {
		update_canvas();
		if (to_kill) {
			training = false;
			to_kill = false;
			kill_callback();
			return;
		};
		setTimeout(function() {
			num_trained += num_per;
			if (num_trained < num_train) {
				net.advance_offset(num_per);
				net.train(num_per, 0, 1, train_individually);   // when to stop?
			} else {
				training = false;
			}
		}, 1);
	};

	function train_all() {
	    net.train(num_train, 0, 1, update_canvas);
	};

	function kill_trainer(kill_callback_) {
		kill_callback = kill_callback_;
		if (!training) {
			kill_callback();
		} else {
			to_kill = true;
		}
	};

	function run_demo(datasetName_, num_hidden_layers_, snapshot) {
		kill_trainer(function(){
			datasetName = datasetName_;
			num_hidden_layers = num_hidden_layers_;
			// if (num_hidden_layers == 0) {
			// 	set_control_panel_height(parent.description_panel_div, 70);
			// 	set_text_panel(parent.description_panel_div, 'This demo shows how convolution works in a convolutional layer. A filter is slid along every horizontal and vertical position of the original image or the previous layer\'s activations, and the dot product is taken in each position. The resulting activation map (on the right) shows the presence of the feature map -- or roughly patterns in the input which resemble the filter itself. Move your mouse around the input to see individual patches, and click \'next sample\' or \'next filter\' to see different convolutional filters and inputs in action.');
			// }
			num_trained = 0;
			data = new dataset(datasetName);
			classes = data.get_classes();
			dim = data.get_dim();
			nc = classes.length;
			net = new convnet(data);
			if (snapshot !== undefined) {
				net.load_from_json(snapshot, update_canvas);
			} else {
				if (num_hidden_layers > 0) {
					net.add_layer({type:'fc', num_neurons:num_hidden_units, activation:'sigmoid'});
				}
				net.add_layer({type:'softmax', num_classes:nc});
				net.setup_trainer({method:'adadelta', learning_rate:learning_rate, batch_size:4, l2_decay:0.001});
				training = true;
				train_individually();
			}
		});
	};

	add_control_panel_menu(["MNIST 2-layer","MNIST 1-layer","CIFAR 2-layer","CIFAR 1-layer"], function() {
		if 		(this.value == "MNIST 1-layer") {run_demo('MNIST', 0);}
		else if (this.value == "MNIST 2-layer") {run_demo('MNIST', 1);}
		else if (this.value == "CIFAR 1-layer") {run_demo('CIFAR', 0);}
		else if (this.value == "CIFAR 2-layer") {run_demo('CIFAR', 1);}
	});

	add_control_panel_action("save", function() {
		net.save_to_json();
	});

	// load preset
	run_demo(datasetName_, 1, snapshot);
};
