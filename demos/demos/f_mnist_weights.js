
var demo = function(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
	// parameters
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');

	var scale = 4.0;
	var margin = 2;
	var fontSize = 16;
	var to_draw_labels = true;
	
	var num_train = 16*100;
	var learning_rate = 0.01;
	
	function update_canvas(){
		ctx.fillStyle = 'rgba(255,255,255,1.0)';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		ctx.textAlign = 'center';
		ctx.fillStyle = 'rgba(0,0,0,1.0)';
		ctx.font = fontSize+'px Arial';
		for (var idx=0; idx<nc; idx++) {
			var x = (dim + margin) * scale * (idx % 5);
			var y = ((dim + margin) * scale + to_draw_labels * (fontSize+5)) * Math.floor(idx / 5);
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
	var data = new dataset('MNIST');
	var net = new convnet(data);
	var classes = data.get_classes();
	var dim = data.get_dim();
	var nc = classes.length;

	//net.add_layer({type:'fc', num_neurons:10, activation:'sigmoid'});
	net.add_layer({type:'softmax', num_classes:10});
	net.setup_trainer({method:'adadelta', learning_rate:learning_rate, batch_size:4, l2_decay:0.001});


	train_all();
	// train_individually();
	
}
