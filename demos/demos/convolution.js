function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
	// parameters
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');

	var datasetName = (datasetName_ === undefined) ? 'MNIST' : datasetName_;
	

	var draw_sample_as_grid = function(sample, x_, y_, cellsize) {
	    var nx = sample.sw;
	    var ny = sample.sh;
	    ctx.save();
	    ctx.translate(x_, y_);  
	    ctx.beginPath();
	    ctx.rect(0, 0, nx * cellsize.x, ny * cellsize.y);
	    ctx.strokeStyle = 'argb(255,0,0,1.0)';
	    ctx.lineJoin = ctx.lineCap = 'round';
	    ctx.lineWidth = 0.5;
	    ctx.stroke();
	    ctx.closePath();
	    for (var y=0; y<ny; y++) {
	        var ty = y * cellsize.y;
	        for (var x=0; x<nx; x++) {
	            var tx = x * cellsize.x;
	            var idx_color = 4*(x+y*nx);
	            ctx.save();
	            ctx.font = (cellsize.x/2.0)+'px Arial';
	            ctx.textAlign = 'center';   
	            ctx.textBaseline = 'middle';
	            ctx.translate(tx+cellsize.x/2.0, ty+cellsize.y/2.0);
	            ctx.fillText(sample.data[idx_color], 0, 0);
	            ctx.restore();
	        }
	    }
	    ctx.restore();
	};



	var net;
	var data;

	//net.add_layer({type:'fc', num_neurons:10, activation:'sigmoid'});
	//net.add_layer({type:'softmax', num_classes:10});
	//net.setup_trainer({method:'adadelta', learning_rate:0.5, batch_size:8, l2_decay:0.0001});

	var crop_amt = 5;

	var filter_size, pad_amt, num_filters, grid_size, sample_size;

	function loadPresetNetwork(callback) {		
		var snapshot;
		if (datasetName == 'MNIST') {
			snapshot = '/demos/datasets/mnist/mnist_snapshot.json';
		} else if (datasetName == 'CIFAR') {
			snapshot = '/demos/datasets/cifar/cifar10_snapshot.json';
		}
		data = new dataset(datasetName);
	    net = new convnet(data);
		net.load_from_json(snapshot, callback);
	};



	// 1) train N samples/preload  2) test M (fwd pass demo, conv demo
	// 2) train + test intermittently  (weights demo)

	var finished_training = function() {
	    net.test(1, finished_testing);
	}

	var finished_testing = function(results) {
	    console.log("done testing");

	    var idx_sample = 2;
	    var sample = data.get_sample_image(idx_sample, function() {
		
		
		        console.log("done testing 111");

			//    data.draw_sample(ctx, idx_sample, 20, 20, 10, 1);    
				data.draw_current_sample(ctx, 20, 20, 10, 1, {x:0, y:0, w:data.get_dim()+2*pad_amt, h:data.get_dim()+2*pad_amt, pad:pad_amt});

				data.draw_current_sample(ctx, 620, 20, 10, 1, {x:10, y:10, w:filter_size, h:filter_size, pad:pad_amt});


				for (var idx=0; idx<num_filters; idx++) {
			        net.draw_filter(ctx, 1, idx, 33*3*idx, 140, 12);
			    }

				for (var idx=0; idx<num_filters; idx++) {
			   		net.draw_activations(ctx, 1, idx, 33*3*idx, 280, 5);
			    }

		        console.log("done testing 222");
		
		});
	


	
	};
	
	
	function finished_loading() {
		pad_amt = net.get_net().layers[1].pad;
	    filter_size = net.get_net().layers[1].filters[0].sx;
	    num_filters = net.get_net().layers[1].filters.length;
	    grid_size = data.get_dim() + 2 * (pad_amt-crop_amt) - filter_size + 1;
	    sample_size = data.get_dim() + 2 * (pad_amt - crop_amt);
		net.test(1, finished_testing);
	}

	//net.train(1000, finished_training);
	loadPresetNetwork(finished_loading);

};
