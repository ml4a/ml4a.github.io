function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
	// parameters
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');

	var datasetName = (datasetName_ === undefined) ? 'MNIST' : datasetName_;
	
	var net;
	var data;
	var dim;

	//net.add_layer({type:'fc', num_neurons:10, activation:'sigmoid'});
	//net.add_layer({type:'softmax', num_classes:10});
	//net.setup_trainer({method:'adadelta', learning_rate:0.5, batch_size:8, l2_decay:0.0001});

	var crop_amt = 5;

	var filter_size, pad_amt, num_filters, grid_size, sample_size;



	var settings = {
	    sample_x: 5,
	    sample_y: 5,
	    sample_scale: 8,
	    sample_grid: 1,
	    sample_thickness: 4,
	};

	var select_x = 0;
	var select_y = 0;

	var idx_sample = 2;


	// 1) train N samples/preload  2) test M (fwd pass demo, conv demo
	// 2) train + test intermittently  (weights demo)

	function draw_sample_as_grid(sample, x_, y_, cellsize) {
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

	function loadPresetNetwork(callback) {		
		var snapshot;
		if (datasetName == 'MNIST') {
			snapshot = '/demos/datasets/mnist/mnist_snapshot.json';
		} else if (datasetName == 'CIFAR') {
			snapshot = '/demos/datasets/cifar/cifar10_snapshot.json';
		}
		data = new dataset(datasetName);
	    net = new convnet(data);
	    dim = data.get_dim();
		net.load_from_json(snapshot, callback);
	};

	function finished_training() {
	    net.test(1, finished_testing);
	}

	function finished_testing(results) {
	    sample = data.get_sample_image(idx_sample, draw);
	};

	function draw() {
		function draw_square(x1, y1, x2, y2, thickness, color) {
			ctx.beginPath();
			ctx.strokeStyle = color;
	    	ctx.lineWidth = settings.sample_thickness;
	    	ctx.moveTo(x1, y1);
	    	ctx.lineTo(x2, y1);
	    	ctx.lineTo(x2, y2);
	    	ctx.lineTo(x1, y2);
	    	ctx.lineTo(x1, y1);
	    	ctx.stroke();
	    	ctx.closePath();
		}

    	var cell_size = settings.sample_scale + settings.sample_grid;
    	var x1 = settings.sample_x + cell_size * select_x;
    	var y1 = settings.sample_y + cell_size * select_y;
    	var x2 = x1 + cell_size * filter_size;
    	var y2 = y1 + cell_size * filter_size;

    	var rx1 = 440;
    	var ry1 = 5;
    	var rs = 8;

		data.draw_current_sample(ctx, settings.sample_x, settings.sample_y, settings.sample_scale, settings.sample_grid, {x:0, y:0, w:data.get_dim()+2*pad_amt, h:data.get_dim()+2*pad_amt, pad:pad_amt});
		data.draw_current_sample(ctx, 320, 5, 16, 1, {x:select_x, y:select_y, w:filter_size, h:filter_size, pad:pad_amt});
		net.draw_activations(ctx, 1, 2, rx1, ry1, rs);

		// green box around subregion of main sample
		draw_square(x1, y1, x2, y2, settings.sample_thickness, 'rgba(0,255,0,1.0)');
		draw_square(320, 5, 320 + (16+1)*filter_size, 5+(16+1)*filter_size, settings.sample_thickness, 'rgba(0,255,0,1.0)');
		draw_square(rx1 + rs*select_x, ry1 + rs*select_y, rx1 + rs*(select_x+1), ry1 + rs*(select_y+1), 1.0, 'rgba(255,0,0,1.0)');
	}

	function mouseMoved(evt) {
		function inside(mx, my, x, y, w, h) {
			return (mx > 0 && my > 0 && mx < w && my < h);
		}
		var canvas_rect = canvas.getBoundingClientRect();
		var mouse_x = evt.clientX - canvas_rect.left;
	    var mouse_y = evt.clientY - canvas_rect.top;

	    var smx = mouse_x - settings.sample_x; 
		var smy = mouse_y - settings.sample_y; 

		if (inside(smx, smy, 0, 0, (dim + 2*pad_amt) * (settings.sample_scale + settings.sample_grid), (dim + 2*pad_amt) * (settings.sample_scale + settings.sample_grid))) {
			select_x = Math.min(dim-1, Math.floor(smx / (settings.sample_scale + settings.sample_grid)));
			select_y = Math.min(dim-1, Math.floor(smy / (settings.sample_scale + settings.sample_grid)));
			draw();
		} 
		else if (inside(smx, smy, 0, 0, 1, 1)) {
			// pick a weight
		}
		else {
			// new sample
		}
	};
	
	function finished_loading() {
		pad_amt = net.get_net().layers[1].pad;
	    filter_size = net.get_net().layers[1].filters[0].sx;
	    num_filters = net.get_net().layers[1].filters.length;
	    grid_size = data.get_dim() + 2 * (pad_amt-crop_amt) - filter_size + 1;
	    sample_size = data.get_dim() + 2 * (pad_amt - crop_amt);
		net.test(1, function() {
			canvas.addEventListener("mousemove", mouseMoved, false);
			finished_testing();
		});
	}

	//net.train(1000, finished_training);
	loadPresetNetwork(finished_loading);
};
