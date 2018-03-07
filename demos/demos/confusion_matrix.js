function demo(parent, width, height, datasetName_, summaryFile_, snapshotFile_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
	// canvas
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');
	
	// parameters
	var datasetName = (datasetName_ === undefined) ? 'MNIST' : datasetName_;
	var summaryFile = (summaryFile_ === undefined) ? '/demos/datasets/mnist/mnist_summary_2layers.json' : summaryFile_;
	var snapshotFile = (snapshotFile_ === undefined) ? true : snapshotFile_;
	var viewTopSamples = (viewTopSamples_ === undefined) ? false : viewTopSamples_;
	var testAll = (testAll_ === undefined) ? true : testAll_;
	var numTrain = (numTrain_ === undefined) ? 40000 : numTrain_;
	var numTest = (numTest_ === undefined) ? 10000 : numTest_;
	var numEpochs = 10;

	var mcw = 45;
	var mch = 36;
	var samples_grid_margin = 2;
	var sample_scale = 1.0;
	var selected = {a:2, p:2};
	var mouseListener = false;

	// variables
	var data, net, classes, nc, dim;
	var cellsize, mx, my;

	function preloadModel(datasetName_, snapshotFile, callback) {
		datasetName = datasetName_;
		data = new dataset(datasetName);
	    classes = data.get_classes();
		nc = classes.length;
		dim = data.get_dim();
	    net = new convnet(data);
	    net.load_from_json(snapshotFile, callback);
	};

	function loadFromSummary(datasetName_, summaryFile, callback) {
		datasetName = datasetName_;
		data = new dataset(datasetName);
	    classes = data.get_classes();
		nc = classes.length;
		dim = data.get_dim();			
	    net = new convnet(data);
	    net.load_summary(summaryFile, callback);
	};

	function createModel(datasetName_, callback) {
		datasetName = datasetName_;
		data = new dataset(datasetName);
		classes = data.get_classes();
		nc = classes.length;
		dim = data.get_dim();
		net = new convnet(data);
		// net.add_layer({type:'fc', num_neurons:15, activation:'sigmoid'});
		net.add_layer({type:'conv', sx:5, filters:8, stride:1, pad:2, activation:'relu'});
		net.add_layer({type:'pool', sx:2, stride:2});
		net.add_layer({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
		net.add_layer({type:'pool', sx:3, stride:3});
		net.add_layer({type:'softmax', num_classes:nc});
		// net.setup_trainer({method:'adadelta', learning_rate:0.01, batch_size:8, l2_decay:0.0001});
		net.setup_trainer({method:'adadelta', batch_size:20, l2_decay:0.001});
		callback();
	};

	function draw_confusion_matrix_box(cellsize){
		ctx.beginPath();
	    ctx.fillStyle = 'rgba(255,255,255,1.0)';
	    ctx.strokeStyle = 'rgba(0,0,0,1.0)';
	    ctx.lineWidth = 1.0;
	    ctx.fillRect(0, 0, nc * cellsize.x, nc * cellsize.y);
	    ctx.rect(0, 0, nc * cellsize.x, nc * cellsize.y);
	    ctx.stroke();
	    ctx.closePath();
	};

	function draw_confusion_matrix_grid(cellsize){
	    ctx.lineWidth = 0.75;
	    ctx.strokeStyle = 'rgba(0,0,0,0.35)';
	    for (var p=0; p<nc+1; p++) {
	    	ctx.beginPath();
	    	ctx.moveTo(0, p * cellsize.y);
	    	ctx.lineTo(nc * cellsize.x, p * cellsize.y);
	    	ctx.stroke();
	    	ctx.closePath();
	    }
	    for (var a=0; a<nc+1; a++) {
	    	ctx.beginPath();
	    	ctx.moveTo(a * cellsize.x, 0);
	    	ctx.lineTo(a * cellsize.x, nc * cellsize.y);
	    	ctx.stroke();
	    	ctx.closePath();
	    }
	};

	function draw_confusion_matrix_labels(cellsize){
	    ctx.font = '16px Arial';
	    ctx.fillStyle = 'rgba(0,0,0,1.0)';
	    ctx.textAlign = 'right';
	    ctx.textBaseline = 'middle';
	    for (var a=0; a<nc; a++) {
	    	ctx.fillText((datasetName=='MNIST'?'actual ':'')+classes[a], -5, (a + 0.5) * cellsize.y);
	    }
	    ctx.textAlign = 'left';
	    for (var p=0; p<nc; p++) {
	    	ctx.save();
	    	ctx.translate((p + 0.5) * cellsize.x, 0);
	    	ctx.rotate(-0.5);
	    	ctx.fillText("predicted "+classes[p], -2, -12);
	    	ctx.restore();
	    }
	};

	function draw_confusion_matrix(x_, y_, cellsize, fontsize) 
	{
		function draw_cell(summary, a, p) {
			var count = summary.confusion[a][p];
			var pct = summary.actuals[a] > 0 ? summary.confusion[a][p] / summary.actuals[a] : 0;
	        ctx.save();
	        ctx.font = fontsize+'px Arial';
	        ctx.textAlign = 'center';   
	        ctx.textBaseline = 'middle';
	        ctx.translate(p * cellsize.x, a * cellsize.y);
	        ctx.fillStyle = (a==p) ? 'rgba(0,255,0,'+pct+')' : 'rgba(255,0,0,'+pct+')';
	    	ctx.fillRect(0, 0, cellsize.x, cellsize.y);
	    	ctx.fillStyle = 'rgba(0,0,0,1.0)';
	        ctx.fillText(count, cellsize.x/2.0, cellsize.y/2.0);
	        ctx.restore();
	    }
	    var summary = net.get_summary();    
		
		ctx.save();
	    ctx.translate(x_, y_);  

	    draw_confusion_matrix_box(cellsize);
	    for (var p=0; p<nc; p++) {
	    	for (var a=0; a<nc; a++) {
	    		draw_cell(summary, a, p);
	    	}
	    }	    
	    draw_confusion_matrix_grid(cellsize);
		draw_confusion_matrix_labels(cellsize);

		// precision, recall, accuracy
	    ctx.save();
	    ctx.textAlign = 'right';
	    ctx.fillText('precision', -5, nc * cellsize.y + 50);
	    ctx.textAlign = 'left';
	    ctx.translate(nc * cellsize.x + 48, -12);
	    ctx.rotate(-0.5);
	    ctx.fillText('recall', 0, 0);
	    ctx.restore();
	    ctx.strokeStyle = 'rgba(0,0,0,1.0)';
	    ctx.textAlign = 'center';
		for (var a=0; a<nc; a++) {
			var recall = summary.actuals[a] == 0 ? 0 : summary.confusion[a][a] / summary.actuals[a];
			ctx.fillStyle = 'rgba('+Math.round(255*(1.0-recall))+','+Math.round(255*recall)+',0,1.0)';
			ctx.fillRect((nc - 0.5) * cellsize.x + 50, a * cellsize.y, cellsize.x, cellsize.y);
			ctx.rect((nc - 0.5) * cellsize.x + 50, a * cellsize.y, cellsize.x, cellsize.y);
			ctx.stroke();
			ctx.fillStyle = 'rgba(0,0,0,1.0)';
	    	ctx.fillText(Math.round(100*recall)+'%', nc * cellsize.x + 50, (a + 0.5) * cellsize.y);
	    }	    
	    for (var p=0; p<nc; p++) {
	    	var precision = summary.predictions[p] == 0 ? 0 : summary.confusion[p][p] / summary.predictions[p];
	    	ctx.fillStyle = 'rgba('+Math.round(255*(1.0-precision))+','+Math.round(255*precision)+',0,1.0)';
	    	ctx.strokeStyle = 'rgba(0,0,0,0.5)';
	    	ctx.fillRect(p * cellsize.x, (nc - 0.5) * cellsize.y + 50, cellsize.x, cellsize.y);
	    	ctx.rect(p * cellsize.x, (nc - 0.5) * cellsize.y + 50, cellsize.x, cellsize.y);
	    	ctx.stroke();
	    	ctx.fillStyle = 'rgba(0,0,0,1.0)';
	    	ctx.fillText(Math.round(100*precision)+'%', (p + 0.5) * cellsize.x, nc * cellsize.y + 50);
	    }	    
	    var accuracy = summary.correct / summary.total;
	    ctx.fillStyle = 'rgba('+Math.round(255*(1.0-accuracy))+','+Math.round(255*accuracy)+',0,1.0)';
	    ctx.fillRect((nc - 0.5 - 0.25) * cellsize.x + 50, (nc - 0.5 - 0.25) * cellsize.y + 50, cellsize.x * 1.5, cellsize.y * 1.5);
	    ctx.rect((nc - 0.5 - 0.25) * cellsize.x + 50, (nc - 0.5 - 0.25) * cellsize.y + 50, cellsize.x * 1.5, cellsize.y * 1.5);
	    ctx.stroke();
	    ctx.fillStyle = 'rgba(0,0,0,1.0)';
	    ctx.fillText('accuracy', nc * cellsize.x + 50, nc * cellsize.y + 50 - 20);
	    ctx.fillText(Math.round(accuracy*100)+'%', nc * cellsize.x + 50, nc * cellsize.y + 50);
	    ctx.restore();
	};

	function draw_confusion_matrix_samples(x_, y_, scale) 
	{
	    function draw_confusion_matrix_samples_grid() {
	    	function draw_cell(summary, a, p) {
				var tops = summary.tops[a][p];
				if (tops.length == 0) return;
				ctx.fillStyle = (a==p) ? 'rgba(0,255,0,'+tops[0].prob+')' : 'rgba(255,0,0,'+tops[0].prob+')';
		    	ctx.fillRect(p * cellsize.x, a * cellsize.y, cellsize.x, cellsize.y);
		    	data.draw_sample(ctx, tops[0].idx, x_ + p * cellsize.x + samples_grid_margin, y_ + a * cellsize.y + samples_grid_margin, scale);
		    }
	    	ctx.save();
		    ctx.translate(x_, y_);  
		    draw_confusion_matrix_box(cellsize);
			for (var p=0; p<nc; p++) {
		    	for (var a=0; a<nc; a++) {
		    		draw_cell(summary, a, p);
		    	}
		    }		    
		    draw_confusion_matrix_grid(cellsize);
		    draw_confusion_matrix_labels(cellsize);
		    ctx.restore();
    	};

    	var summary = net.get_summary();   
    	
		// get inexes of batches we are drawing from
		var batch_idxs = [];
		for (var p=0; p<nc; p++) {
	    	for (var a=0; a<nc; a++) {
	    		var tops = summary.tops[a][p];
	    		if (tops.length > 0) {
		    		batch_idxs.push(data.get_batch_idx_from_sample_idx(tops[0].idx));
		    	}
		    }
		}

		batch_idxs = batch_idxs.filter(function(item, i, ar){ return ar.indexOf(item) === i; });
	    data.load_multiple_batches(batch_idxs, draw_confusion_matrix_samples_grid);
	};

	function draw_confusion_samples(x_, y_, height, p, a, scale)
	{
		function draw_confusion_samples_box() {
	    	for (var i=0; i<Math.min(t.length, cols*rows); i++) {
		    	var c = i % cols;
		    	var r = Math.floor(i / cols);
		    	var x = x_ + margin + c * (dim * scale + margin);
		    	var y = y_ + margin + headerHeight + r * (dim * scale + margin + textHeight);

		    	// draw sample
		    	data.draw_sample(ctx, t[i].idx, x, y, scale);

		    	// prob
		    	ctx.font = fontSizePct+'px Arial';
		    	ctx.textAlign = 'center'
		    	ctx.fillStyle = 'rgba(0,0,0,1.0)';
		    	ctx.fillText(Math.round(t[i].prob*100)+'%', x + (dim * scale)/2.0, y + (dim * scale) + fontSizePct);
	    	}
	    };

		var summary = net.get_summary();
		
		var cols = 4;
		var margin = 8;
		var textHeight = 18;
		var headerHeight = 25;
		var fontSizePct = 12;
		var fontSizeHeader = 14;

		var width = margin + cols * (dim * scale + margin);
		var rows = Math.floor((height - headerHeight) / (dim * scale + textHeight));
		
		ctx.fillStyle = 'rgba(255,255,255,1.0)';
	    ctx.strokeStyle = 'rgba(0,0,0,1.0)';
		ctx.fillRect(x_, y_, width, height);
		ctx.rect(x_, y_, width, height);
		ctx.stroke();
		ctx.font = fontSizeHeader+'px Arial';
		ctx.textAlign = 'left';
		ctx.fillStyle = 'rgba(0,0,0,1.0)';
		ctx.fillText(classes[a] + (a==p?' correctly classified as ':' misclassified as ') + classes[p], x_+5, y_+20);
		
		// all the samples we need to draw
		var t = summary.tops[a][p];
		
		// get inexes of batches we are drawing rom
		var batch_idxs = [];
	    for (var i=0; i<Math.min(t.length, cols*rows); i++) {
	    	batch_idxs.push(data.get_batch_idx_from_sample_idx(t[i].idx))
	    }
		batch_idxs = batch_idxs.filter(function(item, i, ar){ return ar.indexOf(item) === i; });

		// load batches if necessary, then draw the samples
	    data.load_multiple_batches(batch_idxs, draw_confusion_samples_box);
	};

	function update_canvas() {
		if (!mouseListener) {
			canvas.addEventListener("mousemove", mouseMoved, false);
			mouseListener = true;
		}
		toggleView(viewTopSamples);
		ctx.fillStyle = 'rgba(255,255,255,1.0)';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		//draw_confusion_samples(100 + nc*mcw + 100, 24, canvas.height-40, selected.p, selected.a, 2);
		if (viewTopSamples) {
			draw_confusion_matrix_samples(mx, my, sample_scale);
		} else {
			draw_confusion_matrix(mx, my, {x:mcw, y:mch}, 16);
		}
		draw_confusion_samples(100 + nc*mcw + 100, 24, canvas.height-40, selected.p, selected.a, 2);
	};

	function test_all() {
		net.test(numTrain, numTest, update_canvas);
	};

	function test_individually() {
		update_canvas();
		setTimeout(function() {
			net.test(1, test_individually);   // when to stop?
		}, 100);
	};

	function mouseMoved(evt) {
		var canvas_rect = canvas.getBoundingClientRect();
		var mouse_x = evt.clientX - canvas_rect.left;
	    var mouse_y = evt.clientY - canvas_rect.top;
		var mx_ = Math.floor((mouse_x - mx) / cellsize.x);
		var my_ = Math.floor((mouse_y - my) / cellsize.y);
		if (mx_ >= 0 && mx_ < nc && my_ >= 0 && my_ <nc &&
			(mx_ != selected.p || my_ != selected.a)) {
			selected = {a: my_, p: mx_};    
			update_canvas();
		}
	};

	function toggleView(viewTopSamples_) {
		viewTopSamples = viewTopSamples_;
		if (viewTopSamples) {
			cellsize = {x:data.get_dim() * sample_scale + 2 * samples_grid_margin, y:data.get_dim() * sample_scale + 2 * samples_grid_margin};
			mx = 130;
			my = 130;
		} else {
			cellsize = {x:mcw, y:mch};
			mx = 100;
			my = 90;
		}
	};

	add_control_panel_menu(["MNIST ordinary","MNIST convnet","CIFAR ordinary","CIFAR convnet"], function() {
		if 		(this.value == "MNIST ordinary") {loadFromSummary('MNIST', '/demos/datasets/mnist/mnist_summary_2layers.json', update_canvas);}
		else if (this.value == "MNIST convnet") {loadFromSummary('MNIST', '/demos/datasets/mnist/mnist_summary_convnet.json', update_canvas);}
		else if (this.value == "CIFAR ordinary") {loadFromSummary('CIFAR', '/demos/datasets/cifar/cifar10_summary_2layers.json', update_canvas);}
		else if (this.value == "CIFAR convnet") {loadFromSummary('CIFAR', '/demos/datasets/cifar/cifar10_summary_convnet.json', update_canvas);}
	});

	add_control_panel_menu(["View numbers","View top samples"], function() {
		viewTopSamples = (this.value == "View top samples");
		update_canvas();
	});
	
	add_control_panel_action("save", function() {net.save_summary();})

	// mode 1: load everything from summary file
	if (summaryFile !== undefined) {
		console.log("load ",summaryFile)
		loadFromSummary(datasetName, summaryFile, update_canvas);
	}
	// mode 2: load pretrained model and test samples on client
	else if (snapshotFile !== undefined) {
		preloadModel(datasetName, snapshotFile, testAll ? test_all : test_individually);
	} 
	// mode 3: create and train own model and test samples on client
	else {
		createModel(datasetName, function() {
			if (testAll) {
				net.train(numTrain, numTest, numEpochs, test_all);
			} else {
				net.train(numTrain, numTest, numEpochs, test_individually);
			}
		});		
	};
};

