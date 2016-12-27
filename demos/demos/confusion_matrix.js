function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
	// parameters
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');

	var datasetName = (datasetName_ === undefined) ? 'MNIST' : datasetName_;
	var useSummary = (useSummary_ === undefined) ? false : useSummary_;
	var useSnapshot = (useSnapshot_ === undefined) ? true : useSnapshot_;
	var viewTopSamples = (viewTopSamples_ === undefined) ? false : viewTopSamples_;
	var testAll = (testAll_ === undefined) ? true : testAll_;
	var numTrain = (numTrain_ === undefined) ? 10000 : numTrain_;
	var numTest = (numTest_ === undefined) ? 5000 : numTest_;
	var mx = 100;
	var my = 90;
	var mcw = 45;
	var mch = 36;
	var selected = {a:2, p:2};

	// variables
	var data, net, classes, nc, dim;

	function preloadModel(callback) {
		var snapshot;
		if (datasetName == 'MNIST') {
			snapshot = '/demos/datasets/mnist/mnist_snapshot.json';
		} else if (datasetName == 'CIFAR') {
			snapshot = '/demos/datasets/cifar/cifar10_snapshot.json';
		}
		data = new dataset(datasetName);
	    net = new convnet(data);
	    classes = data.get_classes();
		nc = classes.length;
		dim = data.get_dim();
	    net.load_from_json(snapshot, callback);
	};

	function loadFromSummary(callback) {
		var summary_file;
		if (datasetName == 'MNIST') {
			summary_file = '/demos/datasets/mnist/mnist_summary.json';
		} else if (datasetName == 'CIFAR') {
			summary_file = '/demos/datasets/cifar/cifar10_summary.json';
		}
		data = new dataset(datasetName);
	    net = new convnet(data);
		classes = data.get_classes();
		nc = classes.length;
		dim = data.get_dim();			
	    net.load_summary(summary_file, callback);
	};

	function createModel(callback) {
		data = new dataset(datasetName);
		net = new convnet(data);
		net.add_layer({type:'fc', num_neurons:15, activation:'sigmoid'});
		net.add_layer({type:'softmax', num_classes:10});
		net.setup_trainer({method:'adadelta', learning_rate:0.5, batch_size:8, l2_decay:0.0001});
		classes = data.get_classes();
		nc = classes.length;
		dim = data.get_dim();
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
		    	data.draw_sample(ctx, tops[0].idx, x_ + p * cellsize.x + margin, y_ + a * cellsize.y + margin, scale);
		    }

	    	var margin = 2;
			var cellsize = {
				x:data.get_dim() * scale + 2 * margin, 
				y: data.get_dim() * scale + 2 * margin
			};

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

		// get inexes of batches we are drawing rom
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
		ctx.fillStyle = 'rgba(255,255,255,1.0)';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		if (viewTopSamples) {
			draw_confusion_matrix_samples(mx, my, 1.0);
		} else {
			draw_confusion_matrix(mx, my, {x:mcw, y:mch}, 16);
		    draw_confusion_samples(mx + nc*mcw + 100, 24, canvas.height-40, selected.p, selected.a, 2);
		}
	};

	function test_all() {
		net.test(numTest, update_canvas);
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
		var mx_ = Math.floor((mouse_x - mx) / mcw);
		var my_ = Math.floor((mouse_y - my) / mch);
		if (mx_ >= 0 && mx_ < nc && my_ >= 0 && my_ <nc &&
			(mx_ != selected.p || my_ != selected.a)) {
			selected = {a: my_, p: mx_};    
	   		draw_confusion_samples(mx + nc*mcw + 100, 24, canvas.height-40, selected.p, selected.a, 2.0);
		}
	};


	if (!viewTopSamples) {
		canvas.addEventListener("mousemove", mouseMoved, false);
	}

	// mode 1: load everything from summary file
	if (useSummary) {
		loadFromSummary(update_canvas);
	}
	// mode 2: load pretrained model and test samples on client
	else if (useSnapshot) {
		preloadModel(testAll ? test_all : test_individually);
	} 
	// mode 3: create and train own model and test samples on client
	else {
		createModel(function() {
			if (testAll) {
				net.train(numTrain, test_all);
			} else {
				net.train(numTrain, test_individually);
			}
		});		
	};	
};

