function dataset(datasetName_) 
{
	this.get_name = function(){return datasetName;}
	this.get_dim = function(){return sw;}
	this.get_channels = function(){return channels;}
	this.get_samples_per_batch = function(){return samplesPerBatch;}
	this.get_batch_idx = function(){return idxBatch;}
	this.get_sample_idx = function(){return idxSample;}
	this.get_classes = function(){return classes;}

	this.load_MNIST = function() {
		sw = 28;
		sh = 28;
		channels = 1;
		samplesPerBatch = 3000;
		nBatches = 21;
		batchPath = "/demos/datasets/mnist/mnist";
		classes = ["0","1","2","3","4","5","6","7","8","9"];
		labelsFile = "/demos/datasets/mnist/mnist_labels.json";
		initialize();
	};

	this.load_CIFAR = function() {
		sw = 32;
		sh = 32;
		channels = 3;
		samplesPerBatch = 1000;
		nBatches = 51;
		batchPath = "/demos/datasets/cifar/cifar10";
		classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"];
		labelsFile = "/demos/datasets/cifar/cifar10_labels.json";
		initialize();
	};		

	function initialize() {
		labelsLoaded = false;
		idxBatch = -1;
		idxSample = 0;
		lastBatch = -1;
		batchImg = new Image();
		batches = [...Array(nBatches)];
		batchCtx = [...Array(nBatches)];
	};

	this.load_labels = function(callback) { 
		$.getJSON(labelsFile, function(json){
			self.labels = json.labels;
			labelsLoaded = true;
			callback();
		});
	};

	this.load_batch = function(idxBatch_, callback) {
		idxBatch = idxBatch_;
		if (batchCtx[idxBatch] === undefined) {
			batchImg.onload = function() {
				console.log("loaded "+datasetName+" batch "+idxBatch);
				batches[idxBatch] = document.createElement('canvas');
				batches[idxBatch].width = sw * sh;
				batches[idxBatch].height = samplesPerBatch;   
				batchCtx[idxBatch] = batches[idxBatch].getContext('2d');
				batchCtx[idxBatch].drawImage(batchImg, 0, 0);
				callback();
			};
			batchImg.src = batchPath+"_batch_"+idxBatch+".png";
		} else {
			callback();
		}
	};

	this.load_next_batch = function(callback) {
		this.load_batch(idxBatch+1, callback);
	};

	this.load_multiple_batches = function(idxBatches, callback) {
		function load_next_batch_from_sequence() {
			if (idxBatches.length == 0) {
				callback();
			} else {
				var idxBatch_ = idxBatches.splice(0, 1);
				self.load_batch(idxBatch_, load_next_batch_from_sequence);
			}
		};
		load_next_batch_from_sequence();
	};

	this.get_batch_idx_from_sample_idx = function(idxSample) { 
		return Math.floor(idxSample / samplesPerBatch);
	};

	function get_batch_sample(idxSample) {
		var b = Math.floor(idxSample / samplesPerBatch);
		var k = idxSample % samplesPerBatch;
  		if (b != lastBatch) {
	  		currentBatchData = batchCtx[b].getImageData(0, 0, batches[b].width, batches[b].height).data;
	  		lastBatch = b;
	  	}
  		var W = sw * sh;
  		var y = self.labels[idxBatch * samplesPerBatch + k];
  		var x = new convnetjs.Vol(sw, sh, channels, 0.0);
		for(var dc=0; dc<channels; dc++) {
			var idx=0;
		    for(var xc=0; xc<sw; xc++) {
		    	for(var yc=0; yc<sh; yc++) {
		        	var ix = ((W * k) + idx) * 4 + dc;
		        	x.set(yc, xc, dc, currentBatchData[ix]/255.0 - 0.5);
		        	idx++;
		      	}
		    }
		}
		return {x:x, y:y, idx:idxSample};
	};

	this.get_next_sample = function(idx, callback) {
		var returnSample = function(){
			var sample = get_batch_sample(idxSample);
			callback(sample);
		};
		idxSample += 1;
		var b = Math.floor(idxSample / samplesPerBatch);
		if (batches[b] === undefined) {
			this.load_next_batch(returnSample);
		} else {
			returnSample();
		}		
	};

	this.draw_current_sample = function(ctx, x, y, scale, grid_thickness, crop) {
		this.draw_sample(ctx, idxSample, x, y, scale, grid_thickness, crop);
	};

	this.get_sample_image = function(idx, callback) {
		var b = Math.floor(idx / samplesPerBatch);
		var k = idx % samplesPerBatch;
		if (batchCtx[b] === undefined) {
			this.load_batch(b, function() {
				var sample = batchCtx[b].getImageData(0, k, sw*sh, 1);
				callback({data:sample.data, sw:sw, sh:sh, channels:channels});
			})
		}
		else {
			var sample = batchCtx[b].getImageData(0, k, sw*sh, 1);
			callback({data:sample.data, sw:sw, sh:sh, channels:channels});
		}
	};

	this.draw_sample = function(ctx, idx, x, y, scale, grid_thickness, crop) {
		var sampleImg = this.get_sample_image(idx, function(sampleImg){
			var crop_ = (crop === undefined) ? {x:0, y:0, w:sw, h:sh, pad:0} : crop;
			var g = (grid_thickness === undefined) ? 0 : grid_thickness;
			var ny = crop_.h;
			var nx = crop_.w;
			var newImg = ctx.createImageData(nx * (scale + g), ny * (scale + g));
			for (var j=0; j<ny; j++) {
			 	for (var i=0; i<nx; i++) {
					var y_ = crop_.y + j - crop_.pad;
					var x_ = crop_.x + i - crop_.pad;
					var idxS = (y_ * sw + x_) * 4;
					if (y_ < 0 || y_ >= sh || x_ < 0 || x_ >= sw) {
						idxS = -1;	// in the padding
					}
					for (var sj=0; sj<scale+g; sj++) {
			      		for (var si=0; si<scale+g; si++) {
							var idxN = ((j * (scale + g) + sj) * nx * (scale + g) + (i * (scale + g) + si)) * 4;
			      			if (si < scale && sj < scale) {
				        		newImg.data[idxN  ] = idxS == -1 ? 0   : sampleImg.data[idxS  ];
				        		newImg.data[idxN+1] = idxS == -1 ? 0   : sampleImg.data[idxS+1];
				        		newImg.data[idxN+2] = idxS == -1 ? 0   : sampleImg.data[idxS+2];
				        		newImg.data[idxN+3] = idxS == -1 ? 255 : sampleImg.data[idxS+3];                						
				        	} else {
				        		newImg.data[idxN  ] = 127;
				        		newImg.data[idxN+1] = 127;
				        		newImg.data[idxN+2] = 127;
				        		newImg.data[idxN+3] = 255;
				        	}
			      		}
			    	}
			  	}
			}
			ctx.putImageData(newImg, x, y);
		});
	};



	this.draw_sample_grid = function(ctx, rows, cols, scale, margin, label) {
		var draw_next_sample = function(n, idx, label) {
			if (self.labels[idx] == label || label == null) {
				var y = margin + (sh * scale + margin) * Math.floor(n / cols);
		    	var x = margin + (sw * scale + margin) * (n % cols);
		    	self.draw_sample(ctx, idx, x, y, scale);
		    	n++;
		    	if (n==rows*cols) return;
		  	}
		  	if (idx+1 < samplesPerBatch) {
		    	draw_next_sample(n, idx+1, label);
		  	} 
		  	else if (idxBatch+1 < nBatches) {
		    	self.load_batch(idxBatch+1, function() {           
		    		draw_next_sample(n, 0, label);
		    	}); 
		  	}
		};
		draw_next_sample(0, 0, label);
	};

	// initialize
	var self = this;
	var datasetName = datasetName_;
	var batchPath, idxBatch;
  	var sw, sh, channels, samplesPerBatch, nBatches;
  	var labelsFile, labelsLoaded, classes;
  	var idxSample;

	// setup canvases
	var batchImg;
	var batches;
	var batchCtx;
	var currentBatchData;
	var lastBatch;
  
  	// set dataset
  	if (datasetName == 'MNIST') {
		this.load_MNIST();
	} else if (datasetName == 'CIFAR') {
		this.load_CIFAR();
	}
};
