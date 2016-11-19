function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
	// parameters
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');



	var data = new dataset('CIFAR');
	data.load_labels(function(){
		data.load_batch(0, function(){
			data.draw_sample_grid(ctx, 5, 30, 2, 2, null);
		});
	});

};