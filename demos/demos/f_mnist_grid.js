var demo = function(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
	// parameters
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');



	var data = new dataset('MNIST');
	data.load_labels(function(){
		data.load_batch(0, function() {
			data.draw_sample_grid(ctx, 18, 58, 1, 0, 0);
		});
	});
}