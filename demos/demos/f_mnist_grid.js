var demo = function(canvas_, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
	var canvas = canvas_;
	//document.getElementById("post").appendChild(canvas);
//	document.body.appendChild(canvas);

	var ctx = canvas.getContext('2d');
	ctx.imageSmoothingEnabled = true;



	var data = new dataset('MNIST');
	data.load_labels(function(){
		data.load_batch(0, function() {
			data.draw_sample_grid(ctx, 18, 58, 1, 0, 0);
		});
	});
}