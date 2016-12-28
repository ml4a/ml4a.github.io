function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
    // parameters
    var canvas = parent.canvas;
    var ctx = canvas.getContext('2d');

    function draw_bezier(ctx, x1, y1, x2, y2, x3, y3, x4, y4) {
        ctx.beginPath();
        ctx.lineWidth = 3.0;
        ctx.moveTo(x1,y1);
        ctx.bezierCurveTo(
            x2,y2,
            x3,y3,
            x4,y4);
        ctx.stroke();
        ctx.closePath();
    };

    var settings = {
        context: ctx,
        width: 630, 
        height: 540,
        architecture: [784,10],
        visible: [21,10],
        neuronStyle: {
            color: 'rgba(0,0,0,1.0)',
            thickness: 1,
            radius: 7,
        },
        connectionStyle: {
            color: 'rgba(50,50,50,0.9)',
            arrowLen: 0,
            thickness: 0.5,
        }
    };

    var y1 = 10;
    var y2 = settings.height-5;
    var x1 = 245;
    var x2 = 400;

    var data = new dataset('MNIST');
    var net = new NetworkVisualization(settings);

    net.setNeuronStyle({
        leftLabelText: "pixel",
        leftLabelCounter: true,
        leftLabelMargin: 30,
        leftLabelSize: 20}, 0);
    net.setNeuronStyle({radius:17}, 1);

    data.load_next_batch(function(){
        var idx_sample = 299;    
        data.draw_sample(ctx, idx_sample, 5, 155, 7, 1);    
        net.draw(370, 5);
        var xm = (x1 + x2) / 2;
        var ym = (y1 + y2) / 2;
        draw_bezier(ctx, x1, ym, xm, ym, x1, y1, xm, y1);
        draw_bezier(ctx, x1, ym, xm, ym, x1, y2, xm, y2);
    });

};