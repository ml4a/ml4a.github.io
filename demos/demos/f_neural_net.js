function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
    // parameters
    var canvas = parent.canvas;
    var ctx = canvas.getContext('2d');


    var settings = {
        context: ctx,
        width:600, 
        height:440,
        architecture: [3,2,1],
        visible: [3,2,1],
        heightBounds: [[0, 1], [0.1, 0.9], [0, 1]],
        neuronStyle: {
            color: 'rgba(0,0,0,0.9)',
            thickness: 5,
            radius: 50
        },
        connectionStyle: {
            color: 'rgba(100,100,100,1)',
            arrowLen: 15,
            arrowWidth: 6,
            thickness: 3
        }
    };

    var net = new NetworkVisualization(settings);
    net.draw(20, 45);

    ctx.font = '30px Arial';
    ctx.fillStyle = 'rgba(100, 100, 100, 1.0)';
    ctx.textAlign = 'center';   
    ctx.textBaseline = 'middle';        

    ctx.save();
    ctx.translate(75, 15);
    ctx.fillText("input layer", 0, 0);
    ctx.restore();

    ctx.save();
    ctx.translate(22+0.5*settings.width, 15);
    ctx.fillText("hidden layer", 0, 0);
    ctx.restore();

    ctx.save();
    ctx.translate(20+settings.width - 50, 15);
    ctx.fillText("output layer", 0, 0);
    ctx.restore();

};