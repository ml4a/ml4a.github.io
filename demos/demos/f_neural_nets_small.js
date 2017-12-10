function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
    // canvas
    var canvas = parent.canvas;
    var ctx = canvas.getContext('2d');

    // parameters
    var settings = {
        context: ctx,
        width:100, 
        height:80,
        architecture: [2,1],
        visible: [2,1],
        heightBounds: [[0, 1], [0, 1]],
        neuronStyle: {
            color: 'rgba(0,0,0,0.9)',
            thickness: 2,
            radius: 15
        },
        connectionStyle: {
            color: 'rgba(100,100,100,1)',
            arrowLen: 8,
            arrowWidth: 7,
            thickness: 1
        }
    };


    var net1 = new NetworkVisualization(settings);
    net1.draw(10, 15);

    settings.architecture = [3,1];
    settings.visible = [3,1];
    settings.height = 100;

    var net2 = new NetworkVisualization(settings);
    net2.draw(160, 5);

};