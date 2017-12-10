function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
    // canvas
    var canvas = parent.canvas;
    var ctx = canvas.getContext('2d');

    // parameters
    var settings = {
        context: ctx,
        width:500, 
        height:360,
        architecture: [10,8,8,5],
        visible: [10,8,8,5],
        heightBounds: [[0, 1], [0, 1], [0, 1], [0, 1]],
        neuronStyle: {
            color: 'rgba(0,0,0,0.9)',
            thickness: 2,
            radius: 15
        },
        connectionStyle: {
            color: 'rgba(130,130,130,0.67)',
            arrowLen: 0,
            arrowWidth: 1,
            thickness: 1
        }
    };

    var net = new NetworkVisualization(settings);
    net.setConnectionStyle({color:'rgba(220,10,10,0.85)', thickness:1.5}, 0, 2, 3)
    net.setConnectionStyle({color:'rgba(220,10,10,0.85)', thickness:1.5}, 1, 3);
    net.setConnectionStyle({color:'rgba(220,10,10,0.85)', thickness:1.5}, 2);
    
    net.draw(20, 10);
};