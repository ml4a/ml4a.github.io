function demo(parent, width, height, datasetName_, useSummary_, useSnapshot_ , viewTopSamples_, testAll_, numTrain_, numTest_) 
{
    // parameters
    var canvas = parent.canvas;
    var ctx = canvas.getContext('2d');



    var settings = {
        context: ctx,
        width: 1200, 
        height: 720,
        architecture: [784,15,10],
        visible: [25,15,10],
        neuronStyle: {
            color: 'rgba(0,0,0,1.0)',
            thickness: 1,
            radius: 12,
        },
        connectionStyle: {
            color: 'rgba(50,50,50,0.9)',
            arrowLen: 0,
            thickness: 0.5,
        }
    };

    var net = new NetworkVisualization(settings);
    net.setNeuronStyle({
        leftLabelText: "pixel",
        leftLabelCounter: true,
        leftLabelMargin: 30,
        leftLabelSize: 20}, 0);
    net.setNeuronStyle({radius:21}, 1);
    net.setNeuronStyle({radius:25}, 2);

    net.draw(80, 0);

};