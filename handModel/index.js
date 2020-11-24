let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var posSamples=0, negSamples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(2);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 2, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'binaryCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			posSamples++;
			document.getElementById("positiveamples").innerText = "Positive samples:" + posSamples;
			break;
		case "1":
			negSamples++;
			document.getElementById("negativesamples").innerText = "Negative samples:" + negSamples;
			break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    console.log(classId);
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see a Hand";
			break;
		case 1:
			predictionText = "I do not see a Hand";
			break;
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


async function doTraining(){
  await train();
  alert("Model Trained!!");
}

function startPredicting(){
  isPredicting = true;
  console.log(model.summary());
	predict();
}

function stopPredicting(){
  isPredicting = false;
	predict();
}

async function saveModel(){
  const saveResult = await model.save('downloads://my-model');
  console.log(model.summary());
  alert("Model Saved!!");
}

async function loadModel(){
  const tmpModel = await tf.loadLayersModel('http://127.0.0.1:8887//faceModel//model//my-model.json');
  model = tmpModel;
  console.log(model.summary());
  alert("Model Loaded!!");
}

async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}


init();
