// #include "NetworkExperiment.h"

// #include "../SpikeAnalyser/SpikeAnalyser.h"

// #include "../Helpers/TerminalHelpers.h"


// // NetworkExperiment Constructor
// NetworkExperiment::NetworkExperiment() {

// 	four_layer_vision_spiking_model = NULL;
// 	simulator = NULL;

// 	experiment_prepared = false;
// 	experiment_run = false;


// }


// // NetworkExperiment Destructor
// NetworkExperiment::~NetworkExperiment() {
	
// }

// void NetworkExperiment::prepare_network_experiment(FourLayerVisionSpikingModel * four_layer_vision_spiking_model_param) {

// 	four_layer_vision_spiking_model = four_layer_vision_spiking_model_param;

// 	// Create an instance of the Simulator

// 	simulator = new Simulator();
// 	simulator->SetSpikingModel(four_layer_vision_spiking_model);
	
// 	experiment_prepared = true;

// }