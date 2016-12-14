#include "ConductanceSpikingSynapses.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"

// Used for inclusive scan
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// ConductanceSpikingSynapses Constructor
ConductanceSpikingSynapses::ConductanceSpikingSynapses() {

	synaptic_conductances_g = NULL;
	d_synaptic_conductances_g = NULL;

	biological_conductance_scaling_constants_lambda = NULL;
	d_biological_conductance_scaling_constants_lambda = NULL;

	reversal_potentials_Vhat = NULL;
	d_reversal_potentials_Vhat = NULL;

	decay_terms_tau_g = NULL;
	d_decay_terms_tau_g = NULL;

	d_active_synapses = NULL;
	d_num_active_synapses = NULL;

}

// ConductanceSpikingSynapses Destructor
ConductanceSpikingSynapses::~ConductanceSpikingSynapses() {

	free(synaptic_conductances_g);
	free(biological_conductance_scaling_constants_lambda);
	free(reversal_potentials_Vhat);
	free(decay_terms_tau_g);

	CudaSafeCall(cudaFree(d_synaptic_conductances_g));
	CudaSafeCall(cudaFree(d_biological_conductance_scaling_constants_lambda));
	CudaSafeCall(cudaFree(d_reversal_potentials_Vhat));
	CudaSafeCall(cudaFree(d_decay_terms_tau_g));
	CudaSafeCall(cudaFree(d_active_synapses));
	CudaSafeCall(cudaFree(d_num_active_synapses));
}


// Connection Detail implementation
//	INPUT:
//		Pre-neuron population ID
//		Post-neuron population ID
//		An array of the exclusive sum of neuron populations
//		CONNECTIVITY_TYPE (Constants.h)
//		Boolean value to indicate if population is STDP based
//		Parameter = either probability for random synapses or S.D. for Gaussian
void ConductanceSpikingSynapses::AddGroup(int presynaptic_group_id,
						int postsynaptic_group_id,
						Neurons * neurons,
						Neurons * input_neurons,
						float timestep,
						synapse_parameters_struct * synapse_params) {


	SpikingSynapses::AddGroup(presynaptic_group_id,
							postsynaptic_group_id,
							neurons,
							input_neurons,
							timestep,
							synapse_params);

	conductance_spiking_synapse_parameters_struct * conductance_spiking_synapse_group_params = (conductance_spiking_synapse_parameters_struct*)synapse_params;

	for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses; i++) {
		synaptic_conductances_g[i] = 0.0f;
		biological_conductance_scaling_constants_lambda[i] = conductance_spiking_synapse_group_params->biological_conductance_scaling_constant_lambda;
		reversal_potentials_Vhat[i] = conductance_spiking_synapse_group_params->reversal_potential_Vhat;
		decay_terms_tau_g[i] = conductance_spiking_synapse_group_params->decay_term_tau_g;
	}

}

void ConductanceSpikingSynapses::increment_number_of_synapses(int increment) {

	SpikingSynapses::increment_number_of_synapses(increment);

	synaptic_conductances_g = (float*)realloc(synaptic_conductances_g, total_number_of_synapses * sizeof(float));
	biological_conductance_scaling_constants_lambda = (float*)realloc(biological_conductance_scaling_constants_lambda, total_number_of_synapses * sizeof(float));
	reversal_potentials_Vhat = (float*)realloc(reversal_potentials_Vhat, total_number_of_synapses * sizeof(float));
	decay_terms_tau_g = (float*)realloc(decay_terms_tau_g, total_number_of_synapses * sizeof(float));

}


void ConductanceSpikingSynapses::allocate_device_pointers() {

	SpikingSynapses::allocate_device_pointers();

	CudaSafeCall(cudaMalloc((void **)&d_biological_conductance_scaling_constants_lambda, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_reversal_potentials_Vhat, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_decay_terms_tau_g, sizeof(float)*total_number_of_synapses));

	CudaSafeCall(cudaMalloc((void **)&d_synaptic_conductances_g, sizeof(float)*total_number_of_synapses));

	CudaSafeCall(cudaMalloc((void **)&d_active_synapses, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_num_active_synapses, sizeof(int)));
}

void ConductanceSpikingSynapses::copy_constants_and_initial_efficacies_to_device() {

	SpikingSynapses::copy_constants_and_initial_efficacies_to_device();

	CudaSafeCall(cudaMemcpy(d_biological_conductance_scaling_constants_lambda, biological_conductance_scaling_constants_lambda, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_reversal_potentials_Vhat, reversal_potentials_Vhat, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_decay_terms_tau_g, decay_terms_tau_g, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));

}


void ConductanceSpikingSynapses::reset_synapse_activities() {

	SpikingSynapses::reset_synapse_activities();

	CudaSafeCall(cudaMemcpy(d_synaptic_conductances_g, synaptic_conductances_g, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));

	CudaSafeCall(cudaMemset(d_num_active_synapses, 0, sizeof(int)));

}


void ConductanceSpikingSynapses::shuffle_synapses() {

	SpikingSynapses::shuffle_synapses();

	float * temp_synaptic_conductances_g = (float *)malloc(total_number_of_synapses*sizeof(float));
	float * temp_biological_conductance_scaling_constants_lambda = (float *)malloc(total_number_of_synapses*sizeof(float));
	float * temp_reversal_potentials_Vhat = (float *)malloc(total_number_of_synapses*sizeof(float));
	float * temp_decay_terms_tau_g = (float*)malloc(total_number_of_synapses*sizeof(float));

	for(int i = 0; i < total_number_of_synapses; i++) {

		temp_synaptic_conductances_g[i] = synaptic_conductances_g[original_synapse_indices[i]];
		temp_biological_conductance_scaling_constants_lambda[i] = biological_conductance_scaling_constants_lambda[original_synapse_indices[i]];
		temp_reversal_potentials_Vhat[i] = reversal_potentials_Vhat[original_synapse_indices[i]];
		temp_decay_terms_tau_g[i] = decay_terms_tau_g[original_synapse_indices[i]];
	}

	synaptic_conductances_g = temp_synaptic_conductances_g;
	biological_conductance_scaling_constants_lambda = temp_biological_conductance_scaling_constants_lambda;
	reversal_potentials_Vhat = temp_reversal_potentials_Vhat;
	decay_terms_tau_g = temp_decay_terms_tau_g;

}


void ConductanceSpikingSynapses::set_threads_per_block_and_blocks_per_grid(int threads) {

	SpikingSynapses::set_threads_per_block_and_blocks_per_grid(threads);

}



void ConductanceSpikingSynapses::calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep) {

	// First update the conductances
	update_synaptic_conductances(timestep, current_time_in_seconds);

	conductance_calculate_postsynaptic_current_injection_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_presynaptic_neuron_indices,
																	d_postsynaptic_neuron_indices,
																	d_reversal_potentials_Vhat,
																	neurons->d_current_injections,
																	d_num_active_synapses,
																	d_active_synapses,
																	neurons->d_membrane_potentials_v,
																	d_synaptic_conductances_g);

	CudaCheckError();

}

void ConductanceSpikingSynapses::update_synaptic_conductances(float timestep, float current_time_in_seconds) {

	conductance_update_synaptic_conductances_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(timestep,
											d_synaptic_conductances_g,
											d_synaptic_efficacies_or_weights,
											d_time_of_last_spike_to_reach_synapse,
											d_biological_conductance_scaling_constants_lambda,
											d_num_active_synapses,
											d_active_synapses,
											current_time_in_seconds,
											d_decay_terms_tau_g);

	CudaCheckError();

}


__global__ void conductance_calculate_postsynaptic_current_injection_kernel(int * d_presynaptic_neuron_indices,
							int* d_postsynaptic_neuron_indices,
							float* d_reversal_potentials_Vhat,
							float* d_neurons_current_injections,
							int* d_num_active_synapses,
							int* d_active_synapses,
							float * d_membrane_potentials_v,
							float * d_synaptic_conductances_g){

	int indx = threadIdx.x + blockIdx.x * blockDim.x;
	while (indx < d_num_active_synapses[0]) {
		int idx = d_active_synapses[indx];

		float reversal_potential_Vhat = d_reversal_potentials_Vhat[idx];
		int postsynaptic_neuron_index = d_postsynaptic_neuron_indices[idx];
		float membrane_potential_v = d_membrane_potentials_v[postsynaptic_neuron_index];
		float synaptic_conductance_g = d_synaptic_conductances_g[idx];

		float component_for_sum = synaptic_conductance_g * (reversal_potential_Vhat - membrane_potential_v);
		if (component_for_sum != 0.0) {
			atomicAdd(&d_neurons_current_injections[postsynaptic_neuron_index], component_for_sum);
		}

		indx += blockDim.x * gridDim.x;

	}
	__syncthreads();
}



__global__ void conductance_update_synaptic_conductances_kernel(float timestep,
														float * d_synaptic_conductances_g,
														float * d_synaptic_efficacies_or_weights,
														float * d_time_of_last_spike_to_reach_synapse,
														float * d_biological_conductance_scaling_constants_lambda,
														int* d_num_active_synapses,
														int* d_active_synapses,
														float current_time_in_seconds,
														float * d_decay_terms_tau_g) {

	int indx = threadIdx.x + blockIdx.x * blockDim.x;
	while (indx < d_num_active_synapses[0]) {
		int idx = d_active_synapses[indx];

		float synaptic_conductance_g = d_synaptic_conductances_g[idx];

		float new_conductance = (1.0 - (timestep/d_decay_terms_tau_g[idx])) * synaptic_conductance_g;

		if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
			float timestep_times_synaptic_efficacy = timestep * d_synaptic_efficacies_or_weights[idx];
			float biological_conductance_scaling_constant_lambda = d_biological_conductance_scaling_constants_lambda[idx];
			float timestep_times_synaptic_efficacy_times_scaling_constant = timestep_times_synaptic_efficacy * biological_conductance_scaling_constant_lambda;
			new_conductance += timestep_times_synaptic_efficacy_times_scaling_constant;
		}

		d_synaptic_conductances_g[idx] = new_conductance;

		indx += blockDim.x * gridDim.x;
	}
	__syncthreads();

}




void ConductanceSpikingSynapses::interact_spikes_with_synapses(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) {
	// This function is the first one to run on each loop. We can use this opportunity to detect synapses which should be updated this round and
	CudaSafeCall(cudaMemset(d_num_active_synapses, 0, sizeof(int)));
	// Get Active Synapses
	get_active_synapses_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_presynaptic_neuron_indices,
									d_delays,
									neurons->d_last_spike_time_of_each_neuron,
									input_neurons->d_last_spike_time_of_each_neuron,
									d_synaptic_conductances_g,
									d_decay_terms_tau_g,
									current_time_in_seconds,
									d_num_active_synapses,
									d_active_synapses,
									timestep,
									total_number_of_synapses);
	CudaCheckError();

	// thrust::device_ptr<int> active_syns(d_active_synapses);


	if (neurons->high_fidelity_spike_flag){
		conductance_check_bitarray_for_presynaptic_neuron_spikes<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(
								d_presynaptic_neuron_indices,
								d_delays,
								neurons->d_bitarray_of_neuron_spikes,
								input_neurons->d_bitarray_of_neuron_spikes,
								neurons->bitarray_length,
								neurons->bitarray_maximum_axonal_delay_in_timesteps,
								current_time_in_seconds,
								timestep,
								d_num_active_synapses,
								d_active_synapses,
								d_time_of_last_spike_to_reach_synapse);
		CudaCheckError();
	}
	else{
		conductance_move_spikes_towards_synapses_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_presynaptic_neuron_indices,
																			d_delays,
																			d_spikes_travelling_to_synapse,
																			neurons->d_last_spike_time_of_each_neuron,
																			input_neurons->d_last_spike_time_of_each_neuron,
																			current_time_in_seconds,
																			d_num_active_synapses,
																			d_active_synapses,
																			d_time_of_last_spike_to_reach_synapse);
		CudaCheckError();
	}
}


__global__ void get_active_synapses_kernel(int* d_presynaptic_neuron_indices,
								int* d_delays,
								float* d_last_spike_time_of_each_neuron,
								float* d_input_neurons_last_spike_time,
								float * d_synaptic_conductances_g,
								float * d_decay_terms_tau_g,
								float current_time_in_seconds,
								int* d_num_active_synapses,
								int* d_active_synapses,
								float timestep,
								size_t total_number_of_synapses){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_synapses) {

		int presynaptic_neuron_index = d_presynaptic_neuron_indices[idx];
		bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(presynaptic_neuron_index);
		int delay = d_delays[idx];

		// Check if the presynaptic neuron spiked less than the delay ago
		if (presynaptic_is_input){
			if ((d_input_neurons_last_spike_time[CORRECTED_PRESYNAPTIC_ID(presynaptic_neuron_index, presynaptic_is_input)] + (delay + 1)*timestep + 20.0f*d_decay_terms_tau_g[idx]) > current_time_in_seconds){
				int pos = atomicAdd(&d_num_active_synapses[0], 1);
				d_active_synapses[pos] = idx;
			}
		} else {
			if ((d_last_spike_time_of_each_neuron[presynaptic_neuron_index] + (delay + 1)*timestep + 20.0f*d_decay_terms_tau_g[idx]) > current_time_in_seconds){
				int pos = atomicAdd(&d_num_active_synapses[0], 1);
				d_active_synapses[pos] = idx;
			}
		}
		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();
}


__global__ void conductance_move_spikes_towards_synapses_kernel(int* d_presynaptic_neuron_indices,
								int* d_delays,
								int* d_spikes_travelling_to_synapse,
								float* d_last_spike_time_of_each_neuron,
								float* d_input_neurons_last_spike_time,
								float current_time_in_seconds,
								int* d_num_active_synapses,
								int* d_active_synapses,
								float* d_time_of_last_spike_to_reach_synapse){

	int indx = threadIdx.x + blockIdx.x * blockDim.x;
	while (indx < d_num_active_synapses[0]) {
		int idx = d_active_synapses[indx];

		int timesteps_until_spike_reaches_synapse = d_spikes_travelling_to_synapse[idx];
		timesteps_until_spike_reaches_synapse -= 1;

		if (timesteps_until_spike_reaches_synapse == 0) {
			d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
		}

		if (timesteps_until_spike_reaches_synapse < 0) {

			// Get presynaptic neurons last spike time
			int presynaptic_neuron_index = d_presynaptic_neuron_indices[idx];
			bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(presynaptic_neuron_index);
			float presynaptic_neurons_last_spike_time = presynaptic_is_input ? d_input_neurons_last_spike_time[CORRECTED_PRESYNAPTIC_ID(presynaptic_neuron_index, presynaptic_is_input)] : d_last_spike_time_of_each_neuron[presynaptic_neuron_index];

			if (presynaptic_neurons_last_spike_time == current_time_in_seconds){

				timesteps_until_spike_reaches_synapse = d_delays[idx];

			}
		}

		d_spikes_travelling_to_synapse[idx] = timesteps_until_spike_reaches_synapse;

		indx += blockDim.x * gridDim.x;
	}
	__syncthreads();
}

__global__ void conductance_check_bitarray_for_presynaptic_neuron_spikes(int* d_presynaptic_neuron_indices,
								int* d_delays,
								unsigned char* d_bitarray_of_neuron_spikes,
								unsigned char* d_input_neuron_bitarray_of_neuron_spikes,
								int bitarray_length,
								int bitarray_maximum_axonal_delay_in_timesteps,
								float current_time_in_seconds,
								float timestep,
								int * d_num_active_synapses,
								int * d_active_synapses,
								float* d_time_of_last_spike_to_reach_synapse){

	int indx = threadIdx.x + blockIdx.x * blockDim.x;
	while (indx < d_num_active_synapses[0]) {
		int idx = d_active_synapses[indx];

		int presynaptic_neuron_index = d_presynaptic_neuron_indices[idx];
		bool presynaptic_is_input = PRESYNAPTIC_IS_INPUT(presynaptic_neuron_index);
		int delay = d_delays[idx];

		// Get offset depending upon the current timestep
		int offset_index = ((int)(round(current_time_in_seconds / timestep)) % bitarray_maximum_axonal_delay_in_timesteps) - delay;
		offset_index = (offset_index < 0) ? (offset_index + bitarray_maximum_axonal_delay_in_timesteps) : offset_index;
		int offset_byte = offset_index / 8;
		int offset_bit_pos = offset_index - (8 * offset_byte);

		// Get the correct neuron index
		int neuron_index = CORRECTED_PRESYNAPTIC_ID(presynaptic_neuron_index, presynaptic_is_input);

		// Check the spike
		int neuron_id_spike_store_start = neuron_index * bitarray_length;
		int check = 0;
		if (presynaptic_is_input){
			unsigned char byte = d_input_neuron_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
			check = ((byte >> offset_bit_pos) & 1);
			if (check == 1){
				d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
			}
		} else {
			unsigned char byte = d_bitarray_of_neuron_spikes[neuron_id_spike_store_start + offset_byte];
			check = ((byte >> offset_bit_pos) & 1);
			if (check == 1){
				d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
			}
		}

		indx += blockDim.x * gridDim.x;
	}
	__syncthreads();
}
