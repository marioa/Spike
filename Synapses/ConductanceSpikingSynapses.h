#ifndef ConductanceSPIKINGSYNAPSES_H
#define ConductanceSPIKINGSYNAPSES_H

#include "SpikingSynapses.h"
#include "../Neurons/SpikingNeurons.h"

struct conductance_spiking_synapse_parameters_struct : spiking_synapse_parameters_struct {
	conductance_spiking_synapse_parameters_struct(): biological_conductance_scaling_constant_lambda(1.0), reversal_potential_Vhat(0.0f), decay_term_tau_g(0.001f) { spiking_synapse_parameters_struct(); }

	float biological_conductance_scaling_constant_lambda;
	float reversal_potential_Vhat;
	float decay_term_tau_g;
};

class ConductanceSpikingSynapses : public SpikingSynapses {

public:

	// Constructor/Destructor
	ConductanceSpikingSynapses();
	~ConductanceSpikingSynapses();

	float * synaptic_conductances_g;
	float * d_synaptic_conductances_g;

	float * biological_conductance_scaling_constants_lambda;
	float * d_biological_conductance_scaling_constants_lambda;

	float * reversal_potentials_Vhat;
	float * d_reversal_potentials_Vhat;

	float * decay_terms_tau_g;
	float * d_decay_terms_tau_g;

	int * d_active_synapses;
	int * d_num_active_synapses;

	// Synapse Functions
	virtual void AddGroup(int presynaptic_group_id,
						int postsynaptic_group_id,
						Neurons * neurons,
						Neurons * input_neurons,
						float timestep,
						synapse_parameters_struct * synapse_params);

	virtual void allocate_device_pointers();
	virtual void copy_constants_and_initial_efficacies_to_device();
	virtual void reset_synapse_activities();

	virtual void set_threads_per_block_and_blocks_per_grid(int threads);
	virtual void increment_number_of_synapses(int increment);
	virtual void shuffle_synapses();

	virtual void interact_spikes_with_synapses(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep);
	virtual void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep);
	virtual void update_synaptic_conductances(float timestep, float current_time_in_seconds);
};

__global__ void conductance_calculate_postsynaptic_current_injection_kernel(int * d_presynaptic_neuron_indices,
							int* d_postsynaptic_neuron_indices,
							float* d_reversal_potentials_Vhat,
							float* d_neurons_current_injections,
							int* d_num_active_synapses,
							int* d_active_synapses,
							float * d_membrane_potentials_v,
							float * d_synaptic_conductances_g);


__global__ void conductance_update_synaptic_conductances_kernel(float timestep,
													float * d_synaptic_conductances_g,
													float * d_synaptic_efficacies_or_weights,
													float * d_time_of_last_spike_to_reach_synapse,
													float * d_biological_conductance_scaling_constants_lambda,
													int* d_num_active_synapses,
													int* d_active_synapses,
													float current_time_in_seconds,
													float * d_decay_terms_tau_g);

__global__ void get_active_synapses_kernel(int* d_presynaptic_neuron_indices,
								int* d_delays,
								float* d_last_spike_time_of_each_neuron,
								float* d_input_neurons_last_spike_time,
								float * d_synaptic_conductances_g,
								float current_time_in_seconds,
								int* d_num_active_synapses,
								int* d_active_synapses,
								float timestep,
								size_t total_number_of_synapses);

__global__ void conductance_move_spikes_towards_synapses_kernel(int* d_presynaptic_neuron_indices,
								int* d_delays,
								int* d_spikes_travelling_to_synapse,
								float* d_neurons_last_spike_time,
								float* d_input_neurons_last_spike_time,
								float currtime,
								int* d_num_active_synapses,
								int* d_active_synapses,
								float* d_time_of_last_spike_to_reach_synapse);

__global__ void conductance_check_bitarray_for_presynaptic_neuron_spikes(int* d_presynaptic_neuron_indices,
								int* d_delays,
								unsigned char* d_bitarray_of_neuron_spikes,
								unsigned char* d_input_neuruon_bitarray_of_neuron_spikes,
								int bitarray_length,
								int bitarray_maximum_axonal_delay_in_timesteps,
								float current_time_in_seconds,
								float timestep,
								int* d_num_active_synapses,
								int* d_active_synapses,
								float* d_time_of_last_spike_to_reach_synapse);

#endif
