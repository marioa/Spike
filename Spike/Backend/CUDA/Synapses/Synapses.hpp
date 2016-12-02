#pragma once

#include "Spike/Synapses/Synapses.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class Synapses : public virtual ::Backend::Synapses {
    public:
      ~Synapses();

      int* presynaptic_neuron_indices = nullptr;
      int* postsynaptic_neuron_indices = nullptr;
      int* temp_presynaptic_neuron_indices = nullptr;
      int* temp_postsynaptic_neuron_indices = nullptr;
      int * synapse_postsynaptic_neuron_count_index = nullptr;
      float* synaptic_efficacies_or_weights = nullptr;
      float* temp_synaptic_efficacies_or_weights = nullptr;
      
      // CUDA Specific
      dim3 number_of_synapse_blocks_per_grid;
      dim3 threads_per_block;

      virtual void prepare() {}
      virtual void reset_state() {}

      virtual void allocate_device_pointers();
      virtual void copy_constants_and_initial_efficacies_to_device();
      virtual void set_threads_per_block_and_blocks_per_grid(int threads);

      virtual void set_neuron_indices_by_sampling_from_normal_distribution() {
        printf("TODO Backend::Synapses::set_neuron_indices_by_sampling_from_normal_distribution\n");
      }
    };

    __global__ void compute_yes_no_connection_matrix_for_groups(bool * d_yes_no_connection_vector, 
                                                                int pre_width, 
                                                                int post_width, 
                                                                int post_height, 
                                                                float sigma, 
                                                                int total_pre_neurons, 
                                                                int total_post_neurons);

    __global__ void set_up_neuron_indices_and_weights_for_yes_no_connection_matrix(bool * d_yes_no_connection_vector, 
                                                                                   int pre_width, 
                                                                                   int post_width, 
                                                                                   int post_height, 
                                                                                   int total_pre_neurons, 
                                                                                   int total_post_neurons, 
                                                                                   int * d_presynaptic_neuron_indices, 
                                                                                   int * d_postsynaptic_neuron_indices);

    __global__ void set_neuron_indices_by_sampling_from_normal_distribution(int total_number_of_new_synapses, 
                                                                            int postsynaptic_group_id, 
                                                                            int poststart, 
                                                                            int prestart, 
                                                                            int post_width, 
                                                                            int post_height, 
                                                                            int pre_width, 
                                                                            int pre_height, 
                                                                            int number_of_new_synapses_per_postsynaptic_neuron, 
                                                                            int number_of_postsynaptic_neurons_in_group, 
                                                                            int * d_presynaptic_neuron_indices, 
                                                                            int * d_postsynaptic_neuron_indices, 
                                                                            float * d_synaptic_efficacies_or_weights, 
                                                                            float standard_deviation_sigma, 
                                                                            bool presynaptic_group_is_input,
                                                                            curandState_t* d_states);
  }
}
