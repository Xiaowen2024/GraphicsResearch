#include <iostream>
#include <vector>
#include <random>
#include <utility> // For std::pair
#include <cmath>   // For abs
#include <functional> // For std::function
#include "fractureModelHelpers.cpp"
#include "fractureModelHelpers.h"


struct BoundaryPoint {
    Vec2D p;
    Vec2D n;
};

template <typename RandomEngine, typename Func, typename UniformSampler>
std::pair<BoundaryPoint, double> sample_boundary_ris(
    RandomEngine& rng,
    Func target_distribution_func,
    UniformSampler uniform_sampler,
    unsigned int num_candidates)
{
    // A nested struct for the reservoir, just like in the original code.
    struct Reservoir {
        BoundaryPoint y;            // The chosen sample
        double w_sum = 0.0;         // The sum of weights
        unsigned int M = 0;         // The number of candidates seen
        RandomEngine& rng_ref;      // Reference to the random engine

        Reservoir(RandomEngine& engine) : rng_ref(engine) {}

        // The core update logic is identical to the GPU version.
        void update(const BoundaryPoint& x_i, double w_i) {
            w_sum += w_i;
            M++;
            // Generate a random number to decide if we swap the sample
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            if (dist(rng_ref) <= w_i / w_sum) {
                y = x_i;
            }
        }
    };

    if (num_candidates == 0) {
        return {{}, 0.0}; // Return an empty result if no candidates
    }

    Reservoir reservoir(rng);

    // 1. Generate candidates and populate the reservoir
    for (unsigned int i = 0; i < num_candidates; ++i) {
        // Get a candidate point sampled uniformly from the boundary
        auto [x_i, inv_pdf_uniform] = uniform_sampler();
        
        // 2. Calculate its importance weight
        double weight = target_distribution_func(x_i) * inv_pdf_uniform;

        // 3. Update the reservoir with the new candidate
        reservoir.update(x_i, weight);
    }

    // 4. Finalize the result
    BoundaryPoint result_sample = reservoir.y;
    double target_val_at_result = target_distribution_func(result_sample);

    // The PDF of the chosen sample is estimated as:
    // pdf(y) ≈ target(y) * M / sum_of_weights
    // So the inverse PDF is sum_of_weights / (target(y) * M)
    double inv_pdf = reservoir.w_sum / std::max(
        target_val_at_result * reservoir.M,
        1e-9 // Small epsilon for numerical stability
    );

    return {result_sample, inv_pdf};
}