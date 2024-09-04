#include "OpenSDF/Tier0.h"

extern "C" __global__
void eval_sdSphere(glm::vec3* domain, Shape0* params, float* output)
{
    using namespace glm;
    const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    const vec3 p(domain[idx]);
    output[idx] = sdSphere(p, params[idx].sphere.radius); 
}

extern "C" __global__
void eval_sdBox(glm::vec3* domain, Shape0* params, float* output)
{
    using namespace glm;
    const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    const vec3 p(domain[idx]);
    output[idx] = sdBox(p, params[idx].box.extents); 
}