
#ifdef __CUDACC__
#include <builtin_types.h>
#define __opensdf_shape__ __device__
#define __opensdf_operator__ __device__ __forceinline__
#else
#define __opensdf_shape__
#define __opensdf_operator__
#define __device__
#endif

#include <glm/glm.hpp>

//
// Shape Parameters
//
union Shape0
{
	struct Sphere
	{
		float radius;
	};
	Sphere sphere;

	struct Box
	{
		glm::vec3 extents;
	};
	Box box;
};

//
// Shape Functions
//
__opensdf_shape__
static float sdBox(glm::vec3 const& p, glm::vec3 b)
{
	using namespace glm;

	const vec3 d(abs(p) - b);
	const float a(min(max(d.x, max(d.y, d.z)), 0.0f));
	const vec3 d3(
		max(d.x, 0.0f),
		max(d.y, 0.0f),
		max(d.z, 0.0f)
	);
	return length(d3) + a;
}

__opensdf_shape__
static float sdSphere(glm::vec3 const& p, float r)
{
	using namespace glm;

	const float l(length(p));
	const float sd(l - r);
	return sd;
}

//
// Operator Functions
//
__opensdf_operator__
static float opUnion(float a, float b)
{
    return glm::min(a, b);
}

__opensdf_operator__
static float opDifference(float a, float b)
{
    return glm::max(a, -b);
}

__opensdf_operator__
static float opIntersection(float a, float b)
{
    return glm::max(a, b);
}

