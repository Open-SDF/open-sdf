# Open-SDF

## Purpose 
To define a standardized file format for exchanging 3D models represented as signed distance functions, facilitating interoperability between different software tools while accounting for varying computational capabilities and specific use cases.

## Motivation
Modeling 3D objects with implicit surfaces and more specifically Signed Distance Functions has been around for decades. In recent years, it has become increasingly practical due to the ubiquity of massively parallel hardware, the GPU. Today, various commercial products exist for 3D modeling, graphics design and games. 

TODO: List Products

While development of such products is still young in certain aspects, we intend to define an exchange format such that creators using such products can take advantage of an ecosystem rather than having their content locked into a single application. While encouraging healthy competition with regards to implementation and variety of use cases, we see the opportunity for synergies between various applications.

## Scope
### Included
* A list of functions (the math/code of the shapes)
* A list of operators (the math/code of combining operators)
* A set of global limits (minimum shape count, etc)
* A system to deal with support for evaluation depths
* A set of tiers organized by objective quality measures
* A reference implementation for evaluating the composed functions
* A test suite to automatically evaluate as many of the objective quality measures as possible
* A file format to store and load all parameters of a single model

### Not Included
* A scene graph format. No need to reinvent. It should be possible to define extensions to include OpenSDF models in popular scene formats such as GLTF or USD.
* A definition of a fully CSG tree of arbitrary depth (this is not very friendly for GPU evaluation).
* A definition of a culling algorithm. The functions are evaluated in tiers by quality criteria useful for culling algorithms (e.g. Interval Analysis), however the specific of the algorithms are left to the implementations.
Discretized implicit surfaces (3D textures).
Vendor extensions. Instead we encourage a progression of implementation complexity with tiers. 

## Overview
* Objective: Create a flexible, scalable interchange format for 3D models defined from simple to complex Signed Distance Function shapes, which are combined with Boolean operations.
* Non-destructive: The format is focused on representing Signed Distance Function models that are represented by their individual, parameterized shapes and operations. This is specifically differentiated to implicit surfaces represented by discretized SDF functions (“3D textures of scalar SDF values”), which suffer from large file sizes as well as loss in information (e.g. are piece-wise cubic surface patches) with regards to their originating surface definition. 
* Tiers: The format is tiered to accommodate different levels of desired complexity and specificity in the implementing 3D modeling applications. This specifically moves away from the approach of a “core format” plus “vendor extensions” which we deem as less suitable for an exchange format as it provides mostly arbitrary basis for decision of supporting various extensions. In contrast, tiers allow a well defined and ranked progressive increase in support, if an implementation might choose so.

## Tiers Definition
We intend to rank the list of functions and operators by the following criteria and categorize them into meaningful tiers. This should optimize for implementation simplicity, maximize expressiveness in the lower tiers as well as provide a linear path for an implementation to increase support.

### Objective Criteria
* Runtime / ALU complexity
* Register pressure
* Bandwidth pressure / number of parameters
Quality of the resulting field. Baseline is a pure L2-norm signed distance function that is Lipschitz-continuous, e.g. the length of the gradient is 1 for the whole domain.

### Non-Objective Criteria
* Commonness of shapes vs specificity of shapes. E.g.: A sphere is more commonly used in modeling than a star with 5 vertices.
* what else?

### Redundancies
Shapes defined by known Signed Distance Functions exhibit a certain degree of redundancy. For instance, the function for a rounded box is a superset of the function of a box with sharp edges. A sphere can be represented by a Superprim. While these geometric supersets might rank into a higher tier due to the above criteria, it is up to the implementation to decide on how the shapes are implemented.

## Functions and Operators by Tier

### Shapes

| Tier  | Shape  | Runtime Complexity  | Register Pressure  | Bandwidth Pressure  | Field Quality | Commonness|
|---|---|---|---|---|---|---|
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |

## Reference Implementation

The reference implementation provides the following purposes:

1) Providing a baseline function for various shapes and operators (in many cases these can improve over time by e.g. community contributions)
2) Providing a framework for evaluating the above objective criteria
3) Providing an implementation to load and save Open-SDF models

### Code of Shapes and Operators

We implement the functions in C/C++. TODO...

### Measurement Framework

Since the most likey implementation is on a GPU, we choose our evaluation framework accordingly. Candidates are various APIs and various vendors. Initially we choose CUDA since we can use C++ interchangably on the GPU compute and the CPU to evaluate the above criteria. It also provides ISA dissambly as well as static analysis of register pressure. We can later extend this to AMD, Intel, Apple, etc.

TODO: write the framework

### File Format

TODO: probably JSON with the option to use binary JSON formats.

### Dependencies
 * GLM
 * Premake 5
 * CUDA

## Contributing

We need to define some framework of managing contributions, including clearing code from potential IP rettrictions. TODO: define CLA, licenses etc.
