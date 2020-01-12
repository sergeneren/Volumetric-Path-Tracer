[![Build status](https://ci.appveyor.com/api/projects/status/cbtij8mhtx87wche/branch/master?svg=true)](https://ci.appveyor.com/project/sergeneren/volumetric-path-tracer/branch/master)

# Volumetric Path Tracer

![banner](https://github.com/sergeneren/Volumetric-Path-Tracer/blob/master/img/VPT_Banner.gif)

VPT is a volumetric path tracer to render openvdb files on gpu using Cuda. It uses the [Ray Tracing Gems Vol 28.](https://github.com/Apress/ray-tracing-gems/tree/master/Ch_28_Ray_Tracing_Inhomogeneous_Volumes) as the base, and implements volume rendering algorithms from [PBRT](https://www.pbrt.org/). Features of VPT is listed below 

* Ability to render Open VDB files with thousands of ray depths on gpu
* Realistic lighting with a procedural atmosphere and sun system
* HDRI maps for environmental lighting
* Point lights 
* Eric Bruneton style sky implementation
* Depth of field 
* Volume emission 
* Ability to render planetary atmospheres   
* Instanced rendering of vdb files with custom file format (.ins)
* BVH and Octree structures for fast ray traversal
* Custom instance file writer plugin for Houdini written in HDK 

This repo is currently built and tested only under Windows.

## Release Notes

v 1.0.2 Alpha

*Please see the [releases](https://github.com/sergeneren/Volumetric-Path-Tracer/releases) section for release notes.*

## Installation, Build and Usage

Please see [this](https://sergeneren.com/2020/01/07/using-vpt/) detailed article for installing building and using VPT    

## Author

* **Sergen Eren** - [My website](https://sergeneren.com)

## Status
This project is under active maintenance and development

## License
This project is licensed under BSD 3-Clause License

## Acknowledgments
* [PBRT](https://github.com/mmp/pbrt-v3/) - *Big thanks to Matt Pharr, Wenzel Jakob and Greg Humphreys*
* [GVDB](https://github.com/NVIDIA/gvdb-voxels) - *Nvidia Sparse Voxel Database*
* [Ray Tracing Gems](http://www.realtimerendering.com/raytracinggems/) - *Base of VPT*
* [Walt Disney Animation Studios](https://www.disneyanimation.com/) - *Moana Cloud Dataset*
