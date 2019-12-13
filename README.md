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

*Please see the releases section for release notes.*

## Installation

Either download the source as a zip file or right click to a desired location and use below command with git bash
```
git clone https://github.com/sergeneren/Volumetric-Path-Tracer
```


### Dependencies

VPT depends on following libraries. All the libraries are choosen so they can be easily installed by vcpkg using ```vcpkg.exe install 'package name'``` command. 

* [OpenVDB](https://www.openvdb.org/)
* [GLFW3](https://www.glfw.org/) 
* [Dear Imgui](https://github.com/ocornut/imgui)
* [STB Image](https://github.com/nothings/stb)
* [tinyexr](https://github.com/syoyo/tinyexr)

### Build 
VPT expects [vcpkg](https://github.com/Microsoft/vcpkg), Visual Studio 2017 and CMake to be installed.  

**Step 1:** With CMake Gui select the "VPT" folder as source directory and create a build directory of your choice.

**Step 2:** Choose x64 for optional platform and specify toolchain for cross-compiling

![platform](https://github.com/sergeneren/Volumetric-Path-Tracer/blob/master/img/platform.JPG)

**Step 3:** Specify the location your vcpkg cmake file 

![toolchain](https://github.com/sergeneren/Volumetric-Path-Tracer/blob/master/img/toolchain.JPG)

**Step 4:** Configure with these options. If you would like to render procedural sky sampling textures to a folder before rendering, mark the "RENDER_ENV_TEXTURES" option

![render_textures](https://github.com/sergeneren/Volumetric-Path-Tracer/blob/master/img/render_textures.JPG)

**Step 5:** Generate and open the VS file. Build VPT in "Release" configuration. This will create a "VPT" folder under build directory and vpt.exe. Necessary binaries will be placed here. 
 
### Usage 

VPT has two command line arguments: A vdb file name or a an instance file (.ins format) as the first argument, and a second optional environment texture. If you wish to use an environment map with VPT just specify the hdri in command line, for example: 

```.\vpt.exe dragon.vdb Barce_Rooftop_C_3k.hdr```

Currently VPT expects vdb and hdri files to be under assets directory. This directory is assigned at cmake during configuration. 

You can find couple hdri maps under assets directory which are provided by [sIbl Archive](http://www.hdrlabs.com/sibl/archive.html) and [HDRI Skies](https://hdri-skies.com/).

[The Moana Cloud datasets](https://www.technology.disneyanimation.com/clouds) are Copyright 2017 Disney Enterprises, Inc. and are licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License. A copy of this license is available at http://creativecommons.org/licenses/by-sa/3.0/.

The interactive camera in application uses left mouse for orbiting, middle mouse for panning, and mouse wheel for zooming. keyboard "s" takes a screenshot and places it under "bin/render" folder with .tga or .exr extension. Keyboard "-" and "+" changes FOV and "ESC" key quits the application. "F" key frames the current vdb file so it is in the cameras fov.     

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
