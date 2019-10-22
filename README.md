# Volumetric Path Tracer

![banner](https://github.com/sergeneren/Volumetric-Path-Tracer/blob/master/img/banner.png)

VPT is a path tracer to render openvdb files using Nvidia gvdb library and Cuda. It uses the [Ray Tracing Gems Vol 28.](https://github.com/Apress/ray-tracing-gems/tree/master/Ch_28_Ray_Tracing_Inhomogeneous_Volumes) as the base, and implements volume rendering algorithms from [PBRT](https://www.pbrt.org/). This repo is currently built and tested only under Windows.

## Release Notes

v1.0 Initial Public Release

## Installation

Either download the source as a zip file or right click to a desired location and use below command with git bash
```
git clone https://github.com/sergeneren/Volumetric-Path-Tracer
```
### Dependencies

VPT depends on following libraries. They can be installed using ```vcpkg.exe install 'package name'``` command. 

* [GLFW3](https://www.glfw.org/) 
* [Dear Imgui](https://github.com/ocornut/imgui)
* [STB Image](https://github.com/nothings/stb)

### Build 
VPT expects [vcpkg](https://github.com/Microsoft/vcpkg), Visual Studio 2017 and CMake to be installed.  

**Step 1:** With CMake Gui select the "VPT/source" folder as source directory and create a build directory of your choice.

**Step 2:** Choose x64 for optional platform and specify toolchain for cross-compiling

![platform](https://github.com/sergeneren/Volumetric-Path-Tracer/blob/master/img/platform.JPG)

**Step 3:** Specify the location your vcpkg cmake file 

![toolchain](https://github.com/sergeneren/Volumetric-Path-Tracer/blob/master/img/toolchain.JPG)

**Step 4:** Configure with these options. If you would like to render procedural sky sampling textures to a folder before rendering, mark the "RENDER_ENV_TEXTURES" option

![render_textures](https://github.com/sergeneren/Volumetric-Path-Tracer/blob/master/img/render_textures.JPG)

**Step 5:** Generate and open the VS file. Build VPT in "Release" configuration. This will create a "VPT" folder under build directory and vpt.exe. Necessary binaries will be placed here. 
 
**note:** Currently only windows GVDB libraries (VS 2017, VC 14) are distributed with the repo. If you wish to use the repo under Linux or Mac please use this fork of [GVDB](https://github.com/sergeneren/gvdb-voxels) and compile the necessary binaries to be used with the renderer. 

### Usage 

VPT has two command line arguments: A vdb file name as first argument, and a second optional environment texture. If you wish to use an environment map with VPT just specify the hdri in command line, for example: 

```vpt.exe ../../../assets/dragon.vdb ../../../assets/Barce_Rooftop_C_3k.hdr```

You can find couple hdri maps under assets directory which are provided by [sIbl Archive](http://www.hdrlabs.com/sibl/archive.html) and [HDRI Skies](https://hdri-skies.com/).

[The Moana Cloud datasets](https://www.technology.disneyanimation.com/clouds) are Copyright 2017 Disney Enterprises, Inc. and are licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License. A copy of this license is available at http://creativecommons.org/licenses/by-sa/3.0/.

The interactive camera in application uses left mouse for orbiting, middle mouse for panning, and mouse wheel for zooming. keyboard "s" takes a screenshot and places it under "bin/render" folder with .tga extension. Keyboard "-" and "+" changes FOV and "ESC" key quits the application   

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
