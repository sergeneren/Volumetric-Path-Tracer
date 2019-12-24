# Instancer HDA


Instancer HDA is an *".ins"* file writer ROP plugin for Houdini. This file is used with VPT to render volume instances.     


## Release Notes

v 1.0.0 Alpha

## Installation

Instancer HDA comes with VPT and no further installation is required. 

### Build 

Build instancer_hda with release configuration and it will install itself as a post build step to *$(HOME)/$(Houdini_VERSION)/dso* directory. Currently houdini cmake directory is hardcoded for Houdini 18.0.287 but you can change it in CMake.   

 
### Usage 

Create a point cloud with string attribute *"instancefile"* pointing to the vdb file you want to instance. 

For creating a transformation matrix Instancer uses the following attributes. 

* Point position is written as translation 
* Instancer uses pscale to set the scale of the instance
* For rotations Instancer uses couple options 
	* If *"orient"* quaternion attribute is present it is used with first priority
	* *"rot"* attribute is used if *"orient"* is not present secondary priority
	* If both quaternion attributes are missing, Instancer uses *"N"* and *"up"* to create an orient rotation
	* If none is present default values are written and VPT creates an identity matrix 

An example file is provided under scenes folder. 
	
## Author

* **Sergen Eren** - [My website](https://sergeneren.com)

## Status
This library is under active maintenance

## License
This project is licensed under BSD 3-Clause License
