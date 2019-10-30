//--------------------------------------------------------------------------------
//
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the following conditions are met :
//
//	*Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer.
//
//	* Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution.
//	
//	* Neither the name of the copyright holder nor the names of its
//	contributors may be used to endorse or promote products derived from
//	this software without specific prior written permission.
//	
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//	DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//	OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Copyright(c) 2019, Sergen Eren
// All rights reserved.
//----------------------------------------------------------------------------------
// 
//	Version 1.0: Sergen Eren, 25/10/2019
//
// File: This is the implementation file for GPU_VDB that converts and loads
//		 a vdb file to CUDA 3d texture and loads it into gpu
//
//-----------------------------------------------


#include "gpu_vdb.h"
#include "driver_types.h"
#include <helper_cuda.h>
#include <helper_math.h>

#include <filesystem>


GPU_VDB::GPU_VDB(){}

GPU_VDB::~GPU_VDB(){}


VDB_INFO* GPU_VDB::get_vdb_info() {

	return &vdb_info;

}

void GPU_VDB::fill_texture(openvdb::GridBase::Ptr gridBase, cudaTextureObject_t &texture){

	openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(gridBase);

	openvdb::FloatTree tree = grid->tree();
	openvdb::CoordBBox bbox;
	tree.evalActiveVoxelBoundingBox(bbox);

	//Create dense volume
	openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense(bbox);
	openvdb::tools::copyToDense(tree, dense);
	dense.print();
	
	std::cout << "value count: " << dense.valueCount() << "\n";

	int dim_x = bbox.dim().x();
	int dim_y = bbox.dim().y();
	int dim_z = bbox.dim().z();

	cudaExtent vol_size;
	vol_size.width = dim_x;
	vol_size.height = dim_y;
	vol_size.depth = dim_z;

	float *volume_data_host = (float *)malloc(dim_x * dim_y * dim_z * sizeof(float));

	// Copy vdb values
	for (int z = 0; z < dim_z; z++) {
		for (int y = 0; y < dim_y; y++) {
			for (int x = 0; x < dim_x; x++) {
				int idx = z * dim_x*dim_y + y * dim_x + x;
				volume_data_host[idx] = dense.getValue(idx);
			}
		}
	}


	// create 3D array
	cudaArray *d_volumeArray = 0;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, vol_size));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(volume_data_host, vol_size.width * sizeof(float), vol_size.width, vol_size.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = vol_size;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));


	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_volumeArray;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = true; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear; // linear interpolation

	texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeElementType;
	//texDescr.readMode = cudaReadModeNormalizedFloat;

	checkCudaErrors(cudaCreateTextureObject(&texture, &texRes, &texDescr, NULL));
	   	  
}

bool GPU_VDB::loadVDB(std::string filename, std::string density_channel, std::string emission_channel){

	if (!std::experimental::filesystem::exists(filename)) {

		printf("!File doesn't exists '%s'\n", filename.c_str());
		return false;

	}

	if (filename.empty())
	{
		printf("Error finding file '%s'\n", filename.c_str());
		return false;
	}

	if (density_channel.empty()) {

		printf("Density channel can't be empty!!!");
		return false;

	}

	openvdb::initialize();
	openvdb::io::File file(filename);
	file.open();

	// Print grid attributes
	auto grids = file.readAllGridMetadata();
	for (auto grid : *grids) {
		std::cout << "Grid: " << grid->getName() << " " << grid->valueType() << "\n";
	}

	// Read density and emission channel from the file 
	openvdb::GridBase::Ptr densityGridBase;
	openvdb::GridBase::Ptr emissionGridBase;

	for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter)
	{
		// Read in only the grid we are interested in.
		if (nameIter.gridName() == density_channel) {
			densityGridBase = file.readGrid(nameIter.gridName());
		}
		else if (!emission_channel.empty() && nameIter.gridName() == emission_channel) {
			emissionGridBase = file.readGrid(nameIter.gridName());
		}
		else {
			std::cout << "skipping grid " << nameIter.gridName() << std::endl;
		}
	}
	openvdb::MetaMap::Ptr filemetadata = file.getMetadata();
	file.close();

	// Copy the grids to 3d cuda arrays 

	fill_texture(densityGridBase, vdb_info.density_texture);

	// Fill emission channel if specified
	if (!emission_channel.empty()) {

		fill_texture(emissionGridBase, vdb_info.emission_texture);
	}
	
	// Fill vdb_info
	openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(densityGridBase);
	openvdb::CoordBBox bbox;
	grid->tree().evalActiveVoxelBoundingBox(bbox);

#if 0
	// print all metadata

	openvdb::math::UniformScaleMap::Ptr map = grid->transform().map<openvdb::math::UniformScaleMap>();
	
	std::cout << "meta count: " << filemetadata->metaCount() << "\n";

	// Read all file meta data 

	const openvdb::MetaMap& metamap = static_cast<const openvdb::MetaMap>(*filemetadata);
	metamap.beginMeta();
	for (openvdb::MetaMap::ConstMetaIterator metaIt = metamap.beginMeta(),
		metaEnd = metamap.endMeta(); metaIt != metaEnd; ++metaIt) {
	
		openvdb::Metadata::Ptr meta = metaIt->second;
		std::cout << "meta type: " << meta->typeName() << "\n";
	}


	// Read all grid metadata
	for(openvdb::MetaMap::MetaIterator iter = densityGridBase->beginMeta();
		iter != densityGridBase->endMeta(); ++iter)
	{
		const std::string& name = iter->first;
		openvdb::Metadata::Ptr value = iter->second;
		std::string valueAsString = value->str();
		std::cout << name << " = " << valueAsString << std::endl;
	}


	grid->transform().print();


#endif
	
	set_vec3s(vdb_info.bmin, bbox.min().asVec3s());
	set_vec3s(vdb_info.bmax, bbox.max().asVec3s());
	set_vec3i(vdb_info.dim, bbox.dim().asVec3i());

	
	vdb_info.epsilon = FLT_EPSILON;
	vdb_info.voxelsize = float(grid->voxelSize()[0]);

	openvdb::Mat4R ref_xform = grid->transform().baseMap()->getAffineMap()->getMat4();
	
	set_xform(this->xform, ref_xform);

	return true;
}
