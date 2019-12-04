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
//	Version 1.0: Sergen Eren, 03/12/2019
//
// File: Volume instance file structure
//
//-----------------------------------------------


#ifndef _VOLUME_INSTANCE_H_
#define _VOLUME_INSTANCE_H_

#include <vector>
#include <iostream>
#include <fstream>


constexpr auto MAX_FILE_PATH_LENGTH = 128;

struct instance {
	double position[3] = {0,0,0}; // x,y,z
	double rotation[4] = { 0,0,0,0}; // x,y,z,w
	double scale = 0;
};

struct vdb_instance {
	unsigned int num_instances = 0;
	const char *vdb_file = new char[MAX_FILE_PATH_LENGTH];
	std::vector<instance> instances;
};

std::ostream& operator<<(std::ostream& stream, vdb_instance & vdb);
std::ifstream& operator>>(std::ifstream& stream, vdb_instance &vdb);

std::ostream& operator<<(std::ostream& stream, vdb_instance & vdb) {


	stream.write(reinterpret_cast<char*>(&vdb.num_instances), sizeof(int));
	stream.write(reinterpret_cast<char*>(&vdb.vdb_file), sizeof(char)*MAX_FILE_PATH_LENGTH);
	
	for (int i = 0; i < vdb.num_instances; ++i) {
		stream.write(reinterpret_cast<char*>(&vdb.instances.at(i).position[0]), sizeof(double));
		stream.write(reinterpret_cast<char*>(&vdb.instances.at(i).position[1]), sizeof(double));
		stream.write(reinterpret_cast<char*>(&vdb.instances.at(i).position[2]), sizeof(double));
		
		stream.write(reinterpret_cast<char*>(&vdb.instances.at(i).rotation[0]), sizeof(double));
		stream.write(reinterpret_cast<char*>(&vdb.instances.at(i).rotation[1]), sizeof(double));
		stream.write(reinterpret_cast<char*>(&vdb.instances.at(i).rotation[2]), sizeof(double));
		stream.write(reinterpret_cast<char*>(&vdb.instances.at(i).rotation[3]), sizeof(double));
		
		stream.write(reinterpret_cast<char*>(&vdb.instances.at(i).scale), sizeof(double));
	}

	return stream;
}

std::ifstream& operator>>(std::ifstream& stream, vdb_instance &vdb) {
	
	stream.read(reinterpret_cast<char*>(&vdb.num_instances), sizeof(int));
	vdb.instances.resize(vdb.num_instances);

	stream.read(reinterpret_cast<char*>(&vdb.vdb_file), sizeof(char)*MAX_FILE_PATH_LENGTH);

	for (int i = 0; i < vdb.num_instances; ++i) {
		stream.read(reinterpret_cast<char*>(&vdb.instances.at(i).position[0]), sizeof(double));
		stream.read(reinterpret_cast<char*>(&vdb.instances.at(i).position[1]), sizeof(double));
		stream.read(reinterpret_cast<char*>(&vdb.instances.at(i).position[2]), sizeof(double));

		stream.read(reinterpret_cast<char*>(&vdb.instances.at(i).rotation[0]), sizeof(double));
		stream.read(reinterpret_cast<char*>(&vdb.instances.at(i).rotation[1]), sizeof(double));
		stream.read(reinterpret_cast<char*>(&vdb.instances.at(i).rotation[2]), sizeof(double));
		stream.read(reinterpret_cast<char*>(&vdb.instances.at(i).rotation[3]), sizeof(double));

		stream.read(reinterpret_cast<char*>(&vdb.instances.at(i).scale), sizeof(double));
	}

	return stream;
}


#endif // !_VOLUME_INSTANCE_H_





