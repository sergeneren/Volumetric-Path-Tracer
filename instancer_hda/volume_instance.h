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


struct instance {
	double position[3]; // x,y,z
	double rotation[4]; // x,y,z,w
	double scale = 0;
};

struct vdb_instance {
	unsigned int num_instances = 0;
	const char *vdb_file;
	std::vector<instance> instances;
};

std::ostream& operator<<(std::ostream& stream, instance &inst);
std::ifstream& operator>>(std::ifstream& stream, instance &inst);

std::ostream& operator<<(std::ostream& stream, instance &inst) {


	stream.write(reinterpret_cast<char*>(&inst.position[0]), sizeof(double));
	stream.write(reinterpret_cast<char*>(&inst.position[1]), sizeof(double));
	stream.write(reinterpret_cast<char*>(&inst.position[2]), sizeof(double));

	stream.write(reinterpret_cast<char*>(&inst.rotation[0]), sizeof(double));
	stream.write(reinterpret_cast<char*>(&inst.rotation[1]), sizeof(double));
	stream.write(reinterpret_cast<char*>(&inst.rotation[2]), sizeof(double));
	stream.write(reinterpret_cast<char*>(&inst.rotation[3]), sizeof(double));

	stream.write(reinterpret_cast<char*>(&inst.scale), sizeof(double));

	return stream;
}

std::ifstream& operator>>(std::ifstream& stream, instance &inst) {

	stream.read(reinterpret_cast<char*>(&inst.position[0]), sizeof(double));
	stream.read(reinterpret_cast<char*>(&inst.position[1]), sizeof(double));
	stream.read(reinterpret_cast<char*>(&inst.position[2]), sizeof(double));

	stream.read(reinterpret_cast<char*>(&inst.rotation[0]), sizeof(double));
	stream.read(reinterpret_cast<char*>(&inst.rotation[1]), sizeof(double));
	stream.read(reinterpret_cast<char*>(&inst.rotation[2]), sizeof(double));
	stream.read(reinterpret_cast<char*>(&inst.rotation[3]), sizeof(double));

	stream.read(reinterpret_cast<char*>(&inst.scale), sizeof(double));

	return stream;
}


#endif // !_VOLUME_INSTANCE_H_





