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
// File: Volume instance file exporter main function file implementation
//		 this function creates a json file with given point attributes 
//
//-----------------------------------------------


//#include "nlohmann/json.hpp"
//using json = nlohmann::json;


#include "file_IO.h"
#include "volume_instance.h"

#include <ROP/ROP_API.h>
#include <ROP/ROP_Error.h>
#include <UT/UT_OFStream.h>
#include <vector>


namespace vpt_instance {


	GA_Detail::IOStatus file_save(const GU_Detail *gdp, const char *file_name) {

		UT_OFStream file(file_name);

		GA_ROHandleV3 pos_h(gdp, GA_ATTRIB_POINT, "P");
		UT_Vector3F pos_val(0, 0, 0);

		GA_ROHandleV3 N_h(gdp, GA_ATTRIB_POINT, "N");
		GA_ROHandleV3 up_h(gdp, GA_ATTRIB_POINT, "up");
		GA_ROHandleV4 orient_h(gdp, GA_ATTRIB_POINT, "orient");
		GA_ROHandleV4 rot_h(gdp, GA_ATTRIB_POINT, "rot");

		GA_ROHandleF rad_h(gdp, GA_ATTRIB_POINT, "pscale");
		fpreal32 rad_val(1);

		GA_ROHandleS vdb_h(gdp, GA_ATTRIB_POINT, "instancefile");
		UT_String vdb_val("");

		// First find out how many unique vdb files we need 
		std::vector<std::string> unique_vdb_files;
		GA_Offset lcl_start, lcl_end, ptoff;
		for (GA_Iterator lcl_it(gdp->getPointRange()); lcl_it.blockAdvance(lcl_start, lcl_end); ) {
			for (ptoff = lcl_start; ptoff < lcl_end; ++ptoff) {
				if (vdb_h.isValid()) {
					vdb_val = vdb_h.get(ptoff);
					unique_vdb_files.push_back(vdb_val.c_str());
				}
				else {
					return false;
				}
			}
		}

		
		std::sort(unique_vdb_files.begin(), unique_vdb_files.end());
		std::vector<std::string>::iterator it = std::unique(unique_vdb_files.begin(), unique_vdb_files.end());
		unique_vdb_files.resize(std::distance(unique_vdb_files.begin(), it));
		

		file << unique_vdb_files.size() << std::endl;

		for (auto vdb : unique_vdb_files) {

			vdb_instance new_instance;
			new_instance.vdb_file = vdb;
			file << vdb << std::endl;

			int idx = 0;
			for (GA_Iterator lcl_it(gdp->getPointRange()); lcl_it.blockAdvance(lcl_start, lcl_end); ) {
				for (ptoff = lcl_start; ptoff < lcl_end; ++ptoff) {

					vdb_val = vdb_h.get(ptoff);

					if (vdb_val == UT_String(vdb)) {
						instance ins;

						// Check if pscale exists if not set it to 1
						if (rad_h.isValid()) {
							rad_val = rad_h.get(ptoff);
						}
						else rad_val = 1.0f;
						ins.scale = rad_val;

						// For rotations we first check if orient attribute is peresent 
						
						UT_QuaternionF quat(0, 0, 0, 1); // this will set the instance rotation 
												
						if (orient_h.isValid()) {
							quat = orient_h.get(ptoff);
						}
						else { // orient is not present. 

							if (rot_h.isValid()) { // check if rot attribute is present 
								quat = rot_h.get(ptoff);
							}
							else { // neither attributes are present we should construct our own quaternion 
								UT_Vector3F up(0,1,0);
								if (up_h.isValid())	up = up_h.get(ptoff);

								UT_Vector3F normal(0,0,1);
								if (N_h.isValid())	normal = N_h.get(ptoff);

								UT_Matrix3F rot_matrix;
								rot_matrix.orient(normal, up);
								
								quat.updateFromRotationMatrix(rot_matrix);
							}
						}

						ins.rotation[0] = quat.x();
						ins.rotation[1] = quat.y();
						ins.rotation[2] = quat.z();
						ins.rotation[3] = quat.w();

						// Get the position of point and set instance position
						pos_val = pos_h.get(ptoff);
						ins.position[0] = pos_val.x();
						ins.position[1] = pos_val.y();
						ins.position[2] = pos_val.z();

						new_instance.instances.push_back(ins);

					}

				}
			}


			new_instance.num_instances = new_instance.instances.size();
			file << new_instance.num_instances << std::endl;
			for (int i = 0; i < new_instance.num_instances; ++i) {
				file << new_instance.instances.at(i).position[0];
				file << " " << new_instance.instances.at(i).position[1];
				file << " " << new_instance.instances.at(i).position[2];

				file << " " << new_instance.instances.at(i).rotation[0];
				file << " " << new_instance.instances.at(i).rotation[1];
				file << " " << new_instance.instances.at(i).rotation[2];
				file << " " << new_instance.instances.at(i).rotation[3];

				file << " " << new_instance.instances.at(i).scale;

				file << std::endl;
			}

		}


		file.close();
		return true;

	}


}