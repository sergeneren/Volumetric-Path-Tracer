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

#include "file_IO.h"

#include <ROP/ROP_API.h>
#include <ROP/ROP_Error.h>

#include <vector>

//#include "nlohmann/json.hpp"
//using json = nlohmann::json;

namespace vpt_instance {


	GA_Detail::IOStatus file_save(const GU_Detail *gdp, const char *file_name) {


		GA_ROHandleV3 pos_h(gdp, GA_ATTRIB_POINT, "P");
		UT_Vector3F pos_val(0, 0, 0);

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

		for (std::vector<std::string>::const_iterator i = unique_vdb_files.begin(); i != unique_vdb_files.end(); ++i) {
			std::cout << *i << "\n";
		}

		return true;

	}


}