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
// File: Volume instance file exporter for VPT
//
//-----------------------------------------------


#ifndef _ROP_VPT_INSTANCE_H_
#define _ROP_VPT_INSTANCE_H_


#include <UT/UT_String.h>
#include <UT/UT_Vector2.h>
#include <UT/UT_Vector3.h>
#include <UT/UT_Vector4.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_DSOVersion.h>

#include <ROP/ROP_Node.h>
#include <ROP/ROP_Error.h>
#include <ROP/ROP_Templates.h>
#include <ROP/ROP_API.h>
#include <ROP/ROP_Error.h>

#include <PRM/PRM_Include.h>
#include <PRM/PRM_SpareData.h>
#include <PRM/PRM_ChoiceList.h>

#include <OP/OP_Operator.h>
#include <OP/OP_Director.h>
#include <OP/OP_OperatorTable.h>
#include <OP/OP_AutoLockInputs.h>

#include <SOP/SOP_Node.h>

#include <GU/GU_Detail.h>

#include <GA/GA_AIFTuple.h>
#include <GA/GA_Iterator.h>
#include <GA/GA_Types.h>
#include <GA/GA_AttributeFilter.h>

#include <SYS/SYS_Math.h>



namespace vpt_instance {


	enum {
		ROP_VPT_RENDER,
		ROP_VPT_RENDERBACKGROUND,
		ROP_VPT_RENDER_CTRL,
		ROP_VPT_TRANGE,
		ROP_VPT_FRANGE,
		ROP_VPT_TAKE,
		ROP_VPT_SOPPATH,
		ROP_VPT_SOPOUTPUT,

		//render parameters
		ROP_VPT_INITSIM,
		ROP_VPT_MKPATH,
		ROP_VPT_ALFPROGRESS,
		ROP_VPT_TPRERENDER,
		ROP_VPT_PRERENDER,
		ROP_VPT_LPRERENDER,
		ROP_VPT_TPREFRAME,
		ROP_VPT_PREFRAME,
		ROP_VPT_LPREFRAME,
		ROP_VPT_TPOSTFRAME,
		ROP_VPT_POSTFRAME,
		ROP_VPT_LPOSTFRAME,
		ROP_VPT_TPOSTRENDER,
		ROP_VPT_POSTRENDER,
		ROP_VPT_LPOSTRENDER,

		ROP_VPT_MAXPARMS
	};


	class VPT_INS_ROP : public ROP_Node {


	public:
		static OP_TemplatePair      *getTemplatePair();
		static OP_VariablePair      *getVariablePair();
		static PRM_Template			myTemplateList[];
		static OP_Node              *nodeConstructor(OP_Network *net, const char *name, OP_Operator *op);


	protected:

		VPT_INS_ROP(OP_Network *net, const char *name, OP_Operator *op);

		virtual ~VPT_INS_ROP();



	protected:

		virtual int startRender(int nframes, fpreal s, fpreal e);
		virtual ROP_RENDER_CODE renderFrame(fpreal time, UT_Interrupt *boss);
		virtual ROP_RENDER_CODE endRender();


	private:
		void	OUTPUT(UT_String &str, fpreal t) { return evalString(str, "outputFile", 0, t); }
		void	SOPPATH(UT_String &str, fpreal t) { return evalString(str, "soppath", 0, t); }
		int		INITSIM() { return evalInt("initsim", 0, 0); }
		int		ALFPROGRESS() { return evalInt("alfprogress", 0, 0); }


		fpreal  startTime;
		fpreal  endTime;

	};

}











#endif // !_ROP_VPT_INSTANCE_H_
