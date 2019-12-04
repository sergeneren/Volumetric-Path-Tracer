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


#include "ROP_VPT_Instance.h"


using namespace vpt_instance;




static PRM_Name names[] = {
	PRM_Name("outputFile",     "Output File"),
	PRM_Name("soppath", "SOP Path")
};

static PRM_Default	 theFileDefault(0, "$HIP/Outputs/VPT/defgeo.json");
const char *nameName = "`$OS`";
static PRM_Default	 theNameDefault(0, nameName);


PRM_Template VPT_INS_ROP::myTemplateList[] = {
	PRM_Template(PRM_FILE_E,	1, &names[0], &theFileDefault , 0, 0 , 0 ,  &PRM_SpareData::fileChooserModeWrite),	// file Output
	PRM_Template(PRM_STRING, PRM_TYPE_DYNAMIC_PATH, 1, &names[1], 0, 0, 0, 0, &PRM_SpareData::sopPath),			// sop path
	PRM_Template()																									// placeholder
};

// End custom template list

static PRM_Name         MKPATH_PRM_NAME("mkpath", "Create Intermediate Directories");
static PRM_Name         INITSIM_PRM_NAME("initsim", "Initialize Simulation OPs");
static PRM_Name         ALFPROGRESS_PRM_NAME("alfprogress", "Alfred Style Progress");

static PRM_Template * getTemplates()
{
	static PRM_Template *prmTemplate = 0;

	if (prmTemplate)
	{
		return prmTemplate;
	}

	prmTemplate = new PRM_Template[ROP_VPT_MAXPARMS + 1];

	prmTemplate[ROP_VPT_RENDER] = theRopTemplates[ROP_RENDER_TPLATE];
	prmTemplate[ROP_VPT_RENDERBACKGROUND] = theRopTemplates[ROP_RENDERBACKGROUND_TPLATE];
	prmTemplate[ROP_VPT_RENDER_CTRL] = theRopTemplates[ROP_RENDERDIALOG_TPLATE];
	prmTemplate[ROP_VPT_TRANGE] = theRopTemplates[ROP_TRANGE_TPLATE];
	prmTemplate[ROP_VPT_FRANGE] = theRopTemplates[ROP_FRAMERANGE_TPLATE];
	prmTemplate[ROP_VPT_TAKE] = theRopTemplates[ROP_TAKENAME_TPLATE];
	prmTemplate[ROP_VPT_SOPOUTPUT] = VPT_INS_ROP::myTemplateList[0];
	prmTemplate[ROP_VPT_SOPPATH] = VPT_INS_ROP::myTemplateList[1];

	prmTemplate[ROP_VPT_TPRERENDER] = theRopTemplates[ROP_TPRERENDER_TPLATE];
	prmTemplate[ROP_VPT_PRERENDER] = theRopTemplates[ROP_PRERENDER_TPLATE];
	prmTemplate[ROP_VPT_LPRERENDER] = theRopTemplates[ROP_LPRERENDER_TPLATE];
	prmTemplate[ROP_VPT_TPREFRAME] = theRopTemplates[ROP_TPREFRAME_TPLATE];
	prmTemplate[ROP_VPT_PREFRAME] = theRopTemplates[ROP_PREFRAME_TPLATE];
	prmTemplate[ROP_VPT_LPREFRAME] = theRopTemplates[ROP_LPREFRAME_TPLATE];
	prmTemplate[ROP_VPT_TPOSTFRAME] = theRopTemplates[ROP_TPOSTFRAME_TPLATE];
	prmTemplate[ROP_VPT_POSTFRAME] = theRopTemplates[ROP_POSTFRAME_TPLATE];
	prmTemplate[ROP_VPT_LPOSTFRAME] = theRopTemplates[ROP_LPOSTFRAME_TPLATE];
	prmTemplate[ROP_VPT_TPOSTRENDER] = theRopTemplates[ROP_TPOSTRENDER_TPLATE];
	prmTemplate[ROP_VPT_POSTRENDER] = theRopTemplates[ROP_POSTRENDER_TPLATE];
	prmTemplate[ROP_VPT_LPOSTRENDER] = theRopTemplates[ROP_LPOSTRENDER_TPLATE];
	prmTemplate[ROP_VPT_MKPATH] = theRopTemplates[ROP_MKPATH_TPLATE];
	prmTemplate[ROP_VPT_INITSIM] = theRopTemplates[ROP_INITSIM_TPLATE];
	prmTemplate[ROP_VPT_ALFPROGRESS] = PRM_Template(PRM_TOGGLE, 1, &ALFPROGRESS_PRM_NAME, PRMzeroDefaults);

	prmTemplate[ROP_VPT_MAXPARMS] = PRM_Template();

	return prmTemplate;
};

OP_TemplatePair * VPT_INS_ROP::getTemplatePair()
{
	static OP_TemplatePair *ropPair = 0;

	if (ropPair)
	{
		return ropPair;
	}

	ropPair = new OP_TemplatePair(getTemplates());

	return ropPair;
};

OP_VariablePair * VPT_INS_ROP::getVariablePair()
{
	static OP_VariablePair *varPair = 0;

	if (varPair)
	{
		return varPair;
	}

	varPair = new OP_VariablePair(ROP_Node::myVariableList);

	return varPair;
};

// start, end and render frames are auto invoked by houdini

int VPT_INS_ROP::startRender(int /*nframes*/, fpreal tstart, fpreal tend)
{
	int			 rcode = 1;

	endTime = tend;
	startTime = tstart;

	if (INITSIM())
	{
		initSimulationOPs();
		OPgetDirector()->bumpSkipPlaybarBasedSimulationReset(1);
	}

	if (error() < UT_ERROR_ABORT)
	{
		if (!executePreRenderScript(tstart))
			return 0;
	}

	return rcode;
}


ROP_RENDER_CODE VPT_INS_ROP::renderFrame(fpreal time, UT_Interrupt *)
{

	SOP_Node		*sop;
	UT_String		 soppath, savepath, name;

	OUTPUT(savepath, time);
	//NAME(name, time);

	if (!executePreFrameScript(time))
		return ROP_ABORT_RENDER;

	// From here, establish the SOP that will be rendered, if it cannot
	// be found, return 0.
	// This is needed to be done here as the SOPPATH may be time
	// dependent (ie: OUT$F) or the perframe script may have
	// rewired the input to this node.

	sop = CAST_SOPNODE(getInput(0));
	if (sop)
	{
		// If we have an input, get the full path to that SOP.
		sop->getFullPath(soppath);
	}
	else
	{
		// Otherwise get the SOP Path from our parameter.
		SOPPATH(soppath, time);
	}

	if (!soppath.isstring())
	{
		addError(ROP_MESSAGE, "Invalid SOP path");
		return ROP_ABORT_RENDER;
	}

	sop = getSOPNode(soppath, 1);
	if (!sop)
	{
		addError(ROP_COOK_ERROR, (const char *)soppath);
		return ROP_ABORT_RENDER;
	}
	OP_Context		context(time);
	GU_DetailHandle gdh;
	gdh = sop->getCookedGeoHandle(context);

	GU_DetailHandleAutoReadLock	 gdl(gdh);
	const GU_Detail		*gdp = gdl.getGdp();

	if (!gdp)
	{
		addError(ROP_COOK_ERROR, (const char *)soppath);
		return ROP_ABORT_RENDER;
	}

	if (evalInt("mkpath", 0, 0)) {
		ROP_Node::makeFilePathDirs(savepath);
	}

	//h2a_fileSave(gdp, (const char *)savepath, (const char *)name, PWIDTH(time), SHUTTER(time), MODE(), MOTIONB(), COLOR(), TYPE(), P_RENDER_TYPE(), RADIUS(time), SUBDIV_TYPE(), SUBDIV_ITE(time));

	if (ALFPROGRESS() && (endTime != startTime))
	{
		fpreal		fpercent = (time - startTime) / (endTime - startTime);
		int		percent = (int)SYSrint(fpercent * 100);
		percent = SYSclamp(percent, 0, 100);
		fprintf(stdout, "ALF_PROGRESS %d%%\n", percent);
		fflush(stdout);
	}

	if (error() < UT_ERROR_ABORT)
	{
		if (!executePostFrameScript(time))
			return ROP_ABORT_RENDER;
	}

	return ROP_CONTINUE_RENDER;
}

ROP_RENDER_CODE VPT_INS_ROP::endRender()
{
	if (INITSIM())
		OPgetDirector()->bumpSkipPlaybarBasedSimulationReset(-1);

	if (error() < UT_ERROR_ABORT)
	{
		if (!executePostRenderScript(endTime))
			return ROP_ABORT_RENDER;
	}
	return ROP_CONTINUE_RENDER;
}

VPT_INS_ROP::~VPT_INS_ROP(){}


OP_Node * VPT_INS_ROP::nodeConstructor(OP_Network *net, const char *name, OP_Operator *op)
{
	return new VPT_INS_ROP(net, name, op);
};


VPT_INS_ROP::VPT_INS_ROP(OP_Network *net, const char *name, OP_Operator *op)

	:ROP_Node(net, name, op)
	, endTime(0)

{};


void newDriverOperator(OP_OperatorTable *table)
{
	table->addOperator(new OP_Operator("VPT_Instance",
		"VPT Instance",
		VPT_INS_ROP::nodeConstructor,
		VPT_INS_ROP::getTemplatePair(),
		0,
		0,
		VPT_INS_ROP::getVariablePair(),
		OP_FLAG_GENERATOR));
};
