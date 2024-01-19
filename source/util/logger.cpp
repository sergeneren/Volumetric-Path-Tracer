
#include "logger.h"

void log(const char * message, unsigned int level)
{
	
	if (level == 0) {
		fprintf(stderr, "Error! %s\n", message);
	}
	if (level == 1) {
#if defined(LOG_LEVEL_WARNING) || defined(LOG_LEVEL_LOG)
		printf("Warning: %s\n", message);
#endif
	}
	if (level == 2) {
#if defined(LOG_LEVEL_LOG)
		printf("%s\n", message);
#endif
	}

}

void log(std::string message, unsigned int level)
{

	if (level == VPT_ERROR) {
		fprintf(stderr, "Error! %s\n", message.c_str());
	}
	if (level == VPT_WARNING) {
#if defined(LOG_LEVEL_WARNING) || defined(LOG_LEVEL_LOG)
		printf("Warning: %s\n", message.c_str());
#endif
	}
	if (level == VPT_LOG) {
#if defined(LOG_LEVEL_LOG)
		printf("%s\n", message.c_str());
#endif
	}

}
