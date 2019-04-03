


#define M_PI       3.14159265358979323846   // pi






/*
int main(const int argc, const char* argv[])
{
	Window_context window_context;
	memset(&window_context, 0, sizeof(Window_context));

	GLuint display_buffer = 0;
	GLuint display_tex = 0;
	GLuint program = 0;
	GLuint quad_vertex_buffer = 0;
	GLuint quad_vao = 0;
	GLFWwindow *window = NULL;
	int width = -1;
	int height = -1;


	// Init OpenGL window and callbacks.
	window = init_opengl();
	glfwSetWindowUserPointer(window, &window_context);
	glfwSetKeyCallback(window, handle_key);
	glfwSetScrollCallback(window, handle_scroll);
	glfwSetCursorPosCallback(window, handle_mouse_pos);
	glfwSetMouseButtonCallback(window, handle_mouse_button);


	glGenBuffers(1, &display_buffer);
	glGenTextures(1, &display_tex);
	check_success(glGetError() == GL_NO_ERROR);

	program = create_shader_program();
	quad_vao = create_quad(program, &quad_vertex_buffer);

	init_cuda();

	float3 *accum_buffer = NULL;
	cudaGraphicsResource_t display_buffer_cuda = NULL;

	// Setup initial CUDA kernel parameters.
	Kernel_params kernel_params;
	memset(&kernel_params, 0, sizeof(Kernel_params));
	kernel_params.iteration = 0;
	kernel_params.max_interactions = 100;
	kernel_params.environment_type = 0;
	kernel_params.max_extinction = 10.0f;
	kernel_params.albedo = 1.0f;

	cudaArray_t env_tex_data = 0;
	bool env_tex = false;
	if (argc >= 2)
		env_tex = create_environment(
			&kernel_params.env_tex, &env_tex_data, argv[1]);
	if (env_tex) {
		kernel_params.environment_type = 1;
		window_context.config_type = 2;
	}




	


	while (!glfwWindowShouldClose(window)) {

		// Process events.
		glfwPollEvents();
		Window_context *ctx = static_cast<Window_context *>(glfwGetWindowUserPointer(window));
		kernel_params.exposure_scale = powf(2.0f, ctx->exposure);
		const unsigned int environment_type = env_tex ? ((ctx->config_type >> 1) & 1) : 0;

		if (kernel_params.environment_type != environment_type) {
			kernel_params.environment_type = environment_type;
			kernel_params.iteration = 0;
		}



		// Reallocate buffers if window size changed.
		int nwidth, nheight;
		glfwGetFramebufferSize(window, &nwidth, &nheight);
		if (nwidth != width || nheight != height)
		{
			width = nwidth;
			height = nheight;

			resize_buffers(
				&accum_buffer, &display_buffer_cuda, width, height, display_buffer);
			glViewport(0, 0, width, height);

		}

		// Map GL buffer for access with CUDA.
		check_success(cudaGraphicsMapResources(1, &display_buffer_cuda, 0) == cudaSuccess);
		void *p;
		size_t size_p;
		check_success(
			cudaGraphicsResourceGetMappedPointer(&p, &size_p, display_buffer_cuda) == cudaSuccess);
		kernel_params.display_buffer = reinterpret_cast<unsigned int *>(p);


		// Launch volume rendering kernel.
		dim3 threads_per_block(16, 16);
		dim3 num_blocks((width + 15) / 16, (height + 15) / 16);
		void *params[] = {&gvdb , &kernel_params} ;
		check_success(cudaLaunchKernel(
			(const void *)&path_trace_kernel,
			num_blocks,
			threads_per_block,
			params) == cudaSuccess);
		++kernel_params.iteration;


		// Unmap GL buffer.
		check_success(cudaGraphicsUnmapResources(1, &display_buffer_cuda,0) == cudaSuccess);

		// Update texture for display.
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, display_buffer);
		glBindTexture(GL_TEXTURE_2D, display_tex);
		glTexImage2D(
			GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		check_success(glGetError() == GL_NO_ERROR);

		// Render the quad.
		glClear(GL_COLOR_BUFFER_BIT);
		glBindVertexArray(quad_vao);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		check_success(glGetError() == GL_NO_ERROR);

		glfwSwapBuffers(window);
	}

	check_success(cudaFree(accum_buffer) == cudaSuccess);

	// Cleanup OpenGL.
	glDeleteVertexArrays(1, &quad_vao);
	glDeleteBuffers(1, &quad_vertex_buffer);
	glDeleteProgram(program);
	check_success(glGetError() == GL_NO_ERROR);
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}

*/

