package de.q2web.jocl.gis;

import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clReleaseProgram;
import static org.jocl.CL.clSetKernelArg;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;

import de.q2web.jocl.util.Resources;

public class Transform {

	private static final String GEOGRAPHIC2D_TO_CARTESIAN3D = "geographic2dToCartesian3d";
	private static final String SOURCE = Resources
			.convertStreamToString(Transform.class
					.getResourceAsStream("transform.cl"));

	/**
	 * Geographic2d to cartesian3d.
	 * 
	 * @param context
	 *            the context
	 * @param queue
	 *            the queue
	 * @param semiMajor
	 *            the semi major
	 * @param semiMinor
	 *            the semi minor
	 * @param latitudeX
	 *            the latitude x
	 * @param longitudeY
	 *            the longitude y
	 * @param z
	 *            the z
	 */
	public static void geographic2dToCartesian3d(cl_context context,
			cl_command_queue queue, float semiMajor, float semiMinor,
			float[] latitudeX, float[] longitudeY, float[] z) {
		cl_program program = null;
		cl_kernel kernel = null;
		cl_mem[] memObject = null;
		try {
			int length = latitudeX.length;
			program = clCreateProgramWithSource(context, 1,
					new String[] { SOURCE }, null, null);
			clBuildProgram(program, 0, null, null, null, null);
			kernel = clCreateKernel(program, GEOGRAPHIC2D_TO_CARTESIAN3D, null);

			Pointer pointerX = Pointer.to(latitudeX);
			Pointer pointerY = Pointer.to(longitudeY);
			Pointer pointerZ = Pointer.to(z);

			memObject = new cl_mem[3];
			memObject[0] = clCreateBuffer(context, CL_MEM_READ_WRITE
					| CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * length, pointerX,
					null);
			memObject[1] = clCreateBuffer(context, CL_MEM_READ_WRITE
					| CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * length, pointerY,
					null);
			memObject[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
					Sizeof.cl_float * length, null, null);

			// Set the arguments for the kernel
			clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memObject[0]));
			clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memObject[1]));
			clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memObject[2]));
			clSetKernelArg(kernel, 3, Sizeof.cl_float,
					Pointer.to(new float[] { (Float) semiMajor }));
			clSetKernelArg(kernel, 4, Sizeof.cl_float,
					Pointer.to(new float[] { (Float) semiMinor }));

			final long[] globalWorkSize = new long[] { length };
			final long[] localWorkSize = new long[] { 1 };

			// Execute the kernel
			clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize,
					localWorkSize, 0, null, null);

			// Read the output data
			clEnqueueReadBuffer(queue, memObject[0], CL_TRUE, 0,
					latitudeX.length * Sizeof.cl_float, pointerX, 0, null, null);
			clEnqueueReadBuffer(queue, memObject[1], CL_TRUE, 0,
					longitudeY.length * Sizeof.cl_float, pointerY, 0, null,
					null);
			clEnqueueReadBuffer(queue, memObject[2], CL_TRUE, 0, z.length
					* Sizeof.cl_float, pointerZ, 0, null, null);
		} finally {
			// Release memory objects, kernel and program
			clReleaseMemObject(memObject[0]);
			clReleaseMemObject(memObject[1]);
			clReleaseMemObject(memObject[2]);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
		}

	}

}
