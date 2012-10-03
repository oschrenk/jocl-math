package de.q2web.jocl.num;

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

import java.util.Arrays;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;

import de.q2web.jocl.util.Resources;

public class SplineInterpolation {

	private static final long[] DEFAULT_LOCAL_WORKSIZE = new long[] { 1 };

	private static final String KERNEL_CUBIC_SPLINE = "cubicSpline";

	static final String SOURCE = Resources
			.convertStreamToString(SplineInterpolation.class
					.getResourceAsStream("spline.cl"));

	public static void compute(final cl_context context,
			final cl_command_queue queue, final float[] longitudeX,
			final float[] latitudeY, final float[] distance,
			final int[] currentExcludes) {

		cl_program program = null;
		cl_kernel splineKernel = null;
		cl_mem[] memObject = null;

		try {
			int length = longitudeX.length;
			program = clCreateProgramWithSource(context, 1,
					new String[] { SOURCE }, null, null);
			clBuildProgram(program, 0, null, null, null, null);
			splineKernel = clCreateKernel(program, KERNEL_CUBIC_SPLINE, null);

			Pointer pointerLongitudeX = Pointer.to(longitudeX);
			Pointer pointerLatitudeY = Pointer.to(latitudeY);
			Pointer pointerDistance = Pointer.to(distance);
			Pointer pointerCurrentExcludes = Pointer.to(currentExcludes);

			memObject = new cl_mem[4];
			// longitude / x array
			memObject[0] = clCreateBuffer(context, CL_MEM_READ_WRITE
					| CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * length,
					pointerLongitudeX, null);
			// longitude / y array
			memObject[1] = clCreateBuffer(context, CL_MEM_READ_WRITE
					| CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * length,
					pointerLatitudeY, null);
			// distance array, results will be written here
			memObject[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
					Sizeof.cl_float * length, null, null);
			// currentExcludes
			memObject[3] = clCreateBuffer(context, CL_MEM_READ_WRITE
					| CL_MEM_COPY_HOST_PTR, Sizeof.cl_uint * length,
					pointerCurrentExcludes, null);

			// Set default arguments for the kernels
			// x
			clSetKernelArg(splineKernel, 0, Sizeof.cl_mem,
					Pointer.to(memObject[0]));
			// y
			clSetKernelArg(splineKernel, 1, Sizeof.cl_mem,
					Pointer.to(memObject[1]));
			// distance
			clSetKernelArg(splineKernel, 2, Sizeof.cl_mem,
					Pointer.to(memObject[2]));
			// excludes
			clSetKernelArg(splineKernel, 3, Sizeof.cl_mem,
					Pointer.to(memObject[3]));
			clSetKernelArg(splineKernel, 4, Sizeof.cl_uint,
					Pointer.to(new int[] { length }));

			final long[] globalWorkSize = new long[] { length };

			// Execute the kernel
			clEnqueueNDRangeKernel(queue, splineKernel, 1, null,
					globalWorkSize, DEFAULT_LOCAL_WORKSIZE, 0, null, null);

			// Read the output data
			clEnqueueReadBuffer(queue, memObject[2], CL_TRUE, 0,
					distance.length * Sizeof.cl_float, pointerDistance, 0,
					null, null);

			System.out.println(Arrays.toString(distance));
		} finally {
			// Release memory objects, kernel and program
			clReleaseMemObject(memObject[0]);
			clReleaseMemObject(memObject[1]);
			clReleaseMemObject(memObject[2]);
			clReleaseMemObject(memObject[3]);
			clReleaseKernel(splineKernel);
			clReleaseProgram(program);
		}

	}

}
