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

public class Distance {

	private static final long[] DEFAULT_LOCAL_WORKSIZE = new long[] { 1 };

	private static final String KERNEL_EUCLIDEAN_DISTANCE_POINT_TO_LINE = "euclidean2dPointLineDistance";

	private static final String KERNEL_SPHERICAL_DISTANCE_POINT_TO_LINE = "spherical2dPointLineDistance";

	private static final String SOURCE = Resources
			.convertStreamToString(Distance.class
					.getResourceAsStream("distance.cl"));

	public static void crossTrackEuclideanDistance(final cl_context context,
			final cl_command_queue queue, final float[] latitudeX,
			final float[] longitudeY, final float[] distance,
			final float fromX, final float fromY, final float toX,
			final float toY) {

		abstractCrossTrackDistance(context, queue,
				KERNEL_EUCLIDEAN_DISTANCE_POINT_TO_LINE, latitudeX, longitudeY,
				distance, fromX, fromY, toX, toY);

	}

	public static void crossTrackSphericalDistance(final cl_context context,
			final cl_command_queue queue, final float[] latitudeX,
			final float[] longitudeY, final float[] distance,
			final float fromX, final float fromY, final float toX,
			final float toY) {

		abstractCrossTrackDistance(context, queue,
				KERNEL_SPHERICAL_DISTANCE_POINT_TO_LINE, latitudeX, longitudeY,
				distance, fromX, fromY, toX, toY);

	}

	private static void abstractCrossTrackDistance(final cl_context context,
			final cl_command_queue queue, final String kernel,
			final float[] latitudeX, final float[] longitudeY,
			final float[] distance, final float fromX, final float fromY,
			final float toX, final float toY) {
		cl_program program = null;
		cl_kernel distanceKernel = null;
		cl_mem[] memObject = null;
		try {
			int length = latitudeX.length;
			program = clCreateProgramWithSource(context, 1,
					new String[] { SOURCE }, null, null);
			clBuildProgram(program, 0, null, null, null, null);
			distanceKernel = clCreateKernel(program, kernel, null);

			Pointer pointerX = Pointer.to(latitudeX);
			Pointer pointerY = Pointer.to(longitudeY);
			Pointer pointerZ = Pointer.to(distance);

			memObject = new cl_mem[3];
			memObject[0] = clCreateBuffer(context, CL_MEM_READ_WRITE
					| CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * length, pointerX,
					null);
			memObject[1] = clCreateBuffer(context, CL_MEM_READ_WRITE
					| CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * length, pointerY,
					null);
			memObject[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
					Sizeof.cl_float * length, null, null);

			// Set default arguments for the kernels
			// x
			clSetKernelArg(distanceKernel, 0, Sizeof.cl_mem,
					Pointer.to(memObject[0]));
			// y
			clSetKernelArg(distanceKernel, 1, Sizeof.cl_mem,
					Pointer.to(memObject[1]));
			// distance
			clSetKernelArg(distanceKernel, 2, Sizeof.cl_mem,
					Pointer.to(memObject[2]));

			final long[] globalWorkSize = new long[] { length };
			final int leftOffset = 0;

			setArguments(distanceKernel, leftOffset, fromX, fromY, toX, toY);

			// Execute the kernel
			clEnqueueNDRangeKernel(queue, distanceKernel, 1, null,
					globalWorkSize, DEFAULT_LOCAL_WORKSIZE, 0, null, null);

			// Read the output data
			clEnqueueReadBuffer(queue, memObject[0], CL_TRUE, 0,
					latitudeX.length * Sizeof.cl_float, pointerX, 0, null, null);
			clEnqueueReadBuffer(queue, memObject[1], CL_TRUE, 0,
					longitudeY.length * Sizeof.cl_float, pointerY, 0, null,
					null);
			clEnqueueReadBuffer(queue, memObject[2], CL_TRUE, 0,
					distance.length * Sizeof.cl_float, pointerZ, 0, null, null);
		} finally {
			// Release memory objects, kernel and program
			clReleaseMemObject(memObject[0]);
			clReleaseMemObject(memObject[1]);
			clReleaseMemObject(memObject[2]);
			clReleaseKernel(distanceKernel);
			clReleaseProgram(program);
		}

	}

	private static void setArguments(final cl_kernel distanceKernel,
			final int leftOffset, final float fromX, final float fromY,
			final float toX, final float toY) {

		// compute distances
		// const uint offset,
		clSetKernelArg(distanceKernel, 3, Sizeof.cl_uint,
				Pointer.to(new int[] { leftOffset }));
		// const float fromX,
		clSetKernelArg(distanceKernel, 4, Sizeof.cl_float,
				Pointer.to(new float[] { fromX }));
		// const float fromY,
		clSetKernelArg(distanceKernel, 5, Sizeof.cl_float,
				Pointer.to(new float[] { fromY }));
		// const float toX,
		clSetKernelArg(distanceKernel, 6, Sizeof.cl_float,
				Pointer.to(new float[] { toX }));
		// const float toY
		clSetKernelArg(distanceKernel, 7, Sizeof.cl_float,
				Pointer.to(new float[] { toY }));

	}

}
