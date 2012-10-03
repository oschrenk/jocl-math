package de.q2web.jocl.num;

import static org.jocl.CL.CL_DEVICE_TYPE_GPU;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.junit.Assert.fail;

import org.jocl.CL;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;
import org.jocl.utils.CommandQueues;
import org.jocl.utils.Contexts;
import org.jocl.utils.Devices;
import org.jocl.utils.Platforms;
import org.jocl.utils.Programs;
import org.junit.BeforeClass;
import org.junit.Test;

public class SplineInterpolationTest {

	private static final float X[] = { 0, 10, 30, 50, 70, 90, 100 };
	private static final float Y[] = { 30, 130, 150, 150, 170, 220, 320 };

	private static final int LENGTH = X.length;

	@BeforeClass
	public static void setUp() {
		// Enable exceptions and subsequently omit error checks in this sample
		CL.setExceptionsEnabled(true);
	}

	@Test
	public void testBuildProgram() {
		final cl_platform_id platformId = Platforms.getPlatforms().get(0);
		final cl_device_id deviceId = Devices.getDevices(platformId,
				CL_DEVICE_TYPE_GPU).get(0);
		final cl_context context = Contexts.create(platformId, deviceId);

		final cl_program program = clCreateProgramWithSource(context, 1,
				new String[] { SplineInterpolation.SOURCE }, null, null);
		final int returnCode = clBuildProgram(program, 0, null, null, null,
				null);

		if (returnCode != CL.CL_SUCCESS) {
			final String buildLogs = Programs.obtainBuildLogs(program);
			System.err.println(buildLogs);
			fail();
		}
	}

	@Test
	public void testSpline() {
		final cl_platform_id platformId = Platforms.getPlatforms().get(0);
		final cl_device_id deviceId = Devices.getDevices(platformId,
				CL_DEVICE_TYPE_GPU).get(0);
		final cl_context context = Contexts.create(platformId, deviceId);
		final cl_command_queue queue = CommandQueues.create(context, deviceId);
		try {

			float[] distance = new float[LENGTH - 2];
			int[] currentExcludes = new int[LENGTH];

			SplineInterpolation.compute(context, queue, X, Y, distance,
					currentExcludes);

		} finally {
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
		}
	}
	
	@Test
	public void testSplineWithExcludes() {
		final cl_platform_id platformId = Platforms.getPlatforms().get(0);
		final cl_device_id deviceId = Devices.getDevices(platformId,
				CL_DEVICE_TYPE_GPU).get(0);
		final cl_context context = Contexts.create(platformId, deviceId);
		final cl_command_queue queue = CommandQueues.create(context, deviceId);
		try {

			float[] distance = new float[LENGTH - 2];
			int[] currentExcludes = new int[LENGTH];
			currentExcludes[2] = 1;
			currentExcludes[3] = 1;
			currentExcludes[4] = 1;

			SplineInterpolation.compute(context, queue, X, Y, distance,
					currentExcludes);

		} finally {
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
		}
	}
}
