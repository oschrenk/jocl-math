package de.q2web.jocl.gis;

import static org.jocl.CL.CL_DEVICE_TYPE_GPU;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.Arrays;

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

public class DistanceTest {

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
				new String[] { Distance.SOURCE }, null, null);
		final int returnCode = clBuildProgram(program, 0, null, null, null,
				null);

		if (returnCode != CL.CL_SUCCESS) {
			final String buildLogs = Programs.obtainBuildLogs(program);
			System.err.println(buildLogs);
			fail();
		}
	}

	@Test
	public void testCrossTrackEuclideanDistance() {

		final float fromX = 1;
		final float fromY = 3;

		final float toX = 9;
		final float toY = 3;

		final float pointX = 5;
		final float pointY = 1;

		float[] latitudeX = { pointX, pointX, pointX, pointX, pointX };
		float[] longitudeY = { pointY, pointY, pointY, pointY, pointY };
		float[] distance = new float[latitudeX.length];

		cl_platform_id platformId = Platforms.getPlatforms().get(0);
		cl_device_id deviceId = Devices.getDevices(platformId,
				CL_DEVICE_TYPE_GPU).get(0);
		cl_context context = Contexts.create(platformId, deviceId);
		cl_command_queue queue = CommandQueues.create(context, deviceId);
		try {

			Distance.crossTrackEuclideanDistance(context, queue, latitudeX,
					longitudeY, distance, fromX, fromY, toX, toY);

			for (float f : distance) {
				assertEquals(2.0, f, 0.0);
			}

		} finally {
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
		}

	}

	@Test
	public void testCrossTrackSphericalDistance() {

		// LAX: (33deg 57min N, 118deg 24min W)
		final float latLAX = 33 + 57f / 60;
		final float lonLAX = 118 + 24f / 60;

		// JFK: (40deg 38min N, 73deg 47min W)
		final float latJFK = 40 + 38f / 60;
		final float lonJFK = 73 + 47f / 60;

		// some point D N34:30 W116:30
		final float latD = 34 + 30f / 60;
		final float lonD = 116 + 30f / 60;

		float[] longitudeX = { lonD, lonD, lonD, lonD, lonD };
		float[] latitudeY = { latD, latD, latD, latD, latD };

		float[] distance = new float[longitudeX.length];

		cl_platform_id platformId = Platforms.getPlatforms().get(0);
		cl_device_id deviceId = Devices.getDevices(platformId,
				CL_DEVICE_TYPE_GPU).get(0);
		cl_context context = Contexts.create(platformId, deviceId);
		cl_command_queue queue = CommandQueues.create(context, deviceId);
		try {

			// from LAX to JFK
			Distance.crossTrackSphericalDistance(context, queue, longitudeX,
					latitudeY, distance, lonLAX, latLAX, lonJFK, latJFK);

			System.out.println(Arrays.toString(distance));

		} finally {
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
		}

	}

	@Test
	public void testCrossTrackHaversineDistance() {

		// LAX: (33deg 57min N, 118deg 24min W)
		final float latLAX = 33 + 57f / 60;
		final float lonLAX = 118 + 24f / 60;

		// JFK: (40deg 38min N, 73deg 47min W)
		final float latJFK = 40 + 38f / 60;
		final float lonJFK = 73 + 47f / 60;

		// some point D N34:30 W116:30
		final float latD = 34 + 30f / 60;
		final float lonD = 116 + 30f / 60;

		float[] longitudeX = { lonD, lonD, lonD, lonD, lonD };
		float[] latitudeY = { latD, latD, latD, latD, latD };
		float[] distance = new float[longitudeX.length];

		cl_platform_id platformId = Platforms.getPlatforms().get(0);
		cl_device_id deviceId = Devices.getDevices(platformId,
				CL_DEVICE_TYPE_GPU).get(0);
		cl_context context = Contexts.create(platformId, deviceId);
		cl_command_queue queue = CommandQueues.create(context, deviceId);
		try {

			// from LAX to JFK
			Distance.crossTrackHaversineDistance(context, queue, longitudeX,
					latitudeY, distance, lonLAX, latLAX, lonJFK, latJFK);

			System.out.println(Arrays.toString(distance));

		} finally {
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
		}

	}
}
