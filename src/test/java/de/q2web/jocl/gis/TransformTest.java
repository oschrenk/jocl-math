package de.q2web.jocl.gis;

import static org.jocl.CL.CL_DEVICE_TYPE_GPU;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.junit.Assert.assertEquals;

import org.jocl.CL;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;
import org.jocl.utils.CommandQueues;
import org.jocl.utils.Contexts;
import org.jocl.utils.Devices;
import org.jocl.utils.Platforms;
import org.junit.BeforeClass;
import org.junit.Test;

public class TransformTest {

	@BeforeClass
	public static void setUp() {
		// Enable exceptions and subsequently omit error checks in this sample
		CL.setExceptionsEnabled(true);
	}

	@Test
	public void testGeographic2dToCartesian3d() {

		// Paris
		final float lat = 48 + 50 / 60.0f;
		final float lon = 2 + 20 / 60.0f;

		float[] latitudeX = { lat, lat, lat, lat, lat };
		float[] longitudeY = { lon, lon, lon, lon, lon };
		float[] arrayZ = new float[latitudeX.length];

		final float semiMajor = 6378137.0f;
		final float semiMinor = 6356752.3142f;

		cl_platform_id platformId = Platforms.getPlatforms().get(0);
		cl_device_id deviceId = Devices.getDevices(platformId,
				CL_DEVICE_TYPE_GPU).get(0);
		cl_context context = Contexts.create(platformId, deviceId);
		cl_command_queue queue = CommandQueues.create(context, deviceId);
		try {

			Transform.geographic2dToCartesian3d(context, queue, semiMajor,
					semiMinor, latitudeX, longitudeY, arrayZ);

			final float x = 4202917.917f;
			final float y = 171255.782f;
			final float z = 4778378.571f;

			for (float f : latitudeX) {
				assertEquals(x, f, 0.1f);
			}
			for (float f : longitudeY) {
				assertEquals(y, f, 0.1f);
			}
			for (float f : arrayZ) {
				assertEquals(z, f, 0.5f);
			}

		} finally {
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
		}

	}
}
