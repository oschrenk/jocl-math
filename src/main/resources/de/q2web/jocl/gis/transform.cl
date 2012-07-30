/**
 * Converts geographic coordinates (lat, lng) to Cartesian (x, y, z) given a
 * reference spheroid (semiMajor, semiMinor Axis)
 *
 *
 */
__kernel void geographic2dToCartesian3d(

	__global float *latX,
	__global float *lonY,
	__global float *z,
	const float semiMajor,
	const float semiMinor

) {

	int gid = get_global_id(0);

	float cosPhi = cos(radians(latX[gid]));
	float sinPhi = sin(radians(latX[gid]));
	float cosLamda = cos(radians(lonY[gid]));
	float sinLamda = sin(radians(lonY[gid]));

	float eccentricitySquared = 1 - (semiMinor * semiMinor / semiMajor / semiMajor );

	float denomSquared = 1 - eccentricitySquared * sinPhi * sinPhi;
	float nu = semiMajor / sqrt (denomSquared);

	// assume height = 0
	// write results back in same arrays
	latX[gid] = nu * cosPhi * cosLamda;
	lonY[gid] = nu * cosPhi * sinLamda;
	z[gid] = (nu * (1 - eccentricitySquared)) * sinPhi;
}