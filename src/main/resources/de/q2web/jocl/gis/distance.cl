/**
 * Compute the distance from a point, stored as seperate coordinates in the
 * x, x array to a line, passed as arguments
 *
 * d=|v^^·r|= (|(toX-fromX)(fromY-pointY)-(fromX-pointX)(toY-fromY)|)
 *            /(sqrt((toX-fromX)^2+(toY-fromY)^2))
 */
__kernel void euclidean2dPointLineDistance(
	__global const float *x,
	__global const float *y,
	__global float *distance,
	const uint offset,
	const float fromX,
	const float fromY,
	const float toX,
	const float toY
) {
	int tid = get_global_id(0) + offset;

	float nom = (toX-fromX)*(fromY-y[tid])-(fromX-x[tid])*(toY-fromY);
	float denom = (toX-fromX)*(toX-fromX)+(toY-fromY)*(toY-fromY);

	distance[tid] = fabs(nom)/ sqrt(denom);
}

__kernel void spherical2dPointLineDistance(
	__global const float *longitudeX,
	__global const float *latitudeY,
	__global float *distance,
	const uint offset,
	const float fromX,
	const float fromY,
	const float toX,
	const float toY
) {
	// earth volumetric mean radius in meter
	float radius = 6371000;

	int tid = get_global_id(0) + offset;

	float radX = radians(longitudeX[tid]);
	float radY = radians(latitudeY[tid]);

	float radFromX = radians(fromX);
	float radFromY = radians(fromY);
	float radToX = radians(toX);
	float radToY = radians(toY);

	// final Point aPrime = SphericalGeometry.toCartesian3d(lineStart);
	// final Point bPrime = SphericalGeometry.toCartesian3d(lineEnd);
	// final Point pPrime = SphericalGeometry.toCartesian3d(point);

	// toCartesian3d
	// final double[] c = {
	// 		Math.cos(Math.toRadians(p.get(0)))
	// 				* Math.cos(Math.toRadians(p.get(1))),
	// 		Math.cos(Math.toRadians(p.get(0)))
	// 				* Math.sin(Math.toRadians(p.get(1))),
	// 		Math.sin(Math.toRadians(p.get(0))) };
	// return new Point(p.getTime(), c);

	float4 aPrime = (float4)(
		radFromX * cos(radFromY),
		radFromX * sin(radFromY),
		sin(radFromX),
		0);

	float4 bPrime = (float4)(
		radToX * cos(radToY),
		radToX * sin(radToY),
		sin(radToX),
		0);

	float4 pPrime = (float4)(
		radX * cos(radY),
		radX * sin(radY),
		sin(radX),
		0);

	// distance2d
	// final Point n = Vector.cross(aPrime, bPrime);
	// final double sinPhi = Math.abs(Vector.dot(n, pPrime));
	// final double phi = Math.asin(sinPhi);
	// return radius * phi;

	// opencl naturally supports vector operations
	// when using cross 4th result component is defined as 0
	float4 n = cross(aPrime, bPrime);
	
	float sinPhi = fabs(dot(n, pPrime));
	float phi = asin(sinPhi);

	distance[tid] = fabs(radius * phi);
}

/**
 * Compute orthogonal bearing
 */
inline float orthodromeBearing(float lon1, float lat1, float lon2, float lat2) {
	float radLat1 = radians(lat1);
	float radLat2 = radians(lat2);

	float deltaLongitude = radians(lon2 - lon1);

	float y = sin(deltaLongitude) * cos(radLat2);
	float x = cos(radLat1) * sin(radLat2) - sin(radLat1)
			* cos(radLat2) * cos(deltaLongitude);

	return atan2(y, x);
}

inline float haversineDistance(float longitudeFrom, float latitudeFrom,
			float longitudeTo, float latitudeTo
) {
	float deltaLatitude = radians(latitudeFrom - latitudeTo);
	float deltaLongitude = radians((longitudeFrom - longitudeTo));

	float sinusHalfDeltaLatitude = sin(deltaLatitude / 2);
	float sinusHalfDeltaLongitude = sin(deltaLongitude / 2);

	float a = sinusHalfDeltaLatitude * sinusHalfDeltaLatitude
			+ cos(radians(latitudeFrom))
			* cos(radians(latitudeTo))
			* sinusHalfDeltaLongitude * sinusHalfDeltaLongitude;
	float c = 2 * atan2(sqrt(a), sqrt(1 - a));

	return 6371000 * c;
	}

__kernel void haversine2dPointLineDistance(
	__global const float *longitudeX,
	__global const float *latitudeY,
	__global float *distance,
	const uint offset,
	const float fromLongitudeX,
	const float fromLatitudeY,
	const float toLongitudeX,
	const float toLatitudeY
) {
	// earth volumetric mean radius in meter
	float radius = 6371000;

	int tid = get_global_id(0) + offset;
	
	
	
	float b12 = orthodromeBearing(fromLongitudeX, fromLatitudeY, toLongitudeX, toLatitudeY );
	float b13 = orthodromeBearing(fromLongitudeX, fromLatitudeY, longitudeX[tid], latitudeY[tid] );
	float d13 = haversineDistance(fromLongitudeX, fromLatitudeY, longitudeX[tid], latitudeY[tid] );
	
	float dt = asin( sin(d13 / radius) * sin(b13 - b12)  ) * radius;
	
	distance[tid] = fabs(dt);
}