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
	__global const float *x,
	__global const float *y,
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

	float radX = radians(x[tid]);
	float radY = radians(y[tid]);

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