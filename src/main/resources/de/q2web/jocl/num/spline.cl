__kernel void cubicSpline(
	__global const float *x,
	__global const float *y,
	__global float *distance,
	__global const uint *excludes,
	const uint length
) {
	uint tid = get_global_id(0);

	if (tid <= 0 ) {
		return;
	}

	if (tid >= length-1 ) {
		return;
	}

	// initialize variables
	float x0 = 0, y0 = 0, xa, ya, xb, yb, xc = 0, yc = 0;
	float fPrimeLeft;
	float fPrimeRight;
	float fDoublePrimeLeft;
	float fDoublePrimeRight;

	int i = tid;
	int j = 1;
	int k = 2;
	int g = i - 2;
	int h = i - 1;

	// case iii)
	while (excludes[h] && h >= 1) {
		h = h - 1;
		g = g - 1;
	}
	// case iii, b)
	while (h > 0 && excludes[g] && g > 1) {
		g = g - 1;
	}

	// make sure that i (=tid) is excluded by adding
	j = i + 1;
	k = j + 1;

	// case ii) excludeId == xRight
	while (excludes[j] && k < (length - 1)) {
		j = j + 1;
		k = j + 1;
	}
	// because of case iii, we might have reached left border,
	// if check == false, left border is not reached , so we can set x0
	if (j > 1 && g > 0) {
		x0 = x[g];
		y0 = y[g];
	}
	xa = x[h];
	ya = y[h];
	xb = x[j];
	yb = y[j];

	// case ii,b)
	while (j < length - 1 && excludes[k] && k < length - 1) {
		k = k + 1;
	}
	// if evaluates to true than right border not reached => we can
	// access x[k], y[k]
	if (j < (length - 1)) {
		xc = x[k];
		yc = y[k];
	}
	fPrimeRight = 2 / ((xc - xb) / (yc - yb) + (xb - xa) / (yb - ya));

	// if check == true, left border is reached => change formula of
	// prime left
	fPrimeLeft = (j > 1 && h == 0) ? 3 / 2 * (yb - ya) / (xb - xa)
			- fPrimeRight / 2 : 2 / ((xb - xa) / (yb - ya) + (xa - x0)
			/ (ya - y0));

	// formula of prime right changes if right border is reached
	if (j == length) {
		fPrimeRight = 3 / 2 * ((yb - ya) / (xb - xa)) - fPrimeLeft
				/ 2;
	}

	fDoublePrimeLeft = -(2 * (fPrimeRight + 2 * fPrimeLeft))
			/ (xb - xa) + (6 * (yb - ya) / (xb - xa)) / (xb - xa);
	fDoublePrimeRight = 2 * (2 * fPrimeRight + fPrimeLeft) / (xb - xa)
			- 6 * (yb - ya) / (xb - xa) / (xb - xa);

	// it might be the case that instead of i, j and k is used
	// it is still ok to write the results to a,b,c,d as in the next
	// iteration the next i is tried, which in turn migth also be passed
	// on, because it is being excluded, that is ok because it will
	// store the same result
	float d = 1 / 6 * (fDoublePrimeRight - fDoublePrimeLeft) / (xb - xa);
	float c = 1 / 2 * (xb * fDoublePrimeLeft - xa * fDoublePrimeRight)
			/ (xb - xa);
	float b = ((yb - ya) - c * (xb * xb - xa * xa) - d
			* (xb * xb * xb - xa * xa * xa))
			/ (xb - xa);
	float a = ya - b * xa - c * xa * xa - d * xa * xa * xa;

	float yy = a + b * x[tid]
			+ c * x[tid] * x[tid]
			+ d * x[tid] * x[tid]
			* x[tid];
	distance[tid - 1] = fabs(y[tid] - yy);

}