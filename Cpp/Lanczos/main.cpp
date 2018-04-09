// The following constants LG_g and LG_N are the "g" and "n" parameters
// for the table of coefficients that follows them; several alternative
// coefficients are available at mrob.com/pub/ries/lanczos-gamma.html
#define LG_g 5.0      // Lanczos parameter "g"
#define LG_N 6        // Range of coefficients i=[0..N]
const double lct[LG_N+1] = {
  1.000000000190015,
  76.18009172947146,
  -86.50532032941677,
  24.01409824083091,
  -1.231739572450155,
  0.1208650973866179e-2,
  -0.5395239384953e-5
};
 
 
const double ln_sqrt_2_pi = 0.91893853320467274178;
const double         g_pi = 3.14159265358979323846;
 
// Compute the logarithm of the Gamma function using the Lanczos method.
double lanczos_ln_gamma(double z)
{
  double sum;
  double base;
  double rv;
  int i;
  if (z < 0.5) {
    // Use Euler's reflection formula:
    // Gamma(z) = Pi / [Sin[Pi*z] * Gamma[1-z]];
    return log(g_pi / sin(g_pi * z)) - lanczos_ln_gamma(1.0 - z);
  }
  z = z - 1.0;
  base = z + LG_g + 0.5;  // Base of the Lanczos exponential
  sum = 0;
  // We start with the terms that have the smallest coefficients and largest
  // denominator.
  for(i=LG_N; i>=1; i--) {
    sum += lct[i] / (z + ((double) i));
  }
  sum += lct[0];
  // This printf is just for debugging
  printf("ls2p %7g  l(b^e) %7g   -b %7g  l(s) %7g\n", ln_sqrt_2_pi,
	 log(base)*(z+0.5), -base, log(sum));
  // Gamma[z] = Sqrt(2*Pi) * sum * base^[z + 0.5] / E^base
  return ((ln_sqrt_2_pi + log(sum)) - base) + log(base)*(z+0.5);
}
 
// Compute the Gamma function, which is e to the power of ln_gamma.
double lanczos_gamma(double z)
{
  return(exp(lanczos_ln_gamma(z)));
}
