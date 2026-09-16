#include <stdio.h>
#include <fenv.h>
#include <math.h>
#include <string.h>
#pragma STDC FENV_ACCESS ON
int main(void) {
    unsigned long long bits = 0x7FF80000000007A2ULL;  /* our NA: quiet NaN */
    double na; memcpy(&na, &bits, 8);
    double x = 1.0;
    volatile int r;

    feclearexcept(FE_ALL_EXCEPT);
    r = (na == x);
    printf("na == x  -> invalid=%d\n", fetestexcept(FE_INVALID) != 0);

    feclearexcept(FE_ALL_EXCEPT);
    r = (na > x);
    printf("na >  x  -> invalid=%d\n", fetestexcept(FE_INVALID) != 0);

    feclearexcept(FE_ALL_EXCEPT);
    volatile double s = na + x;
    printf("na +  x  -> invalid=%d\n", fetestexcept(FE_INVALID) != 0);
    (void)r;
    return 0;
}
