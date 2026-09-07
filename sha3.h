#ifndef SHA3_H
#define SHA3_H

#include <stdint.h>

typedef unsigned long long ull;


/******** Round Constants ********/

static const ull RC[24] =
{
    0x0000000000000001ULL,
    0x0000000000008082ULL,
    0x800000000000808AULL,
    0x8000000080008000ULL,
    0x000000000000808BULL,
    0x0000000080000001ULL,
    0x8000000080008081ULL,
    0x8000000000008009ULL,
    0x000000000000008AULL,
    0x0000000000000088ULL,
    0x0000000080008009ULL,
    0x000000008000000AULL,
    0x000000008000808BULL,
    0x800000000000008BULL,
    0x8000000000008089ULL,
    0x8000000000008003ULL,
    0x8000000000008002ULL,
    0x8000000000000080ULL,
    0x000000000000800AULL,
    0x800000008000000AULL,
    0x8000000080008081ULL,
    0x8000000000008080ULL,
    0x0000000080000001ULL,
    0x8000000080008008ULL
};



static inline ull ROTL64(ull x, int n)
{
    if (n == 0)
        return x;

    return (x << n) | (x >> (64 - n));
}



/******** θ ********/

static void theta(ull* A)
{

    ull C[5];
    ull D[5];


    for (int x = 0; x < 5; x++)
    {

        C[x] =
            A[x] ^
            A[x + 5] ^
            A[x + 10] ^
            A[x + 15] ^
            A[x + 20];

    }


    for (int x = 0; x < 5; x++)
    {

        D[x] =
            C[(x + 4) % 5] ^
            ROTL64(C[(x + 1) % 5], 1);

    }



    for (int x = 0; x < 5; x++)
    {

        for (int y = 0; y < 5; y++)
        {

            A[x + 5 * y] ^= D[x];

        }

    }

}



/******** ρ π ********/


static const int ROT[25] =
{
    0,1,62,28,27,
    36,44,6,55,20,
    3,10,43,25,39,
    41,45,15,21,8,
    18,2,61,56,14
};



static void rho_pi(ull* A)
{

    ull B[25];


    for (int x = 0; x < 5; x++)
    {

        for (int y = 0; y < 5; y++)
        {

            B[
                y + 5 * ((2 * x + 3 * y) % 5)
            ]
                =
                ROTL64(
                    A[x + 5 * y],
                    ROT[x + 5 * y]
                );

        }

    }


    for (int i = 0; i < 25; i++)
        A[i] = B[i];

}



/******** χ ********/


static void chi(ull* A)
{

    ull B[25];


    for (int x = 0; x < 5; x++)
    {

        for (int y = 0; y < 5; y++)
        {

            B[x + 5 * y]
                =
                A[x + 5 * y]
                ^
                (
                    (~A[(x + 1) % 5 + 5 * y])
                    &
                    A[(x + 2) % 5 + 5 * y]
                    );

        }

    }


    for (int i = 0; i < 25; i++)
        A[i] = B[i];

}



/******** ι ********/


static void iota(ull* A, int r)
{
    A[0] ^= RC[r];
}



#endif