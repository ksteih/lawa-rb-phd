/*
 *   Copyright (c) 2009, Michael Lehn
 *
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1) Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2) Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *   3) Neither the name of the FLENS development group nor the names of
 *      its contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <flens/aux/macros.h>
#include <flens/blas/debugmacro.h>
#include <flens/storage/storageinfo.h>
#include <flens/vectortypes/impl/densevector.h>

// Pfusch: TODO: remove
namespace flens {

struct OpMult;

template <typename T>
    class ScalarValue;

template <typename Op, typename L, typename R>
    class MatrixClosure;

} // namespace flens

namespace flens { namespace blas {

//-- common interface for vectors ----------------------------------------------
template <typename ALPHA, typename VX, typename VY>
void
axpy(const ALPHA &alpha, const Vector<VX> &x, Vector<VY> &y)
{
    axpy(alpha, x.impl(), y.impl());
}

// Pfusch !!! TODO: remove
//-- axpy
// B += alpha*A
template <typename ALPHA, typename T, typename MA, typename MB>
    void
    axpy(cxxblas::Transpose trans, const ALPHA &alpha,
         const MatrixClosure<OpMult, ScalarValue<T>, MA> &aA, Matrix<MB> &B);

//-- common interface for matrices ---------------------------------------------
template <typename ALPHA, typename MA, typename MB>
void
axpy(cxxblas::Transpose trans,
     const ALPHA &alpha, const Matrix<MA> &A, Matrix<MB> &B)
{
    axpy(trans, alpha, A.impl(), B.impl());
}

//-- axpy
template <typename ALPHA, typename VX, typename VY>
void
axpy(const ALPHA &alpha, const DenseVector<VX> &x, DenseVector<VY> &y)
{
    FLENS_CLOSURELOG_ADD_ENTRY_AXPY(alpha, x, y);

    if (y.length()==0) {
        y.engine().resize(x.engine(), 0);
    }
    ASSERT(y.length()==x.length());
    cxxblas::axpy(x.length(), alpha,
                  x.engine().data(), x.engine().stride(),
                  y.engine().data(), y.engine().stride());

    FLENS_CLOSURELOG_END_ENTRY;
}

//-- geaxpy
template <typename ALPHA, typename MA, typename MB>
void
axpy(cxxblas::Transpose trans,
     const ALPHA &alpha, const GeMatrix<MA> &A, GeMatrix<MB> &B)
{
    FLENS_CLOSURELOG_ADD_ENTRY_AXPY(alpha, A, B);


    if ((A.numRows()!=B.numRows())
     || (A.numCols()!=B.numCols())) {
        B.engine().resize(A.engine(), 0);
    }
    trans = (StorageInfo<MA>::Order==StorageInfo<MB>::Order)
          ? cxxblas::Transpose(trans ^ cxxblas::NoTrans)
          : cxxblas::Transpose(trans ^ cxxblas::Trans);
    cxxblas::geaxpy(StorageInfo<MB>::Order, trans,
                    B.numRows(), B.numCols(), alpha,
                    A.engine().data(), A.engine().leadingDimension(),
                    B.engine().data(), B.engine().leadingDimension());

    FLENS_CLOSURELOG_END_ENTRY;
}

} } // namespace blas, flens
