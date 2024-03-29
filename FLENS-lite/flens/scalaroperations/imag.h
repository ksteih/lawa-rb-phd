/*
 *   Copyright (c) 2010, Michael Lehn
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

#ifndef FLENS_SCALAROPERATIONS_IMAG_H
#define FLENS_SCALAROPERATIONS_IMAG_H 1

#include <cxxblas/aux/complex.h>
#include <flens/aux/aux.h>
#include <flens/scalartypes/impl/scalarclosure.h>

namespace flens {

struct ScalarOpImag {};

template <typename S>
struct ElementType<ScalarClosure<ScalarOpImag, S, S> >
{
    typedef typename S::ElementType::value_type Type;
};

template <typename S>
const typename ScalarClosure<ScalarOpImag, S, S>::ElementType
evalScalarClosure(const ScalarClosure<ScalarOpImag, S, S> &exp)
{
    return cxxblas::imag(exp.left().value());
}

//-- operator overloading
template <typename S>
const ScalarClosure<ScalarOpImag,
                    typename S::Impl,
                    typename S::Impl>
Imag(const Scalar<S> &s)
{
    typedef ScalarClosure<ScalarOpImag, typename S::Impl, typename S::Impl>  SC;
    return SC(s.impl(), s.impl());
}

} // namespace flens

#include <flens/scalaroperations/imag.tcc>

#endif // FLENS_SCALAROPERATIONS_IMAG_H
