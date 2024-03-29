/*
  LAWA - Library for Adaptive Wavelet Applications.
  Copyright (C) 2008-2014  Sebastian Kestler, Mario Rometsch, Kristina Steih, 
  Alexander Stippler.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

#ifndef LAWA_FUNCTIONTYPES_SEPARABLEFUNCTION3D_H
#define LAWA_FUNCTIONTYPES_SEPARABLEFUNCTION3D_H 1

#include <lawa/functiontypes/function.h>

namespace lawa {

using namespace flens;

template<typename T>
struct SeparableFunction3D
{
    SeparableFunction3D(Function<T> _F_x, Function<T>  _F_y, Function<T>  _F_z);

    SeparableFunction3D(T (*_f_x)(T), const DenseVector<Array<T> > &_singularPts_x,
                        T (*_f_y)(T), const DenseVector<Array<T> > &_singularPts_y,
                        T (*_f_z)(T), const DenseVector<Array<T> > &_singularPts_z);

    T
    operator()(T x, T y, T z) const;

    Function<T> F_x;
    Function<T> F_y;
    Function<T> F_z;
};

} // namespace lawa

#include <lawa/functiontypes/separablefunction3d.tcc>

#endif // LAWA_FUNCTIONTYPES_SEPARABLEFUNCTION3D_H

