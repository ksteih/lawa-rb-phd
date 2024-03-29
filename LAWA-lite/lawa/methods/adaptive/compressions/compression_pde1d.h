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

#ifndef  LAWA_METHODS_ADAPTIVE_COMPRESSIONS_COMPRESSION_PDE1D_H
#define  LAWA_METHODS_ADAPTIVE_COMPRESSIONS_COMPRESSION_PDE1D_H 1

#include <lawa/methods/adaptive/datastructures/index.h>
#include <lawa/methods/adaptive/datastructures/indexset.h>
#include <lawa/aux/timer.h>

namespace lawa {

template <typename T, typename Basis>
struct CompressionPDE1D
{
    const Basis &basis;
    short s_tilde, jmin, jmax;

    CompressionPDE1D(const Basis &_basis);

    void
    setParameters(const IndexSet<Index1D> &LambdaRow);

    IndexSet<Index1D>
    SparsityPattern(const Index1D &lambda_col, const IndexSet<Index1D> &LambdaRow, int J=-1);
};

} // namespace lawa

#include <lawa/methods/adaptive/compressions/compression_pde1d.tcc>

#endif //  LAWA_METHODS_ADAPTIVE_COMPRESSIONS_COMPRESSION_PDE1D_H

