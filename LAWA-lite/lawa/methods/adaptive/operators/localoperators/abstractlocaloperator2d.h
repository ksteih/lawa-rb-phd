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

#ifndef LAWA_METHODS_ADAPTIVE_OPERATORS_LOCALOPERATORS_ABSTRACTLOCALOPERATOR2D_H_
#define LAWA_METHODS_ADAPTIVE_OPERATORS_LOCALOPERATORS_ABSTRACTLOCALOPERATOR2D_H_

namespace lawa {

template <typename _T>
struct AbstractLocalOperator2D {

    typedef _T T;

    virtual void
    eval(const Coefficients<Lexicographical,T,Index2D> &input,
    		   Coefficients<Lexicographical,T,Index2D> &output) = 0;

    virtual T
    operator()(const Index2D &row_index, const Index2D &col_index) = 0;
    
    virtual void
    clear() = 0;

};


} // namespace lawa

#endif /* LAWA_METHODS_ADAPTIVE_OPERATORS_LOCALOPERATORS_ABSTRACTLOCALOPERATOR2D_H_ */
