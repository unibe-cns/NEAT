/*
 *  Tools.cc
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <complex>
#include <tuple>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <time.h>


inline void arr2vec(std::vector< int > &vec, int64_t *arr, int dim){
    vec.clear();
    vec = std::vector< int > (dim, 0);
    for(int i=0; i<dim; i++){
        vec[i] = int(arr[i]);
    }
}

inline void arr2vec(std::vector< double > &vec, double *arr, int dim){
    vec.clear();
    vec = std::vector< double > (dim, 0.);
    for(int i=0; i<dim; i++){
        vec[i] = arr[i];
    }
}

inline void arr2vec(std::vector< std::complex< double > > &vec, double *arr, int dim){
    vec.clear();
    for(int i=0; i<dim; i++){
        std::complex< double > c(arr[2*i], arr[2*i+1]);
        vec.push_back(c);
    }
}

template< typename T >
inline std::string vec2string(std::vector< T > &vec){
    std::stringstream stream;
    stream << "{";
    for(int ii=0; ii<vec.size(); ii++){
        stream << vec[ii];
        if(ii < vec.size()-1){stream << ", ";}
    }
    stream << "}";
    return stream.str();
};