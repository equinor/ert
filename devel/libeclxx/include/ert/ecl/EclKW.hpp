/*
  Copyright 2015 Statoil ASA.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_ERT_ECL_KW
#define OPM_ERT_ECL_KW

#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>

#include <ert/ecl/ecl_kw.h>
#include <ert/ecl/ecl_util.h>

#include <ert/util/ert_unique_ptr.hpp>
#include <ert/ecl/FortIO.hpp>


namespace ERT {
    template< typename > struct ecl_type {};

    template<> struct ecl_type< float >
    { static const ecl_type_enum type { ECL_FLOAT_TYPE }; };

    template<> struct ecl_type< double >
    { static const ecl_type_enum type { ECL_DOUBLE_TYPE }; };

    template<> struct ecl_type< int >
    { static const ecl_type_enum type { ECL_INT_TYPE }; };

    template<> struct ecl_type< char* >
    { static const ecl_type_enum type { ECL_CHAR_TYPE }; };

    template<> struct ecl_type< const char* >
    { static const ecl_type_enum type { ECL_CHAR_TYPE }; };

    template <typename T>
    class EclKW {
    public:
        EclKW( const std::string& kw, int size_ ) noexcept :
            m_kw( ecl_kw_alloc( kw.c_str(), size_, ecl_type< T >::type ) )
        {}

        EclKW( const std::string& kw, const std::vector< T >& data ) noexcept :
            EclKW( kw, data.size() )
        {
            ecl_kw_set_memcpy_data( m_kw.get(), data.data() );
        }

        template< typename U >
        EclKW( const std::string& kw, const std::vector< U >& data ) noexcept :
            EclKW( kw, data.size() )
        {
            T* target = static_cast< T* >( ecl_kw_get_ptr( this->m_kw.get() ) );

            for( size_t i = 0; i < data.size(); ++i )
                target[ i ] = T( data[ i ] );
        }

        EclKW() noexcept {}

        static EclKW load(FortIO& fortio) {
            return checkedLoad( fortio, ecl_type< T >::type );
        }

        size_t size() const {
            return static_cast<size_t>( ecl_kw_get_size( m_kw.get() ));
        }

        void fwrite(FortIO& fortio) const {
            ecl_kw_fwrite( m_kw.get() , fortio.get() );
        }

        T at( size_t i ) const {
            return *static_cast< T* >( ecl_kw_iget_ptr( m_kw.get(), i ) );
        }

        ecl_kw_type* get() const {
            return m_kw.get();
        }

    private:
        EclKW(ecl_kw_type * c_ptr) {
            m_kw.reset( c_ptr );
        }


        static EclKW checkedLoad(FortIO& fortio, ecl_type_enum expectedType) {
            ecl_kw_type * c_ptr = ecl_kw_fread_alloc( fortio.get() );
            if (c_ptr) {
                if (ecl_kw_get_type( c_ptr ) == expectedType)
                    return EclKW( c_ptr );
                else
                    throw std::invalid_argument("Type error");
            } else
                throw std::invalid_argument("fread kw failed - EOF?");
        }


        ert_unique_ptr<ecl_kw_type , ecl_kw_free> m_kw;
    };

template<>
inline const char* EclKW< const char* >::at( size_t i ) const {
    return ecl_kw_iget_char_ptr( m_kw.get(), i );
}

template<> inline
EclKW< const char* >::EclKW( const std::string& kw,
                                const std::vector< const char* >& data )
    noexcept : EclKW( kw, data.size() )
{
    auto* ptr = this->get();
    for( size_t i = 0; i < data.size(); ++i )
        ecl_kw_iset_string8( ptr, i, data[ i ] );
}

}

#endif
