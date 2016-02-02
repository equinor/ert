#include <ert/ecl/ecl_smspec.h>
#include <ert/ecl/Smspec.hpp>

namespace ERT {

    smspec_node::smspec_node(
            ecl_smspec_var_type var_type,
            const std::string& name,
            const std::string& kw
            ) : smspec_node( var_type, name, kw, std::string( ":" ) )
    {}

    static const int dummy_dims[ 3 ] = { -1, -1, -1 };
    smspec_node::smspec_node(
            ecl_smspec_var_type var_type,
            const std::string& name,
            const std::string& kw,
            const std::string& join
            ) : smspec_node( var_type, name.c_str(), kw.c_str(),
                            "", join.c_str(), dummy_dims, 0 )
    {}

    smspec_node::smspec_node(
            ecl_smspec_var_type type,
            const char* wgname,
            const char* keyword,
            const char* unit,
            const char* join,
            const int grid_dims[ 3 ],
            int num, int index, float default_value ) :
        node( smspec_node_alloc( type, wgname, keyword, unit,
                    join, grid_dims, num, index, default_value ) )
    {}

    const char* smspec_node::wgname() const {
        return smspec_node_get_wgname( this->node.get() );
    }
}
