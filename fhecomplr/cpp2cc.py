import re
from fhecomplr import Imageplain

class OpenPEGASUSGenerator():
    def __init__(self, function_file_path):
        self.function_str = open(function_file_path, 'r').read()
        self.function_name, self.param_names, self.statements = self.parse_function()
        self.var_types = self.determine_var_types()
        self.contains_comparelut = any(stmt[0] == 'comparelut' for stmt in self.statements)
        self.pegasus_code = self.generate_pegasus_code(self.function_name, self.param_names, self.statements)

    def parse_function(self):
        lines = self.function_str.strip().split('\n')
        function_header = lines[0]
        function_body = lines[1:-1] 
        match = re.match(r'Ctx\s+(\w+)\s*\((.*)\)\s*{', function_header)
        if not match:
            raise ValueError("Invalid function header")
        function_name = match.group(1)
        params_str = match.group(2)
        params = [param.strip() for param in params_str.split(',')]
        param_names = [param.split()[-1] for param in params]

        statements = []
        for line in function_body:
            line = line.strip()
            if line.startswith('Ctx '):
                comparelut_match = re.match(r'Ctx\s+(\w+)\s*=\s*comparelut<(.+?)>\((\w+)\);', line)
                if comparelut_match:
                    var_name = comparelut_match.group(1)
                    template_args = comparelut_match.group(2).split(',')
                    input_var = comparelut_match.group(3)
                    statements.append(('comparelut', var_name, input_var, template_args))
                else:
                    match = re.match(r'Ctx\s+(\w+)\s*=\s*(\w+)\((.*)\);', line)
                    if match:
                        var_name = match.group(1)
                        operation = match.group(2)
                        operands = [operand.strip() for operand in match.group(3).split(',')]
                        statements.append(('declare_assign', var_name, operation, operands))
            elif line.startswith('copy('):
                match = re.match(r'copy\((\w+),\s*(\w+)\);', line)
                if match:
                    src = match.group(1)
                    dest = match.group(2)
                    statements.append(('copy', src, dest))
            elif line.startswith('return '):
                match = re.match(r'return\s+(\w+);', line)
                if match:
                    ret_var = match.group(1)
                    statements.append(('return', ret_var))
        return function_name, param_names, statements

    def determine_var_types(self):
        var_types = {param: 'Ctx' for param in self.param_names}
        for stmt in self.statements:
            if stmt[0] == 'declare_assign':
                var_name = stmt[1]
                var_types[var_name] = 'Ctx'
            elif stmt[0] == 'comparelut':
                var_name = stmt[1]
                var_types[var_name] = 'std::vector<lwe::Ctx_st>'
            elif stmt[0] == 'copy':
                src = stmt[1]
                dest = stmt[2]
                if src in var_types:
                    var_types[dest] = var_types[src]
            elif stmt[0] == 'return':
                pass 
        return var_types

    def generate_pegasus_code(self, function_name: str, param_names: list, statements: list[list]):
        var_types = self.var_types
        ret_var = statements[-1][1] if statements and statements[-1][0] == 'return' else None
        if ret_var and var_types.get(ret_var) == 'std::vector<lwe::Ctx_st>':
            return_type = 'std::vector<lwe::Ctx_st>'
        else:
            return_type = 'Ctx'

        adjusted_params = []
        for param in param_names:
            param_type = var_types.get(param, 'Ctx')
            adjusted_params.append(f'{param_type} &{param}')

        code = f'{return_type} {function_name}(' + ', '.join(adjusted_params) + ', PegasusRunTime &pg_rt)\n'
        code += '{\n'

        for stmt in statements:
            if stmt[0] == 'declare_assign':
                var_name = stmt[1]
                operation = stmt[2]
                operands = stmt[3]
                var_type = var_types.get(var_name, 'Ctx')
                code += f'    {var_type} {var_name};\n'
                if operation == 'RotateLeft':
                    code += f'    {var_name} = RotateLeft({operands[0]}, {operands[1]}, pg_rt);\n'
                elif operation == 'Add':
                    code += f'    {var_name} = rlwe_addition({operands[0]}, {operands[1]}, pg_rt);\n'
                    for operand in operands[2:]:
                        code += f'    {var_name} = rlwe_addition({var_name}, {operand}, pg_rt);\n'
                elif operation == 'Sub':
                    code += f'    {var_name} = rlwe_substraction({operands[0]}, {operands[1]}, pg_rt);\n'
                elif operation == 'lwe_multiply':
                    if operands[0] == operands[1]:
                        code += f'    pg_rt.Square({operands[0]});\n'
                        code += f'    {var_name} = {operands[0]};\n'
                        code += f'    pg_rt.RelinThenRescale({var_name});\n'
                    else:
                        code += f'    {var_name} = rlwe_multiply({operands[0]}, {operands[1]}, pg_rt);\n'
            elif stmt[0] == 'comparelut':
                var_name = stmt[1]
                input_var = stmt[2]
                template_args = stmt[3]
                threshold = template_args[-1]
                match = re.search(r'[-+]?\d*\.?\d+([eE][-+]?\d+)?', threshold)
                threshold = float(match.group())/255
                code += f'    std::vector<lwe::Ctx_st> {var_name};\n'
                code += f'    pg_rt.ExtraAllCoefficients({input_var}, {var_name});\n'
                code += f'    pg_rt.Binary({var_name}.data(), {var_name}.size(), {threshold});\n'
            elif stmt[0] == 'copy':
                src = stmt[1]
                dest = stmt[2]
                src_type = var_types.get(src, 'Ctx')
                var_types[dest] = src_type
                code += f'    {dest} = {src};\n'
            elif stmt[0] == 'return':
                ret_var = stmt[1]
                code += f'    return {ret_var};\n'
        code += '}\n'
        return code


    def cpptocc(self, output_file_path: str, output_txt_path: str, image: Imageplain):

        height = image.height
        width = image.width
        nslots = height * width
        if self.contains_comparelut:
            slots = [i/255 for i in image.data]
        else: 
            slots = image.data

        code = ''
        code += '#include "pegasus/pegasus_runtime.h"\n'
        code += '#include "pegasus/timer.h"\n'
        code += '#include <iostream>\n#include <fstream>\n'
        code += '#include <vector>\n'
        code += '#include <random>\n'
        code += '#include <type_traits>\n'
        code += 'using namespace gemini;\n'
        code += 'using namespace std;\n\n'

        helper_functions = '''
Ctx rlwe_multiply(Ctx &a, Ctx &b, PegasusRunTime &pg_rt)
{
    Ctx result = a;
    pg_rt.Mul(result, b);
    pg_rt.RelinThenRescale(result);
    return result;
}

Ctx rlwe_addition(Ctx &a, Ctx &b, PegasusRunTime &pg_rt)
{
    Ctx result = a;
    pg_rt.Add(result, b);
    return result;
}

Ctx rlwe_substraction(Ctx &a, Ctx &b, PegasusRunTime &pg_rt)
{
    Ctx result = a;
    pg_rt.Sub(result, b);
    return result;
}

Ctx RotateLeft(Ctx &a, int step, PegasusRunTime &pg_rt)
{
    Ctx result = a;
    pg_rt.RotateLeft(result, (size_t)abs(step));
    return result;
}
        '''
        code += helper_functions + '\n'

        code += self.pegasus_code + '\n'

        code += 'int main() {\n'
        code += '    PegasusRunTime::Parms pp;\n'
        code += '    pp.lvl0_lattice_dim = lwe::params::n();\n'
        code += '    pp.lvl1_lattice_dim = 1 << 12;\n'
        code += '    pp.lvl2_lattice_dim = 1 << 16;\n'
        code += '    pp.nlevels = 4;\n'
        code += '    pp.scale = std::pow(2., 40);\n'
        code += f'    pp.nslots = {nslots};\n'
        code += '    pp.s2c_multiplier = 1.;\n'
        code += '    pp.enable_repacking = false;\n\n'
        code += '    PegasusRunTime pg_rt(pp, 4);\n\n'
        code += '    F64Vec slots = {' + ', '.join(map(str, slots)) + '};\n'
        for param in self.param_names:
            param_type = self.var_types.get(param, 'Ctx')
            if param_type == 'Ctx':
                code += f'    Ctx {param};\n'
                code += f'    CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(slots, {param}));\n'
                if self.contains_comparelut:
                    code += f'    CHECK_AND_ABORT(pg_rt.SlotsToCoeffs({param}));\n'
            elif param_type == 'std::vector<lwe::Ctx_st>':
                code += f'    std::vector<lwe::Ctx_st> {param};\n'

        if self.contains_comparelut:
            result_type = 'std::vector<lwe::Ctx_st>'
            code += f'    {result_type} result = {self.function_name}(' + ', '.join([param for param in self.param_names]) + ', pg_rt);\n\n'
            code += '    F64Vec output;\n'
            code += '    for (auto &lwe_ct : result) {\n'
            code += '        double value = pg_rt.DecryptLWE(lwe_ct);\n'
            code += '        output.push_back(value);\n'
            code += '    }\n\n'
        else:
            result_type = 'Ctx'
            code += f'    {result_type} result = {self.function_name}(' + ', '.join([param for param in self.param_names]) + ', pg_rt);\n\n'
            code += '    F64Vec output;\n'
            code += '    CHECK_AND_ABORT(pg_rt.DecryptThenDecode(result, pp.nslots, output));\n\n'

        code += f'    std::ofstream outfile("{output_txt_path}");\n'
        code += '    if (outfile.is_open()) {\n'
        code += '        size_t index = 0;\n'
        code += f'        for (size_t i = 0; i < {height}; ++i) {{\n'
        code += f'            for (size_t j = 0; j < {width}; ++j) {{\n'
        if self.contains_comparelut:
            code += '                outfile << output[index++]*255 << " ";\n'
        else:
            code += '                outfile << output[index++] << " ";\n'
        code += '            }\n'
        code += '            outfile << std::endl;\n'
        code += '        }\n'
        code += '        outfile.close();\n'
        code += '        std::cout << "Data saved in output_image.txt" << std::endl;\n'
        code += '    } else {\n'
        code += '        std::cerr << "Can\'t write" << std::endl;\n'
        code += '    }\n'
        code += '    return 0;\n'
        code += '}'

        with open(output_file_path, 'w') as outfile:
            outfile.write(code)
