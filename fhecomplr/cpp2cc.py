import re

class OpenPEGASUSGenerator():
    def __init__(self, function_file_path):
        self.function_str = open(function_file_path, 'r').read()
        self.function_name, self.param_names, self.statements = self.parse_function()
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

    def generate_pegasus_code(self, function_name, param_names, statements):
        code = f'Ctx {function_name}(' + ', '.join([f'Ctx &{param}' for param in param_names]) + ', PegasusRunTime &pg_rt)\n'
        code += '{\n'
        var_defs = {}
        for stmt in statements:
            if stmt[0] == 'declare_assign':
                var_name = stmt[1]
                operation = stmt[2]
                operands = stmt[3]
                if operation == 'RotateLeft':
                    code += f'    Ctx {var_name} = RotateLeft({operands[0]}, {operands[1]}, pg_rt);\n'
                elif operation == 'Add':
                    code += f'    Ctx {var_name} = rlwe_addition({operands[0]}, {operands[1]}, pg_rt);\n'
                    for operand in operands[2:]:
                        code += f'    {var_name} = rlwe_addition({var_name}, {operand}, pg_rt);\n'
                elif operation == 'Sub':
                    code += f'    Ctx {var_name} = rlwe_substraction({operands[0]}, {operands[1]}, pg_rt);\n'
                elif operation == 'lwe_multiply':
                    if operands[0] == operands[1]:
                        code += f'    pg_rt.Square({operands[0]});\n'
                        code += f'    Ctx {var_name} = {operands[0]};\n'
                        code += f'    pg_rt.RelinThenRescale({var_name});\n'
                    else:
                        code += f'    Ctx {var_name} = rlwe_multiply({operands[0]}, {operands[1]}, pg_rt);\n'
                var_defs[var_name] = True
                var_defs[var_name] = True
            elif stmt[0] == 'copy':
                src = stmt[1]
                dest = stmt[2]
                code += f'    {dest} = {src};\n'
            elif stmt[0] == 'return':
                ret_var = stmt[1]
                code += f'    return {ret_var};\n'
        code += '}\n'
        return code


    def cpptocc(self, output_file_path, output_txt_path, image):

        height = image.height
        width = image.width
        nslots = height * width
        slots = image.data

        code = ''
        code += '#include "pegasus/pegasus_runtime.h"\n'
        code += '#include "pegasus/timer.h"\n'
        code += '#include <iostream>\n#include <fstream>\n'
        code += '#include <vector>\n'
        code += '#include <random>\n'
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
        # code += '    F64Vec slots;\n'
        # code += f'    slots.resize(pp.nslots, 1.0);\n\n'

        for param in self.param_names:
            code += f'    Ctx {param};\n'
            code += f'    CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(slots, {param}));\n'

        code += '\n    // Call the transformed function\n'
        code += f'    Ctx result = {self.function_name}(' + ', '.join([param for param in self.param_names]) + ', pg_rt);\n\n'

        code += '    // Decrypt and decode result\n'
        code += '    F64Vec output(pp.nslots);\n'
        code += '    CHECK_AND_ABORT(pg_rt.DecryptThenDecode(result, pp.nslots, output));\n\n'

        code += '    // Output result to a file\n'
        code += f'    std::ofstream outfile("{output_txt_path}");\n'
        code += '    if (outfile.is_open()) {\n'
        code += f'        for (size_t i = 0; i < {height}; ++i) {chr(123)}\n'
        code += f'            for (size_t j = 0; j < {width}; ++j) {chr(123)}\n'
        code += f'                size_t index = i * {width} + j;\n'
        code += '                outfile << output[index] << " ";\n'
        code += '            }\n'
        code += '            outfile << std::endl;\n'
        code += '        }\n'
        code += '        outfile.close();\n'
        code += '        std::cout << "Data successfully saved to output_image.txt" << std::endl;\n'
        code += '    } else {\n'
        code += '        std::cerr << "Unable to open file for writing." << std::endl;\n'
        code += '    }\n'
        code += '    return 0;\n'
        code += '}'

        with open(output_file_path, 'w') as outfile:
            outfile.write(code)
