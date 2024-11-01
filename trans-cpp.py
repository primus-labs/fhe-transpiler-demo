import re
import sys
import getopt

def parse_function(function_str):
    lines = function_str.strip().split('\n')
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

def generate_pegasus_code(function_name, param_names, statements, nslots):
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
                    code += f'    Ctx {var_name} = pg_rt.Square({operands[0]});\n'
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

# Test Fuction
# Ctx encryptedBoxBlur_8x8(Ctx v1, Ctx v2) {
    # Ctx v3 = RotateLeft(v1, 55);
    # Ctx v4 = RotateLeft(v1, 63);
    # Ctx v5 = RotateLeft(v1, 7);
    # Ctx v6 = RotateLeft(v1, 56);
    # Ctx v7 = RotateLeft(v1, 8);
    # Ctx v8 = RotateLeft(v1, 57);
    # Ctx v9 = RotateLeft(v1, 9);
    # Ctx v10 = RotateLeft(v1, 1);
    # Ctx v11 = Add(v3, v4, v5, v6, v1, v7, v8, v9, v10);
    # copy(v11, v2);
    # return v2;
    # }
# Ctx encryptedRobertsCross_32x32(Ctx v1, Ctx v2) {
    # Ctx v3 = RotateLeft(v1, 991);
    # Ctx v4 = Sub(v1, v3);
    # Ctx v5 = lwe_multiply(v4, v4);
    # Ctx v6 = RotateLeft(v1, 993);
    # Ctx v7 = Sub(v1, v6);
    # Ctx v8 = lwe_multiply(v7, v7);
    # Ctx v9 = RotateLeft(v5, 33);
    # Ctx v10 = RotateLeft(v8, 32);
    # Ctx v11 = Add(v9, v10);
    # copy(v11, v2);
    # return v2;
    # }
def main(argv):
    input_file = ''
    output_file = ''
    try:
        opts, args = getopt.getopt(argv, "h", ["input=", "output="])
        if len(args) == 2:
            input_file = args[0]
            output_file = args[1]
        elif len(args) == 3:
            input_file = args[0]
            output_file = args[1]
            nslots_input = args[2]
        else:
            raise getopt.GetoptError("Not enough arguments")
    except getopt.GetoptError as e:
        print("Usage: python trans-cpp.py <inputname> <outputname> (listlength)")
        sys.exit(2)

    try:
        with open(input_file, 'r') as infile:
            function_str = infile.read()
    except FileNotFoundError:
        print(f"File {input_file} not found.")
        sys.exit(1)
    try:
        nslots_input
    except UnboundLocalError:
        nslots_input = input("Please input the value for nslots: ")
    try:
        nslots = int(nslots_input)
    except ValueError:
        print("Invalid input for nslots. Using default value 1024.")
        nslots = 1024

    function_name, param_names, statements = parse_function(function_str)

    code = ''
    code += '#include "pegasus/pegasus_runtime.h"\n'
    code += '#include "pegasus/timer.h"\n'
    code += '#include <iostream>\n'
    code += '#include <vector>\n'
    code += '#include <random>\n'
    code += 'using namespace gemini;\n'
    code += 'using namespace std;\n\n'

    code += helper_functions + '\n'

    code += generate_pegasus_code(function_name, param_names, statements, nslots) + '\n'

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

    code += '    F64Vec slots;\n'
    code += f'    slots.resize(pp.nslots, 1.0);\n\n'

    for param in param_names:
        code += f'    Ctx {param};\n'
        code += f'    CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(slots, {param}));\n'

    code += '\n    // Call the transformed function\n'
    code += f'    Ctx result = {function_name}(' + ', '.join([param for param in param_names]) + ', pg_rt);\n\n'

    code += '    // Decrypt and decode result\n'
    code += '    F64Vec output(pp.nslots);\n'
    code += '    CHECK_AND_ABORT(pg_rt.DecryptThenDecode(result, pp.nslots, output));\n\n'

    code += '    // Output result\n'
    code += '    for (size_t i = 0; i < pp.nslots; ++i) {\n'
    code += '        std::cout << output[i] << " ";\n'
    code += '        if ((i + 1) % 16 == 0) std::cout << std::endl;\n'
    code += '    }\n'
    code += '    return 0;\n'
    code += '}'

    try:
        with open(output_file, 'w') as outfile:
            outfile.write(code)
        print(f"Generated code written to {output_file}")
    except IOError:
        print(f"Failed to write to {output_file}")
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])
