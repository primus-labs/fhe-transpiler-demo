import re
from fhecomplr.value import Imageplain
from typing import Tuple

class OpenPEGASUSGenerator():
    """
    OpenPEGASUSGenerator class that generates PEGASUS code from a given function file.
    It parses the function, determines variable types, and generates the corresponding PEGASUS code.
    """
    def __init__(self, function_file_path = None):
        if function_file_path != None:
            self.function_str = open(function_file_path, 'r').read()
            self.function_name, self.param_names, self.statements = self.parse_function()
            self.var_types = self.determine_var_types()
            self.contains_comparelut = any(stmt[0] == 'comparelut' for stmt in self.statements)
            self.pegasus_code = self.generate_pegasus_code(self.function_name, self.param_names, self.statements)

    def parse_function(self) -> Tuple:
        """
        Gets the function name, parameter names, and statements from the function string.

        Returns:
            tuple: A tuple containing the function name, parameter names, and statements.
        """
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

    def determine_var_types(self) -> dict:
        """
        Determines the types of the variables in the function.
        
        Returns:
            dict: A dictionary containing the variable names as keys and the variable types as values.
        """
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

    def generate_pegasus_code(self, function_name: str, param_names: list, statements: list[list]) -> str:
        """
        Generates the Pegasus code for the function.

        Args:
            function_name (str): The name of the function.
            param_names (list): The names of the parameters.
            statements (list): The statements in the function.

        Returns:
            str: The generated Pegasus code.
        """
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

    def cc_encrypt(self, output_file_path: str, image: Imageplain, output_cipher_path: str):
        """
        Generates a C++ file for encrypting an image using Pegasus.

        Args:
            output_file_path (str): Path to the output C++ file.
            image (Imageplain): The image to be encrypted.
            output_cipher_path (str): Path to save the encrypted cipher.
        """

        height = image.height
        width = image.width
        nslots = height * width
        slots = [i/255 for i in image.data]

        code = ''
        code += '#include "pegasus/pegasus_runtime.h"\n'
        code += '#include "pegasus/timer.h"\n'
        code += '#include <iostream>\n#include <fstream>\n'
        code += '#include <vector>\n'
        code += '#include <random>\n'
        code += '#include <type_traits>\n'
        code += 'using namespace gemini;\n'
        code += 'using namespace std;\n\n'

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
        code += '    Ctx image;\n'
        code += '    CHECK_AND_ABORT(pg_rt.EncodeThenEncrypt(slots, image));\n'
        code += f'    std::ofstream ofs("{output_cipher_path}", std::ios::binary);\n'
        code += '    image.save(ofs);\n'
        code += '    ofs.close();\n'
        code += '    return 0;\n'
        code += '}'
        with open(output_file_path, 'w') as outfile:
            outfile.write(code)

    def cc_decrypt(self, output_file_path: str, input_cipher_path: str, output_txt_path: str, width: int, height: int):
        """
        Generates a C++ file for decrypting an image using Pegasus.

        Args:
            output_file_path (str): Path to the output C++ file.
            input_cipher_path (str): Path to the input cipher.
            output_txt_path (str): Path to save the decrypted image.
            width (int): Width of the image.
            height (int): Height of the image.
        """
        nslots = height * width
        code = ''
        code += '#include "pegasus/pegasus_runtime.h"\n'
        code += '#include "pegasus/timer.h"\n'
        code += '#include <iostream>\n#include <fstream>\n'
        code += '#include <vector>\n'
        code += '#include <random>\n'
        code += '#include <type_traits>\n'
        code += 'using namespace gemini;\n'
        code += 'using namespace std;\n\n'

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
        code += '    Ctx image;\n'
        code += f'    std::ifstream ifs("{input_cipher_path}", std::ios::binary);\n'
        code += '    pg_rt.load_cipher(image, ifs);\n'
        code += '    ifs.close();\n'
        code += '    F64Vec output;\n'
        code += '    CHECK_AND_ABORT(pg_rt.DecryptThenDecode(image, pp.nslots, output));\n'
        code += f'    std::ofstream outfile("{output_txt_path}");\n'
        code += '    if (outfile.is_open()) {\n'
        code += '        size_t index = 0;\n'
        code += f'        for (size_t i = 0; i < {height}; ++i) {{\n'
        code += f'            for (size_t j = 0; j < {width}; ++j) {{\n'
        code += '                outfile << output[index++]*255 << " ";\n'
        code += '            }\n'
        code += '            outfile << std::endl;\n'
        code += '        }\n'
        code += '        outfile.close();\n'
        code += '        std::cout << "Data saved in decrypted.txt" << std::endl;\n'
        code += '    } else {\n'
        code += '        std::cerr << "Can\'t write" << std::endl;\n'
        code += '    }\n'
        code += '    return 0;\n'
        code += '}'
        with open(output_file_path, 'w') as outfile:
            outfile.write(code)

    def cpptocc(self, output_file_path: str, output_txt_path: str, image: Imageplain):
        """
        Generates a C++ file for decrypting an image using Pegasus.

        Args:
            output_file_path (str): Path to the output C++ file.
            output_txt_path (str): Path to save the decrypted image.
            image (Imageplain): The image to be decrypted.
        """
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

    def enc_cpptocc(self, input_cipher_path: str, cc_path: str, output_cipher_path: str, slots: int):
        """
        Generates a C++ file for evaluating the encrypted image cipher by Pegasus.
        
        Args:
            input_cipher_path (str): Path to the input cipher.
            cc_path (str): Path to the output C++ file.
            output_cipher_path (str): Path to save the encrypted cipher.
            slots (int): Number of slots.
        """
        
        nslots = slots

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
        temp = 0
        for param in self.param_names:
            param_type = self.var_types.get(param, 'Ctx')
            if param_type == 'Ctx':
                temp += 1
                if temp == 1:
                    code += f'    Ctx {param};\n'
                    code += f'    std::ifstream ifs("{input_cipher_path}", std::ios::binary)\n;'
                    code += f'    pg_rt.load_cipher({param}, ifs);\n'
                    code += '    ifs.close();\n'
                else:
                    code += f'    Ctx {param};\n'
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
        code += '    Ctx image;\n'
        code += '    pg_rt.EncodeThenEncrypt(output, image);\n'
        code += f'    std::ofstream ofs("{output_cipher_path}", std::ios::binary);\n'
        code += '    image.save(ofs);\n'
        code += '    ofs.close();\n'
        code += '    return 0;\n'
        code += '}'


        with open(cc_path, 'w') as outfile:
            outfile.write(code)



class OpenFHEGenerator():
    """
    OpenFHEGenerator class that generates OpenFHE code from a given function file.
    It parses the function, determines if it contains comparelut statements, and generates the corresponding OpenFHE code.
    """
    def __init__(self, function_file_path = None):
        if function_file_path != None:
            self.function_str = open(function_file_path, 'r').read()
            self.function_name, self.param_names, self.statements = self.parse_function()
            self.contains_comparelut = any(stmt[0] == 'comparelut' for stmt in self.statements)
            self.openfhe_code = self.generate_openfhe_code(self.function_name, self.param_names, self.statements)

    def parse_function(self) -> Tuple:
        """
        Gets the function name, parameter names, and statements from the function string.

        Returns:
            tuple: A tuple containing the function name, parameter names, and statements.
        """
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

    def generate_openfhe_code(self, function_name: str, param_names: list, statements: list[list]) -> str:
        """
        Generates C++ code that uses the OpenFHE library to perform the operations specified in the function.

        Args:
            function_name (str): Name of the function.
            param_names (list): List of parameter names.
            statements (list): List of statements in the function.

        Returns:
            str: C++ code that uses the OpenFHE library to perform the operations specified in the function.
        """
        ret_var = statements[-1][1] if statements and statements[-1][0] == 'return' else None
        return_type = 'Ciphertext<DCRTPoly>'

        adjusted_params = []
        for param in param_names:
            adjusted_params.append(f'Ciphertext<DCRTPoly> &{param}')

        code = f'{return_type} {function_name}(' + ', '.join(adjusted_params) + ', const CryptoContext<DCRTPoly> &cryptoContext)\n'
        code += '{\n'

        for stmt in statements:
            if stmt[0] == 'declare_assign':
                var_name = stmt[1]
                operation = stmt[2]
                operands = stmt[3]
                if operation == 'RotateLeft':
                    step = operands[1]
                    code += f'    Ciphertext<DCRTPoly> {var_name} = cryptoContext->EvalRotate({operands[0]}, {step});\n'
                elif operation == 'Add':
                    code += f'    Ciphertext<DCRTPoly> {var_name} = cryptoContext->EvalAdd({operands[0]}, {operands[1]});\n'
                    for operand in operands[2:]:
                        code += f'    {var_name} = cryptoContext->EvalAdd({var_name}, {operand});\n'
                elif operation == 'Sub':
                    code += f'    Ciphertext<DCRTPoly> {var_name} = cryptoContext->EvalSub({operands[0]}, {operands[1]});\n'
                elif operation == 'lwe_multiply':
                    code += f'    Ciphertext<DCRTPoly> {var_name} = cryptoContext->EvalMult({operands[0]}, {operands[1]});\n'
            elif stmt[0] == 'comparelut':
                var_name = stmt[1]
                input_var = stmt[2]
                template_args = stmt[3]
                threshold = template_args[-1]
                match = re.search(r'[-+]?\d*\.?\d+([eE][-+]?\d+)?', threshold)
                threshold = float(match.group()) / 255
                code += f'    Ciphertext<DCRTPoly> {var_name} = cryptoContext->EvalChebyshevFunction('
                code += '[](double x) -> double { return x > %.6f ? 1.0 : 0.0; }, ' % threshold
                code += f'{input_var}, 0.0, 1.0, 1000);\n'
            elif stmt[0] == 'copy':
                src = stmt[1]
                dest = stmt[2]
                code += f'    {dest} = {src};\n'
            elif stmt[0] == 'return':
                ret_var = stmt[1]
                code += f'    return {ret_var};\n'
        code += '}\n'
        return code

    def encrypt(self, output_file_path: str, image: Imageplain, rotate_steps: list[int], build_path: str, image_data_path: str):
        """
        Generates a C++ program that uses the OpenFHE library to encrypt an image with specified rotation steps.

        Args:
            output_file_path (str): Path to save the generated C++ code.
            image (Imageplain): An object containing image data with attributes `height`, `width`, and `data`.
            rotate_steps (list[int]): A list of integers specifying the rotation steps for key generation.
            build_path (str): The build directory path.
            image_data_path (str): Path to the image data txt file.
        """
        height = image.height
        width = image.width
        nslots = height * width

        code = ''
        code += '#include "openfhe.h"\n'
        code += '#include "ciphertext-ser.h"\n'
        code += '#include "cryptocontext-ser.h"\n'
        code += '#include "key/key-ser.h"\n'
        code += '#include "scheme/ckksrns/ckksrns-ser.h"\n'
        code += '#include <vector>\n'
        code += '#include <iostream>\n'
        code += '#include <fstream>\n'
        code += '#include <cmath>\n'
        code += '#include <functional>\n'
        code += '#include <complex>\n\n'

        code += 'using namespace lbcrypto;\n\n'

        code += f'const std::string ccLocation = "{build_path}/cryptocontext.bin";\n'
        code += f'const std::string pubKeyLocation = "{build_path}/key_pub.bin";\n'
        code += f'const std::string secKeyLocation = "{build_path}/key_sec.bin";\n'
        code += f'const std::string multKeyLocation = "{build_path}/key_mult.bin";\n'
        code += f'const std::string rotKeyLocation = "{build_path}/key_rot.bin";\n'
        code += f'const std::string cipherEncLocation = "{build_path}/ciphertextenc.bin";\n'
        code += f'const std::string imageDataPath = "{image_data_path}";\n\n'

        code += 'int main() {\n'
        code += '    // Initialize Crypto Parameters\n'
        code += '    CCParams<CryptoContextCKKSRNS> parameters;\n\n'

        code += f'    size_t BatchSize = {nslots};\n'
        code += '    parameters.SetMultiplicativeDepth(12);\n'
        code += '    parameters.SetScalingModSize(50);\n'
        code += '    parameters.SetBatchSize(BatchSize);\n'
        code += '    parameters.SetSecurityLevel(HEStd_128_classic);\n'
        code += '    parameters.SetRingDim(65536);\n'
        code += '    parameters.SetScalingTechnique(FLEXIBLEAUTO);\n'
        code += '    parameters.SetFirstModSize(60);\n\n'

        code += '    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);\n'
        code += '    cryptoContext->Enable(PKE);\n'
        code += '    cryptoContext->Enable(KEYSWITCH);\n'
        code += '    cryptoContext->Enable(LEVELEDSHE);\n'
        code += '    cryptoContext->Enable(ADVANCEDSHE);\n\n'

        code += '    auto keyPair = cryptoContext->KeyGen();\n'
        code += '    cryptoContext->EvalMultKeyGen(keyPair.secretKey);\n'
        rotate_steps_str = ', '.join(map(str, rotate_steps))
        code += f'    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {{{rotate_steps_str}}});\n\n'

        code += '    std::vector<double> image;\n'
        code += '    std::ifstream imgFile(imageDataPath);\n'
        code += '    double value;\n'
        code += '    while (imgFile >> value) {\n'
        code += '        image.push_back(value);\n'
        code += '    }\n'
        code += '    imgFile.close();\n\n'

        code += '    Plaintext ptxt_image = cryptoContext->MakeCKKSPackedPlaintext(image);\n'
        code += '    Ciphertext<DCRTPoly> ciphertext_image = cryptoContext->Encrypt(keyPair.publicKey, ptxt_image);\n\n'

        code += '    // Serialize ciphertext and keys\n'
        code += '    Serial::SerializeToFile(cipherEncLocation, ciphertext_image, SerType::BINARY);\n'
        code += '    Serial::SerializeToFile(ccLocation, cryptoContext, SerType::BINARY);\n'
        code += '    Serial::SerializeToFile(secKeyLocation, keyPair.secretKey, SerType::BINARY);\n'
        code += '    Serial::SerializeToFile(pubKeyLocation, keyPair.publicKey, SerType::BINARY);\n\n'

        code += '    std::ofstream multKeyFile(multKeyLocation, std::ios::out | std::ios::binary);\n'
        code += '    cryptoContext->SerializeEvalMultKey(multKeyFile, SerType::BINARY);\n'
        code += '    multKeyFile.close();\n\n'

        code += '    std::ofstream rotKeyFile(rotKeyLocation, std::ios::out | std::ios::binary);\n'
        code += '    cryptoContext->SerializeEvalAutomorphismKey(rotKeyFile, SerType::BINARY);\n'
        code += '    rotKeyFile.close();\n\n'

        code += '    std::cout << "Encryption complete. Serialized files saved." << std::endl;\n'
        code += '    return 0;\n'
        code += '}\n'

        with open(output_file_path, 'w') as outfile:
            outfile.write(code)

    def decrypt(self, output_file_path: str, output_txt_path: str, cipher_path: str, width: int, height: int, build_path: str):
        """
        Generate a C++ program that uses the OpenFHE library to decrypt an encrypted image and save the result to a text file.

        Args:
            output_file_path (str): The path where the generated C++ code file is saved.
            output_txt_path (str): The path where the decrypted image data is saved (text file).
            width (int): The width of the image.
            height (int): The height of the image.
            build_path (str): The build directory path.
        """
        nslots = width * height 

        code = ''
        code += '#include "openfhe.h"\n'
        code += '#include "ciphertext-ser.h"\n'
        code += '#include "cryptocontext-ser.h"\n'
        code += '#include "key/key-ser.h"\n'
        code += '#include "scheme/ckksrns/ckksrns-ser.h"\n'
        code += '#include <vector>\n'
        code += '#include <iostream>\n'
        code += '#include <fstream>\n'
        code += '#include <cmath>\n'
        code += '#include <functional>\n'
        code += '#include <complex>\n\n'

        code += 'using namespace lbcrypto;\n\n'

        code += f'const std::string ccLocation = "{build_path}/cryptocontext.bin";\n'
        code += f'const std::string secKeyLocation = "{build_path}/key_sec.bin";\n'
        code += f'const std::string cipherEvalLocation = "{cipher_path}";\n'
        code += f'const std::string outputTxtLocation = "{output_txt_path}";\n\n'

        code += 'int main() {\n'
        code += f'    size_t Slots = {nslots};\n'
        code += f'    int height = {height};\n'
        code += f'    int width = {width};\n\n'

        code += '    CryptoContext<DCRTPoly> cryptoContextClient;\n'
        code += '    PrivateKey<DCRTPoly> sk;\n'
        code += '    Serial::DeserializeFromFile(ccLocation, cryptoContextClient, SerType::BINARY);\n'
        code += '    Serial::DeserializeFromFile(secKeyLocation, sk, SerType::BINARY);\n\n'

        code += '    Ciphertext<DCRTPoly> ciphertexteval;\n'
        code += '    Serial::DeserializeFromFile(cipherEvalLocation, ciphertexteval, SerType::BINARY);\n\n'

        code += '    Plaintext result_ptxt;\n'
        code += '    cryptoContextClient->Decrypt(sk, ciphertexteval, &result_ptxt);\n'
        code += f'    result_ptxt->SetLength(Slots);\n\n'

        code += '    std::vector<double> decvec = result_ptxt->GetRealPackedValue();\n\n'

        code += '    std::ofstream outfile(outputTxtLocation);\n'
        code += '    if (outfile.is_open()) {\n'
        code += '        size_t index = 0;\n'
        code += '        for (int i = 0; i < height; ++i) {\n'
        code += '            for (int j = 0; j < width; ++j) {\n'
        code += '                outfile << decvec[index++] * 255 << " ";\n'
        code += '            }\n'
        code += '            outfile << std::endl;\n'
        code += '        }\n'
        code += '        outfile.close();\n'
        code += '        std::cout << "Decryption complete. Data saved to " << outputTxtLocation << std::endl;\n'
        code += '    } else {\n'
        code += '        std::cerr << "Can\'t open file:" << outputTxtLocation << std::endl;\n'
        code += '    }\n'
        code += '    return 0;\n'
        code += '}\n'

        with open(output_file_path, 'w') as outfile:
            outfile.write(code)

    def cpptoof(self, input_cipher_path: str, output_cpp_path: str, output_cipher_path: str, build_path: str):
        """
        Generates a C++ program that loads an encrypted ciphertext, performs operations, and saves the resulting ciphertext.

        Args:
            input_cipher_path (str): Path to the input encrypted ciphertext file.
            cc_path (str): Path to the serialized CryptoContext and keys.
            output_cipher_path (str): Path where the output ciphertext will be saved.
            build_path (str): Build folder of openfhe back-end.
        """

        code = ''
        code += '#include "openfhe.h"\n'
        code += '#include "ciphertext-ser.h"\n'
        code += '#include "cryptocontext-ser.h"\n'
        code += '#include "key/key-ser.h"\n'
        code += '#include "scheme/ckksrns/ckksrns-ser.h"\n'
        code += '#include <vector>\n'
        code += '#include <iostream>\n'
        code += '#include <fstream>\n'
        code += '#include <cmath>\n'
        code += '#include <functional>\n'
        code += '#include <complex>\n\n'

        code += 'using namespace lbcrypto;\n\n'

        code += f'std::string ccLocation = "{build_path}/cryptocontext.bin";\n'
        code += f'std::string multKeyLocation = "{build_path}/key_mult.bin";\n'
        code += f'std::string rotKeyLocation = "{build_path}/key_rot.bin";\n'

        code += f'std::string inputCipherLocation = "{input_cipher_path}";\n'
        code += f'std::string outputCipherLocation = "{output_cipher_path}";\n\n'

        code += self.openfhe_code + '\n\n'

        code += 'int main() {\n'
        code += '    // Deserialize CryptoContext and Keys\n'
        code += '    CryptoContext<DCRTPoly> cryptoContextClient;\n'
        code += '    Serial::DeserializeFromFile(ccLocation, cryptoContextClient, SerType::BINARY);\n\n'

        code += '    // Deserialize Evaluation Keys\n'
        code += f'    std::ifstream multKeyIStream("{build_path}/key_mult.bin", std::ios::in | std::ios::binary);\n'
        code += '    cryptoContextClient->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY);\n'
        code += f'    std::ifstream rotKeyIStream("{build_path}/key_rot.bin", std::ios::in | std::ios::binary);\n'
        code += '    cryptoContextClient->DeserializeEvalAutomorphismKey(rotKeyIStream, SerType::BINARY);\n\n'

        code += '    // Deserialize Input Ciphertext\n'
        code += '    Ciphertext<DCRTPoly> ciphertext_input;\n'
        code += '    Serial::DeserializeFromFile(inputCipherLocation, ciphertext_input, SerType::BINARY);\n\n'

        if self.param_names:
            for param in self.param_names:
                code += f'    Ciphertext<DCRTPoly> {param} = ciphertext_input;\n'

        code += f'    Ciphertext<DCRTPoly> result = {self.function_name}(' + ', '.join(self.param_names) + ', cryptoContextClient);\n\n'

        code += f'    Serial::SerializeToFile(outputCipherLocation, result, SerType::BINARY);\n\n'

        code += '    std::cout << "Evaluation complete. Result ciphertext saved to " << outputCipherLocation << std::endl;\n'
        code += '    return 0;\n'
        code += '}\n'

        with open(output_cpp_path, 'w') as outfile:
            outfile.write(code)

        print(f"C++ encrypted evaluation code has been generated and saved to {output_cpp_path}")
