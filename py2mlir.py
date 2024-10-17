#   Steps:
#   
#   git clone https://github.com/llvm/llvm-project.git
#   cd llvm-project
#   cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS="mlir"    \
#   -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host"          \
#   -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3) \
#   -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DLLVM_ENABLE_RTTI=ON
#   cd build
#   ninja install
#   export PYTHONPATH=tools/mlir/python_packages/mlir_core:${PYTHONPATH}
#
#   python3 py2mlir.py


import ast
from mlir import ir
from mlir.dialects import func, arith, scf
from mlir.dialects import memref
from mlir.ir import ShapedType
import numpy as np

code = '''
def main():
    a = 0
    b = 3
    for i in range(b):
        if b > 0:
            a += b
    return a
'''
code1 = '''
def inner_product(a: list[float], b:list[float]) -> float:
    result = a[0]*b[0]
    for i in range(3):
        result += a[i+1]*b[i+1]
    return result
'''
code2 = '''
def BoxBlur(img: list[float], img2: list[float]) -> float:
    imgSize = 4
    kerSize = 3
    weightMatrix = [1.0, 1, 1, 1, 1, 1, 1, 1, 1]
    for x in range(imgSize):
        for y in range(imgSize):
            value = 0
            for i in range(3):
                for j in range(3):
                    weightRow = i*kerSize
                    weightIndex = weightRow + j
                    imgRow = (x+i-1)*imgSize
                    imgIndex = imgRow + y + j - 1
                    ImgIndex = imgIndex % 16
                    value += weightMatrix[weightIndex] * img[ImgIndex];
            resRow = imgSize*x
            resIndex = resRow + y
            img2[resIndex] = value
    return_value = img2[0]
    return return_value
'''
code3 = '''
def BoxBlur(img: list[float], img2: list[float]) -> float:
    imgSize = 4
    kerSize = 3
    weightMatrix = [1.0, 1, 1, 1, 1, 1, 1, 1, 1]
    for x in range(imgSize):
        for y in range(imgSize):
            value = 0
            for i in range(3):
                for j in range(3):
                    weightRow = i*kerSize
                    weightIndex = weightRow + j
                    imgRow = (x+i-1)*imgSize
                    imgIndex = imgRow + y + j - 1
                    ImgIndex = imgIndex % 16
                    value += weightMatrix[weightIndex] * img[ImgIndex];
            resRow = imgSize*x
            resIndex = resRow + y
            img2[resIndex] = value
    return_value = img2[0]
    return return_value
'''
code4 = '''
def BoxBlur(img: list[float], img2: list[float]) -> float:
    imgSize = 4
    kerSize = 3
    weightMatrix = [1, 1]
    value = 0
    for x in range(imgSize):
        for y in range(imgSize):
            for i in range(3):
                for j in range(3):
                    value += weightMatrix[1];
            img2[x] = value
    return img2[0]
'''
code5 = '''
def temp():
    a = [0]
    a = 1.0
    b = 2.0
    return a
'''
code6 = '''
def a(b:int):
    if b > 0:
        a = 0
        b = 3
        for i in range(b):
            a += b
    return b
'''
code7 = '''
def test_mullist():
    a = [2, 3]
    d = [[1,1.0],[2,2]]
    b = [3, 4.0]
    c = 0
    for i in range(2):
        a[i] /= a[i]/b[i]
        b[i] %= b[i]
    return c
'''

parsed_ast = ast.parse(code3)
# print(ast.dump(parsed_ast, indent=2))

class MLIRGenerator(ast.NodeVisitor):
    def __init__(self, module):
        self.module = module
        self.symbol_table = {}
        self.block_stack = []
        self.func_table = {}

    def flatten_list(self, elts):
        if all(isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) for e in elts):
            values = [e.value for e in elts]
            shape = [len(elts)]
            return values, shape
        elif all(isinstance(e, ast.List) for e in elts):
            values_list = []
            shapes = []
            for e in elts:
                values, shape = self.flatten_list(e.elts)
                values_list.extend(values)
                shapes.append(shape)
            if not all(s == shapes[0] for s in shapes):
                raise ValueError("List sublists have inconsistent shapes")
            shape = [len(elts)] + shapes[0]
            return values_list, shape
        else:
            raise NotImplementedError("Unsupported list element type")

    def cast_to_index(self, value):
        index_type = ir.IndexType.get()
        with ir.InsertionPoint(self.block_stack[-1]):
            if value.type == index_type:
                return value
            elif isinstance(value.type, ir.IntegerType):
                return arith.IndexCastOp(index_type, value).result
            elif isinstance(value.type, ir.FloatType):
                int_value = arith.FPToSIOp(ir.IntegerType.get_signless(64), value).result
                return arith.IndexCastOp(index_type, int_value).result
            else:
                raise TypeError(f"Cannot cast type {value.type} to index")

    def cast_to_int64(self, value):
        int64_type = ir.IntegerType.get_signless(64)
        with ir.InsertionPoint(self.block_stack[-1]):
            if value.type == int64_type:
                return value
            elif isinstance(value.type, ir.IndexType):
                return arith.IndexCastOp(int64_type, value).result
            elif isinstance(value.type, ir.IntegerType):
                return arith.ExtSIOp(int64_type, value).result
            elif isinstance(value.type, ir.F64Type):
                return arith.FPToSIOp(int64_type, value).result
            else:
                raise TypeError(f"Cannot cast type {value.type} to int64")

    def cast_to_f64(self, value):
        f64_type = ir.F64Type.get()
        with ir.InsertionPoint(self.block_stack[-1]):
            if value.type == f64_type:
                return value
            elif isinstance(value.type, ir.IntegerType) or isinstance(value.type, ir.IndexType):
                return arith.SIToFPOp(f64_type, value).result
            elif isinstance(value.type, ir.F32Type):
                return arith.ExtFOp(f64_type, value).result
            else:
                raise TypeError(f"Cannot cast type {value.type} to f64")

    def parse_type_annotation(self, annotation):
        if isinstance(annotation, ast.Name):
            if annotation.id == 'float':
                return ir.F64Type.get()
            elif annotation.id == 'int':
                return ir.IntegerType.get_signless(64)
            else:
                raise NotImplementedError(f"Unsupported type annotation: {annotation.id}")
        elif isinstance(annotation, ast.Subscript):
            if (isinstance(annotation.value, ast.Name) and annotation.value.id == 'list'):
                element_type = self.parse_type_annotation(annotation.slice)
                dynamic_size = ir.ShapedType.get_dynamic_size()
                memref_type = ir.MemRefType.get([dynamic_size], element_type)
                return memref_type
            else:
                raise NotImplementedError(f"Unsupported type annotation: {ast.dump(annotation)}")
        else:
            raise NotImplementedError(f"Unsupported type annotation: {ast.dump(annotation)}")

    def create_subview(self, base, indices):
        base_type = base.type
        memref_shape = base_type.shape
        memref_rank = len(memref_shape)
        index_type = ir.IndexType.get()
        dynamic = ShapedType.get_dynamic_stride_or_offset()
        offsets = []
        sizes = []
        strides = []
        static_offsets = []
        static_sizes = []
        static_strides = []
        zero = arith.ConstantOp(index_type, 0).result
        one = arith.ConstantOp(index_type, 1).result
        for i in range(memref_rank):
            if i < len(indices):
                offsets.append(indices[i])
                static_offsets.append(dynamic) 
                sizes.append(one) 
                static_sizes.append(1) 
            else:
                offsets.append(zero)
                static_offsets.append(0)
                if memref_shape[i] == -1:
                    dim_size = memref.DimOp(base, i).result
                    sizes.append(dim_size)
                    static_sizes.append(dynamic) 
                else:
                    size = memref_shape[i]
                    size_value = arith.ConstantOp(index_type, size).result
                    sizes.append(size_value)
                    static_sizes.append(size)
            strides.append(one)
            static_strides.append(1)
        subview_shape = []
        for size in static_sizes:
            if size == dynamic:
                subview_shape.append(-1)
            else:
                subview_shape.append(size)
        subview_type = ir.MemRefType.get(
            subview_shape,
            base_type.element_type,
            memory_space=base_type.memory_space
        )
        subview_op = memref.SubViewOp(
            subview_type,
            base,
            offsets,
            sizes,
            strides,
            static_offsets=static_offsets,
            static_sizes=static_sizes,
            static_strides=static_strides
        )
        return subview_op

    def store_to_subscript(self, target, value):
        base = self.visit(target.value)
        if not isinstance(base.type, ir.MemRefType):
            raise TypeError(f"Expected a memref type for base, got {base.type}")
        if isinstance(target.slice, ast.Slice):
            raise NotImplementedError("Slicing assignment not supported yet")
        elif isinstance(target.slice, ast.ExtSlice):
            raise NotImplementedError("Extended slicing assignment not supported yet")
        elif isinstance(target.slice, ast.Tuple):
            indices = []
            for index_node in target.slice.elts:
                idx = self.visit(index_node)
                idx = self.cast_to_index(idx)
                indices.append(idx)
        else:
            idx = self.visit(target.slice)
            idx = self.cast_to_index(idx)
            indices = [idx]

        with ir.InsertionPoint(self.block_stack[-1]):
            memref.StoreOp(value, base, indices)

    def visit_Module(self, node):
        self.block_stack.append(self.module.body)
        for stmt in node.body:
            self.visit(stmt)
        self.block_stack.pop()

    def visit_FunctionDef(self, node):
        arg_types = []
        for arg in node.args.args:
            if arg.annotation:
                arg_type = self.parse_type_annotation(arg.annotation)
            else:
                arg_type = ir.F64Type.get()
            arg_types.append(arg_type)
        self.return_values = []
        placeholder_func_type = ir.FunctionType.get(arg_types, [])
        with ir.InsertionPoint(self.module.body):
            func_op = func.FuncOp(name=node.name, type=placeholder_func_type)
            entry_block = func_op.add_entry_block()
            self.block_stack.append(entry_block)
            self.symbol_table = {}
            for i, arg in enumerate(node.args.args):
                arg_name = arg.arg
                mlir_arg = entry_block.arguments[i]
                self.symbol_table[arg_name] = mlir_arg
            for stmt in node.body:
                self.visit(stmt)
            if self.return_values:
                return_types = [value.type for value in self.return_values]
            else:
                return_types = []
            func_type = ir.FunctionType.get(arg_types, return_types)
            func_op.attributes["function_type"] = ir.TypeAttr.get(func_type)
            self.func_table[node.name] = func_type
            self.block_stack.pop()

    def visit_If(self, node):
        condition = self.visit(node.test)
        assigned_vars = set()
        def collect_assigned_vars(stmts):
            for stmt in stmts:
                if isinstance(stmt, (ast.Assign, ast.AugAssign)):
                    target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)
                elif isinstance(stmt, ast.For):
                    collect_assigned_vars(stmt.body)
                elif isinstance(stmt, ast.If):
                    collect_assigned_vars(stmt.body)
                    collect_assigned_vars(stmt.orelse)
        collect_assigned_vars(node.body)
        collect_assigned_vars(node.orelse)
        input_vars = []
        input_values = []
        for var in assigned_vars:
            if var in self.symbol_table:
                input_vars.append(var)
                input_values.append(self.symbol_table[var])
        result_types = [v.type for v in input_values]
        with ir.InsertionPoint(self.block_stack[-1]):
            if_op = scf.IfOp(condition, result_types, hasElse=True)
        self.block_stack.append(if_op.then_block)
        with ir.InsertionPoint(if_op.then_block):
            for stmt in node.body:
                self.visit(stmt)
            then_results = [self.symbol_table[var] for var in input_vars]
            scf.YieldOp(then_results)
        self.block_stack.pop()
        self.block_stack.append(if_op.else_block)
        with ir.InsertionPoint(if_op.else_block):
            if node.orelse:
                for stmt in node.orelse:
                    self.visit(stmt)
                else_results = [self.symbol_table.get(var, input_values[idx]) for idx, var in enumerate(input_vars)]
            else:
                else_results = input_values
            scf.YieldOp(else_results)
        self.block_stack.pop()
        for idx, var in enumerate(input_vars):
            new_value = if_op.results[idx]
            self.symbol_table[var] = new_value

    def visit_Subscript(self, node):
        base = self.visit(node.value)
        base_type = base.type
        if not isinstance(base_type, ir.MemRefType):
            raise TypeError(f"Expected a memref type, got {base_type}")
        memref_shape = base_type.shape
        memref_rank = len(memref_shape)
        indices = []
        if isinstance(node.slice, ast.Slice):
            raise NotImplementedError("Slicing not supported yet")
        elif isinstance(node.slice, ast.ExtSlice):
            raise NotImplementedError("Extended slicing not supported yet")
        elif isinstance(node.slice, ast.Tuple):
            for index_node in node.slice.elts:
                idx = self.visit(index_node)
                idx = self.cast_to_index(idx)
                indices.append(idx)
        else:
            idx = self.visit(node.slice)
            idx = self.cast_to_index(idx)
            indices.append(idx)
        with ir.InsertionPoint(self.block_stack[-1]):
            if len(indices) == memref_rank:
                load_op = memref.LoadOp(base, indices)
                return load_op.result
            elif len(indices) < memref_rank:
                subview_op = self.create_subview(base, indices)
                desired_shape = list(base_type.shape[len(indices):])
                desired_type = ir.MemRefType.get(
                    desired_shape,
                    base_type.element_type,
                    memory_space=base_type.memory_space
                )
                cast_op = memref.CastOp(desired_type, subview_op.result)
                return cast_op.result
            else:
                raise IndexError("Too many indices for memref")

    def visit_List(self, node):
        values, shape = self.flatten_list(node.elts)
        if all(isinstance(v, int) for v in values):
            element_type = ir.IntegerType.get_signless(64)
            eltype = 0
        else:
            element_type = ir.F64Type.get()
            eltype = 1
        memref_type = ir.MemRefType.get(shape, element_type)
        dynamic_sizes = []
        symbol_operands = []
        with ir.InsertionPoint(self.block_stack[-1]):
            memref_alloc = memref.AllocOp(memref_type, dynamic_sizes, symbol_operands).result
            if eltype == 0:
                np_values = np.array(values, dtype=np.int64).reshape(shape)
            else:
                np_values = np.array(values, dtype=np.float64).reshape(shape)
            for idx, value in np.ndenumerate(np_values):
                index = [arith.ConstantOp(ir.IndexType.get(), i).result for i in idx]
                memref.StoreOp(arith.ConstantOp(element_type, value.item()), memref_alloc, index)
        return memref_alloc

    def visit_Call(self, node):
        func_name = node.func.id
        if func_name not in self.func_table:
            raise NameError(f"Function '{func_name}' is not defined.")
        func_type = self.func_table[func_name]
        return_types = func_type.results
        args = [self.visit(arg) for arg in node.args]
        with ir.InsertionPoint(self.block_stack[-1]):
            call_op = ir.Operation.create(
                "func.call",
                results=return_types,
                operands=args,
                attributes={"callee": ir.FlatSymbolRefAttr.get(func_name)}
            )
        if len(return_types) == 0:
            return None
        elif len(return_types) == 1:
            return call_op.result
        else:
            return tuple(call_op.results)

    def visit_Return(self, node):
        return_value = self.visit(node.value)
        with ir.InsertionPoint(self.block_stack[-1]):
            func.ReturnOp([return_value])
        self.return_values.append(return_value)

    def visit_Name(self, node):
        value = self.symbol_table.get(node.id)
        if value is None:
            raise NameError(f"Value '{node.id}' is not defined")
        return value

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(left.type, ir.F64Type) or isinstance(right.type, ir.F64Type):
            left = self.cast_to_f64(left)
            right = self.cast_to_f64(right)
            with ir.InsertionPoint(self.block_stack[-1]):
                if isinstance(node.op, ast.Add):
                    op = arith.AddFOp(left, right)
                elif isinstance(node.op, ast.Sub):
                    op = arith.SubFOp(left, right)
                elif isinstance(node.op, ast.Mult):
                    op = arith.MulFOp(left, right)
                elif isinstance(node.op, ast.Div):
                    op = arith.DivFOp(left, right)
                elif isinstance(node.op, ast.Mod):
                    op = arith.RemFOp(left, right)
                else:
                    raise NotImplementedError(f"Unsupported operator: {type(node.op)}")
        elif isinstance(left.type, (ir.IntegerType, ir.IndexType)) and isinstance(right.type, (ir.IntegerType, ir.IndexType)):
            left = self.cast_to_int64(left)
            right = self.cast_to_int64(right)
            with ir.InsertionPoint(self.block_stack[-1]):
                if isinstance(node.op, ast.Add):
                    op = arith.AddIOp(left, right)
                elif isinstance(node.op, ast.Sub):
                    op = arith.SubIOp(left, right)
                elif isinstance(node.op, ast.Mult):
                    op = arith.MulIOp(left, right)
                elif isinstance(node.op, ast.Div):
                    op = arith.DivSIOp(left, right)
                elif isinstance(node.op, ast.Mod):
                    op = arith.RemSIOp(left, right)
                else:
                    raise NotImplementedError(f"Unsupported operator: {type(node.op)}")
        else:
            raise TypeError("Operand types do not match for binary operation.")

        return op.result

    def visit_Compare(self, node):
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        left = self.cast_to_f64(left)
        right = self.cast_to_f64(right)
        with ir.InsertionPoint(self.block_stack[-1]):
            if isinstance(node.ops[0], ast.Gt):
                op = arith.CmpFOp(arith.CmpFPredicate.OGT, left, right)
            elif isinstance(node.ops[0], ast.Lt):
                op = arith.CmpFOp(arith.CmpFPredicate.OLT, left, right)
            elif isinstance(node.ops[0], ast.Eq):
                op = arith.CmpFOp(arith.CmpFPredicate.OEQ, left, right)
            else:
                raise NotImplementedError(f"Unsupported comparison operator: {type(node.ops[0])}")
        return op.result

    def visit_For(self, node):
        if not (isinstance(node.iter, ast.Call) and
                isinstance(node.iter.func, ast.Name) and
                node.iter.func.id == 'range'):
            raise NotImplementedError("Only loops of the form 'for ... in range(...)' are supported, where range can contain variables")
        range_args = node.iter.args
        index_type = ir.IndexType.get()
        with ir.InsertionPoint(self.block_stack[-1]):
            if len(range_args) == 1:
                start = arith.ConstantOp(index_type, 0).result
                end = self.cast_to_index(self.visit(range_args[0]))
                step = arith.ConstantOp(index_type, 1).result
            elif len(range_args) == 2:
                start = self.cast_to_index(self.visit(range_args[0]))
                end = self.cast_to_index(self.visit(range_args[1]))
                step = arith.ConstantOp(index_type, 1).result
            elif len(range_args) == 3:
                start = self.cast_to_index(self.visit(range_args[0]))
                end = self.cast_to_index(self.visit(range_args[1]))
                step = self.cast_to_index(self.visit(range_args[2]))
            else:
                raise NotImplementedError("range() only supports 1 to 3 parameters")
            assigned_vars = set()
            def collect_assigned_vars(stmts):
                for stmt in stmts:
                    if isinstance(stmt, (ast.Assign, ast.AugAssign)):
                        target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
                        if isinstance(target, ast.Name):
                            assigned_vars.add(target.id)
                    elif isinstance(stmt, ast.For):
                        collect_assigned_vars(stmt.body)
                    elif isinstance(stmt, ast.If):
                        collect_assigned_vars(stmt.body)
                        collect_assigned_vars(stmt.orelse)
            collect_assigned_vars(node.body)
            loop_carried_vars = []
            init_vals = []
            for var_name in assigned_vars:
                if var_name in self.symbol_table:
                    init_val = self.symbol_table[var_name]
                    loop_carried_vars.append(var_name)
                    init_vals.append(init_val)
            loop = scf.ForOp(start, end, step, iter_args=init_vals)
            loop_body = loop.body
        self.block_stack.append(loop_body)
        with ir.InsertionPoint(loop_body):
            loop_var = loop.induction_variable
            self.symbol_table[node.target.id] = loop_var
            for idx, var_name in enumerate(loop_carried_vars):
                self.symbol_table[var_name] = loop_body.arguments[idx + 1]
            for stmt in node.body:
                self.visit(stmt)
            yield_vals = [self.symbol_table[var_name] for var_name in loop_carried_vars]
            scf.YieldOp(yield_vals)
        self.block_stack.pop()
        for idx, var_name in enumerate(loop_carried_vars):
            self.symbol_table[var_name] = loop.results[idx]

    def visit_Assign(self, node):
        value = self.visit(node.value)
        target = node.targets[0]
        if isinstance(target, ast.Name):
            target_id = target.id
            self.symbol_table[target_id] = value
        else:
            if isinstance(target, ast.Subscript):
                self.store_to_subscript(target, value)
            else:
                raise NotImplementedError(f"Unsupported assignment target: {type(target)}")

    def visit_AugAssign(self, node):
        target = node.target
        value = self.visit(node.value)
        if isinstance(target, ast.Name):
            var_name = target.id
            current_value = self.symbol_table[var_name]
            if isinstance(current_value.type, ir.F64Type):
                value = self.cast_to_f64(value)
                with ir.InsertionPoint(self.block_stack[-1]):
                    if isinstance(node.op, ast.Add):
                        op = arith.AddFOp(current_value, value).result
                    elif isinstance(node.op, ast.Sub):
                        op = arith.SubFOp(current_value, value).result
                    elif isinstance(node.op, ast.Mult):
                        op = arith.MulFOp(current_value, value).result
                    elif isinstance(node.op, ast.Div):
                        op = arith.DivFOp(current_value, value).result
                    elif isinstance(node.op, ast.Mod):
                        op = arith.RemFOp(current_value, value).result
                    else:
                        raise NotImplementedError(f"Unsupported operator: {type(node.op)}")
            elif isinstance(current_value.type, (ir.IntegerType)):
                value = self.cast_to_int64(value)
                with ir.InsertionPoint(self.block_stack[-1]):
                    if isinstance(node.op, ast.Add):
                        op = arith.AddIOp(current_value, value).result
                    elif isinstance(node.op, ast.Sub):
                        op = arith.SubIOp(current_value, value).result
                    elif isinstance(node.op, ast.Mult):
                        op = arith.MulIOp(current_value, value).result
                    elif isinstance(node.op, ast.Div):
                        op = arith.DivSIOp(current_value, value).result
                    elif isinstance(node.op, ast.Mod):
                        op = arith.RemSIOp(current_value, value).result
                    else:
                        raise NotImplementedError(f"Unsupported operator: {type(node.op)}")
            self.symbol_table[var_name] = op
        elif isinstance(target, ast.Subscript):
            base = self.visit(target.value)
            indices = [self.cast_to_index(self.visit(target.slice))]
            current_value = memref.LoadOp(base, indices).result
            with ir.InsertionPoint(self.block_stack[-1]):
                if isinstance(current_value.type, ir.F64Type):
                    value = self.cast_to_f64(value)
                    if isinstance(node.op, ast.Add):
                        result = arith.AddFOp(current_value, value).result
                    elif isinstance(node.op, ast.Sub):
                        result = arith.SubFOp(current_value, value).result
                    elif isinstance(node.op, ast.Mult):
                        result = arith.MulFOp(current_value, value).result
                    elif isinstance(node.op, ast.Div):
                        result = arith.DivFOp(current_value, value).result
                    elif isinstance(node.op, ast.Mod):
                        result = arith.RemFOp(current_value, value).result
                    else:
                        raise NotImplementedError(f"Unsupported compound assignment operation: {type(node.op)}")
                elif isinstance(current_value.type, (ir.IntegerType)):
                    value = self.cast_to_int64(value)
                    if isinstance(node.op, ast.Add):
                        result = arith.AddIOp(current_value, value).result
                    elif isinstance(node.op, ast.Sub):
                        result = arith.SubIOp(current_value, value).result
                    elif isinstance(node.op, ast.Mult):
                        result = arith.MulIOp(current_value, value).result
                    elif isinstance(node.op, ast.Div):
                        result = arith.DivSIOp(current_value, value).result
                    elif isinstance(node.op, ast.Mod):
                        result = arith.RemSIOp(current_value, value).result
                    else:
                        raise NotImplementedError(f"Unsupported compound assignment operation: {type(node.op)}")
            memref.StoreOp(result, base, indices)
        else:
            raise NotImplementedError(f"Unsupported assignment target: {type(target)}")

    def visit_Constant(self, node):
        value = node.value
        with ir.InsertionPoint(self.block_stack[-1]):
            if isinstance(value, int):
                int_type = ir.IntegerType.get_signless(64)
                const_op = arith.ConstantOp(int_type, value)
            elif isinstance(value, float):
                f64 = ir.F64Type.get()
                const_op = arith.ConstantOp(f64, value)
            else:
                raise NotImplementedError(f"Unsupported constant type: {type(value)}")
        return const_op.result

with ir.Context() as ctx:
    with ir.Location.unknown():
        module = ir.Module.create()
        generator = MLIRGenerator(module)
        generator.visit(parsed_ast)
        print(module)
