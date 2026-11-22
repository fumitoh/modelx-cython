"""Native integer boundaries reject lossy conversions with rebuild guidance."""

import json
import operator
import subprocess
import sys
import textwrap

import modelx as mx
import pytest


def run_python(work, *args):
    result = subprocess.run([sys.executable, "-B", *args], cwd=work,
                            capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def integer_domain_build(tmp_path_factory):
    # these checks are the checked mode's (owner's rule): the small model would resolve unchecked
    return _integer_domain_build(tmp_path_factory, "checked")


@pytest.fixture(scope="module")
def integer_domain_build_unchecked(tmp_path_factory):
    return _integer_domain_build(tmp_path_factory, "unchecked")


def _integer_domain_build(tmp_path_factory, mode):
    work = tmp_path_factory.mktemp(f"integer-domains-{mode}")
    model = mx.new_model("IntegerDomains")
    try:
        projection = model.new_space("Outer").new_space(
            "Projection", refs={"payload": {"value": 2}, "amount": 30, "divisor": -1, "shift": 1,
                                "get_item": operator.getitem, "reads": []})
        # An object result keeps the argument checks independent of native
        # result admission and dense cache range checks.
        projection.new_cells("receive", 'lambda count: {"count": count}')
        projection.new_cells("receive_default", 'lambda count=4: {"count": count}')
        projection.new_cells("internal_argument", 'lambda: receive(payload["value"])')
        projection.new_cells("internal_operand", 'lambda: receive(payload["value"] + 1)')
        projection.new_cells("internal_keyword", 'lambda: receive_default(count=payload["value"])')
        projection.new_cells("object_result", 'lambda: payload["value"]')
        projection.new_cells("conditional_result", 'lambda: payload["value"] if amount > 0 else 0')
        projection.new_cells("helper_result", 'lambda: get_item(payload, "value")')
        projection.new_cells("read_amount", "lambda: amount")
        projection.new_cells("read_divisor", """def read_divisor():
    reads.append(divisor)
    return divisor
""").is_cached = False
        for name, result in (("terminal_product", "multiplier * (read_divisor() - 1)"),
                             ("reordered_product", "(read_divisor() - 1) * multiplier")):
            projection.new_cells(name, f"def {name}():\n    multiplier = read_amount()\n    return {result}\n")
        projection.new_cells("increment_amount", "lambda: amount + 1")
        projection.new_cells("decrement_amount", "lambda: amount - 1")
        projection.new_cells("square_amount", "lambda: amount * amount")
        projection.new_cells("divide_amount", "lambda: amount // divisor")
        projection.new_cells("left_shift", "lambda: amount << shift")
        projection.new_cells("right_shift", "lambda: amount >> shift")
        projection.new_cells("remainder", "lambda: amount % divisor")
        projection.new_cells("negate_amount", "lambda: -amount")
        projection.new_cells("local_amount", """def local_amount():
    adjusted = amount - 1
    return adjusted + 1
""")
        projection.new_cells("reassigned_local", """def reassigned_local():
    adjusted = amount
    adjusted = adjusted - 1
    return adjusted + 1
""")
        projection.new_cells("conditional_local", """def conditional_local():
    if divisor < 0:
        adjusted = amount - 1
    else:
        adjusted = amount + 1
    return adjusted
""")
        projection.new_cells("single_assignment_after_guard", """def single_assignment_after_guard():
    if divisor == 0:
        return 0
    adjusted = amount
    return adjusted + 1
""")
        projection.new_cells("mixed_local", """def mixed_local():
    if divisor < 0:
        adjusted = amount
    else:
        adjusted = payload["value"]
    return adjusted
""")
        projection.new_cells("possibly_unbound_local", """def possibly_unbound_local():
    if divisor < 0:
        adjusted = amount
    elif divisor == 0:
        adjusted = amount
    return adjusted
""")
        projection.new_cells("python_context_local", """def python_context_local():
    adjusted = amount
    adjusted = adjusted + 1
    if adjusted is payload["value"]:
        return 0
    return adjusted
""")
        projection.new_cells("prefix_identity_local", """def prefix_identity_local():
    adjusted = amount
    saved = [adjusted]
    if adjusted is saved[0]:
        return adjusted + 1
    return adjusted
""")
        projection.new_cells("native_operand", "lambda: receive(amount + 1)")
        projection.new_cells("inline_receive", "lambda count: count + 1")
        projection.new_cells("inline_operand", 'lambda: {"count": inline_receive(amount * 4)}')
        projection.new_cells("shift_argument", 'lambda: {"count": inline_receive(amount << shift)}')
        policies = model.new_space("Policies", formula="lambda policy: {}")
        policies.new_cells("number", "lambda: policy")
        assert policies[4].number() == 4
        assert projection.receive(2) == projection.internal_argument() == {"count": 2}
        assert projection.internal_operand() == {"count": 3}
        assert projection.receive_default() == {"count": 4}
        assert projection.internal_keyword() == {"count": 2}
        assert projection.conditional_result() == 2
        assert projection.object_result() == projection.helper_result() == 2
        assert projection.read_amount() == 30
        model.export(work / "exported")
    finally:
        model.close()

    (work / "sample.py").write_text(textwrap.dedent("""\
        import numpy as np
        from exported import mx_model

        projection = mx_model.Outer.Projection
        for count in (0, 2, 6, np.int64(4)):
            projection.receive(count)
            projection.receive_default(count)
            mx_model.Policies[count].number()
        projection.receive_default()
        projection.internal_argument()
        projection.internal_operand()
        projection.internal_keyword()
        projection.object_result()
        projection.conditional_result()
        projection.helper_result()
        projection.read_amount()
        projection.terminal_product()
        projection.reordered_product()
        projection.increment_amount()
        projection.decrement_amount()
        projection.square_amount()
        projection.divide_amount()
        projection.left_shift()
        projection.right_shift()
        projection.remainder()
        projection.negate_amount()
        projection.local_amount()
        projection.reassigned_local()
        projection.conditional_local()
        projection.single_assignment_after_guard()
        projection.mixed_local()
        projection.possibly_unbound_local()
        projection.python_context_local()
        projection.prefix_identity_local()
        projection.native_operand()
        projection.inline_receive(0)
        projection.inline_operand()
        projection.shift_argument()
        """), encoding="utf-8")
    (work / "spec.py").write_text(repr({"integer_checks": mode}), encoding="utf-8")
    run_python(work, "-m", "modelx_cython", "exported", "--sample", "sample.py", "--spec", "spec.py")

    report = json.loads((work / "exported_cy/_mx_admission.json").read_text(encoding="utf-8"))

    def admission(cell):
        result, = (row for name, row in report.items() if name.endswith(f"._f_{cell}"))
        return result

    assert admission("receive")["arguments"]["count"] == "long long"
    assert admission("receive")["return_type"] == "object"
    assert admission("receive_default")["arguments"]["count"] == "long long"
    assert admission("inline_receive")["inline_entry"]
    for name in ("object_result", "conditional_result", "helper_result", "read_amount"):
        assert admission(name)["return_type"] == "long long"

    (work / "check_boundaries.py").write_text(textwrap.dedent(r"""
        from importlib import import_module
        from importlib.machinery import ExtensionFileLoader
        import math
        import re
        import sys

        import numpy as np
        from exported import mx_model as exported
        from exported_cy import mx_model as compiled

        case = sys.argv[1]
        prefix = "Outer.Projection."
        positional_cases = {"public_argument", "internal_argument", "internal_operand"}
        keyword_cases = {"public_keyword", "internal_keyword"}
        receiving_cases = positional_cases | keyword_cases
        if case in positional_cases:
            cell_paths = (prefix + "receive",)
        elif case in keyword_cases:
            cell_paths = (prefix + "receive_default",)
        elif case == "reference":
            # An eager setter can name the Reference instead of the consuming Cell.
            cell_paths = (prefix + "amount", prefix + "read_amount")
        else:
            cell_paths = (prefix + case,)

        classes = import_module(type(compiled.Outer.Projection).__module__)
        assert isinstance(classes.__loader__, ExtensionFileLoader), classes.__file__

        def calculate(model, value, warm_cache=False):
            projection = type(model)().Outer.Projection
            if warm_cache:
                if case in keyword_cases:
                    assert projection.receive_default() == {"count": 4}
                    projection.receive_default(2)
                else:
                    projection.receive(3 if case == "internal_operand" else 2)
            if case == "public_argument":
                return projection.receive(value)["count"]
            if case == "public_keyword":
                assert projection.receive_default() == {"count": 4}
                return projection.receive_default(count=value)["count"]
            if case == "reference":
                projection.amount = value
                return projection.read_amount()
            projection.payload = {"value": value}
            result = getattr(projection, case)()
            return result["count"] if case in receiving_cases else result

        def expected(value):
            return value + 1 if case == "internal_operand" else value

        valid = (2, np.int64(4))
        if case not in receiving_cases:
            valid += (-1, -(1 << 63), np.int64((1 << 63) - 1))
        for value in valid:
            assert calculate(exported, value) == expected(value), (case, value)
            assert calculate(compiled, value) == expected(value), (case, value)

        invalid = (2.5, np.float64(2.5), float("nan"), 1 << 70,
                   -(1 << 70), np.uint64(1 << 63))
        for value in invalid:
            raw = expected(value)
            exported_result = calculate(exported, value)
            if math.isnan(raw):
                assert math.isnan(exported_result), (case, value)
            else:
                assert exported_result == raw, (case, value)
            for warm_cache in ((False, True) if case in receiving_cases else (False,)):
                detail = f"{case}, value={value!r}, warm_cache={warm_cache}"
                if case == "reference":     # as upstream: a plain cdef public integer Reference (Cython's conversion)
                    if isinstance(value, float) and math.isfinite(value):
                        assert calculate(compiled, value, warm_cache) == int(value), detail
                    else:
                        try:
                            actual = calculate(compiled, value, warm_cache)
                        except (ValueError, OverflowError):
                            continue
                        raise AssertionError(f"{detail}: compiled calculation returned {actual!r}")
                    continue
                try:
                    actual = calculate(compiled, value, warm_cache)
                except (TypeError, ValueError, OverflowError) as error:
                    message = str(error)
                    assert any(path in message for path in cell_paths), (detail, message)
                    assert re.search(
                        r"\b(?:sample|spec(?:ification)?|return_type|param_type)\b", message, re.I
                    ), (detail, message)
                    assert re.search(
                        r"\b(?:rebuild|recompile|retrace|wider|widen|broader)\b", message, re.I
                    ), (detail, message)
                else:
                    raise AssertionError(f"{detail}: compiled calculation returned {actual!r}")
        """), encoding="utf-8")
    return work


@pytest.mark.parametrize("case", [
    "public_argument",
    "public_keyword",
    "internal_argument",
    "internal_operand",
    "internal_keyword",
    "object_result",
    "conditional_result",
    "helper_result",
    "reference",
])
def test_native_integer_boundary(integer_domain_build, case):
    run_python(integer_domain_build, "check_boundaries.py", case)


def test_integer_itemspace_parameter(integer_domain_build):
    run_python(integer_domain_build, "-c", textwrap.dedent("""
        import numpy as np
        from exported_cy import mx_model

        policies = mx_model.Policies
        assert policies[np.int64(4)].number() == 4
        for bad in (2.5, float("nan"), 1 << 70, np.uint64(1 << 63)):
            try:
                policies[bad]
            except (TypeError, OverflowError) as error:
                assert "Policies" in str(error) and "wider sample or spec" in str(error)
            else:
                raise AssertionError(f"ItemSpace accepted an out-of-domain key {bad!r}")
        """))


def test_private_native_entry_checks_integer_argument(integer_domain_build):
    run_python(integer_domain_build, "-c", textwrap.dedent("""\
        from exported import mx_model as exported
        from exported_cy import mx_model as compiled

        reference = type(exported)().Outer.Projection
        projection = type(compiled)().Outer.Projection
        assert reference.inline_operand() == projection.inline_operand() == {"count": 121}

        reference = type(exported)().Outer.Projection
        projection = type(compiled)().Outer.Projection
        reference.amount = projection.amount = 1 << 62
        assert reference.inline_operand() == {"count": (1 << 64) + 1}
        assert projection.inline_receive(0) == 1
        try:
            projection.inline_operand()
        except OverflowError as error:
            # the checked mode raises where the argument is computed (amount * 4), naming that Cell
            assert "Outer.Projection.inline_operand" in str(error)
        else:
            raise AssertionError("private entry accepted a wrapped integer argument")
        """))


def test_native_signed_arithmetic_cannot_wrap_result(integer_domain_build):
    run_python(integer_domain_build, "-c", textwrap.dedent("""\
        from exported import mx_model as exported
        from exported_cy import mx_model as compiled

        maximum, minimum = (1 << 63) - 1, -(1 << 63)
        cases = (
            ("increment_amount", maximum, maximum - 1),
            ("decrement_amount", minimum, minimum + 1),
            ("square_amount", 3037000500, 1 << 31),
            ("divide_amount", minimum, minimum + 1),
            ("negate_amount", minimum, minimum + 1),
        )
        for cell, overflow, valid in cases:
            reference = type(exported)().Outer.Projection
            projection = type(compiled)().Outer.Projection
            reference.amount = projection.amount = valid
            assert getattr(projection, cell)() == getattr(reference, cell)()
            reference = type(exported)().Outer.Projection
            projection = type(compiled)().Outer.Projection
            reference.amount = projection.amount = overflow
            assert not minimum <= getattr(reference, cell)() <= maximum
            try:
                getattr(projection, cell)()
            except OverflowError as error:
                assert f"Outer.Projection.{cell}" in str(error)
            else:
                raise AssertionError(f"{cell} wrapped before conversion")

        reference = type(exported)().Outer.Projection
        projection = type(compiled)().Outer.Projection
        reference.amount = projection.amount = maximum
        assert reference.native_operand() == {"count": maximum + 1}
        try:
            projection.native_operand()
        except OverflowError as error:
            # raised where the operand is computed (amount + 1), naming that Cell
            assert "Outer.Projection.native_operand" in str(error)
        else:
            raise AssertionError("A native operand wrapped before argument conversion")

        reference = type(exported)().Outer.Projection
        projection = type(compiled)().Outer.Projection
        reference.amount = projection.amount = minimum
        assert reference.local_amount() == minimum
        try:
            projection.local_amount()
        except OverflowError as error:
            assert "Outer.Projection.local_amount" in str(error)
        else:
            raise AssertionError("The native local accepted an overflowing intermediate")
        """))


def test_signed_local_flow(integer_domain_build):
    report = json.loads((integer_domain_build / "exported_cy/_mx_admission.json").read_text(encoding="utf-8"))

    def object_locals(cell):
        entry = next(v for k, v in report.items() if k.endswith(f"._c_Projection._f_{cell}"))
        return {item["local"] for item in entry["object_locals"]}

    # the build report: a local compared by identity stays a Python object; the others do not
    for cell in ("reassigned_local", "conditional_local", "single_assignment_after_guard"):
        assert "adjusted" not in object_locals(cell), cell
    for cell in ("python_context_local", "prefix_identity_local"):
        assert "adjusted" in object_locals(cell), cell
    run_python(integer_domain_build, "-c", textwrap.dedent("""\
        from exported import mx_model as exported
        from exported_cy import mx_model as compiled

        maximum, minimum = (1 << 63) - 1, -(1 << 63)
        for amount, divisor in ((minimum + 1, -1), (maximum - 1, 1)):
            reference = type(exported)().Outer.Projection
            projection = type(compiled)().Outer.Projection
            reference.amount = projection.amount = amount
            reference.divisor = projection.divisor = divisor
            assert reference.reassigned_local() == projection.reassigned_local() == amount
            assert reference.conditional_local() == projection.conditional_local()
            assert reference.single_assignment_after_guard() == projection.single_assignment_after_guard()
            assert reference.prefix_identity_local() == projection.prefix_identity_local() == amount + 1

        for cell, amount, divisor in (("reassigned_local", minimum, -1),
                                      ("conditional_local", minimum, -1),
                                      ("conditional_local", maximum, 1),
                                      ("single_assignment_after_guard", maximum, 1)):
            projection = type(compiled)().Outer.Projection
            projection.amount, projection.divisor = amount, divisor
            try:
                getattr(projection, cell)()
            except OverflowError as error:
                assert f"Outer.Projection.{cell}" in str(error)
            else:
                raise AssertionError(f"{cell} accepted an overflowing intermediate")

        projection = type(compiled)().Outer.Projection
        projection.divisor, projection.payload = 1, {"value": 3}
        assert projection.mixed_local() == 3
        projection = type(compiled)().Outer.Projection
        projection.divisor, projection.payload = 1, {"value": 3.5}
        try:
            projection.mixed_local()
        except TypeError as error:
            assert "Outer.Projection.mixed_local" in str(error)
        else:
            raise AssertionError("mixed local truncated an object result")
        """))


def test_terminal_product_checks_the_final_signed_result(integer_domain_build):
    run_python(integer_domain_build, "-c", textwrap.dedent("""\
        from exported import mx_model as exported
        from exported_cy import mx_model as compiled

        minimum = -(1 << 63)
        for name in ('terminal_product', 'reordered_product'):
            # (0, minimum): the intermediate divisor - 1 leaves the range although the product
            # is 0; the exact terminal arithmetic is gone (owner's rule), it raises naming the Cell
            projection = type(compiled)().Outer.Projection
            projection.amount, projection.divisor, projection.reads = 0, minimum, []
            try:
                getattr(projection, name)()
            except OverflowError as error:
                assert f"Outer.Projection.{name}" in str(error)
            else:
                raise AssertionError(f"{name}: intermediate beyond the signed range did not raise")
            for amount, divisor in ((2, 3), (1, minimum + 1)):
                reference = type(exported)().Outer.Projection
                projection = type(compiled)().Outer.Projection
                reference.amount = projection.amount = amount
                reference.divisor = projection.divisor = divisor
                reference.reads, projection.reads = [], []
                assert getattr(projection, name)() == getattr(reference, name)() == amount * (divisor - 1)
                assert projection.reads == reference.reads == [divisor]

            projection = type(compiled)().Outer.Projection
            projection.amount, projection.divisor, projection.reads = 1, minimum, []
            try:
                getattr(projection, name)()
            except OverflowError as error:
                assert f"Outer.Projection.{name}" in str(error)
            else:
                raise AssertionError(f"{name} accepted a result below the signed range")
            assert projection.reads == [minimum]
        """))


@pytest.mark.parametrize("cell, amount, other, expected", [
    ("left_shift", 1 << 62, 1, "overflow"),
    ("left_shift", 30, -1, "negative shift"),
    ("left_shift", -3, 2, -12),
    ("left_shift", -1, 63, -(1 << 63)),
    ("left_shift", 0, 64, 0),
    ("left_shift", 1, 64, "overflow"),
    ("right_shift", -30, -1, "negative shift"),
    ("right_shift", -(1 << 63), 1, -(1 << 62)),
    ("right_shift", -30, 64, -1),
    ("right_shift", 30, 64, 0),
    ("shift_argument", 30, 1, {"count": 61}),
    ("shift_argument", 1 << 62, 2, "overflow"),
    ("remainder", -(1 << 63), -1, 0),
    ("remainder", -30, 7, 5),
    ("remainder", 30, 0, "division by zero"),
])
def test_native_signed_shift_and_remainder(integer_domain_build, cell, amount, other, expected):
    run_python(integer_domain_build, "-c", textwrap.dedent(f"""\
        from exported_cy import mx_model

        projection = mx_model.Outer.Projection
        projection.amount = {amount}
        projection.shift = projection.divisor = {other}
        cell = {cell!r}
        expected = {expected!r}
        try:
            actual = getattr(projection, cell)()
        except OverflowError as error:
            assert expected == "overflow", str(error)
            name = "inline_receive" if cell == "shift_argument" else cell
            assert f"Outer.Projection.{{name}}" in str(error)
        except ValueError as error:
            assert expected == "negative shift" and str(error) == "negative shift count"
        except ZeroDivisionError:
            assert expected == "division by zero"
        else:
            assert actual == expected, (cell, actual, expected)
        """))


def test_unchecked_mode_arithmetic_is_plain_c(integer_domain_build_unchecked):
    """Unchecked mode (owner's rule): the overflowing arithmetic of the checked-mode tests above is
    plain C (no check, no exception, as on upstream); values inside the range, the conversions at
    the Python boundary and the shift and division rules stay as in the checked mode."""
    run_python(integer_domain_build_unchecked, "-c", textwrap.dedent("""\
        import json
        from exported import mx_model as exported
        from exported_cy import mx_model as compiled

        report = json.load(open("exported_cy/_mx_integer_checks.json"))
        assert report["mode"] == "unchecked" and "not traced" in report["limit"]
        maximum, minimum = (1 << 63) - 1, -(1 << 63)
        for cell, overflow, valid in (("increment_amount", maximum, maximum - 1),
                                      ("decrement_amount", minimum, minimum + 1),
                                      ("square_amount", 3037000500, 1 << 31),
                                      ("negate_amount", minimum, minimum + 1),
                                      ("local_amount", minimum + 1, 5)):
            reference = type(exported)().Outer.Projection
            projection = type(compiled)().Outer.Projection
            reference.amount = projection.amount = valid
            assert getattr(projection, cell)() == getattr(reference, cell)(), cell
            projection = type(compiled)().Outer.Projection
            projection.amount = overflow
            getattr(projection, cell)()          # plain C: returns, whatever C gives
        projection = type(compiled)().Outer.Projection
        projection.amount = 1 << 62
        projection.inline_operand()              # amount * 4 in plain C, then a checked argument
        projection = type(compiled)().Outer.Projection
        projection.amount, projection.divisor = minimum, -1
        projection.reassigned_local()
        # boundary conversions stay checked and named
        projection = type(compiled)().Outer.Projection
        projection.payload = {"value": 1 << 70}
        try:
            projection.object_result()
        except OverflowError as error:
            assert "Outer.Projection.object_result" in str(error)
        else:
            raise AssertionError("unchecked mode skipped a boundary conversion")
        # Python's shift and division rules: right shift helper, division check
        projection = type(compiled)().Outer.Projection
        projection.amount, projection.shift = -(1 << 63), 1
        assert projection.right_shift() == -(1 << 62)
        projection.amount, projection.divisor = 30, 0
        try:
            projection.remainder()
        except ZeroDivisionError:
            pass
        else:
            raise AssertionError("remainder by zero")
        """))
