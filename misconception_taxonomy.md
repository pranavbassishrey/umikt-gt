# Misconception Taxonomy — Programming Fundamentals

**Domain:** Programming Fundamentals / Data Structures  
**Version:** 1.0  
**Last updated:** 2026-09

## Purpose

This taxonomy defines the set of misconception categories used by the UMiKT-GAT misconception head and the LLM semantic extraction component. It is domain-specific, hand-authored, and used to label both the synthetic training data and the hand-written Q&A evaluation set.

**Scope limitation:** This taxonomy is intentionally small and domain-scoped. It is NOT claimed to be a universal or validated educational taxonomy. It is designed for feasibility of the research pipeline and should be validated by a domain expert before use in any real educational deployment.

---

## Taxonomy Classes

The misconception label is a categorical variable with 7 possible values:

| Class ID | Label | Description |
|---|---|---|
| 0 | `none` | No misconception detected; error (if any) is due to knowledge gap, carelessness, or insufficient evidence. |
| 1 | `syntax_misunderstanding` | Student misunderstands language syntax rules (e.g., confuses `=` and `==`, misplaces brackets, or uses invalid token sequences). |
| 2 | `variable_scope_misunderstanding` | Student does not correctly model variable scope or lifetime (e.g., assumes a local variable is globally accessible, or reuses a variable name believing it persists). |
| 3 | `loop_termination_misunderstanding` | Student incorrectly reasons about when a loop terminates (e.g., off-by-one errors in loop bounds, infinite loop due to missing update, confusion about while-loop vs for-loop semantics). |
| 4 | `function_parameter_misunderstanding` | Student confuses formal and actual parameters, pass-by-value vs pass-by-reference, or misunderstands how return values propagate. |
| 5 | `recursion_base_case_misunderstanding` | Student fails to correctly identify or implement the base case in a recursive function, or misunderstands the call-stack mechanism (e.g., believes a recursive call modifies the same variable in-place). |
| 6 | `reference_vs_value_misunderstanding` | Student conflates reference semantics and value semantics (e.g., believes copying a list variable makes an independent copy, or modifies a passed list without understanding in-place mutation). |

---

## Concept-to-Misconception Affinity Map

This table shows which misconceptions are most commonly associated with which concept nodes in the concept graph. This is used to weight misconception detection by concept context.

| Concept | Primary Misconceptions | Secondary Misconceptions |
|---|---|---|
| Variables (0) | `syntax_misunderstanding`, `variable_scope_misunderstanding` | `reference_vs_value_misunderstanding` |
| Conditions (1) | `syntax_misunderstanding` | `variable_scope_misunderstanding` |
| Loops (2) | `loop_termination_misunderstanding`, `syntax_misunderstanding` | `variable_scope_misunderstanding` |
| Functions (3) | `function_parameter_misunderstanding`, `variable_scope_misunderstanding` | `reference_vs_value_misunderstanding` |
| Recursion (4) | `recursion_base_case_misunderstanding`, `function_parameter_misunderstanding` | `loop_termination_misunderstanding` |
| Data Structures (5) | `reference_vs_value_misunderstanding` | `loop_termination_misunderstanding` |
| Algorithms (6) | `loop_termination_misunderstanding`, `reference_vs_value_misunderstanding` | `recursion_base_case_misunderstanding` |

---

## Canonical Examples

### Class 1: `syntax_misunderstanding`

**Question:** What is the output of the following code?
```python
x = 5
if x = 5:
    print("yes")
```
**Canonical wrong answer:** "The output is yes" (student does not recognize the syntax error with `=` in a conditional).  
**Correct answer:** "This is a SyntaxError; `=` is assignment, `==` is comparison."

---

### Class 2: `variable_scope_misunderstanding`

**Question:** After calling `increment()`, what is the value of `x`?
```python
x = 10
def increment():
    x = x + 1
increment()
print(x)
```
**Canonical wrong answer:** "x is 11" (student believes the local assignment modifies the outer variable).  
**Correct answer:** "This raises an UnboundLocalError inside `increment()` because `x` is treated as local; the outer `x` remains 10."

---

### Class 3: `loop_termination_misunderstanding`

**Question:** How many times does this loop print?
```python
for i in range(5):
    print(i)
```
**Canonical wrong answer:** "5 times, printing 1 2 3 4 5" (off-by-one: student does not know range starts at 0).  
**Correct answer:** "5 times, printing 0 1 2 3 4."

---

### Class 4: `function_parameter_misunderstanding`

**Question:** What is the output?
```python
def double(n):
    n = n * 2
x = 5
double(x)
print(x)
```
**Canonical wrong answer:** "10" (student believes the function modifies the caller's variable).  
**Correct answer:** "5 — Python passes integers by value; `n` inside `double` is a local copy."

---

### Class 5: `recursion_base_case_misunderstanding`

**Question:** What is wrong with this recursive function?
```python
def countdown(n):
    print(n)
    countdown(n - 1)
```
**Canonical wrong answer:** "Nothing, it counts down from n to 0" (student does not see the missing base case and infinite recursion).  
**Correct answer:** "Missing base case — the function will recurse infinitely and cause a RecursionError."

---

### Class 6: `reference_vs_value_misunderstanding`

**Question:** What is the output?
```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)
```
**Canonical wrong answer:** "[1, 2, 3]" (student believes `b = a` copies the list).  
**Correct answer:** "[1, 2, 3, 4] — both `a` and `b` reference the same list object."

---

## Limitations

1. **Hand-authored, not validated**: This taxonomy has not been validated by educational experts or compared against existing computer-science misconception literature (e.g., Sorva 2013, Pea 1986). It should be treated as a domain-specific proxy for research purposes only.
2. **Not exhaustive**: Many misconceptions in programming (e.g., floating-point precision, concurrency, memory management) are not covered.
3. **Single-label assumption**: The model uses single-label classification per interaction. Real student responses may exhibit multiple simultaneous misconceptions. This is a simplification.
4. **Label noise**: Both the synthetic data and LLM extraction labels have known noise. Results should be interpreted accordingly.
